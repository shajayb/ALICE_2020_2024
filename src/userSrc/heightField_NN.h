#ifndef _HEIGHT_FIELD_NN_
#define _HEIGHT_FIELD_NN_


#include <vector>
#include <algorithm>
#include <cmath>
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;

#include "scalarField.h"
#include "genericMLP.h"


float radius = 2.0f; // =  SDF loss ( blendedCircle radius)

struct Pose2D
{
    zVector c;   // center (x, y, 0)
    zVector v;   // 2D vector (vx, vy, 0)
};

zVector gradientAt(zVector& p,  std::vector<zVector>& poly)
{
    float minDist = 1e6f;
    int n = poly.size();
    zVector grad(0, 0, 0);

    for (int i = 0; i < n; ++i)
    {
        zVector a = poly[i];
        zVector b = poly[(i + 1) % n];

        // project p onto edge ab
        zVector ab = b - a;
        zVector ap = p - a;

        float t = std::max(0.0f, std::min(1.0f, ap * ab / (ab * ab)));
        zVector proj = a + ab * t;

        float d = p.distanceTo(proj);
        if (d < minDist)
        {
            minDist = d;
            grad = (proj - p);
            grad.normalize();
        }
    }

    // Signed: negative if outside
    return grad;
}

float signedDistanceToPolygon(zVector& p,  std::vector<zVector>& poly)
{
    float minDist = 1e6f;
    int n = poly.size();

    for (int i = 0; i < n; ++i)
    {
        zVector a = poly[i];
        zVector b = poly[(i + 1) % n];

        // project p onto edge ab
        zVector ab = b - a;
        zVector ap = p - a;

        float t = std::max(0.0f, std::min(1.0f, ap * ab / (ab * ab)));
        zVector proj = a + ab * t;

        float d = p.distanceTo(proj);
        if (d < minDist) minDist = d;
    }

    // Signed: negative if outside
    return pointInsidePolygon(p, poly) ? -minDist : minDist;
}

// Evaluates the SDF of the given polygon
float evalPolygonSDF(zVector& p, std::vector<zVector>& poly)
{
    return signedDistanceToPolygon(p, poly);
}

// Computes blended SDF from all circles defined by pose centers
float evalBlendedCircleSDF(zVector& p,  std::vector<Pose2D>& poses, float radius)
{
    float sdf = 1e6f;


    for (auto& pose : poses)
    {
        zVector temp = pose.c;

        float d = p.distanceTo(temp) - radius;
        sdf = smin(sdf, d, 3); // union of circles
    }



    return sdf;
}

// ------------------------------------------------------------
//  Oriented Rectangle SDF (same length & width for all boxes)
// ------------------------------------------------------------
inline float sdfOrientedRectangle
(
    zVector& p,  zVector& center,
     zVector& axis, float halfLength,
    float halfWidth
)
{
    // Build orthonormal basis (u = axis direction, v = perpendicular)
    zVector u = axis;
    u.normalize();

    zVector v = zVector(-u.y, u.x, 0);   // 90° rotation in XY

    // Local point coordinates
    zVector d = p - center;

    float x = d * u;
    float y = d * v;

    // Rectangle half extents
    zVector q(fabs(x), fabs(y), 0);

    // Standard rectangle SDF
    zVector ext(halfLength, halfWidth, 0);

    zVector q_minus_e(q.x - ext.x, q.y - ext.y, 0);

    float outsideDist = zVector(
        std::max(q_minus_e.x, 0.0f),
        std::max(q_minus_e.y, 0.0f),
        0.0f
    ).length();

    float insideDist = std::min(std::max(q_minus_e.x, q_minus_e.y), 0.0f);

    return outsideDist + insideDist;
}

// ------------------------------------------------------------
//  Blended SDF of multiple oriented rectangles
//  centers[i] : zVector
//  poses[i].v : orientation vector
// ------------------------------------------------------------
inline float evalBlendedOrientedRectSDF
(
    zVector& p,
    std::vector<Pose2D>& poses,
    float halfLength = 3,
    float halfWidth = 1.5,
    float k = 0.25f       // blending softness
)
{
    float d = 1e6f;

    int n = poses.size();

    for (int i = 0; i < n; i++)
    {
        float di = sdfOrientedRectangle(p, poses[i].c, poses[i].v,
            halfLength, halfWidth);

        d = min(d, di);// smin(d, di, k);
    }

    return d;
}

inline zVector gradientAT_BlendOrientedRectSDF
(
    zVector& p,
    std::vector<Pose2D>& poses,
    float halfLength = 3,
    float halfWidth = 1.5
)
{
    float bestD = 1e6f;
    int bestIdx = -1;
    float eps = 1e-4f; // Small perturbation for numerical gradient

    // 1. Find the winning SDF primitive index
    for (int i = 0; i < poses.size(); i++)
    {
        float di = sdfOrientedRectangle(p, poses[i].c, poses[i].v, halfLength, halfWidth);
        if (di < bestD)
        {
            bestD = di;
            bestIdx = i;
        }
    }

    if (bestIdx == -1) return zVector(0, 0, 0);

    // Get the parameters of the winning primitive
    zVector& c = poses[bestIdx].c;
    zVector& v = poses[bestIdx].v;

    // 2. Compute Numerical Gradient (Central Difference)
    zVector finalGrad;
    zVector p_plus;
    zVector p_minus;

    // --- X component (∂D/∂x)
    p_plus = p; p_plus.x += eps;
    p_minus = p; p_minus.x -= eps;
    float D_plus_x = sdfOrientedRectangle(p_plus, c, v, halfLength, halfWidth);
    float D_minus_x = sdfOrientedRectangle(p_minus, c, v, halfLength, halfWidth);
    finalGrad.x = (D_plus_x - D_minus_x) / (2.0f * eps);

    // --- Y component (∂D/∂y)
    p_plus = p; p_plus.y += eps;
    p_minus = p; p_minus.y -= eps;
    float D_plus_y = sdfOrientedRectangle(p_plus, c, v, halfLength, halfWidth);
    float D_minus_y = sdfOrientedRectangle(p_minus, c, v, halfLength, halfWidth);
    finalGrad.y = (D_plus_y - D_minus_y) / (2.0f * eps);

    // Gradient of SDF is the surface normal. It should be normalized.
    finalGrad.normalize();

    return finalGrad;
}


zVector centroid(vector<zVector> &poly)
{
    zVector centroid;
    for (auto& p : poly)centroid += p;

    centroid /= poly.size();
    return centroid;
}


struct sdfSamples
{
    zVector pt;
    float val;
};

class heightfieldNN : public MLP
{
public:
    int n = 4;
    vector< vector<zVector>> polygons;
    vector<sdfSamples> sdfSamplePoints;
    zVector sdfSample_centroid;

    std::vector<Pose2D> poses;
    vector<float> output, input;
    double loss;
    float sdfLoss = 0.0f;
    float orientationLoss = 0.0f;
    double o_weight;
    bool o_flip_dir;

    HeightField2D *correspondingHeightField;

    heightfieldNN() {}

    heightfieldNN(int _n)
    {
        n = _n;
        initialize( n, { 32,4,32 }, 4 * n); // dummy input = 1; output = n × (center + dir)

        input.assign(inputDim, 0);
        output = forward(input);
    }

    // ------------------
    void generateSamplesInRange(HeightField2D& htField, float minZ = 18.0f, float maxZ = 20.0f)
    {

        // check if htField.rescaleInRange had -ve range prior t rescaling, if so, use {-1,1} below
        minZ = ofMap(minZ, htField.MLS_zMin, htField.MLS_zMax, 0, 1);
        maxZ = ofMap(maxZ, htField.MLS_zMin, htField.MLS_zMax, 0, 1);

        sdfSamplePoints.clear();

        sdfSample_centroid = zVector(0, 0, 0);
        for (int i = 0; i < SF_RES; i += 2)
        {
            for (int j = 0; j < SF_RES; j += 2)
            {
                float val = htField.field[i][j];

                // Optional: Only if the Z component stores elevation
                //if (val >= minZ && val <= maxZ)
                {
                    sdfSamples sample;
                    sample.pt = htField.gridPoints[i][j];
                    sample.val = (val >= minZ && val <= maxZ) ? -val : +val;
                    sdfSamplePoints.push_back(sample);
                    //
                    sdfSample_centroid += sample.pt;
                }
            }
        }
        sdfSample_centroid /= sdfSamplePoints.size();
    }

    void set_field_values_from_polygon(vector<zVector>& polygon, HeightField2D& htField)
    {

        for (int i = 0; i < SF_RES; ++i)
        {
            for (int j = 0; j < SF_RES; ++j)
            {
                zVector pt = htField.gridPoints[i][j];

                if (pointInsidePolygon(pt, polygon))
                    htField.field[i][j] = evalPolygonSDF(pt, polygon);

            }
        }

        htField.normalise();
        htField.rescaleFieldToRange(-1, 1);
    }

    void generateSDFSamplePointsFromPolygons()
    {
        sdfSamplePoints.clear();

        if (polygons.empty()) return;

        // Compute bounding box of polygon
        zVector bmin(1e6, 1e6, 1e6);
        zVector bmax = bmin * -1;

        for( auto &polygon : polygons)
        {

            for (auto& p : polygon)
            {
                bmin = zMin(bmin, p);
                bmax = zMax(bmax, p);
            }
        }

            int gridResX = 50;
            int gridResY = 50;

            for (int i = 0; i < gridResX; i += 1)
            {
                for (int j = 0; j < gridResY; j += 1)
                {
                    float u = (float)i / (gridResX - 1);
                    float v = (float)j / (gridResY - 1);

                    zVector pt;
                    pt.x = zLerp(bmin.x, bmax.x, u);
                    pt.y = zLerp(bmin.y, bmax.y, v);
                    pt.z = 0;

                    for (auto& poly : polygons)
                    if (pointInsidePolygon(pt, poly))
                    {
                        sdfSamples sample;
                        sample.pt = pt;
                        sample.val = evalPolygonSDF(pt, poly);
                        sdfSamplePoints.push_back(sample);
                        sdfSample_centroid += sample.pt;
                    }
                }
            }

            sdfSample_centroid /= sdfSamplePoints.size();

            if (!pointInsidePolygon(sdfSample_centroid, polygons[0]))
            {
                float mn = 1e6;
                for (auto& s : sdfSamplePoints)
                {
                    if (s.val < mn)
                    {
                        mn = s.val;
                        sdfSample_centroid = s.pt;

                        //if (s.val < 1e-1)break;
                    }

                }
            }

            printf("Sample points generated: %zu\n", sdfSamplePoints.size());
        }

    void setTargetPolygons( vector< vector<zVector>> & polys)
    {
        polygons = polys;
    }
    // ------------------

    void computePolygonBBox( std::vector<zVector>& polygon, zVector& bmin, zVector& bmax)
    {
        if (polygon.empty()) return;

        bmin = zVector(1e6, 1e6, 0);
        bmax = zVector(-1e6, -1e6, 0);

        for ( auto& p : polygon)
        {
            bmin.x = std::min(bmin.x, p.x);
            bmin.y = std::min(bmin.y, p.y);
            bmax.x = std::max(bmax.x, p.x);
            bmax.y = std::max(bmax.y, p.y);
        }
    }

    void extractPoses(std::vector<float>& output, std::vector<Pose2D>& poses, bool rawCenter = false)
    {
        poses.resize(n);
        if (polygons.empty()) return;

        // --- Compute polygon bounding box (target range)
        zVector bmin(1e6, 1e6, 0);
        zVector bmax(-1e6, -1e6, 0);
        for (auto& poly : polygons)
            for ( auto& p : poly)
            {
                bmin.x = std::min(bmin.x, p.x);
                bmin.y = std::min(bmin.y, p.y);
                bmax.x = std::max(bmax.x, p.x);
                bmax.y = std::max(bmax.y, p.y);
            }

        /* bmin *= 0.5;
         bmax *= 0.5;*/
         // --- Collect raw centers from network outputs
        std::vector<zVector> rawCenters;
        rawCenters.reserve(n);
        for (int i = 0; i < n; ++i)
        {
            rawCenters.push_back(zVector(output[i * 4 + 0], output[i * 4 + 1], 0));
        }

        // --- Compute input range of raw centers
        zVector inMin(1e6, 1e6, 0);
        zVector inMax(-1e6, -1e6, 0);
        for ( auto& c : rawCenters)
        {
            inMin.x = std::min(inMin.x, c.x);
            inMin.y = std::min(inMin.y, c.y);
            inMax.x = std::max(inMax.x, c.x);
            inMax.y = std::max(inMax.y, c.y);
        }

        // --- Avoid degenerate ranges
        float rangeX = std::max(1e-6f, inMax.x - inMin.x);
        float rangeY = std::max(1e-6f, inMax.y - inMin.y);

        // --- Remap raw centers to polygon bounding box
        for (int i = 0; i < n; ++i)
        {
             zVector& raw = rawCenters[i];

            float u = (raw.x - inMin.x) / rangeX;
            float v = (raw.y - inMin.y) / rangeY;

            zVector mapped;
            mapped.x = bmin.x + u * (bmax.x - bmin.x);
            mapped.y = bmin.y + v * (bmax.y - bmin.y);
            mapped.z = 0;

            poses[i].c = rawCenter ? raw : mapped;
            
            // cycle through polygon and add their centroids;.. so that the startign points are distributed;
            int poly_idx = i % polygons.size();
            poses[i].c += centroid(polygons[poly_idx]);;


            zVector rawDir(output[i * 4 + 2], output[i * 4 + 3], 0);

            if (rawDir.length2() < 1e-6f)rawDir = zVector(1, 0, 0);

            poses[i].v = rawDir;// gradientAt(poses[i].c, polygons[0]);
            poses[i].v.normalize();
        }
    }

    /// ----------------
    /*float computeLoss(std::vector<float>& y_pred, std::vector<float>& y_dummy) override
    {
        return coverageLoss(y_pred);
    }*/

    float computeLoss(std::vector<float>& y_pred, std::vector<float>& y_dummy) override
    {
        return hybridCoverageLoss(y_pred);
    }

    float hybridCoverageLoss(std::vector<float>& output)
    {
        if (polygons.empty()) return 1e6f;

        poses.clear();
        extractPoses(output, poses, true);

        // -----------------------------
        // 1. INITIALISE LOSS TERMS
        // -----------------------------
        sdfLoss = 0.0;
        orientationLoss = 0.0;

        const int N = sdfSamplePoints.size();
        if (N == 0) return 1e6f;

        // -----------------------------
        // 2. EULERIAN COVERAGE LOSS  (SDF reconstruction)
        // -----------------------------
        for (auto& sample : sdfSamplePoints)
        {
            float sdfTarget = sample.val;
            float sdfPred = evalBlendedCircleSDF(sample.pt, poses, radius);
            float diff = sdfTarget - sdfPred;

            if (!isnan(diff))
                sdfLoss +=  diff* diff;
        }
        sdfLoss /= (float)N;

        // -----------------------------
        // 3. LAGRANGIAN ORIENTATION LOSS  (pose alignment)
        // -----------------------------
        for (auto& pose : poses)
        {
            // Target direction: gradient of polygon SDF
            zVector targetDir1, targetDir2;
            

            zVector gradAt = pose.c;
            gradAt.x += 2;// gradient explodes at the center of the rectangle;

            targetDir1 = pose.c - zVector(-35, 50, 0);// gradientAt(pose.c, polygons[0]);// should gradient of corresponding polygon
            targetDir2 = correspondingHeightField->gradientAt(pose.c);// gradientAT_BlendOrientedRectSDF(gradAt, poses, 12 * 0.5 * 0.5, 5.5 * 0.5 * 0.5);
            targetDir1 = targetDir2 ^ zVector(0, 0, o_flip_dir ? -1: 1); // tangent

            zVector targetDir =  targetDir1* o_weight + targetDir2 * (1 - o_weight); ;
            targetDir.normalize();

            // Predicted direction: network output
            zVector predDir = pose.v;
            predDir.normalize();

            float alignment = predDir * targetDir;     // cos(theta)
            float angular_error = 1.0f - fabs(alignment);
            orientationLoss += angular_error * angular_error;
        }
        orientationLoss /= (float)poses.size();

        // -----------------------------
        // 4. COMBINE (STABLE SCALING)
        // -----------------------------
        // Log compression to dampen large residuals but preserve gradient direction
        // log1p(sdfLoss) subdues the gradients; 
        // initially we need strong gradient signal from the coverage loss to push the poses out of their starting locations.
        float Lsdf = (sdfLoss);//
        float Lori = log1p(orientationLoss);

        // Fixed, empirically robust weights
        const float λE = 1.0f;   // field coverage
        const float λL = 0.1f;   // orientation (usually smaller gradient magnitude)

        float totalLoss = λE * Lsdf + λL * Lori;

        // -----------------------------
        // 5. STORE FOR VISUALISATION
        // -----------------------------
        sdfLoss = Lsdf;
        orientationLoss = Lori;
        loss = totalLoss;

        printf("Loss[E=%.4f, L=%.4f] → Total=%.4f\n", Lsdf, Lori, totalLoss);

        return totalLoss;
    }

    float coverageLoss(std::vector<float>& output)
    {
        if (polygons.empty()) return 1e6f;

        //std::vector<Pose2D> poses;
        poses.clear();
        extractPoses(output, poses, true);// for Term 0, raw centers work better.

        //visualiseBlendedSDFs(poses);
        // -- Term 0: SDF field mismatch at fixed sample points : a coverage objective function
        sdfLoss = 0.0f;
        orientationLoss = 0.0f;
        
        if (!sdfSamplePoints.empty())
        {
            for (auto& sample : sdfSamplePoints)
            {
                float sdfTarget = sample.val;
                float sdfPred = evalBlendedCircleSDF(sample.pt, poses, radius); ;// evalBlendedOrientedRectSDF(sample.pt, poses);// evalBlendedCircleSDF(sample.pt, poses, radius);

                float diff = sdfTarget - sdfPred;
                sdfLoss += isnan(diff) ? 0 : (diff * diff);

            }

            sdfLoss /= sdfSamplePoints.size(); // average

        }

        //
        for (auto& sample : sdfSamplePoints)
        {

            zVector targetDir = zVector(-50, 50, 0);
            zVector predDir = gradientAT_BlendOrientedRectSDF(sample.pt, poses);

            //printf(" %.4f,%.4f,%.4f \n", predDir.x, predDir.y, predDir.z);

            predDir = predDir ^ zVector(0, 0, 1);
            targetDir.normalize();
            predDir.normalize();

            float alignment = std::fabs(predDir * targetDir);

            // 4. Use Angular Error Loss (Minimizing 1 - |cos(theta)|)
            // Loss is minimized when alignment approaches 1.0 (angle approaches 0° or 180°).
            float angular_error = 1.0f - alignment;
            orientationLoss += angular_error * angular_error; // Squared error for stab
                
        }
        orientationLoss /= sdfSamplePoints.size(); // average
        orientationLoss *= 10;
        if(isnan(orientationLoss))orientationLoss = 0;

        loss = sdfLoss + orientationLoss * 5;
        return loss;
    }

    void computeGradient(std::vector<float>& x, std::vector<float>& y_dummy, std::vector<float>& gradOut) override
    {
        // 1) Single forward to get the baseline outputs
        std::vector<float> y0 = forward(x);

        // 2) Central-difference step. 
        //    Because you scale centers by 100 in extractPoses(), use a slightly larger eps.
         float eps = 1e-1f;

        gradOut.assign(outputDim, 0.0f);

        // 3) Compute baseline loss once
        float baseLoss = computeLoss(y0, y_dummy);

        //for (size_t i = 0; i < y0.size(); ++i)
        //{
        //    if (std::isnan(y0[i]))
        //    {
        //        printf("NAN detected in network output y0 at index %zu\n", i);
        //        // Stop execution here to investigate
        //    }
        //}

        //for (float v : y0) printf(" y0  %.4f \n ", v);

        // 4) For each output dimension, perturb that output value and re-evaluate loss
        for (int i = 0; i < outputDim; ++i)
        {
            // Positive perturbation
            std::vector<float> y_plus = y0;
            y_plus[i] += eps;
            float L_plus = computeLoss(y_plus, y_dummy);

            // Negative perturbation
            std::vector<float> y_minus = y0;
            y_minus[i] -= eps;
            float L_minus = computeLoss(y_minus, y_dummy);

            // Central difference derivative dL/dy_i
            gradOut[i] = (L_plus - L_minus) / (2.0f * eps);

            // printf("%.4f,%.4f \n", L_plus, baseLoss);
        }

        // Note:
        // - Only the center coords (indices 4*k + 0 and 4*k + 1) affect your current loss,
        //   so gradients for the orientation slots (4*k + 2, 4*k + 3) will be ~0 — this is expected.
        // - Call backward(gradOut, learningRate) after this to update weights.

        /*printf("GRAD: [");
            for (float v : gradOut) printf("  %.4f \n ", v);
        printf("]\n");*/

    }


    ///


    void drawCoverageSamples()
    {
        for (auto& sample : sdfSamplePoints)
        {
            drawPoint(zVecToAliceVec(sample.pt));
        }
    }

    void draw_output_and_loss()
    {


        glPointSize(5);
        glColor3f(0, 0, 0);
            //for (auto& pose : poses)
            //{
            //    drawPoint(zVecToAliceVec(pose.c));
            //    // drawLine(zVecToAliceVec(pose.c), zVecToAliceVec(pose.c + pose.v * 5.0));

            //    // ---------- predicted dir in red
            //    zVector dir = pose.v;
            //    dir.normalize();
            //    glColor3f(0, 0, 0);
            //    drawLine(zVecToAliceVec(pose.c), zVecToAliceVec(pose.c + dir * 2.0));
            //    drawCircle(zVecToAliceVec(pose.c), radius, 32);

            //    // ---------- gradient in red
            //    glColor3f(1, 0, 0);
            //        dir = gradientAT_BlendOrientedRectSDF(pose.c + zVector(1,0,0), poses);
            //        dir.normalize();
            //
            //    drawLine(zVecToAliceVec(pose.c), zVecToAliceVec(pose.c + dir * 2.0));

            //    // ---------- view Dir in blue
            //    dir = zVector(-50, 50, 0);// target Dir << check in coverage loss
            //    dir.normalize();
            //    glColor3f(0, 0, 1);
            //    drawLine(zVecToAliceVec(pose.c), zVecToAliceVec(pose.c + dir * 2.0));
            //}
        glPointSize(1);

        //
        setup2d();

            char s[200];
            sprintf(s, "%.4f", loss);
            drawText(string(s), 50, 450);

            sprintf(s, "coverage constraint %.2f", sdfLoss);
            drawText(string(s), 50, 470);

            sprintf(s, "orientation constraint %.2f", orientationLoss);
            drawText(string(s), 50, 490);

        restore3d();

    }

    void drawCoveragePolygon()
    {
        if (polygons.empty()) return;

        glColor3f(0.1f, 0.1f, 0.1f);
        glLineWidth(2.0f);

       
        for (auto & poly : polygons)
        {
            glBegin(GL_LINE_LOOP);
            for (zVector& pt : poly)
            {
                glVertex3f(pt.x, pt.y, pt.z);
            }
            glEnd();
        }
        

        glLineWidth(1.0f);

        if (sdfSamplePoints.empty())return;

        /*for (auto& sample : sdfSamplePoints)
            drawPoint(zVecToAliceVec(sample.pt));*/

        //     --- SDF Sample Points: Jet Color Visualization
        //{
        //    // Compute min-max val
        //    float vmin = 1e6;
        //    float vmax = -vmin;

        //    for (auto& s : sdfSamplePoints)
        //    {
        //        float val = evalBlendedOrientedRectSDF(s.pt, poses);// evalBlendedCircleSDF(s.pt, poses, radius);
        //        vmin = std::min(vmin, val );
        //        vmax = std::max(vmax, val);
        //    }


        //    glPointSize(1);
        //    for (auto& s : sdfSamplePoints)
        //    {

        //        float r, g, b;
        //        float val = evalBlendedOrientedRectSDF(s.pt, poses); // evalBlendedCircleSDF(s.pt, poses, radius);
        //        getJetColor(ofMap(val,vmin,vmax,-1,1), r, g, b); // map to [-1,1] before jetColor
        //        
        //        glColor3f(r, g, b);
        //        drawPoint(zVecToAliceVec(s.pt));
        //    }
        //    glPointSize(1);
        //    glColor3f(0, 0, 0);

        //}

    }

};


#endif // !_HEIGHT_FIELD_NN_
