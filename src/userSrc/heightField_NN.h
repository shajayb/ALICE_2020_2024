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


float radius = 1.0f; // =  SDF loss ( blendedCircle radius)

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

struct sdfSamples
{
    zVector pt;
    float val;
};

class heightfieldNN : public MLP
{
public:
    int n = 4;
    std::vector<zVector> polygon;
    std::vector<sdfSamples> sdfSamplePoints;
    zVector sdfSample_centroid;

    heightfieldNN() {}

    heightfieldNN(int _n)
    {
        n = _n;
        initialize(2 * n, { 16 }, 4 * n); // dummy input = 1; output = n × (center + dir)
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

    void generateSDFSamplePointsFromPolygon()
    {
        sdfSamplePoints.clear();

        if (polygon.empty()) return;

        // Compute bounding box of polygon
        zVector bmin = polygon[0];
        zVector bmax = polygon[0];

        for (auto& p : polygon)
        {
            bmin = zMin(bmin, p);
            bmax = zMax(bmax, p);
        }

        int gridResX = 50;
        int gridResY = 50;

        for (int i = 0; i < gridResX; i+= 1)
        {
            for (int j = 0; j < gridResY; j+=1)
            {
                float u = (float)i / (gridResX - 1);
                float v = (float)j / (gridResY - 1);

                zVector pt;
                pt.x = zLerp(bmin.x, bmax.x, u);
                pt.y = zLerp(bmin.y, bmax.y, v);
                pt.z = 0;

                if (pointInsidePolygon(pt, polygon))
                {
                    sdfSamples sample;
                    sample.pt = pt;
                    sample.val = evalPolygonSDF(pt, polygon);
                    sdfSamplePoints.push_back(sample);
                    sdfSample_centroid += sample.pt;
                }
            }
        }

        sdfSample_centroid /= sdfSamplePoints.size();

        if (!pointInsidePolygon(sdfSample_centroid, polygon))
        {
            float mn = 1e6;
            for (auto& s : sdfSamplePoints)
            {
                if (s.val < mn)
                {
                    mn = s.val;
                    sdfSample_centroid = s.pt;
                }

            }
        }


        printf("Sample points generated: %zu\n", sdfSamplePoints.size());
    }

    void translate_SDFPolygon_and_samples_to_origin()
    {
        if (!pointInsidePolygon(sdfSample_centroid, polygon))
        {
            float mn = 1e6;
            for (auto& s : sdfSamplePoints)
            {
                if (s.val < mn)
                {
                    mn = s.val;
                    sdfSample_centroid = s.pt;
                }

            }
        }


        for (auto& s : sdfSamplePoints)
        {
            s.pt -= sdfSample_centroid;
        }

        for (auto& s : polygon)
        {
            s -= sdfSample_centroid;
        }
    }

    void translate_SDFPolygon_and_samples_to_original()
    {
        for (auto& s : sdfSamplePoints)
        {
            s.pt += sdfSample_centroid;
        }

        for (auto& s : polygon)
        {
            s += sdfSample_centroid;
        }
    }


    void setTargetPolygon( std::vector<zVector>& poly)
    {
        polygon = poly;
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
        if (polygon.empty()) return;

        // --- Compute polygon bounding box (target range)
        zVector bmin(1e6, 1e6, 0);
        zVector bmax(-1e6, -1e6, 0);
        for ( auto& p : polygon)
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

            zVector rawDir(output[i * 4 + 2], output[i * 4 + 3], 0);

            poses[i].c = rawCenter ? raw : mapped;
            poses[i].c += sdfSample_centroid;

            //if (!pointInsidePolygon(poses[i].c, polygon))
            //{
            //  zVector grad=  gradientAt(poses[i].c, polygon);
            //  float d = evalPolygonSDF(poses[i].c, polygon);
            //  grad *= (d * 2);

            //  //poses[i].c += grad;
            //  //poses[i].c = sdfSample_centroid;
            // // poses[i].c = polygon[int(ofRandom(0, polygon.size()-1))];
            //}

            poses[i].v = rawDir;
            poses[i].v.normalize();
        }
    }

    /// ----------------
    float computeLoss(std::vector<float>& y_pred, std::vector<float>& y_dummy) override
    {
        return coverageLoss(y_pred);
    }

    float coverageLoss(std::vector<float>& output)
    {
        if (polygon.empty()) return 1e6f;

        std::vector<Pose2D> poses;
        extractPoses(output, poses, true);// for Term 0, raw centers work better.

        //visualiseBlendedSDFs(poses);
        // -- Term 0: SDF field mismatch at fixed sample points : a coverage objective function
        float sdfLoss = 0.0f;
        if (!sdfSamplePoints.empty())
        {
            for (auto& sample : sdfSamplePoints)
            {
                float sdfTarget = sample.val;
                float sdfPred = evalBlendedCircleSDF(sample.pt, poses, radius);

                float diff = sdfTarget - sdfPred;
                sdfLoss += isnan(diff) ? 0 : (diff * diff);

            }

            sdfLoss /= sdfSamplePoints.size(); // average

        }


        return sdfLoss;
    }

    void computeGradient(std::vector<float>& x, std::vector<float>& y_dummy, std::vector<float>& gradOut) override
    {
        // 1) Single forward to get the baseline outputs
        std::vector<float> y0 = forward(x);

        // 2) Central-difference step. 
        //    Because you scale centers by 100 in extractPoses(), use a slightly larger eps.
         float eps = 1e-2f;

        gradOut.assign(outputDim, 0.0f);

        // 3) Compute baseline loss once
        float baseLoss = computeLoss(y0, y_dummy);

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

      /*  printf("GRAD: [");
            for (float v : gradOut) printf("  %.4f \n ", v);
        printf("]\n");*/

    }


    ///

    void visualiseBlendedSDFs(vector<Pose2D>& poses)
    {
        for (auto& sample : sdfSamplePoints)
            sample.val = evalBlendedCircleSDF(sample.pt, poses, radius);

    }

    void drawPolygon()
    {
        if (polygon.empty()) return;

        glColor3f(0.1f, 0.1f, 0.1f);
        glLineWidth(2.0f);

        glBegin(GL_LINE_LOOP);
        for ( zVector& pt : polygon)
        {
            glVertex3f(pt.x, pt.y, pt.z);
        }
        glEnd();

        glLineWidth(1.0f);

        if (sdfSamplePoints.empty())return;

        /*for (auto& sample : sdfSamplePoints)
            drawPoint(zVecToAliceVec(sample.pt));*/

            // --- SDF Sample Points: Jet Color Visualization
        {
            // Compute min-max val
            float vmin = 1e6;
            float vmax = -vmin;

            for (auto& s : sdfSamplePoints)
            {
                vmin = std::min(vmin, s.val);
                vmax = std::max(vmax, s.val);
            }


            //glPointSize(3);
            //for (auto& s : sdfSamplePoints)
            //{

            //    float r, g, b;
            //    getJetColor(ofMap(s.val,vmin,vmax,0,1), r, g, b); // map to [-1,1] before jetColor
            //    
            //    glColor3f(r, g, b);
            //    drawPoint(zVecToAliceVec(s.pt));
            //}
            //glPointSize(1);
            //glColor3f(0, 0, 0);

        }

    }

};


#endif // !_HEIGHT_FIELD_NN_
