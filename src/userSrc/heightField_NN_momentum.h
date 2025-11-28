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
// Ensure this matches the filename you saved the previous step as. 
// If you saved it as genericMLP.h, change this include.
#include "genericMLP.h" 

float radius = 1.0f; // = SDF loss (blendedCircle radius)

struct Pose2D
{
    zVector c;   // center (x, y, 0)
    zVector v;   // 2D vector (vx, vy, 0)
};

// -----------------------------------------------------------------------------
// GEOMETRY HELPERS
// -----------------------------------------------------------------------------

zVector gradientAt(zVector& p, std::vector<zVector>& poly)
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
    return grad;
}

float signedDistanceToPolygon(zVector& p, std::vector<zVector>& poly)
{
    float minDist = 1e6f;
    int n = poly.size();

    for (int i = 0; i < n; ++i)
    {
        zVector a = poly[i];
        zVector b = poly[(i + 1) % n];

        zVector ab = b - a;
        zVector ap = p - a;

        float t = std::max(0.0f, std::min(1.0f, ap * ab / (ab * ab)));
        zVector proj = a + ab * t;

        float d = p.distanceTo(proj);
        if (d < minDist) minDist = d;
    }

    return pointInsidePolygon(p, poly) ? -minDist : minDist;
}

float evalPolygonSDF(zVector& p, std::vector<zVector>& poly)
{
    return signedDistanceToPolygon(p, poly);
}

float evalBlendedCircleSDF(zVector& p, std::vector<Pose2D>& poses, float radius)
{
    float sdf = 1e6f;
    for (auto& pose : poses)
    {
        float d = p.distanceTo(pose.c) - radius;
        sdf = smin(sdf, d, 3); // Smooth minimum (union of circles)
    }
    return sdf;
}

struct sdfSamples
{
    zVector pt;
    float val;
};

// -----------------------------------------------------------------------------
// NEURAL NETWORK CLASS
// -----------------------------------------------------------------------------

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
        // dummy input = 2 (though unused); output = n * 4 (center x,y + dir x,y)
        initialize(2, { 16, 16 }, 4 * n);
    }

    // ------------------ DATA GENERATION ------------------

    void generateSamplesInRange(HeightField2D& htField, float minZ = 18.0f, float maxZ = 20.0f)
    {
        minZ = ofMap(minZ, htField.MLS_zMin, htField.MLS_zMax, 0, 1);
        maxZ = ofMap(maxZ, htField.MLS_zMin, htField.MLS_zMax, 0, 1);

        sdfSamplePoints.clear();
        sdfSample_centroid = zVector(0, 0, 0);

        for (int i = 0; i < SF_RES; i += 2)
        {
            for (int j = 0; j < SF_RES; j += 2)
            {
                float val = htField.field[i][j];
                sdfSamples sample;
                sample.pt = htField.gridPoints[i][j];
                sample.val = (val >= minZ && val <= maxZ) ? -val : +val;
                sdfSamplePoints.push_back(sample);
                sdfSample_centroid += sample.pt;
            }
        }
        if (!sdfSamplePoints.empty()) sdfSample_centroid /= sdfSamplePoints.size();
    }

    void set_field_values_from_polygon(std::vector<zVector>& polygon, HeightField2D& htField)
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

        zVector bmin = polygon[0];
        zVector bmax = polygon[0];

        for (auto& p : polygon)
        {
            bmin = zMin(bmin, p);
            bmax = zMax(bmax, p);
        }

        int gridResX = 50;
        int gridResY = 50;
        sdfSample_centroid = zVector(0, 0, 0);

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

        if (!sdfSamplePoints.empty()) sdfSample_centroid /= sdfSamplePoints.size();

        printf("Sample points generated: %zu\n", sdfSamplePoints.size());
    }

    // ------------------ UTILS ------------------

    void setTargetPolygon(std::vector<zVector>& poly)
    {
        polygon = poly;
    }

    void extractPoses(std::vector<float>& output, std::vector<Pose2D>& poses, bool rawCenter = false)
    {
        poses.resize(n);
        if (polygon.empty()) return;

        // Compute polygon bounding box
        zVector bmin(1e6, 1e6, 0);
        zVector bmax(-1e6, -1e6, 0);
        for (auto& p : polygon)
        {
            bmin.x = std::min(bmin.x, p.x);
            bmin.y = std::min(bmin.y, p.y);
            bmax.x = std::max(bmax.x, p.x);
            bmax.y = std::max(bmax.y, p.y);
        }

        // Collect raw centers
        std::vector<zVector> rawCenters;
        rawCenters.reserve(n);
        for (int i = 0; i < n; ++i)
        {
            rawCenters.push_back(zVector(output[i * 4 + 0], output[i * 4 + 1], 0));
        }

        // Compute input range
        zVector inMin(1e6, 1e6, 0);
        zVector inMax(-1e6, -1e6, 0);
        for (auto& c : rawCenters)
        {
            inMin.x = std::min(inMin.x, c.x);
            inMin.y = std::min(inMin.y, c.y);
            inMax.x = std::max(inMax.x, c.x);
            inMax.y = std::max(inMax.y, c.y);
        }

        float rangeX = std::max(1e-6f, inMax.x - inMin.x);
        float rangeY = std::max(1e-6f, inMax.y - inMin.y);

        // Remap
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

            poses[i].v = rawDir;
            poses[i].v.normalize();
        }
    }

    // ------------------ LOSS & GRADIENT ------------------

    float computeLoss(std::vector<float>& y_pred, std::vector<float>& y_dummy) override
    {
        return coverageLoss(y_pred);
    }

    float coverageLoss(std::vector<float>& output)
    {
        if (polygon.empty() || sdfSamplePoints.empty()) return 1e6f;

        std::vector<Pose2D> poses;
        extractPoses(output, poses, true); // Use raw centers for optimization stability

        float sdfLoss = 0.0f;
        int batchSize = 50; // Mini-batch size for SGD
        int totalPoints = sdfSamplePoints.size();

        // --- BATCHING LOGIC ---
        // If isGradientPass is true (set by computeGradientNumerical), we must
        // use a deterministic subset of points to ensure valid finite diffs.
        // If false (visualization/debug), we can use all points or a subset.

        int start = 0;
        int end = totalPoints;

        if (isGradientPass)
        {
            // Since the main loop (trainSGD) shuffles the vector, taking the first N
            // is equivalent to taking a random batch.
            // We use batchOffset to rotate through if shuffling isn't happening,
            // but taking 0..batchSize is safe given your .cpp structure.

            // To be robust:
            start = (batchOffset * batchSize) % totalPoints;
            end = start + batchSize;
            if (end > totalPoints) end = totalPoints;
        }

        // If strictly visualizing, you might want to average ALL points for accuracy.
        // But for training speed, we limit the loop.

        int count = 0;
        for (int i = start; i < end; ++i)
        {
            // Handle wrap-around if using offset logic
            int idx = i % totalPoints;
            auto& sample = sdfSamplePoints[idx];

            float sdfTarget = sample.val;
            float sdfPred = evalBlendedCircleSDF(sample.pt, poses, radius);

            float diff = sdfTarget - sdfPred;

            // Squared Error
            sdfLoss += isnan(diff) ? 0 : (diff * diff);
            count++;
        }

        return (count > 0) ? sdfLoss / (float)count : 0.0f;
    }

    void computeGradient(std::vector<float>& x, std::vector<float>& y_dummy, std::vector<float>& gradOut) override
    {
        // Use the robust numerical gradient from the base class.
        // This handles locking 'isGradientPass' to true/false automatically.
        computeGradientNumerical(x, y_dummy, gradOut);
    }

    // ------------------ VISUALIZATION ------------------

    void visualiseBlendedSDFs(std::vector<Pose2D>& poses)
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
        for (zVector& pt : polygon)
            glVertex3f(pt.x, pt.y, pt.z);
        glEnd();

        glLineWidth(1.0f);

        if (sdfSamplePoints.empty()) return;

        // Visualization of points (Commented out to reduce clutter, uncomment if needed)
        /*
        glPointSize(3);
        float vmin = 1e6, vmax = -1e6;
        for (auto& s : sdfSamplePoints) { vmin = std::min(vmin, s.val); vmax = std::max(vmax, s.val); }

        for (auto& s : sdfSamplePoints)
        {
            float r, g, b;
            float norm = (s.val - vmin) / (vmax - vmin + 1e-5);
            getJetColor(norm, r, g, b);
            glColor3f(r, g, b);
            drawPoint(zVecToAliceVec(s.pt));
        }
        glPointSize(1);
        */
    }
};

#endif // !_HEIGHT_FIELD_NN_