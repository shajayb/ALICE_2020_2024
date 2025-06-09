
#pragma once

#include "genericMLP.h"
#include <cmath>
#include <vector>


using namespace zSpace;

class SDFAlignMLP : public MLP
{
public:
    std::vector<zVector> samplePts;
    std::vector<zVector> polyGradRef;
    zVector sunDir = zVector(1, 1, 0);

    int nEllipse = 4;
    int nBox = 4;

    void decodeMLPOutput(const std::vector<float>& out, std::vector<float>& decoded)
    {
        decoded.clear();
        int idx = 0;

        // Decode ellipses
        for (int i = 0; i < nEllipse; i++)
        {
            float x = out[idx++];
            float y = out[idx++];
            float angle = out[idx++];
            float a = std::max(0.01f, out[idx++]);  // major axis
            float b = std::max(0.01f, out[idx++]);  // minor axis

            decoded.push_back(x);
            decoded.push_back(y);
            decoded.push_back(angle);
            decoded.push_back(a);
            decoded.push_back(b);
        }

        // Decode boxes
        for (int i = 0; i < nBox; i++)
        {
            float x = out[idx++];
            float y = out[idx++];
            float angle = out[idx++];
            float ex = std::max(0.01f, out[idx++]);  // extent x
            float ey = std::max(0.01f, out[idx++]);  // extent y

            decoded.push_back(x);
            decoded.push_back(y);
            decoded.push_back(angle);
            decoded.push_back(ex);
            decoded.push_back(ey);
        }
    }

    float blendedSDF(zVector pt, const std::vector<float>& params)
    {
        float k = 2.0f;
        float sdf = 1e6f;
        int idx = 0;

        for (int i = 0; i < nEllipse; i++)
        {
            zVector c(params[idx], params[idx + 1], 0);
            float angle = params[idx + 2];
            float a = params[idx + 3], b = params[idx + 4];
            float cosT = cos(angle), sinT = sin(angle);

            zVector p = pt - c;
            float x = cosT * p.x + sinT * p.y;
            float y = -sinT * p.x + cosT * p.y;
            float d = sqrt((x * x) / (a * a) + (y * y) / (b * b)) - 1.0f;

            float h = std::max(k - fabs(sdf - d), 0.0f) / k;
            sdf = std::min(sdf, d) - h * h * k * 0.25f;

            idx += 5;
        }

        for (int i = 0; i < nBox; i++)
        {
            zVector c(params[idx], params[idx + 1], 0);
            float angle = params[idx + 2];
            zVector halfExt(params[idx + 3], params[idx + 4], 0);

            zVector p = pt - c;
            float cosT = cos(angle), sinT = sin(angle);
            zVector q(cosT * p.x + sinT * p.y, -sinT * p.x + cosT * p.y, 0);
            q = zVector(fabs(q.x), fabs(q.y), 0) - halfExt;

            float d = std::min(std::max(q.x, q.y), 0.0f) + zVector(std::max(q.x, 0.0f), std::max(q.y, 0.0f), 0).length();

            float h = std::max(k - fabs(sdf - d), 0.0f) / k;
            sdf = std::min(sdf, d) - h * h * k * 0.25f;

            idx += 5;
        }

        return sdf;
    }

    zVector computeSDFGradient(zVector pt, const std::vector<float>& params)
    {
        float eps = 1e-2;
        float dx = blendedSDF(pt + zVector(eps, 0, 0), params) - blendedSDF(pt - zVector(eps, 0, 0), params);
        float dy = blendedSDF(pt + zVector(0, eps, 0), params) - blendedSDF(pt - zVector(0, eps, 0), params);
        zVector grad(dx, dy, 0);
        grad.normalize();
        return grad;
    }

    float computeCustomLoss(const std::vector<zVector>& samplePts,
                            const std::vector<zVector>& polyGradRef,
                            const std::vector<float>& params,
                            int nEllipse, int nBox)
    {
        float loss = 0.0f;
        for (int i = 0; i < samplePts.size(); i++)
        {
            zVector grad = computeSDFGradient(samplePts[i], params);
            float dot1 = grad * sunDir;
            float dot2 = grad * polyGradRef[i];
            loss += (1.0f - dot1) + 0.5f * (1.0f - dot2);
        }

        return loss / samplePts.size();
    }

    float computeLoss(std::vector<float>& out) override
    {
        std::vector<float> params;
        decodeMLPOutput(out, params);
        return computeCustomLoss(samplePts, polyGradRef, params, nEllipse, nBox);
    }

    void computeGradient(std::vector<float>& out, std::vector<float>& grad_out) override
    {
        float eps = 1e-3f;
        grad_out.assign(out.size(), 0.0f);

        std::vector<float> paramsP, paramsM;

        for (int i = 0; i < out.size(); i++)
        {
            float orig = out[i];

            out[i] = orig + eps;
            decodeMLPOutput(out, paramsP);
            float lossP = computeCustomLoss(samplePts, polyGradRef, paramsP, nEllipse, nBox);

            out[i] = orig - eps;
            decodeMLPOutput(out, paramsM);
            float lossM = computeCustomLoss(samplePts, polyGradRef, paramsM, nEllipse, nBox);

            out[i] = orig;
            grad_out[i] = (lossP - lossM) / (2.0f * eps);
        }
    }
};
