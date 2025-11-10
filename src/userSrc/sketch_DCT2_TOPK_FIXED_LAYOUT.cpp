#define _MAIN_
#ifdef _MAIN_

#include "main.h"

#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <map>
#include <random>
#include <limits>

using namespace zSpace;

// ------------------------------------------------------------
// Config
// ------------------------------------------------------------
#define RES 100
#define NUM_SHAPES 5
#define TOP_K 1024              // number of fixed-layout DCT coeffs
#define LATENT_DIM 4          // bottleneck dimension for AE : 

// ------------------------------------------------------------
// Jet colormap
// ------------------------------------------------------------
inline void getJetColor(float value, float& r, float& g, float& b)
{
    value = std::clamp(value, -1.0f, 1.0f);
    float x = (value + 1.0f) * 0.5f;
    float fourValue = 4.0f * x;

    r = std::clamp(std::min(fourValue - 1.5f, -fourValue + 4.5f), 0.0f, 1.0f);
    g = std::clamp(std::min(fourValue - 0.5f, -fourValue + 3.5f), 0.0f, 1.0f);
    b = std::clamp(std::min(fourValue + 0.5f, -fourValue + 2.5f), 0.0f, 1.0f);
}

inline Alice::vec zVecToAliceVec( zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

// ------------------------------------------------------------
// Polygon + SDF helpers
// ------------------------------------------------------------
inline float clampFloat(float v, float vmin, float vmax)
{
    return (v < vmin) ? vmin : (v > vmax) ? vmax : v;
}

inline float lengthSq( zVector& v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

float distancePointToSegment( zVector& p,  zVector& a,  zVector& b)
{
    zVector ab = b - a;
    float denom = lengthSq(ab);
    float t = 0.0f;

    if (denom > 1e-12f)
    {
        zVector ap = p - a;
        t = (ap.x * ab.x + ap.y * ab.y + ap.z * ab.z) / denom;
        t = clampFloat(t, 0.0f, 1.0f);
    }

    zVector proj = a + ab * t;
    zVector d = p - proj;
    return std::sqrt(lengthSq(d));
}

bool pointInPolygon( zVector& p,  std::vector<zVector>& poly)
{
    bool inside = false;
    int n = (int)poly.size();

    for (int i = 0, j = n - 1; i < n; j = i++)
    {
         zVector& pi = poly[i];
         zVector& pj = poly[j];

        bool intersect =
            ((pi.y > p.y) != (pj.y > p.y)) &&
            (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y + 1e-12f) + pi.x);

        if (intersect) inside = !inside;
    }

    return inside;
}

float sdf_Polygon( zVector& p,  std::vector<zVector>& poly)
{
    float minDist = 1e9f;
    int n = (int)poly.size();

    for (int i = 0; i < n; i++)
    {
        int j = (i + 1) % n;
        float d = distancePointToSegment(p, poly[i], poly[j]);
        if (d < minDist) minDist = d;
    }

    bool inside = pointInPolygon(p, poly);
    return inside ? -minDist : minDist;
}

float sdf_Voronoi(float x, float y)
{
    // Number of sites
    int NUM_SITES = 64;

    // Static random site positions
    static bool init = false;
    static std::vector<zVector> sites;

    if (!init)
    {
        srand(42); // fixed seed for reproducibility
        for (int i = 0; i < NUM_SITES; i++)
        {
            float sx = -1.0f + 2.0f * ((float)rand() / RAND_MAX);
            float sy = -1.0f + 2.0f * ((float)rand() / RAND_MAX);
            sites.push_back(zVector(sx, sy, 0));
        }
        init = true;
    }

    // Find nearest and second nearest site
    float d1 = 1e9f, d2 = 1e9f;
    for (int i = 0; i < NUM_SITES; i++)
    {
        float dx = x - sites[i].x;
        float dy = y - sites[i].y;
        float d = sqrt(dx * dx + dy * dy);
        if (d < d1)
        {
            d2 = d1;
            d1 = d;
        }
        else if (d < d2)
        {
            d2 = d;
        }
    }

    // SDF value: distance to nearest cell ridge
    // (ridge halfway between two closest sites)
    float sdf = 0.5f * (d2 - d1);

    // Optional: make ridges visible with a threshold
    // sdf = fabs(sdf) - 0.02f; // uncomment for thin ridge lines

    return sdf;
}


std::vector<zVector> randomPolygon(int n, float radiusMin = 0.3f, float radiusMax = 0.7f)
{
    std::vector<zVector> poly;
    poly.reserve(n);

    for (int i = 0; i < n; i++)
    {
        float t = (float)i / (float)n;
        float ang = 2.0f * (float)PI * t;
        float jitter = 0.8f + 0.4f * ((float)rand() / RAND_MAX);
        float r = radiusMin + (radiusMax - radiusMin) * ((float)rand() / RAND_MAX);
        r *= jitter;

        float x = r * std::cos(ang);
        float y = r * std::sin(ang);

        poly.push_back(zVector(x, y, 0));
    }

    return poly;
}

// ------------------------------------------------------------
// DCT-II (orthonormal) and inverse
// ------------------------------------------------------------
void computeDCT(float in[RES][RES], float out[RES][RES])
{
    for (int u = 0; u < RES; u++)
    {
        float Cu = (u == 0) ? (1.0f / std::sqrt(2.0f)) : 1.0f;

        for (int v = 0; v < RES; v++)
        {
            float Cv = (v == 0) ? (1.0f / std::sqrt(2.0f)) : 1.0f;

            float sum = 0.0f;
            for (int x = 0; x < RES; x++)
            {
                float ax = std::cos(PI * (2.0f * x + 1.0f) * (float)u / (2.0f * RES));
                for (int y = 0; y < RES; y++)
                {
                    float ay = std::cos(PI * (2.0f * y + 1.0f) * (float)v / (2.0f * RES));
                    sum += in[x][y] * ax * ay;
                }
            }

            float cN = std::sqrt(2.0f / (float)RES);
            float cM = std::sqrt(2.0f / (float)RES);
            out[u][v] = cN * cM * Cu * Cv * sum;
        }
    }
}

void computeInverseDCT(float in[RES][RES], float out[RES][RES])
{
    for (int x = 0; x < RES; x++)
    {
        for (int y = 0; y < RES; y++)
        {
            float sum = 0.0f;
            for (int u = 0; u < RES; u++)
            {
                float Cu = (u == 0) ? (1.0f / std::sqrt(2.0f)) : 1.0f;
                float ax = std::cos(PI * (2.0f * x + 1.0f) * (float)u / (2.0f * RES));

                for (int v = 0; v < RES; v++)
                {
                    float Cv = (v == 0) ? (1.0f / std::sqrt(2.0f)) : 1.0f;
                    float ay = std::cos(PI * (2.0f * y + 1.0f) * (float)v / (2.0f * RES));

                    float cN = std::sqrt(2.0f / (float)RES);
                    float cM = std::sqrt(2.0f / (float)RES);
                    sum += cN * cM * Cu * Cv * in[u][v] * ax * ay;
                }
            }
            out[x][y] = sum;
        }
    }
}

// ------------------------------------------------------------
// Data structures
// ------------------------------------------------------------
struct DCTSample
{
    std::vector<zVector> poly;
    float sdf[RES][RES];
    float dct[RES][RES];

    // shape-specific top-K (by magnitude)
    std::vector<int> topU;
    std::vector<int> topV;
    std::vector<float> topValues;
};

// Global dataset
std::vector<DCTSample> g_samples;

// Fixed layout from global distribution
std::vector<int> g_fixedU;
std::vector<int> g_fixedV;

// Fixed-layout features: [NUM_SHAPES][TOP_K]
std::vector<std::vector<float>> g_fixedFeatures;

// Normalisation for AE training
float g_featMean = 0.0f;
float g_featStd = 1.0f;
// Feature-wise normalization parameters
std::vector<float> g_featMeanVec;
std::vector<float> g_featStdVec;

std::vector<std::vector<float>> g_trainX; // normalised features

// AE + training
#include "genericMLP.h"
MLP g_autoencoder;

bool g_isTraining = false;
float g_lastLoss = 0.0f;

enum TrainMode
{
    TRAIN_SGD = 0,
    TRAIN_ADAM = 1
};

TrainMode g_trainMode = TRAIN_ADAM;

// Current selection & viz fields
int g_currentShape = 0;
float g_reconFixed[RES][RES];
float g_reconAE[RES][RES];

// ------------------------------------------------------------
// Utility: draw SDF field
// ------------------------------------------------------------
void drawSDF(float fld[RES][RES], float px, float py, float scale)
{
    float vmin = 1e9f;
    float vmax = -1e9f;

    for (int i = 0; i < RES; i++)
    {
        for (int j = 0; j < RES; j++)
        {
            float v = fld[i][j];
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
    }

    float mid = 0.5f * (vmax + vmin);
    float range = 0.5f * (vmax - vmin);
    if (range < 1e-6f) range = 1.0f;

    //setup2d();

    glPointSize(2.0f);
    glBegin(GL_POINTS);

    for (int i = 0; i < RES; i += 1)
    {
        for (int j = 0; j < RES; j += 1)
        {
            float nval = (fld[i][j] - mid) / range; // [-1,1]
            float r, g, b;
            getJetColor(nval, r, g, b);
            glColor3f(r, g, b);

            float x = px + i * scale;
            float y = py + j * scale;
            glVertex2f(x, y);
        }
    }

    glEnd();
   // restore3d();
}

// ------------------------------------------------------------
// 1) Generate 5 polygons + SDF + DCT + per-shape Top-K
// ------------------------------------------------------------
void generateDataset()
{
    g_samples.clear();
    g_samples.resize(NUM_SHAPES);

    for (int s = 0; s < NUM_SHAPES; s++)
    {
        // polygon in [-0.8,0.8]^2
        int nv = 5 + (rand() % 4);
        g_samples[s].poly = randomPolygon(nv);

        // SDF on [-1,1]^2
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                float x = (float)i / (RES - 1) * 2.0f - 1.0f;
                float y = (float)j / (RES - 1) * 2.0f - 1.0f;
                zVector p(x, y, 0);
                g_samples[s].sdf[i][j] = (s == 0) ? sdf_Voronoi(x,y) : sdf_Polygon(p, g_samples[s].poly);
            }
        }

        // DCT
        computeDCT(g_samples[s].sdf, g_samples[s].dct);

        // Per-shape Top-K by magnitude
        std::vector<std::tuple<float, int, int>> coeffs;
        coeffs.reserve(RES * RES);
        for (int u = 0; u < RES; u++)
        {
            for (int v = 0; v < RES; v++)
            {
                coeffs.push_back(std::make_tuple(std::fabs(g_samples[s].dct[u][v]), u, v));
            }
        }

        std::sort(coeffs.begin(), coeffs.end(),
            []( auto& a,  auto& b)
            {
                return std::get<0>(a) > std::get<0>(b);
            });

        int K = std::min(TOP_K, (int)coeffs.size());
        g_samples[s].topU.resize(K);
        g_samples[s].topV.resize(K);
        g_samples[s].topValues.resize(K);

        for (int k = 0; k < K; k++)
        {
            float mag = std::get<0>(coeffs[k]);
            int u = std::get<1>(coeffs[k]);
            int v = std::get<2>(coeffs[k]);
            g_samples[s].topU[k] = u;
            g_samples[s].topV[k] = v;
            g_samples[s].topValues[k] = g_samples[s].dct[u][v]; // signed
        }
    }

    printf("Generated %d shapes with per-shape Top-%d DCT.\n", NUM_SHAPES, TOP_K);
}

// ------------------------------------------------------------
// 2) Compute global fixed U,V from Top-K distributions
// ------------------------------------------------------------
void computeFixedLayout()
{
    std::map<std::pair<int, int>, float> energySum;

    for (auto& s : g_samples)
    {
        int K = (int)s.topU.size();
        for (int k = 0; k < K; k++)
        {
            std::pair<int, int> key(s.topU[k], s.topV[k]);
            energySum[key] += std::fabs(s.topValues[k]);
        }
    }

    std::vector<std::pair<std::pair<int, int>, float>> ranked;
    ranked.reserve(energySum.size());

    for (auto& kv : energySum)
    {
        ranked.push_back(kv);
    }

    std::sort(ranked.begin(), ranked.end(),
        []( auto& a,  auto& b)
        {
            return a.second > b.second;
        });

    g_fixedU.clear();
    g_fixedV.clear();

    int K = std::min(TOP_K, (int)ranked.size());
    for (int i = 0; i < K; i++)
    {
        g_fixedU.push_back(ranked[i].first.first);
        g_fixedV.push_back(ranked[i].first.second);
    }

    printf("Computed fixed layout with %d modes.\n", (int)g_fixedU.size());
}

// ------------------------------------------------------------
// 3) Build per-shape fixed-layout feature vectors
// ------------------------------------------------------------
void buildFixedLayoutFeatures()
{
    g_fixedFeatures.clear();
    g_fixedFeatures.resize(NUM_SHAPES);

    for (int s = 0; s < NUM_SHAPES; s++)
    {
        // map per-shape (u,v)->value for quick lookup
        std::map<std::pair<int, int>, float> local;
        int K = (int)g_samples[s].topU.size();

        for (int k = 0; k < K; k++)
        {
            local[std::make_pair(g_samples[s].topU[k], g_samples[s].topV[k])] =
                g_samples[s].topValues[k];
        }

        std::vector<float> feat(TOP_K, 0.0f);
        for (int i = 0; i < TOP_K; i++)
        {
            int u = g_fixedU[i];
            int v = g_fixedV[i];
            auto it = local.find(std::make_pair(u, v));
            if (it != local.end())
            {
                feat[i] = it->second;
            }
            else
            {
                feat[i] = 0.0f;
            }
        }

        g_fixedFeatures[s] = feat;
    }

    printf("Built fixed-layout feature vectors.\n");
}

// ------------------------------------------------------------
// 4) Global normalization for AE
// ------------------------------------------------------------
//void buildTrainingData()
//{
//    double sum = 0.0;
//    double sumSq = 0.0;
//    size_t count = 0;
//
//    for (auto& f : g_fixedFeatures)
//    {
//        for (float v : f)
//        {
//            sum += v;
//            sumSq += (double)v * (double)v;
//            count++;
//        }
//    }
//
//    if (count > 0)
//    {
//        g_featMean = (float)(sum / count);
//        double var = (sumSq / count) - (double)g_featMean * (double)g_featMean;
//        g_featStd = (var > 1e-12) ? (float)std::sqrt(var) : 1.0f;
//    }
//    else
//    {
//        g_featMean = 0.0f;
//        g_featStd = 1.0f;
//    }
//
//    g_trainX.clear();
//    g_trainX.resize(NUM_SHAPES);
//
//    for (int s = 0; s < NUM_SHAPES; s++)
//    {
//        g_trainX[s].resize(TOP_K);
//        for (int i = 0; i < TOP_K; i++)
//        {
//            g_trainX[s][i] = (g_fixedFeatures[s][i] - g_featMean) / g_featStd;
//        }
//    }
//
//    printf("Normalised features: mean=%.6f std=%.6f\n", g_featMean, g_featStd);
//}

void buildTrainingData()
{
    int N = (int)g_fixedFeatures.size();
    if (N == 0) return;
    int K = (int)g_fixedFeatures[0].size();

    g_featMeanVec.assign(K, 0.0f);
    g_featStdVec.assign(K, 1.0f);

    // --- Compute mean per feature ---
    for (int i = 0; i < K; i++)
    {
        double sum = 0.0;
        for (int s = 0; s < N; s++)
        {
            sum += g_fixedFeatures[s][i];
        }
        g_featMeanVec[i] = (float)(sum / N);
    }

    // --- Compute std per feature ---
    for (int i = 0; i < K; i++)
    {
        double sumSq = 0.0;
        for (int s = 0; s < N; s++)
        {
            float d = g_fixedFeatures[s][i] - g_featMeanVec[i];
            sumSq += (double)d * (double)d;
        }
        float stdv = (float)std::sqrt(sumSq / N);
        if (stdv < 1e-6f) stdv = 1.0f;
        g_featStdVec[i] = stdv;
    }

    // --- Normalize dataset ---
    g_trainX.clear();
    g_trainX.resize(N);

    for (int s = 0; s < N; s++)
    {
        g_trainX[s].resize(K);
        for (int i = 0; i < K; i++)
        {
            g_trainX[s][i] = (g_fixedFeatures[s][i] - g_featMeanVec[i]) / g_featStdVec[i];
        }
    }

    printf("Per-feature normalization applied (K=%d, N=%d)\n", K, N);
}


// ------------------------------------------------------------
// 5) AE training: SGD and Adam (from your previous sketch style)
// ------------------------------------------------------------
float trainSGD(MLP& net,
    std::vector<std::vector<float>>& X,
    std::vector<std::vector<float>>& Y,
    int epochs,
    float lr,
    int batchSize)
{
    std::vector<int> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(std::random_device{}());

    float totalLoss = 0.0f;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        totalLoss = 0.0f;

        for (int b = 0; b < (int)X.size(); b += batchSize)
        {
            int end = std::min((int)X.size(), b + batchSize);
            for (int ii = b; ii < end; ii++)
            {
                int idx = indices[ii];

                std::vector<float> gradOut;
                net.computeGradient(X[idx], Y[idx], gradOut);
                net.backward(gradOut, lr);

                std::vector<float> y_pred = net.forward(X[idx]);
                totalLoss += net.computeLoss(y_pred, Y[idx]);
            }
        }

        totalLoss /= (float)X.size();
        printf("[SGD] Epoch %d | Loss: %.6f\n", epoch, totalLoss);
    }

    return totalLoss;
}

float trainAdam(MLP& net,
    std::vector<std::vector<float>>& X,
    std::vector<std::vector<float>>& Y,
    int epochs = 200,
    float lr = 1e-3f,
    float beta1 = 0.9f,
    float beta2 = 0.999f,
    float eps = 1e-8f,
    int batchSize = 4)
{
    // Adam states shaped like W,b
    std::vector<std::vector<std::vector<float>>> mW = net.W;
    std::vector<std::vector<std::vector<float>>> vW = net.W;
    std::vector<std::vector<float>> mB = net.b;
    std::vector<std::vector<float>> vB = net.b;

    for (auto& L : mW)
    {
        for (auto& r : L)
        {
            std::fill(r.begin(), r.end(), 0.0f);
        }
    }
    for (auto& L : vW)
    {
        for (auto& r : L)
        {
            std::fill(r.begin(), r.end(), 0.0f);
        }
    }
    for (auto& b : mB) std::fill(b.begin(), b.end(), 0.0f);
    for (auto& b : vB) std::fill(b.begin(), b.end(), 0.0f);

    std::vector<int> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(std::random_device{}());

    float totalLoss = 0.0f;
    int t = 0;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        totalLoss = 0.0f;

        for (int b = 0; b < (int)X.size(); b += batchSize)
        {
            int end = std::min((int)X.size(), b + batchSize);

            for (int ii = b; ii < end; ii++)
            {
                int idx = indices[ii];
                t++;

                // Forward
                std::vector<float> y_pred = net.forward(X[idx]);
                float loss = net.computeLoss(y_pred, Y[idx]);
                totalLoss += loss;

                // dL/dy for MSE
                std::vector<float> gradOut(y_pred.size());
                for (int j = 0; j < (int)y_pred.size(); j++)
                {
                    gradOut[j] = (y_pred[j] - Y[idx][j]);// / (float)y_pred.size();
                }

                // Allocate grads
                std::vector<std::vector<std::vector<float>>> gradW = net.W;
                std::vector<std::vector<float>> gradB = net.b;
                for (auto& L : gradW)
                {
                    for (auto& r : L)
                    {
                        std::fill(r.begin(), r.end(), 0.0f);
                    }
                }
                for (auto& bL : gradB)
                {
                    std::fill(bL.begin(), bL.end(), 0.0f);
                }

                // Backprop deltas
                std::vector<std::vector<float>> deltas(net.W.size());
                deltas.back() = gradOut;

                for (int l = (int)net.W.size() - 1; l >= 0; l--)
                {
                    std::vector<float>& delta = deltas[l];
                    std::vector<float> prevActiv = net.activations[l];

                    gradB[l].resize(net.b[l].size());

                    for (int iN = 0; iN < (int)net.W[l].size(); iN++)
                    {
                        for (int jIn = 0; jIn < (int)net.W[l][iN].size(); jIn++)
                        {
                            gradW[l][iN][jIn] += delta[iN] * prevActiv[jIn];
                        }
                        gradB[l][iN] += delta[iN];
                    }

                    if (l > 0)
                    {
                        std::vector<float> prevDelta(net.W[l][0].size(), 0.0f);
                        for (int jIn = 0; jIn < (int)net.W[l][0].size(); jIn++)
                        {
                            float sum = 0.0f;
                            for (int iN = 0; iN < (int)net.W[l].size(); iN++)
                            {
                                sum += delta[iN] * net.W[l][iN][jIn];
                            }
                            float a = net.activations[l][jIn];
                            prevDelta[jIn] = sum * (1.0f - a * a); // tanh'
                        }
                        deltas[l - 1] = prevDelta;
                    }
                }

                // Adam update
                for (int l = 0; l < (int)net.W.size(); l++)
                {
                    for (int iN = 0; iN < (int)net.W[l].size(); iN++)
                    {
                        for (int jIn = 0; jIn < (int)net.W[l][iN].size(); jIn++)
                        {
                            float g = gradW[l][iN][jIn];
                            //g = std::clamp(g, -10.0f, 10.0f); // optional

                            mW[l][iN][jIn] = beta1 * mW[l][iN][jIn] + (1.0f - beta1) * g;
                            vW[l][iN][jIn] = beta2 * vW[l][iN][jIn] + (1.0f - beta2) * g * g;

                            float m_hat = mW[l][iN][jIn] / (1.0f - std::pow(beta1, t));
                            float v_hat = vW[l][iN][jIn] / (1.0f - std::pow(beta2, t));

                            net.W[l][iN][jIn] -= lr * m_hat / (std::sqrt(v_hat) + eps);
                        }

                        float gb = gradB[l][iN];
                        gb = std::clamp(gb, -1.0f, 1.0f);

                        mB[l][iN] = beta1 * mB[l][iN] + (1.0f - beta1) * gb;
                        vB[l][iN] = beta2 * vB[l][iN] + (1.0f - beta2) * gb * gb;

                        float m_hatb = mB[l][iN] / (1.0f - std::pow(beta1, t));
                        float v_hatb = vB[l][iN] / (1.0f - std::pow(beta2, t));

                        net.b[l][iN] -= lr * m_hatb / (std::sqrt(v_hatb) + eps);
                    }
                }
            }
        }

        totalLoss /= (float)X.size();
        printf("[Adam] Epoch %d | Loss: %.6f\n", epoch, totalLoss);
    }

    return totalLoss;
}

// ------------------------------------------------------------
// 6) Reruction from fixed layout (ground truth, no AE)
// ------------------------------------------------------------
void reconstruct_from_fixed_layout_truth(int s, float out[RES][RES])
{
    if (s < 0 || s >= NUM_SHAPES) return;

    float dctTmp[RES][RES] = { 0 };

    // Use the fixed-layout feature vector directly
     std::vector<float>& feat = g_fixedFeatures[s];
    for (int i = 0; i < TOP_K; i++)
    {
        int u = g_fixedU[i];
        int v = g_fixedV[i];
        dctTmp[u][v] = feat[i];
    }

    computeInverseDCT(dctTmp, out);
}

// ------------------------------------------------------------
// 7) Reruction from AE output
// ------------------------------------------------------------
//void reconstruct_from_AE_output(int s, float out[RES][RES])
//{
//    if (s < 0 || s >= NUM_SHAPES) return;
//
//    // Input to AE is normalised fixed-layout features
//     std::vector<float>& xNorm = g_trainX[s];
//
//    // Forward through AE
//    std::vector<float> yNorm = g_autoencoder.forward(xNorm);
//
//    // De-normalise
//    std::vector<float> y(yNorm.size());
//    for (int i = 0; i < (int)yNorm.size(); i++)
//    {
//        y[i] = yNorm[i] * g_featStd + g_featMean;
//    }
//
//    // Fill DCT grid at fixed positions
//    float dctTmp[RES][RES] = { 0 };
//    int K = std::min(TOP_K, (int)y.size());
//
//    for (int i = 0; i < K; i++)
//    {
//        int u = g_fixedU[i];
//        int v = g_fixedV[i];
//        dctTmp[u][v] = y[i];
//    }
//
//    // IDCT -> SDF
//    computeInverseDCT(dctTmp, out);
//}

void reconstruct_from_AE_output(int s, float out[RES][RES])
{
    if (s < 0 || s >= NUM_SHAPES) return;
    if (g_featMeanVec.empty() || g_featStdVec.empty()) return;

    // Input to AE is normalized fixed-layout features
    std::vector<float>& xNorm = g_trainX[s];

    // Forward pass through AE
    std::vector<float> yNorm = g_autoencoder.forward(xNorm);

    // --- De-normalize per feature ---
    std::vector<float> y(yNorm.size());
    for (int i = 0; i < (int)yNorm.size(); i++)
    {
        y[i] = yNorm[i] * g_featStdVec[i] + g_featMeanVec[i];
    }

    // --- Fill DCT grid at fixed positions ---
    float dctTmp[RES][RES] = { 0 };
    int K = std::min(TOP_K, (int)y.size());
    for (int i = 0; i < K; i++)
    {
        int u = g_fixedU[i];
        int v = g_fixedV[i];
        dctTmp[u][v] = y[i];
    }

    // --- Inverse DCT -> spatial SDF ---
    computeInverseDCT(dctTmp, out);
}


// ------------------------------------------------------------
// Setup + wiring
// ------------------------------------------------------------
void rebuildAll()
{
    generateDataset();
    computeFixedLayout();
    buildFixedLayoutFeatures();
    buildTrainingData();

    // Initialize AE: input=TOP_K, hidden -> LATENT -> hidden, output=TOP_K
    // Using tanh in hidden, linear output in genericMLP (as per your previous usage)
    g_autoencoder.initialize
    (
        TOP_K,
        { 32, LATENT_DIM, 32 },
        TOP_K
    );

    g_lastLoss = 0.0f;

    reconstruct_from_fixed_layout_truth(g_currentShape, g_reconFixed);
    reconstruct_from_AE_output(g_currentShape, g_reconAE);
}

void setup()
{
    backGround(0.9f);
    drawGrid(100);
    srand(1);

    rebuildAll();
}

void update(int value)
{
    if (!g_isTraining) return;
    if (g_trainX.empty()) return;

    if (g_trainMode == TRAIN_SGD)
    {
        g_lastLoss = trainSGD(g_autoencoder, g_trainX, g_trainX, 50, 0.1, 4);
    }
    else
    {
        g_lastLoss = trainAdam(g_autoencoder, g_trainX, g_trainX, 50, 1e-3f, 0.9f, 0.99f, 1e-8f, 4);
    }

    // Update reconstructn for current shape after training chunk
    /*reconstruct_from_fixed_layout_truth(g_currentShape, g_reconFixed);
    reconstruct_from_AE_output(g_currentShape, g_reconAE);*/
}

void draw()
{
    backGround(0.9f);
    drawGrid(100);

    // Draw current polygon
    glColor3f(0, 0, 0);
    auto& poly = g_samples[g_currentShape].poly;
    int n = (int)poly.size();
    for (int i = 0; i < n; i++)
    {
        int j = (i + 1) % n;
        drawLine(zVecToAliceVec(poly[i]), zVecToAliceVec(poly[j]));
    }

    // Original SDF
    drawSDF(g_samples[g_currentShape].sdf, -(float(RES*2+10)), -float(RES * 0.5), 1.0f);

    // Fixed-layout reconstructionruction (truth coefficients)
    drawSDF(g_reconFixed, -(float(RES * 0.5 )), -float(RES * 0.5), 1.0f);

    // AE reruction
    drawSDF(g_reconAE, (float(RES + 10)), -float(RES * 0.5), 1.0f);

    // UI
    setup2d();
    glColor3f(0, 0, 0);

    char buf[256];
    sprintf(buf, "Shape %d / %d   (keys 1-5 to switch)", g_currentShape + 1, NUM_SHAPES);
    drawString(buf, 20, 40);

    drawString("Left: Original SDF", 20, 60);
    drawString("Middle: Fixed-layout reruction (true coeffs)", 20, 80);
    drawString("Right: Autoencoder reruction", 20, 100);

    drawString(std::string("Training [t]: ") + (g_isTraining ? "ON" : "OFF"), 20, 130);
    drawString(std::string("Mode [m]: ") + ((g_trainMode == TRAIN_SGD) ? "SGD" : "Adam"), 20, 150);

    sprintf(buf, "Last loss: %.6f", g_lastLoss);
    drawString(buf, 20, 170);

    drawString("Press 'r' to regenerate shapes & reset", 20, 190);
    drawString("Press 't' to toggle training", 20, 210);
    drawString("Press 'm' to toggle SGD/Adam", 20, 230);

    restore3d();
}

void keyPress(unsigned char k, int xm, int ym)
{
    if (k == 'r')
    {
        rebuildAll();
    }

    if (k == 't')
    {
        g_isTraining = !g_isTraining;
    }

    if (k == 'm')
    {
        g_trainMode = (g_trainMode == TRAIN_SGD) ? TRAIN_ADAM : TRAIN_SGD;
    }

    if (k >= '1' && k <= '5')
    {
        int idx = (int)(k - '1');
        if (idx >= 0 && idx < NUM_SHAPES)
        {
            g_currentShape = idx;
            reconstruct_from_fixed_layout_truth(g_currentShape, g_reconFixed);
            reconstruct_from_AE_output(g_currentShape, g_reconAE);
        }
    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif
