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
#define RES 128
#define NUM_SHAPES 5

// Fixed low-frequency block size U x U
#define BLOCK_U 30
#define TOP_K (BLOCK_U * BLOCK_U)

#define LATENT_DIM 16

#ifndef PI
#define PI 3.14159265358979323846
#endif

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

inline Alice::vec zVecToAliceVec(zVector& in)
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

inline float lengthSq(zVector& v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

float distancePointToSegment(zVector& p, zVector& a, zVector& b)
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

bool pointInPolygon(zVector& p, std::vector<zVector>& poly)
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

float sdf_Polygon(zVector& p, std::vector<zVector>& poly)
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
    int NUM_SITES = 64;

    static bool init = false;
    static std::vector<zVector> sites;

    if (!init)
    {
        srand(42);
        for (int i = 0; i < NUM_SITES; i++)
        {
            float sx = -1.0f + 2.0f * ((float)rand() / RAND_MAX);
            float sy = -1.0f + 2.0f * ((float)rand() / RAND_MAX);
            sites.push_back(zVector(sx, sy, 0));
        }
        init = true;
    }

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

    float sdf = 0.5f * (d2 - d1);
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
// Fast separable DCT / iDCT (orthonormal)
// ------------------------------------------------------------
void computeDCT(float in[RES][RES], float out[RES][RES])
{
    static float cosTable[RES][RES];
    static bool init = false;

    if (!init)
    {
        for (int n = 0; n < RES; n++)
        {
            for (int k = 0; k < RES; k++)
            {
                cosTable[n][k] = cos(PI * (2.0f * n + 1.0f) * k / (2.0f * RES));
            }
        }
        init = true;
    }

    float temp[RES][RES];

    // 1D DCT on rows
    for (int y = 0; y < RES; y++)
    {
        for (int u = 0; u < RES; u++)
        {
            float Cu = (u == 0) ? 1.0f / sqrt(2.0f) : 1.0f;
            float sum = 0.0f;
            for (int x = 0; x < RES; x++)
            {
                sum += in[x][y] * cosTable[x][u];
            }
            temp[u][y] = sum * Cu * sqrt(2.0f / (float)RES);
        }
    }

    // 1D DCT on columns
    for (int u = 0; u < RES; u++)
    {
        for (int v = 0; v < RES; v++)
        {
            float Cv = (v == 0) ? 1.0f / sqrt(2.0f) : 1.0f;
            float sum = 0.0f;
            for (int y = 0; y < RES; y++)
            {
                sum += temp[u][y] * cosTable[y][v];
            }
            out[u][v] = sum * Cv * sqrt(2.0f / (float)RES);
        }
    }
}

void computeInverseDCT(float in[RES][RES], float out[RES][RES])
{
    static float cosTable[RES][RES];
    static bool init = false;

    if (!init)
    {
        for (int n = 0; n < RES; n++)
        {
            for (int k = 0; k < RES; k++)
            {
                cosTable[n][k] = cos(PI * (2.0f * n + 1.0f) * k / (2.0f * RES));
            }
        }
        init = true;
    }

    float temp[RES][RES];

    // 1D inverse DCT on rows
    for (int x = 0; x < RES; x++)
    {
        for (int n = 0; n < RES; n++)
        {
            float sum = 0.0f;
            for (int k = 0; k < RES; k++)
            {
                float Cu = (k == 0) ? 1.0f / sqrt(2.0f) : 1.0f;
                sum += Cu * in[k][x] * cosTable[n][k];
            }
            temp[n][x] = sum * sqrt(2.0f / (float)RES);
        }
    }

    // 1D inverse DCT on columns
    for (int y = 0; y < RES; y++)
    {
        for (int x = 0; x < RES; x++)
        {
            float sum = 0.0f;
            for (int k = 0; k < RES; k++)
            {
                float Cv = (k == 0) ? 1.0f / sqrt(2.0f) : 1.0f;
                sum += Cv * temp[x][k] * cosTable[y][k];
            }
            out[x][y] = sum * sqrt(2.0f / (float)RES);
        }
    }
}

//



///

// ------------------------------------------------------------
// Data structures
// ------------------------------------------------------------
struct DCTSample
{
    std::vector<zVector> poly;
    float sdf[RES][RES];
    float dct[RES][RES];
};

std::vector<DCTSample> g_samples;

// For fixed U x U block
std::vector<int> g_fixedU;
std::vector<int> g_fixedV;

// Features: NUM_SHAPES x TOP_K (fixed block coeffs)
std::vector<std::vector<float>> g_fixedFeatures;

// Per-feature normalization
std::vector<float> g_featMeanVec;
std::vector<float> g_featStdVec;
std::vector<std::vector<float>> g_trainX;

// AE + training
#include "genericMLP.h"
MLP g_autoencoder;

bool g_isTraining = false;
float g_lastLoss = 0.0f;

enum TrainMode
{
    TRAIN_ADAM = 0,
    TRAIN_SGD = 1
};

TrainMode g_trainMode = TRAIN_SGD;

// Current selection & viz
int g_currentShape = 0;
float g_reconFixed[RES][RES];
float g_reconAE[RES][RES];

// Encoder–decoder wrapper
class AutoEncoder : public MLP
{
public:
    MLP encoder, decoder;

    void initializeAE(int inDim, int bottleneckDim, int hidden = 64)
    {
        encoder.initialize(inDim, { hidden }, bottleneckDim);
        decoder.initialize(bottleneckDim, { hidden }, inDim);
    }

    std::vector<float> encode(std::vector<float>& x) { return encoder.forward(x); }
    std::vector<float> decode(std::vector<float>& z) { return decoder.forward(z); }

    void train(std::vector<std::vector<float>>& data, int epochs = 500, float lr = 0.01f)
    {
        for (int e = 0; e < epochs; e++)
        {
            float totalLoss = 0;
            for (auto& x : data)
            {
                auto z = encoder.forward(x);
                auto x_hat = decoder.forward(z);

                std::vector<float> gradOut(x.size());
                for (int i = 0; i < x.size(); i++)
                {
                    float diff = x_hat[i] - x[i];
                    gradOut[i] = 2 * diff;
                    totalLoss += diff * diff;
                }

                decoder.backward(gradOut, lr);

                std::vector<float> gradZ = gradOut;
                encoder.backward(gradZ, lr);
            }
            if (e % 50 == 0) printf("epoch %d  loss %.4f\n", e, totalLoss / data.size());
        }
    }

};

// ------------------------------------------------------------
// Draw SDF
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

    glPointSize(2.0f);
    glBegin(GL_POINTS);

    for (int i = 0; i < RES; i++)
    {
        for (int j = 0; j < RES; j++)
        {
            float nval = (fld[i][j] - mid) / range;
            float r, g, b;
            getJetColor(nval, r, g, b);
            glColor3f(r, g, b);

            float x = px + i * scale;
            float y = py + j * scale;
            glVertex2f(x, y);
        }
    }

    glEnd();
}

// ------------------------------------------------------------
// 1) Generate dataset: polygons + SDF + DCT
// ------------------------------------------------------------
void generateDataset()
{
    g_samples.clear();
    g_samples.resize(NUM_SHAPES);

    for (int s = 0; s < NUM_SHAPES; s++)
    {
        int nv = 5 + (rand() % 4);
        g_samples[s].poly = randomPolygon(nv);

        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                float x = (float)i / (RES - 1) * 2.0f - 1.0f;
                float y = (float)j / (RES - 1) * 2.0f - 1.0f;
                zVector p(x, y, 0);

                if (s == 0)
                {
                    g_samples[s].sdf[i][j] = sdf_Voronoi(x, y);
                }
                else
                {
                    g_samples[s].sdf[i][j] = sdf_Polygon(p, g_samples[s].poly);
                }
            }
        }

        computeDCT(g_samples[s].sdf, g_samples[s].dct);
    }

    printf("Generated %d shapes.\n", NUM_SHAPES);
}

// ------------------------------------------------------------
// 2) Define fixed U x U low-frequency layout
// ------------------------------------------------------------
void computeFixedBlockLayout()
{
    g_fixedU.clear();
    g_fixedV.clear();
    g_fixedU.reserve(TOP_K);
    g_fixedV.reserve(TOP_K);

    for (int u = 0; u < BLOCK_U; u++)
    {
        for (int v = 0; v < BLOCK_U; v++)
        {
            g_fixedU.push_back(u);
            g_fixedV.push_back(v);
        }
    }

    printf("Fixed block layout: %d x %d = %d coeffs.\n", BLOCK_U, BLOCK_U, TOP_K);
}

// ------------------------------------------------------------
// 3) Build fixed-layout features from DCT (direct block sample)
// ------------------------------------------------------------
void buildFixedLayoutFeatures()
{
    g_fixedFeatures.clear();
    g_fixedFeatures.resize(NUM_SHAPES);

    for (int s = 0; s < NUM_SHAPES; s++)
    {
        std::vector<float> feat(TOP_K, 0.0f);

        for (int i = 0; i < TOP_K; i++)
        {
            int u = g_fixedU[i];
            int v = g_fixedV[i];
            feat[i] = g_samples[s].dct[u][v];
        }

        g_fixedFeatures[s] = feat;
    }

    printf("Built fixed UxU block feature vectors.\n");
}

// ------------------------------------------------------------
// 4) Per-feature normalization for AE
// ------------------------------------------------------------
void buildTrainingData()
{
    int N = (int)g_fixedFeatures.size();
    if (N == 0) return;
    int K = (int)g_fixedFeatures[0].size();

    g_featMeanVec.assign(K, 0.0f);
    g_featStdVec.assign(K, 1.0f);

    for (int i = 0; i < K; i++)
    {
        double sum = 0.0;
        for (int s = 0; s < N; s++)
        {
            sum += g_fixedFeatures[s][i];
        }
        g_featMeanVec[i] = (float)(sum / N);
    }

    for (int i = 0; i < K; i++)
    {
        double sumSq = 0.0;
        for (int s = 0; s < N; s++)
        {
            float d = g_fixedFeatures[s][i] - g_featMeanVec[i];
            sumSq += d * d;
        }
        float stdv = (float)std::sqrt(sumSq / N);
        if (stdv < 1e-6f) stdv = 1.0f;
        g_featStdVec[i] = stdv;
    }

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

    printf("Per-feature normalization applied: N=%d, K=%d\n", N, K);
}

// ------------------------------------------------------------
// 5) AE training: SGD and Adam (same interfaces, identity target)
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

//float trainSGD(MLP& net,
//    std::vector<std::vector<float>>& X,
//    std::vector<std::vector<float>>& Y,
//    int epochs,
//    float lr,
//    int batchSize)
//{
//    if (X.empty()) return 0.0f;
//
//    int inputDim = (int)X[0].size();
//    int latentDim = std::min(LATENT_DIM, inputDim);
//    int hiddenDim = 0;
//
//    static AutoEncoder pcaAE;
//    pcaAE.initializeAE(inputDim, latentDim, hiddenDim);
//
//    printf("Reducing dimensionality from %d → %d using AutoEncoder (PCA mode)\n",
//        inputDim, latentDim);
//
//    // Train the AE to reconstruct X (identity mapping)
//    pcaAE.train(X, epochs, lr);
//
//    // Get latent representation and reconstruction error
//    float totalLoss = 0.0f;
//    for (int i = 0; i < X.size(); i++)
//    {
//        std::vector<float> z = pcaAE.encode(X[i]);
//        std::vector<float> x_recon = pcaAE.decode(z);
//
//        // Mean squared reconstruction error
//        float mse = 0.0f;
//        for (int j = 0; j < X[i].size(); j++)
//        {
//            float diff = x_recon[j] - X[i][j];
//            mse += diff * diff;
//        }
//        mse /= (float)X[i].size();
//        totalLoss += mse;
//    }
//
//    totalLoss /= (float)X.size();
//    printf("[PCA] Reconstruction Loss: %.6f\n", totalLoss);
//
//    // Optionally replace your existing MLP weights with the trained AE (for later reconstruction)
//    net = pcaAE.encoder;
//
//    return totalLoss;
//}
//

float trainAdam(MLP& net,
    std::vector<std::vector<float>>& X,
    std::vector<std::vector<float>>& Y,
    int epochs,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int batchSize)
{
    std::vector<std::vector<std::vector<float>>> mW = net.W;
    std::vector<std::vector<std::vector<float>>> vW = net.W;
    std::vector<std::vector<float>> mB = net.b;
    std::vector<std::vector<float>> vB = net.b;

    for (int l = 0; l < (int)mW.size(); l++)
    {
        for (int i = 0; i < (int)mW[l].size(); i++)
        {
            std::fill(mW[l][i].begin(), mW[l][i].end(), 0.0f);
            std::fill(vW[l][i].begin(), vW[l][i].end(), 0.0f);
        }
        std::fill(mB[l].begin(), mB[l].end(), 0.0f);
        std::fill(vB[l].begin(), vB[l].end(), 0.0f);
    }

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

                std::vector<float> y_pred = net.forward(X[idx]);
                float loss = net.computeLoss(y_pred, Y[idx]);
                totalLoss += loss;

                std::vector<float> gradOut(y_pred.size());
                for (int j = 0; j < (int)y_pred.size(); j++)
                {
                    gradOut[j] = (y_pred[j] - Y[idx][j]);
                }

                std::vector<std::vector<std::vector<float>>> gradW = net.W;
                std::vector<std::vector<float>> gradB = net.b;
                for (int l = 0; l < (int)gradW.size(); l++)
                {
                    for (int iN = 0; iN < (int)gradW[l].size(); iN++)
                    {
                        std::fill(gradW[l][iN].begin(), gradW[l][iN].end(), 0.0f);
                    }
                    std::fill(gradB[l].begin(), gradB[l].end(), 0.0f);
                }

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
                            prevDelta[jIn] = sum * (1.0f - a * a);
                        }
                        deltas[l - 1] = prevDelta;
                    }
                }

                for (int l = 0; l < (int)net.W.size(); l++)
                {
                    for (int iN = 0; iN < (int)net.W[l].size(); iN++)
                    {
                        for (int jIn = 0; jIn < (int)net.W[l][iN].size(); jIn++)
                        {
                            float g = gradW[l][iN][jIn];

                            mW[l][iN][jIn] = beta1 * mW[l][iN][jIn] + (1.0f - beta1) * g;
                            vW[l][iN][jIn] = beta2 * vW[l][iN][jIn] + (1.0f - beta2) * g * g;

                            float m_hat = mW[l][iN][jIn] / (1.0f - pow(beta1, t));
                            float v_hat = vW[l][iN][jIn] / (1.0f - pow(beta2, t));

                            net.W[l][iN][jIn] -= lr * m_hat / (sqrt(v_hat) + eps);
                        }

                        float gb = gradB[l][iN];

                        mB[l][iN] = beta1 * mB[l][iN] + (1.0f - beta1) * gb;
                        vB[l][iN] = beta2 * vB[l][iN] + (1.0f - beta2) * gb * gb;

                        float m_hatb = mB[l][iN] / (1.0f - pow(beta1, t));
                        float v_hatb = vB[l][iN] / (1.0f - pow(beta2, t));

                        net.b[l][iN] -= lr * m_hatb / (sqrt(v_hatb) + eps);
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
// 6) Reconstruction from fixed block (ground truth coeffs)
// ------------------------------------------------------------
void reconstruct_from_fixed_block_truth(int s, float out[RES][RES])
{
    if (s < 0 || s >= NUM_SHAPES) return;

    float dctTmp[RES][RES] = { 0 };

    std::vector<float>& feat = g_fixedFeatures[s];
    int K = std::min(TOP_K, (int)feat.size());

    for (int i = 0; i < K; i++)
    {
        int u = g_fixedU[i];
        int v = g_fixedV[i];
        dctTmp[u][v] = feat[i];
    }

    computeInverseDCT(dctTmp, out);
}

// ------------------------------------------------------------
// 7) Reconstruction from AE output
// ------------------------------------------------------------
void reconstruct_from_AE_output(int s, float out[RES][RES])
{
    if (s < 0 || s >= NUM_SHAPES) return;
    if (g_featMeanVec.empty() || g_featStdVec.empty()) return;

    std::vector<float> xNorm = g_trainX[s];
    std::vector<float> yNorm = g_autoencoder.forward(xNorm);

    std::vector<float> y(yNorm.size());
    for (int i = 0; i < (int)yNorm.size(); i++)
    {
        if (i < (int)g_featMeanVec.size())
        {
            y[i] = yNorm[i] * g_featStdVec[i] + g_featMeanVec[i];
        }
        else
        {
            y[i] = yNorm[i];
        }
    }

    float dctTmp[RES][RES] = { 0 };
    int K = std::min(TOP_K, (int)y.size());

    for (int i = 0; i < K; i++)
    {
        int u = g_fixedU[i];
        int v = g_fixedV[i];
        dctTmp[u][v] = y[i];
    }

    computeInverseDCT(dctTmp, out);
}

void drawEnergySpectrum(float dct[RES][RES], int blockU, float px, float py, float scale)
{
    // compute min/max log-energy for colour scaling
    float minE = 1e9f, maxE = -1e9f;
    static float logE[RES][RES];

    for (int u = 0; u < RES; u++)
    {
        for (int v = 0; v < RES; v++)
        {
            float e = fabs(dct[u][v]);
            e = log(1.0f + e);  // log magnitude
            logE[u][v] = e;
            minE = std::min(minE, e);
            maxE = std::max(maxE, e);
        }
    }

    float range = std::max(1e-6f, maxE - minE);

    // draw heatmap (u = horizontal, v = vertical)
    glPointSize(3.0f);
    glBegin(GL_POINTS);
    for (int u = 0; u < RES; u++)
    {
        for (int v = 0; v < RES; v++)
        {
            float val = (logE[u][v] - minE) / range * 2.0f - 1.0f; // [-1,1]
            float r, g, b;
            getJetColor(val, r, g, b);
            ( r < 1e-4f && g < 1e-4f) ? glColor3f(1,1,1): glColor3f(r, g, b);

            float x = px + u * scale;
            float y = py + v * scale;
            glVertex2f(x, y);
        }
    }
    glEnd();

    // overlay rectangle for UxU block
    float x0 = px;
    float y0 = py;
    float x1 = px + blockU * scale;
    float y1 = py + blockU * scale;

    glColor3f(1.0f, 0,0);
    glLineWidth(5.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x0, y0);
    glVertex2f(x1, y0);
    glVertex2f(x1, y1);
    glVertex2f(x0, y1);
    glEnd();

    // axes labels
    setup2d();
    char buf[64];
    sprintf(buf, "Energy Spectrum (%dx%d block)", blockU, blockU);
    drawString(buf, px + 10, py - 20);
    restore3d();
}


// ------------------------------------------------------------
// Setup + wiring
// ------------------------------------------------------------
void rebuildAll()
{
    generateDataset();
    computeFixedBlockLayout();
    buildFixedLayoutFeatures();
    buildTrainingData();

    g_autoencoder.initialize
    (
        TOP_K,
        { 32, LATENT_DIM, 32 },
        TOP_K
    );

    g_lastLoss = 0.0f;

    reconstruct_from_fixed_block_truth(g_currentShape, g_reconFixed);
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
        g_lastLoss = trainSGD(g_autoencoder, g_trainX, g_trainX, 20, 0.1f, 5);
    }
    else
    {
        g_lastLoss = trainAdam(g_autoencoder, g_trainX, g_trainX, 20, 1e-3f, 0.9f, 0.999f, 1e-8f, 5);
    }

    reconstruct_from_fixed_block_truth(g_currentShape, g_reconFixed);
    reconstruct_from_AE_output(g_currentShape, g_reconAE);
}

void draw()
{
    backGround(0.9f);
    drawGrid(100);

    glColor3f(0, 0, 0);
    std::vector<zVector>& poly = g_samples[g_currentShape].poly;
    int n = (int)poly.size();
    for (int i = 0; i < n; i++)
    {
        int j = (i + 1) % n;
        drawLine(zVecToAliceVec(poly[i]), zVecToAliceVec(poly[j]));
    }

    drawSDF(g_samples[g_currentShape].sdf, -(float)(RES * 2 + 10), -float(RES * 0.5f), 1.0f);
    drawSDF(g_reconFixed, -(float)(RES * 0.5f), -float(RES * 0.5f), 1.0f);
    drawSDF(g_reconAE, (float)(RES + 10), -float(RES * 0.5f), 1.0f);

    drawEnergySpectrum(g_samples[g_currentShape].dct, BLOCK_U, -(float)(RES * 0.5f), float(RES * 0.5 + 10) , 1.0f);


    setup2d();
    glColor3f(0, 0, 0);

    char buf[256];
    sprintf(buf, "Shape %d / %d   (keys 1-5)", g_currentShape + 1, NUM_SHAPES);
    drawString(buf, 20, 40);

    sprintf(buf, "BLOCK_U = %d  (TOP_K = %d)", BLOCK_U, TOP_K);
    drawString(buf, 20, 60);

    drawString("Left: Original SDF", 20, 80);
    drawString("Middle: Fixed UxU block reconstruction", 20, 100);
    drawString("Right: Autoencoder reconstruction", 20, 120);

    drawString(std::string("Training [t]: ") + (g_isTraining ? "ON" : "OFF"), 20, 150);
    drawString(std::string("Mode [m]: ") + ((g_trainMode == TRAIN_SGD) ? "SGD" : "Adam"), 20, 170);

    sprintf(buf, "Last loss: %.6f", g_lastLoss);
    drawString(buf, 20, 190);

    drawString("Press 'r' to regenerate shapes", 20, 210);

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
            reconstruct_from_fixed_block_truth(g_currentShape, g_reconFixed);
            reconstruct_from_AE_output(g_currentShape, g_reconAE);
        }
    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif
