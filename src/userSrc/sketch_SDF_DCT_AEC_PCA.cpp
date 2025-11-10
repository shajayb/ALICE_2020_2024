#define _MAIN_
#ifdef _MAIN_

#include "main.h"


#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace zSpace;



//---------------------------------------------------------------------------
// Config
//---------------------------------------------------------------------------

static  int RES = 128;          // SDF grid resolution
static  int NUM_POLYGONS = 5;   // number of training shapes
static  int TOP_K = 64*64;       // number of DCT coeffs used
static  int LATENT_DIM = NUM_POLYGONS - 1;    // PCA dimensionality ( number of data points - 1)

//---------------------------------------------------------------------------
// Helpers
//---------------------------------------------------------------------------
// --------------------------------------------------------------
// Jet colormap: maps value in [-1, 1] to RGB
// --------------------------------------------------------------
inline void getJetColor(float value, float& r, float& g, float& b)
{
    // Clamp to [-1, 1]
    value = std::clamp(value, -1.0f, 1.0f);

    // Normalize to [0, 1]
    float x = (value + 1.0f) * 0.5f;

    float fourValue = 4.0f * x;

    r = std::clamp(std::min(fourValue - 1.5f, -fourValue + 4.5f), 0.0f, 1.0f);
    g = std::clamp(std::min(fourValue - 0.5f, -fourValue + 3.5f), 0.0f, 1.0f);
    b = std::clamp(std::min(fourValue + 0.5f, -fourValue + 2.5f), 0.0f, 1.0f);
}



Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

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

// point-in-polygon (ray casting, 2D on XY)
bool pointInPolygon(zVector& p, std::vector<zVector>& poly)
{
    bool inside = false;
    int n = (int)poly.size();

    for (int i = 0, j = n - 1; i < n; j = i++)
    {
        zVector& pi = poly[i];
        zVector& pj = poly[j];

        bool intersect = ((pi.y > p.y) != (pj.y > p.y)) &&
            (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y + 1e-12f) + pi.x);

        if (intersect) inside = !inside;
    }

    return inside;
}

// signed distance to polygon: negative inside, positive outside
float signedDistanceToPolygon(zVector& p, std::vector<zVector>& poly)
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

// 2D DCT-II (orthonormal) on square input[N][N]
void DCT2(std::vector<std::vector<float>>& input,
    std::vector<std::vector<float>>& output)
{
    int N = (int)input.size();
    int M = (int)input[0].size();

    output.assign(N, std::vector<float>(M, 0.0f));

    float cN = std::sqrt(2.0f / (float)N);
    float cM = std::sqrt(2.0f / (float)M);

    for (int u = 0; u < N; u++)
    {
        float Cu = (u == 0) ? (1.0f / std::sqrt(2.0f)) : 1.0f;

        for (int v = 0; v < M; v++)
        {
            float Cv = (v == 0) ? (1.0f / std::sqrt(2.0f)) : 1.0f;
            float sum = 0.0f;

            for (int x = 0; x < N; x++)
            {
                float ax = std::cos((float)PI * (2.0f * x + 1.0f) * (float)u / (2.0f * (float)N));

                for (int y = 0; y < M; y++)
                {
                    float ay = std::cos((float)PI * (2.0f * y + 1.0f) * (float)v / (2.0f * (float)M));
                    sum += input[x][y] * ax * ay;
                }
            }

            output[u][v] = cN * cM * Cu * Cv * sum;
        }
    }
}

// Inverse 2D DCT (for visualization)
void IDCT2(std::vector<std::vector<float>>& input,
    std::vector<std::vector<float>>& output)
{
    int N = (int)input.size();
    int M = (int)input[0].size();

    output.assign(N, std::vector<float>(M, 0.0f));

    float cN = std::sqrt(2.0f / (float)N);
    float cM = std::sqrt(2.0f / (float)M);

    for (int x = 0; x < N; x++)
    {
        for (int y = 0; y < M; y++)
        {
            float sum = 0.0f;
            for (int u = 0; u < N; u++)
            {
                float Cu = (u == 0) ? (1.0f / std::sqrt(2.0f)) : 1.0f;
                float ax = std::cos((float)PI * (2.0f * x + 1.0f) * (float)u / (2.0f * (float)N));

                for (int v = 0; v < M; v++)
                {
                    float Cv = (v == 0) ? (1.0f / std::sqrt(2.0f)) : 1.0f;
                    float ay = std::cos((float)PI * (2.0f * y + 1.0f) * (float)v / (2.0f * (float)M));

                    sum += Cu * Cv * input[u][v] * ax * ay;
                }
            }
            output[x][y] = cN * cM * sum;
        }
    }
}

// Flatten full DCT (row-major)
std::vector<float> flatten(std::vector<std::vector<float>>& mat)
{
    int N = (int)mat.size();
    int M = (int)mat[0].size();

    std::vector<float> out;
    out.reserve(N * M);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            out.push_back(mat[i][j]);
        }
    }
    return out;
}

// Take top-K by magnitude, keeping positions implicit (PCA style on coefficient vector)
std::vector<float> topKByMagnitude(std::vector<float>& in, int K)
{
    std::vector<float> tmp = in;
    if (K > (int)tmp.size()) K = (int)tmp.size();

    std::nth_element(
        tmp.begin(),
        tmp.begin() + K,
        tmp.end(),
        [](float a, float b)
        {
            return std::fabs(a) > std::fabs(b);
        });

    tmp.resize(K);
    return tmp;
}

// Normalize vector (zero mean, unit variance) for better training stability
void normalizeVector(std::vector<float>& v)
{
    if (v.empty()) return;

    float mean = 0.0f;
    for (float x : v) mean += x;
    mean /= (float)v.size();

    float var = 0.0f;
    for (float x : v)
    {
        float d = x - mean;
        var += d * d;
    }
    var /= (float)v.size();
    float stdDev = (var > 1e-12f) ? std::sqrt(var) : 1.0f;

    for (float& x : v)
    {
        x = (x - mean) / stdDev;
    }
}


#include "genericMLP.h"

//---------------------------------------------------------------------------
// Global State
//---------------------------------------------------------------------------

std::vector<std::vector<zVector>> g_polygons;      // [NUM_POLYGONS][verts]
std::vector<std::vector<float>>  g_dctFeatures;    // [NUM_POLYGONS][TOP_K]

MLP g_autoencoder;

bool g_isTraining = false;
float g_lastLoss = 0.0f;
float g_featMean = 0.0f;
float g_featStd = 1.0f;

int   g_featureDim = 0;   // length of each feature vector
int   g_blockN = 0;   // low-frequency block size (g_blockN * g_blockN == g_featureDim)





// For visualization of reruction:
std::vector<std::vector<float>> g_sdfOriginal;     // RES x RES of polygon[0]
std::vector<std::vector<float>> g_sdfReructed;
std::vector<std::vector<float>> g_sdfFromTopK;
int g_currentIndex = 2;   // which polygon to visualise

//---------------------------------------------------------------------------
// Pipeline Steps
//---------------------------------------------------------------------------

void generatePolygons()
{
    g_polygons.clear();
    g_polygons.reserve(NUM_POLYGONS);

    // simple random star/convex-ish polygons
    for (int i = 0; i < NUM_POLYGONS; i++)
    {
        int n = 5 + (rand() % 5); // 5-9 vertices
        float radius = 20.0f + (rand() % 1000) / 1000.0f * 15.0f; // 20-35
        zVector c(
            -40.0f + (rand() % 1000) / 1000.0f * 80.0f,
            -40.0f + (rand() % 1000) / 1000.0f * 80.0f,
            0.0f
        );

        std::vector<zVector> poly;
        poly.reserve(n);

        for (int k = 0; k < n; k++)
        {
            float t = (float)k / (float)n;
            float ang = 2.0f * (float)PI * t;
            float jitter = 0.85f + (rand() % 1000) / 1000.0f * 0.3f;
            float r = radius * jitter;

            zVector p(
                c.x + r * std::cos(ang),
                c.y + r * std::sin(ang),
                0.0f
            );
            poly.push_back(p);
        }

        g_polygons.push_back(poly);
    }
}

//void computeSDFandDCT()
//{
//    g_dctFeatures.clear();
//    g_dctFeatures.reserve(NUM_POLYGONS);
//
//    g_sdfOriginal.assign(RES, std::vector<float>(RES, 0.0f));
//
//    // Domain mapping: grid indices -> world coords
//    // Here, [-64, 64] x [-64, 64]
//     float half = (float)RES * 0.5f;
//
//    for (int pi = 0; pi < NUM_POLYGONS; pi++)
//    {
//         auto& poly = g_polygons[pi];
//
//        // 1) SDF
//        std::vector<std::vector<float>> sdf(RES, std::vector<float>(RES, 0.0f));
//
//        for (int i = 0; i < RES; i++)
//        {
//            for (int j = 0; j < RES; j++)
//            {
//                float x = (float)i - half;
//                float y = (float)j - half;
//                zVector p(x, y, 0);
//                sdf[i][j] = signedDistanceToPolygon(p, poly);
//            }
//        }
//
//        if (pi == 0)
//        {
//            g_sdfOriginal = sdf;
//        }
//
//        // 2) DCT2
//        std::vector<std::vector<float>> dct;
//        DCT2(sdf, dct);
//
//        // 3) flatten and keep TOP_K by magnitude
//        std::vector<float> flat = flatten(dct);
//        std::vector<float> feat = topKByMagnitude(flat, TOP_K);
//
//        // 4) normalize for training
//        normalizeVector(feat);
//
//        g_dctFeatures.push_back(feat);
//    }
//}


void computeSDFandDCT()
{
    g_dctFeatures.clear();
    g_dctFeatures.reserve(NUM_POLYGONS);
    g_sdfOriginal.assign(RES, std::vector<float>(RES, 0.0f));

     float half = (float)RES * 0.5f;

    // Temporary store all raw features for later global normalisation
    std::vector<std::vector<float>> rawFeatures;
    rawFeatures.reserve(NUM_POLYGONS);

    for (int pi = 0; pi < NUM_POLYGONS; pi++)
    {
        auto& poly = g_polygons[pi];

        // -------------------------------------------------------------
        // 1) Compute SDF grid
        // -------------------------------------------------------------
        std::vector<std::vector<float>> sdf(RES, std::vector<float>(RES, 0.0f));
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                float x = (float)i - half;
                float y = (float)j - half;
                zVector p(x, y, 0);
                sdf[i][j] = signedDistanceToPolygon(p, poly);
            }
        }

        if (pi == 0) g_sdfOriginal = sdf;

        // -------------------------------------------------------------
        // 2) Forward DCT
        // -------------------------------------------------------------
        std::vector<std::vector<float>> dct;
        DCT2(sdf, dct);

        // -------------------------------------------------------------
        // 3) Extract fixed low-frequency block
        // -------------------------------------------------------------
        int blockN = (int)std::sqrt(TOP_K);
        if (blockN * blockN > TOP_K) blockN--;

        std::vector<float> feat;
        feat.reserve(blockN * blockN);
        for (int u = 0; u < blockN; u++)
        {
            for (int v = 0; v < blockN; v++)
            {
                feat.push_back(dct[u][v]);
            }
        }

        rawFeatures.push_back(feat);
    }

    // -------------------------------------------------------------
    // 4) Global mean / std normalisation
    // -------------------------------------------------------------
    double sum = 0.0, sumSq = 0.0;
    size_t total = 0;
    for (auto& f : rawFeatures)
        for (float v : f) { sum += v; sumSq += v * v; total++; }

    g_featMean = (total > 0) ? (float)(sum / total) : 0.0f;
    double var = (total > 0) ? (sumSq / total) - (g_featMean * g_featMean) : 1.0;
    g_featStd = (var > 1e-12) ? (float)std::sqrt(var) : 1.0f;

    // -------------------------------------------------------------
    // 5) Normalise each feature vector
    // -------------------------------------------------------------
    for (auto& f : rawFeatures)
    {
        std::vector<float> normFeat = f;
        for (float& v : normFeat) v = (v - g_featMean) / g_featStd;
        g_dctFeatures.push_back(normFeat);
    }

    std::cout << "Computed SDF and DCT features for "
        << NUM_POLYGONS << " polygons. Global mean="
        << g_featMean << " std=" << g_featStd << std::endl;
}


void reructSampleSDF(int idx)
{
    if (g_dctFeatures.empty()) return;
    idx = std::clamp(idx, 0, (int)g_dctFeatures.size() - 1);

     std::vector<float>& featNorm = g_autoencoder.forward( g_dctFeatures[idx] );

    // 1) Un-normalise
    std::vector<float> feat(featNorm.size());
    for (size_t i = 0; i < feat.size(); i++)
        feat[i] = featNorm[i] * g_featStd + g_featMean;

    // 2) Fill DCT grid
    std::vector<std::vector<float>> dctRec(RES, std::vector<float>(RES, 0.0f));
    int blockN = (int)std::sqrt((int)feat.size());
    if (blockN * blockN > (int)feat.size()) blockN--;

    int k = 0;
    for (int u = 0; u < blockN; u++)
        for (int v = 0; v < blockN; v++)
            if (k < feat.size()) dctRec[u][v] = feat[k++];

    // 3) IDCT
    IDCT2(dctRec, g_sdfReructed);

    // 5) Original SDF for comparison
    std::vector<std::vector<float>> sdf(RES, std::vector<float>(RES, 0.0f));
    float half = (float)RES * 0.5f;
    auto& poly = g_polygons[idx];
    for (int i = 0; i < RES; i++)
        for (int j = 0; j < RES; j++)
        {
            float xw = (float)i - half;
            float yw = (float)j - half;
            zVector p(xw, yw, 0);
            sdf[i][j] = signedDistanceToPolygon(p, poly);
        }
    g_sdfOriginal = sdf;

    std::cout << "Reructed AE SDF for polygon[" << idx << "]\n";
}


void reructFromTopKDCT(int idx)
{
    if (g_polygons.empty()) return;
    idx = std::clamp(idx, 0, (int)g_polygons.size() - 1);

    // 1) Original SDF
    std::vector<std::vector<float>> sdf(RES, std::vector<float>(RES, 0.0f));
    float half = (float)RES * 0.5f;
    auto& poly = g_polygons[idx];
    for (int i = 0; i < RES; i++)
        for (int j = 0; j < RES; j++)
        {
            float xw = (float)i - half;
            float yw = (float)j - half;
            zVector p(xw, yw, 0);
            sdf[i][j] = signedDistanceToPolygon(p, poly);
        }
    g_sdfOriginal = sdf;

    // 2) DCT
    std::vector<std::vector<float>> dct;
    DCT2(sdf, dct);

    // 3) Zero everything except low-frequency block
    std::vector<std::vector<float>> dctTrunc = dct;
    int blockN = (int)std::sqrt(TOP_K);
    if (blockN * blockN > TOP_K) blockN--;
    for (int u = 0; u < RES; u++)
        for (int v = 0; v < RES; v++)
            if (u >= blockN || v >= blockN)
                dctTrunc[u][v] = 0.0f;

    // 4) IDCT
    IDCT2(dctTrunc, g_sdfFromTopK);

    std::cout << "Reructed SDF from low-freq block for polygon[" << idx << "]\n";
}


void reructFromStoredFeatures(int idx)
{
    if (g_dctFeatures.empty()) return;
    idx = std::clamp(idx, 0, (int)g_dctFeatures.size() - 1);

     std::vector<float>& featNorm = g_dctFeatures[idx];

    // 1) Un-normalise
    std::vector<float> feat(featNorm.size());
    for (size_t i = 0; i < feat.size(); i++)
        feat[i] = featNorm[i] * g_featStd + g_featMean;

    // 2) Fill DCT grid
    std::vector<std::vector<float>> dctRec(RES, std::vector<float>(RES, 0.0f));
    int blockN = (int)std::sqrt((int)feat.size());
    if (blockN * blockN > (int)feat.size()) blockN--;

    int k = 0;
    for (int u = 0; u < blockN; u++)
        for (int v = 0; v < blockN; v++)
            if (k < feat.size()) dctRec[u][v] = feat[k++];

    // 3) IDCT
    IDCT2(dctRec, g_sdfFromTopK);

    // 4) Original SDF for comparison
    std::vector<std::vector<float>> sdf(RES, std::vector<float>(RES, 0.0f));
    float half = (float)RES * 0.5f;
    auto& poly = g_polygons[idx];
    for (int i = 0; i < RES; i++)
        for (int j = 0; j < RES; j++)
        {
            float xw = (float)i - half;
            float yw = (float)j - half;
            zVector p(xw, yw, 0);
            sdf[i][j] = signedDistanceToPolygon(p, poly);
        }
    g_sdfOriginal = sdf;

    std::cout << "Reructed SDF from stored normalized features[" << idx << "]\n";
}

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
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        totalLoss = 0.0f;

        for (int b = 0; b < X.size(); b += batchSize)
        {
            int end = std::min((int)X.size(), b + batchSize);
            for (int i = b; i < end; ++i)
            {
                int idx = indices[i];
                std::vector<float> gradOut;

                // Forward + backward
                net.computeGradient(X[idx], Y[idx], gradOut);
                net.backward(gradOut, lr);

                // Compute loss for tracking
                auto y_pred = net.forward(X[idx]);
                totalLoss += net.computeLoss(y_pred, Y[idx]);
            }
        }

        totalLoss /= X.size();
        printf("Epoch %d | Loss: %.6f\n", epoch, totalLoss);
    }

    return totalLoss;
}

// -----------------------------------------------------------------------------
// Adam Optimizer-based training loop for MLP
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Stable Adam Optimizer training for MLP
// -----------------------------------------------------------------------------
float trainAdam(MLP& net,
    std::vector<std::vector<float>>& X,
    std::vector<std::vector<float>>& Y,
    int epochs = 500,
    float lr = 1e-4f,           // smaller LR for stability
    float beta1 = 0.9f,
    float beta2 = 0.999f,
    float eps = 1e-8f,
    int batchSize = 8)
{
    // Allocate Adam states (same shapes as weights/biases)
    std::vector<std::vector<std::vector<float>>> mW = net.W;
    std::vector<std::vector<std::vector<float>>> vW = net.W;
    std::vector<std::vector<float>> mB = net.b;
    std::vector<std::vector<float>> vB = net.b;

    // Initialize to zeros
    for (auto& L : mW) for (auto& r : L) std::fill(r.begin(), r.end(), 0.0f);
    for (auto& L : vW) for (auto& r : L) std::fill(r.begin(), r.end(), 0.0f);
    for (auto& b : mB) std::fill(b.begin(), b.end(), 0.0f);
    for (auto& b : vB) std::fill(b.begin(), b.end(), 0.0f);

    std::vector<int> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(std::random_device{}());

    float totalLoss = 0.0f;
    int t = 0; // timestep for bias correction

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        totalLoss = 0.0f;

        for (int b = 0; b < X.size(); b += batchSize)
        {
            int end = std::min((int)X.size(), b + batchSize);

            for (int i = b; i < end; ++i)
            {
                int idx = indices[i];
                t++;

                // Forward pass
                std::vector<float> y_pred = net.forward(X[idx]);

                // Compute loss and simple output gradient
                float loss = net.computeLoss(y_pred, Y[idx]);
                totalLoss += loss;

                std::vector<float> gradOut(y_pred.size());
                for (int j = 0; j < y_pred.size(); ++j)
                {
                    // Scaled derivative of MSE wrt output
                    gradOut[j] = (y_pred[j] - Y[idx][j]) / (float)y_pred.size();
                }

                // ---- Backpropagate manually to get true gradients ----
                std::vector<std::vector<std::vector<float>>> gradW = net.W;
                std::vector<std::vector<float>> gradB = net.b;
                for (auto& L : gradW) for (auto& r : L) std::fill(r.begin(), r.end(), 0.0f);
                for (auto& bL : gradB) std::fill(bL.begin(), bL.end(), 0.0f);

                std::vector<std::vector<float>> deltas(net.W.size());
                deltas.back() = gradOut; // last layer delta

                // backward pass to compute per-layer gradients
                for (int l = net.W.size() - 1; l >= 0; --l)
                {
                    std::vector<float>& delta = deltas[l];
                    std::vector<float> prevActiv = net.activations[l];
                    gradB[l].resize(net.b[l].size());
                    for (int iNeuron = 0; iNeuron < net.W[l].size(); ++iNeuron)
                    {
                        for (int jInput = 0; jInput < net.W[l][iNeuron].size(); ++jInput)
                        {
                            gradW[l][iNeuron][jInput] += delta[iNeuron] * prevActiv[jInput];
                        }
                        gradB[l][iNeuron] += delta[iNeuron];
                    }

                    // Compute delta for previous layer (except first)
                    if (l > 0)
                    {
                        std::vector<float> prevDelta(net.W[l][0].size(), 0.0f);
                        for (int jInput = 0; jInput < net.W[l][0].size(); ++jInput)
                        {
                            float sum = 0.0f;
                            for (int iNeuron = 0; iNeuron < net.W[l].size(); ++iNeuron)
                            {
                                sum += delta[iNeuron] * net.W[l][iNeuron][jInput];
                            }
                            float a = net.activations[l][jInput];
                            prevDelta[jInput] = sum * (1 - a * a); // tanh' for hidden layers
                        }
                        deltas[l - 1] = prevDelta;
                    }
                }

                // ---- Apply Adam updates ----
                for (int l = 0; l < net.W.size(); ++l)
                {
                    for (int iNeuron = 0; iNeuron < net.W[l].size(); ++iNeuron)
                    {
                        for (int jInput = 0; jInput < net.W[l][iNeuron].size(); ++jInput)
                        {
                            float g = gradW[l][iNeuron][jInput];
                            // Gradient clipping for safety
                            g = std::clamp(g, -1.0f, 1.0f);

                            mW[l][iNeuron][jInput] = beta1 * mW[l][iNeuron][jInput] + (1 - beta1) * g;
                            vW[l][iNeuron][jInput] = beta2 * vW[l][iNeuron][jInput] + (1 - beta2) * g * g;

                            float m_hat = mW[l][iNeuron][jInput] / (1 - std::pow(beta1, t));
                            float v_hat = vW[l][iNeuron][jInput] / (1 - std::pow(beta2, t));

                            net.W[l][iNeuron][jInput] -= lr * m_hat / (std::sqrt(v_hat) + eps);
                        }

                        float gb = gradB[l][iNeuron];
                        gb = std::clamp(gb, -1.0f, 1.0f);
                        mB[l][iNeuron] = beta1 * mB[l][iNeuron] + (1 - beta1) * gb;
                        vB[l][iNeuron] = beta2 * vB[l][iNeuron] + (1 - beta2) * gb * gb;

                        float m_hatb = mB[l][iNeuron] / (1 - std::pow(beta1, t));
                        float v_hatb = vB[l][iNeuron] / (1 - std::pow(beta2, t));

                        net.b[l][iNeuron] -= lr * m_hatb / (std::sqrt(v_hatb) + eps);
                    }
                }
            }
        }

        totalLoss /= X.size();
        printf("Epoch %d | Adam Loss: %.6f\n", epoch, totalLoss);
    }

    return totalLoss;
}


//---------------------------------------------------------------------------
// zSpace Hooks
//---------------------------------------------------------------------------

void setup()
{
    backGround(0.9);
    drawGrid(100);

    srand(1);

    generatePolygons();
    computeSDFandDCT();

    // PCA-style linear AE: 2048 -> 16 -> 2048
    // (Assumes genericMLP is in linear mode: no tanh)
    g_autoencoder.initialize(TOP_K, { 8,LATENT_DIM,8 }, TOP_K);

    g_isTraining = false;
    g_lastLoss = 0.0f;

    reructSampleSDF(g_currentIndex);
}

void update(int value)
{

    if (!g_isTraining || g_dctFeatures.empty()) return;

    float lr = 0.1f;
    float totalLoss = 0.0f;
    int count = 0;

    g_lastLoss = trainSGD(g_autoencoder, g_dctFeatures, g_dctFeatures, 500, lr, 8);
   // g_lastLoss = trainAdam(g_autoencoder, g_dctFeatures, g_dctFeatures, 800, 0.001f, 0.9f, 0.999f, 1e-8f, 8);
   
}

//void drawSDF( std::vector<std::vector<float>>& sdf, float px, float py, float scale)
//{
//    if (sdf.empty()) return;
//
//    int N = (int)sdf.size();
//    int M = (int)sdf[0].size();
//
//    float vmin = 1e9f, vmax = -1e9f;
//    for (int i = 0; i < N; i++)
//    {
//        for (int j = 0; j < M; j++)
//        {
//            float v = sdf[i][j];
//            if (v < vmin) vmin = v;
//            if (v > vmax) vmax = v;
//        }
//    }
//    float invRange = (vmax > vmin + 1e-6f) ? 1.0f / (vmax - vmin) : 1.0f;
//
//    setup2d();
//
//    for (int i = 0; i < N; i += 2)
//    {
//        for (int j = 0; j < M; j += 2)
//        {
//            float v = sdf[i][j];
//            float t = (v - vmin) * invRange;  // [0,1]
//            t = clampFloat(t, 0.0f, 1.0f);
//
//            // grayscale
//            float g = t;
//            glColor3f(g, g, g);
//
//            float x = px + i * scale;
//            float y = py + j * scale;
//
//            glBegin(GL_POINTS);
//            glVertex2f(x, y);
//            glEnd();
//        }
//    }
//
//    restore3d();
//}

void drawSDF( std::vector<std::vector<float>>& sdf,
    float px, float py, float scale)
{
    if (sdf.empty()) return;

    int N = (int)sdf.size();
    int M = (int)sdf[0].size();

    // find range for normalization
    float vmin = 1e9f, vmax = -1e9f;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
        {
            float v = sdf[i][j];
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }

    // normalise to [-1,1] for Jet color
    float mid = 0.5f * (vmax + vmin);
    float range = (vmax - vmin) * 0.5f;
    if (range < 1e-6f) range = 1.0f;

    setup2d();

    glPointSize(2.0f);
    glBegin(GL_POINTS);

    for (int i = 0; i < N; i += 2)
    {
        for (int j = 0; j < M; j += 2)
        {
            float normVal = (sdf[i][j] - mid) / range;  // [-1,1]
            float r, g, b;
            getJetColor(normVal, r, g, b);
            glColor3f(r, g, b);

            float x = px + i * scale;
            float y = py + j * scale;
            glVertex2f(x, y);
        }
    }

    glEnd();
    restore3d();
}


void draw()
{
    backGround(0.9);
    drawGrid(100);

    // Draw polygons
    glColor3f(0, 0, 0);
    for (int pi = 0; pi < (int)g_polygons.size(); pi++)
    {
        auto& poly = g_polygons[pi];
        int n = (int)poly.size();
        for (int i = 0; i < n; i++)
        {
            int j = (i + 1) % n;
            drawLine(zVecToAliceVec(poly[i]), zVecToAliceVec(poly[j]));
        }
    }

    // Visualize original vs reructed SDF of polygon[0]
    glColor3f(0, 0, 0);
        drawSDF(g_sdfOriginal, 20.0f, 20.0f, 1.5f);   // left
        drawSDF(g_sdfFromTopK, 220.0f, 20.0f, 1.5f);   // g_sdfFromTopK is either reructed from stored dctFeatures (reructFromStoredFeatures)or recomputed (reructFromTopKDCT)
        drawSDF(g_sdfReructed, 420.0f, 20.0f, 1.5f);   // right

        // Text UI
        char s[250];
        sprintf(s, "Left: original SDF[%i], Middle: g_sdfFromTopK,  Right: AE-PCA reructed", g_currentIndex);
    setup2d();

        drawString(s, 20, 410);
        drawString(std::string("Training [t]: ") + (g_isTraining ? "ON" : "OFF"), 20, 430);
        drawString("Last avg loss: " + std::to_string(g_lastLoss), 20, 450);
        drawString("Press 'r' to regenerate polygons", 20, 470);

    restore3d();
}

void keyPress(unsigned char k, int xm, int ym)
{
    if (k == 'r')
    {
        generatePolygons();
        computeSDFandDCT();
        reructSampleSDF(g_currentIndex);
    }

    if (k == 't')
    {
        g_isTraining = !g_isTraining;
       // reructSampleSDF(g_currentIndex);
    }

    if (k >= '0' && k <= '4')
    {
        int idx = k - '0';
        if (idx < NUM_POLYGONS)
        {
            g_currentIndex = idx;

            reructSampleSDF(g_currentIndex);
            // reructFromTopKDCT(g_currentIndex);
            reructFromStoredFeatures(g_currentIndex);

            //



            std::vector<float> x = g_dctFeatures[g_currentIndex];
            std::vector<float> y0 = g_autoencoder.forward(x);

            float minX = std::numeric_limits<float>::max();
            float maxX = std::numeric_limits<float>::lowest();
            float minY = std::numeric_limits<float>::max();
            float maxY = std::numeric_limits<float>::lowest();

            double diffSum = 0.0;
            float diffMin = std::numeric_limits<float>::max();
            float diffMax = std::numeric_limits<float>::lowest();

            for (int i = 0; i < (int)y0.size(); i++)
            {
                float diff = y0[i] - x[i];

                if(diff > 1e-2)
                printf("%.4f, %.4f, %.4f\n", y0[i], x[i], diff);

                diffSum += std::fabs(diff);
                diffMin = std::min(diffMin, diff);
                diffMax = std::max(diffMax, diff);

                minX = std::min(minX, x[i]);
                maxX = std::max(maxX, x[i]);
                minY = std::min(minY, y0[i]);
                maxY = std::max(maxY, y0[i]);
            }

            std::cout << "--------------------------\n";
            std::cout << "AE output vs input summary:\n";
            std::cout << "  X range  = [" << minX << ", " << maxX << "]\n";
            std::cout << "  Y range  = [" << minY << ", " << maxY << "]\n";
            std::cout << "  Diff min = " << diffMin
                << "  max = " << diffMax
                << "  mean |diff| = "
                << diffSum / (double)y0.size() << std::endl;
        }
    }

    if (k == 'f')
    {

        std::vector<float> x = g_dctFeatures[0];
        std::vector<float> y0 = g_autoencoder.forward(x);

        x = g_dctFeatures[1];
        std::vector<float> y1 = g_autoencoder.forward(x);

        for (int i = 0; i < y1.size(); i++)
        {
            printf("%.4f,%.4f,%.4f,\n", y0[i], y1[i], y0[i] - y1[i]);
        }

    }

    if (k == 'u')
    {

        for (int n = 0; n < g_dctFeatures.size(); n++)
        {
            std::vector<float> x = g_dctFeatures[n];
            std::vector<float> y0 = g_autoencoder.forward(x);

            float minX = std::numeric_limits<float>::max();
            float maxX = std::numeric_limits<float>::lowest();
            float minY = std::numeric_limits<float>::max();
            float maxY = std::numeric_limits<float>::lowest();

            double diffSum = 0.0;
            float diffMin = std::numeric_limits<float>::max();
            float diffMax = std::numeric_limits<float>::lowest();

            for (int i = 0; i < (int)y0.size(); i++)
            {
                float diff = y0[i] - x[i];
                printf("%.4f, %.4f, %.4f\n", y0[i], x[i], diff);

                diffSum += std::fabs(diff);
                diffMin = std::min(diffMin, diff);
                diffMax = std::max(diffMax, diff);

                minX = std::min(minX, x[i]);
                maxX = std::max(maxX, x[i]);
                minY = std::min(minY, y0[i]);
                maxY = std::max(maxY, y0[i]);
            }

            std::cout << "--------------------------\n";
            std::cout << "AE output vs input summary:\n";
            std::cout << "  X range  = [" << minX << ", " << maxX << "]\n";
            std::cout << "  Y range  = [" << minY << ", " << maxY << "]\n";
            std::cout << "  Diff min = " << diffMin
                << "  max = " << diffMax
                << "  mean |diff| = "
                << diffSum / (double)y0.size() << std::endl;

        }



    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif