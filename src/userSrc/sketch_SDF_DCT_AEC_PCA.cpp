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
static  int TOP_K = 2025;       // number of DCT coeffs used
static  int LATENT_DIM = NUM_POLYGONS -1 ;    // PCA dimensionality ( number of data points - 1)

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



Alice::vec zVecToAliceVec( zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

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

// point-in-polygon (ray casting, 2D on XY)
bool pointInPolygon( zVector& p,  std::vector<zVector>& poly)
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
float signedDistanceToPolygon( zVector& p,  std::vector<zVector>& poly)
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
void DCT2( std::vector<std::vector<float>>& input,
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
void IDCT2( std::vector<std::vector<float>>& input,
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
std::vector<float> flatten( std::vector<std::vector<float>>& mat)
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
std::vector<float> topKByMagnitude( std::vector<float>& in, int K)
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

    const float half = (float)RES * 0.5f;

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


void reconstructSampleSDF(int idx)
{
    if (g_dctFeatures.empty()) return;
    idx = std::clamp(idx, 0, (int)g_dctFeatures.size() - 1);

    // 1) Forward AE
    std::vector<float> x = g_dctFeatures[idx];
    std::vector<float> yNorm = g_autoencoder.forward(x);

    // 2) Un-normalise back to DCT coefficient scale
    std::vector<float> y(yNorm.size());
    for (size_t i = 0; i < y.size(); i++)
        y[i] = yNorm[i] * g_featStd + g_featMean;

    // 3) Fill low-frequency block
    std::vector<std::vector<float>> dctRec(RES, std::vector<float>(RES, 0.0f));
    int blockN = (int)std::sqrt((int)y.size());
    if (blockN * blockN > (int)y.size()) blockN--;

    int k = 0;
    for (int u = 0; u < blockN; u++)
        for (int v = 0; v < blockN; v++)
            if (k < (int)y.size()) dctRec[u][v] = y[k++];

    // 4) IDCT
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

    std::cout << "Reconstructed AE SDF for polygon[" << idx << "]\n";
}


void reconstructFromTopKDCT(int idx)
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

    std::cout << "Reconstructed SDF from low-freq block for polygon[" << idx << "]\n";
}


void reconstructFromStoredFeatures(int idx)
{
    if (g_dctFeatures.empty()) return;
    idx = std::clamp(idx, 0, (int)g_dctFeatures.size() - 1);

    const std::vector<float>& featNorm = g_dctFeatures[idx];

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

    std::cout << "Reconstructed SDF from stored normalized features[" << idx << "]\n";
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
    g_autoencoder.initialize(TOP_K, { LATENT_DIM }, TOP_K);

    g_isTraining = false;
    g_lastLoss = 0.0f;

    reconstructSampleSDF(g_currentIndex);
}

void update(int value)
{

    if (!g_isTraining || g_dctFeatures.empty()) return;

    float lr = 0.1f;
    float totalLoss = 0.0f;
    int count = 0;

    // One epoch over all samples per frame
    for (auto& x : g_dctFeatures)
    {
        std::vector<float> y_pred = g_autoencoder.forward(x);
        float loss = g_autoencoder.computeLoss(y_pred, x);
        totalLoss += loss;
        count++;

        std::vector<float> gradOut;
        g_autoencoder.computeGradient(x, x, gradOut);
        g_autoencoder.backward(gradOut, lr);
    }

    if (count > 0)
    {
        g_lastLoss = totalLoss / (float)count;
    }



    cout << g_lastLoss << endl;

  //  reructSample0SDF();
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

void drawSDF(const std::vector<std::vector<float>>& sdf,
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
    drawSDF(g_sdfOriginal, 20.0f, 20.0f, 1.5f);   // left
    drawSDF(g_sdfFromTopK, 220.0f, 20.0f, 1.5f);   // g_sdfFromTopK is either reconstructed from stored dctFeatures (reconstructFromStoredFeatures)or recomputed (reconstructFromTopKDCT)
    drawSDF(g_sdfReructed, 420.0f, 20.0f, 1.5f);   // right

    // Text UI
    char s[250];
    sprintf(s, "Left: original SDF[%i], Middle: g_sdfFromTopK,  Right: AE-PCA reconstructed", g_currentIndex);
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
        reconstructSampleSDF(g_currentIndex);
    }

    if (k == 't')
    {
        g_isTraining = !g_isTraining;
        reconstructSampleSDF(g_currentIndex);
    }

    if (k >= '0' && k <= '5')
    {
        int idx = k - '0';
        if (idx < NUM_POLYGONS)
        {
            g_currentIndex = idx;
           
            reconstructSampleSDF(g_currentIndex);
           // reconstructFromTopKDCT(g_currentIndex);
            reconstructFromStoredFeatures(g_currentIndex);
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

        for( int n = 0 ; n < g_dctFeatures.size() ; n++)
        {
            std::vector<float> x = g_dctFeatures[0];
            std::vector<float> y0 = g_autoencoder.forward(x);

            for (int i = 0; i < y0.size(); i++)
            {
                printf("%.4f,%.4f,%.4f,\n", y0[i], x[i], y0[i] - x[i]);
            }

            cout << " -------------------------- " << endl;
        }

        

    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif
