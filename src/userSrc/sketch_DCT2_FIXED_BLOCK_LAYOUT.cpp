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

#include <fstream>
#include <sstream>
#include <string>
#include <cctype>
#include <vector>


using namespace zSpace;

// ------------------------------------------------------------
// Config
// ------------------------------------------------------------
#define RES 128
#define NUM_SHAPES 5

// Fixed low-frequency block size U x U
#define BLOCK_U 12
#define TOP_K (BLOCK_U * BLOCK_U)

#define LATENT_DIM 2

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

//--------------------------------------------------------------
// SDF : Union of two circles
//--------------------------------------------------------------
//--------------------------------------------------------------
// SDF : Union of multiple circles (similar style to sdf_Polygon / sdf_Voronoi)

//--------------------------------------------------------------

//--------------------------------------------------------------
// Helper : distribute circles around the center of the SDF domain
//--------------------------------------------------------------

static bool init = false;
static std::vector<zVector> centers;
static std::vector<float> radii;

inline void setupConcentricCircles(std::vector<zVector>& centers,
    std::vector<float>& radii,
    int numCircles = 6,
    float parentRadius = 0.3f,
    float minR = 0.05f,
    float maxR = 0.15f)
{
    centers.resize(numCircles);
    radii.resize(numCircles);

    for (int i = 0; i < numCircles; i++)
    {
        // evenly spaced angles around origin
        float angle = (2.0f * PI * i) / numCircles;

        // small offset from origin so they stay near the center
        float x = parentRadius * cos(angle);
        float y = parentRadius * sin(angle);

        centers[i] = zVector(x, y, 0);
        radii[i] = ofRandom(minR, maxR);
    }
}

float sdf_CirclesUnion(zVector& p)
{

    float dMin = 1e9f;
    for (int i = 0; i < centers.size(); i++)
    {
        zVector d = p - centers[i];
        float sd = sqrt(d.x * d.x + d.y * d.y) - radii[i];
        dMin = std::min(dMin, sd);
    }

    return dMin;
}


std::vector<zVector> randomPolygon(int n, float radiusMin = 0.1f, float radiusMax = 0.15f)
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

inline void normaliseSDF(float field[RES][RES], double targetMin = -1.0, double targetMax = 1.0)
{
    double fMin = 1e9, fMax = -1e9;

    for (int i = 0; i < RES; i++)
        for (int j = 0; j < RES; j++)
        {
            fMin = std::min(fMin, (double)field[i][j]);
            fMax = std::max(fMax, (double)field[i][j]);
        }

    double range = (fMax - fMin);
    if (range < 1e-9) range = 1.0;

    for (int i = 0; i < RES; i++)
        for (int j = 0; j < RES; j++)
        {
            double t = (field[i][j] - fMin) / range;
            field[i][j] = (float)(targetMin + t * (targetMax - targetMin));
        }
}


///

// ------------------------------------------------------------
// Data structures
// ------------------------------------------------------------
//struct DCTSample
//{
//    std::vector<zVector> poly;
//    float sdf[RES][RES];
//    float dct[RES][RES];
//};

struct DCTSample
{
    std::vector<std::vector<zVector>> polys;   // multiple polygons (outer + inner)
    float zValue = 0.0f;                       // shared Z height
    float sdf[RES][RES];
    float dct[RES][RES];
};


std::vector<std::vector<zVector>> readPolygonsFromInShapes( const std::string& path = "data/inShapes.json")
{
    std::ifstream file(path);
    std::vector<std::vector<zVector>> polygons;
    if (!file.is_open())
    {
        std::cerr << "Cannot open " << path << std::endl;
        return polygons;
    }

    std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    size_t pos = 0;
    while ((pos = data.find("\"polys\"", pos)) != std::string::npos)
    {
        pos = data.find('[', pos); // move to first [
        if (pos == std::string::npos) break;

        int depth = 0;
        std::string polyBlock;
        for (size_t i = pos; i < data.size(); ++i)
        {
            char c = data[i];
            if (c == '[') depth++;
            if (c == ']') depth--;
            polyBlock += c;
            if (depth == 0)
            {
                pos = i;
                break;
            }
        }

        // --- Extract coordinates from polyBlock
        std::vector<zVector> currentPoly;
        std::string num;
        float x = 0, y = 0, z = 0;
        int count = 0;
        for (size_t i = 0; i < polyBlock.size(); i++)
        {
            char c = polyBlock[i];
            if (std::isdigit(c) || c == '-' || c == '.' || c == 'e' || c == 'E')
            {
                num += c;
            }
            else
            {
                if (!num.empty())
                {
                    float val = std::stof(num);
                    num.clear();

                    if (count == 0) x = val;
                    else if (count == 1) y = val;
                    else if (count == 2)
                    {
                        z = val;
                        currentPoly.push_back(zVector(x, y, z));
                        count = -1;
                    }
                    count++;
                }
            }
        }

        if (!currentPoly.empty())
            polygons.push_back(currentPoly);
    }

    std::cout << "Parsed " << polygons.size() << " polygons from " << path << std::endl;
    return polygons;
}

struct PolyWithRole
{
    std::vector<zVector> pts;
    bool isHole = false;
};

std::vector<PolyWithRole> classifyPolygons(std::vector<std::vector<zVector>>& polys)
{
    std::vector<PolyWithRole> result;
    result.reserve(polys.size());

    for (int i = 0; i < polys.size(); i++)
    {
        PolyWithRole entry;
        entry.pts = polys[i];
        entry.isHole = false;
        result.push_back(entry);
    }

    // determine which polygons are holes
    for (int i = 0; i < result.size(); i++)
    {
        // compute centroid of polygon i
        zVector c(0, 0, 0);
        for (auto& p : result[i].pts) c += p;
        c /= (float)result[i].pts.size();

        // check if centroid lies inside another polygon
        for (int j = 0; j < result.size(); j++)
        {
            if (i == j) continue;
            if (pointInPolygon(c, result[j].pts))
            {
                result[i].isHole = true;
                break;
            }
        }
    }

    return result;
}

// Updated SDF computation respecting outer vs inner polygons
float sdf_MultiPolygon(zVector& p, std::vector<std::vector<zVector>>& polys)
{
    auto classified = classifyPolygons(polys);

    if (classified.empty()) return 1e6f;

    float d = sdf_Polygon(p, classified[0].pts);

    for (int i = 1; i < classified.size(); i++)
    {
        float di = sdf_Polygon(p, classified[i].pts);
        if (classified[i].isHole)
            d = std::max(d, -di);
        else
            d = std::min(d, di);
    }

    return d;
}




// ------------------------------------------------------------
// Rescale polygons to fit within [-1,1] box (or any target box)
// ------------------------------------------------------------
void normalizePolygonsToBBox(std::vector<std::vector<zVector>>& polys,
     zVector& targetMin = zVector(-1, -1, 0),
     zVector& targetMax = zVector(1, 1, 0))
{
    if (polys.empty()) return;

    zVector bmin(1e9, 1e9, 0);
    zVector bmax(-1e9, -1e9, 0);

    // find current bounding box (XY only)
    for (auto& poly : polys)
    {
        for (auto& p : poly)
        {
            bmin.x = std::min(bmin.x, p.x);
            bmin.y = std::min(bmin.y, p.y);
            bmax.x = std::max(bmax.x, p.x);
            bmax.y = std::max(bmax.y, p.y);
        }
    }

    zVector srcSize = bmax - bmin;
    if (srcSize.x < 1e-6f) srcSize.x = 1.0f;
    if (srcSize.y < 1e-6f) srcSize.y = 1.0f;

    zVector dstSize = targetMax - targetMin;
    zVector scale(dstSize.x / srcSize.x, dstSize.y / srcSize.y, 1);

    zVector srcCenter = (bmax + bmin) * 0.5;
    zVector dstCenter = (targetMax + targetMin)  * 0.5;

    // apply uniform scaling (keep aspect)
    float uniformScale = std::min(scale.x, scale.y);

    for (auto& poly : polys)
    {
        for (auto& p : poly)
        {
            // center + scale + flatten
            p.x = dstCenter.x + (p.x - srcCenter.x) * uniformScale;
            p.y = dstCenter.y + (p.y - srcCenter.y) * uniformScale;
            p.z = 0.0f;
        }
    }
}


std::vector<DCTSample> groupPolygonsByZ(std::vector<std::vector<zVector>>& allPolygons)
{
    std::map<int, DCTSample> groups;
    const float zTol = 1e-3f;
    const float closeTol = 1e-6f;

    // 1) Group input chains by (quantized) z
    for (auto& poly : allPolygons)
    {
        if (poly.empty())
        {
            continue;
        }

        float z = poly[0].z;
        int zKey = (int)std::round(z * 1000.0f);

        DCTSample& sample = groups[zKey];
        sample.zValue = z;

        // 2) Split this chain into multiple closed loops
        std::vector<zVector> currentLoop;
        zVector firstPt;
        bool haveFirst = false;

        for (int i = 0; i < (int)poly.size(); i++)
        {
            zVector p = poly[i];
            p.z = 0.0f; // flatten for 2D SDF

            if (!haveFirst)
            {
                currentLoop.clear();
                currentLoop.push_back(p);
                firstPt = p;
                haveFirst = true;
            }
            else
            {
                currentLoop.push_back(p);

                // check if we closed the loop (back to first point)
                if (currentLoop.size() > 2 && p.distanceTo(firstPt) < closeTol)
                {
                    // enforce exact closure
                    currentLoop.back() = firstPt;

                    // only keep non-degenerate loops
                    if ((int)currentLoop.size() >= 4)
                    {
                        sample.polys.push_back(currentLoop);
                    }

                    haveFirst = false; // next point (if any) starts a new loop
                }
            }
        }

        // If last loop was not explicitly closed but has enough points, close it
        if (haveFirst && currentLoop.size() >= 3)
        {
            if (currentLoop.back().distanceTo(firstPt) >= closeTol)
            {
                currentLoop.push_back(firstPt);
            }

            if ((int)currentLoop.size() >= 4)
            {
                sample.polys.push_back(currentLoop);
            }
        }
    }

    // 3) Normalize each z-layer and collect outputs
    std::vector<DCTSample> out;
    out.reserve(groups.size());

    for (auto& kv : groups)
    {
        DCTSample sample = kv.second;

        // normalize polygons of this layer into [-1,1]^2
        normalizePolygonsToBBox(sample.polys, zVector(-1, -1, 0), zVector(1, 1, 0));

        out.push_back(sample);

        printf(" %d num polys per layer\n", (int)sample.polys.size());
    }

    printf("Grouped into %zu SDF layers based on z-values.\n", out.size());
    return out;
}

// ------------------------------------------------------------

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

std::vector<std::vector<zVector>> g_polygons;



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

void generateDatasetFromInShapes( const std::string& jsonPath = "data/inShapes.json")
{
    g_samples.clear();

    std::vector<std::vector<zVector>> allPolys = readPolygonsFromInShapes(jsonPath);
    std::vector<DCTSample> grouped = groupPolygonsByZ(allPolys);

    for (auto& sample : grouped)
    {
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                float gx = (float)i / (RES - 1) * 2.0f - 1.0f;
                float gy = (float)j / (RES - 1) * 2.0f - 1.0f;
                zVector p(gx, gy, 0);

                sample.sdf[i][j] = sdf_MultiPolygon(p, sample.polys);
            }
        }

        normaliseSDF(sample.sdf);
        computeDCT(sample.sdf, sample.dct);
        g_samples.push_back(sample);
    }

    printf("Generated %zu SDF layers from %s.\n", g_samples.size(), jsonPath.c_str());
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

//--------------------------------------------------------------
// Encode: map normalized input feature x to latent code z
// Assumes MLP: [TOP_K] -> [LATENT_DIM] -> [TOP_K]
//--------------------------------------------------------------
//--------------------------------------------------------------
// Encode input -> latent
// Automatically detects latent layer by smallest layer width
//--------------------------------------------------------------
//--------------------------------------------------------------
// Encode input -> latent
// Robust to arbitrary hidden layer counts (symmetric AE)
//--------------------------------------------------------------

//--------------------------------------------------------------
// Helper: get latent activation layer index for symmetric AE
// activations: [0]=input, [L]=output, middle = latent
//--------------------------------------------------------------
int getLatentActivationIndex( MLP& net)
{
    int L = (int)net.activations.size(); // activations filled after forward()
    if (L < 3)
    {
        return L - 1; // degenerate, fallback to output
    }

    int latentIndex = L / 2; // middle
    if (latentIndex <= 0) latentIndex = 1;
    if (latentIndex >= L - 1) latentIndex = L - 2;

    return latentIndex;
}
//--------------------------------------------------------------
// Encode: input x -> latent z
//--------------------------------------------------------------
std::vector<float> encodeToLatent(MLP& net, std::vector<float>& x)
{
    net.forward(x);

    int latentIndex = getLatentActivationIndex(net);
    return net.activations[latentIndex];
}


//--------------------------------------------------------------
// Decode: map latent code z to normalized output y
// Uses the last layer weights/biases of the AE
// Activation: tanh (matches genericMLP training)
//--------------------------------------------------------------
//--------------------------------------------------------------
// Decode latent -> output
// Traverses decoder layers dynamically
//--------------------------------------------------------------
//--------------------------------------------------------------
// Decode latent -> output
// Robust to arbitrary symmetric AE depth
//--------------------------------------------------------------
//--------------------------------------------------------------
// Decode: latent z -> output y (normalized coeffs)
// Assumes symmetric AE; latent is middle activation layer
//--------------------------------------------------------------
std::vector<float> decodeFromLatent(MLP& net, std::vector<float>& z)
{
    if (net.W.empty() || net.b.empty())
    {
        return std::vector<float>();
    }

    // We need a latent index in *activation* space.
    // Reconstruct a fake activations.size() from architecture:
    int numWeightLayers = (int)net.W.size();      // e.g. 4 for [in,16,L,16,out]
    int numActivations = numWeightLayers + 1;    // e.g. 5

    int latentIndex = numActivations / 2;         // e.g. 2
    if (latentIndex <= 0) latentIndex = 1;
    if (latentIndex >= numActivations - 1) latentIndex = numActivations - 2;

    // First decoder layer index in W:
    // W[l] maps activations[l] -> activations[l+1]
    int startLayer = latentIndex;

    std::vector<float> a = z;

    for (int l = startLayer; l < numWeightLayers; l++)
    {
        int outSize = (int)net.W[l].size();
        std::vector<float> next(outSize, 0.0f);

        for (int i = 0; i < outSize; i++)
        {
            float sum = net.b[l][i];

            for (int j = 0; j < (int)net.W[l][i].size(); j++)
            {
                sum += net.W[l][i][j] * a[j];
            }

            // hidden layers: tanh, final layer: linear
            if (l < numWeightLayers - 1)
            {
                next[i] = std::tanh(sum);
            }
            else
            {
                next[i] = sum;
            }
        }

        a = next;
    }

    return a; // normalized AE output
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
// Linear blend between two SDFs using their coefficients
// ------------------------------------------------------------
//void blendSDFs(
//    float coeffsA[RES][RES],
//    float coeffsB[RES][RES],
//    float t,
//    float outSDF[RES][RES]
//)
//{
//    // interpolate coefficients
//    float interp[RES][RES];
//    for (int i = 0; i < RES; i++)
//    {
//        for (int j = 0; j < RES; j++)
//        {
//            interp[i][j] = (1.0f - t) * coeffsA[i][j] + t * coeffsB[i][j];
//        }
//    }
//
//    // decode back to image / field
//    computeInverseDCT(interp, outSDF);
//}

void blendSDFs(
    float coeffsA[RES][RES],
    float coeffsB[RES][RES],
    float t,
    float outSDF[RES][RES]
)
{
    // Start from all zeros
    float interp[RES][RES] = { 0 };

    // Blend ONLY the fixed TOP_K coefficients (e.g. UxU block)
    int K = std::min(TOP_K, (int)g_fixedU.size());
    for (int i = 0; i < K; i++)
    {
        int u = g_fixedU[i];
        int v = g_fixedV[i];

        interp[u][v] = (1.0f - t) * coeffsA[u][v] + t * coeffsB[u][v];
    }

    // Decode back to spatial domain
    computeInverseDCT(interp, outSDF);
}


// ------------------------------------------------------------
// visualisation
// ------------------------------------------------------------
void visualizeInterpolatedSDFs(int idxA, int idxB, int numSteps = 5)
{
    if (idxA < 0 || idxB < 0 ||
        idxA >= g_samples.size() || idxB >= g_samples.size())
        return;

    float blended_COEFFS[RES][RES];

    for (int s = 0; s <= numSteps; s++)
    {
        float t = (float)s / (float)numSteps;

        // linear blend of DCT coefficients
        blendSDFs(g_samples[idxA].dct, g_samples[idxB].dct, t, blended_COEFFS);

        // visualize
        float offsetX = (s - numSteps / 2.0f) * (RES + 10);
        drawSDF(blended_COEFFS, offsetX * 0.5, -float(RES) * 0.5f - RES - 20, 0.5f);
    }

    glColor3f(0, 0, 0);
    drawString("interpolated coefficients ", -float(RES) * 0.5f, -float(RES) * 0.5f - RES - 20 - 10);
}

//--------------------------------------------------------------
// Naive interpolation directly on SDF field values
// (no DCT coefficient blending)
//--------------------------------------------------------------
void visualizeInterpolatedSDFValues(int idxA, int idxB, int numSteps = 5)
{
    if (idxA < 0 || idxB < 0) return;
    if (idxA >= g_samples.size() || idxB >= g_samples.size()) return;

    float interp_pixels[RES][RES];

    // Loop through interpolation steps
    for (int s = 0; s <= numSteps; s++)
    {
        float t = (float)s / (float)numSteps;

        // --- 1) Linear blend of SDF values ---
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                float a = g_samples[idxA].sdf[i][j];
                float b = g_samples[idxB].sdf[i][j];
                interp_pixels[i][j] = (1.0f - t) * a + t * b;
            }
        }

        // --- 2) Visualize interpolated SDF directly ---
        float offsetX = (s - numSteps / 2.0f) * (RES + 10);
        float offsetY = -float(RES) * 0.5f - RES * 4.0f - 40.0f;
        drawSDF(interp_pixels, offsetX * 0.5, offsetY, 0.5f);
    }

    glColor3f(0, 0, 0);
    drawString("interpolated pixels ", -float(RES) * 0.5f, -float(RES) * 0.5f - RES * 4.0f - 40.0f -10);
}


//--------------------------------------------------------------
// Latent-space interpolation for monolithic MLP autoencoder
// (no explicit encoder/decoder separation)
//--------------------------------------------------------------
//--------------------------------------------------------------
// Latent-space interpolation for monolithic MLP autoencoder
//--------------------------------------------------------------
void visualizeLatentInterpolatedSDFs_MLP(int idxA, int idxB, int numSteps = 8)
{
    if (idxA < 0 || idxB < 0) return;
    if (idxA >= (int)g_trainX.size() || idxB >= (int)g_trainX.size()) return;
    if (g_featMeanVec.empty() || g_featStdVec.empty()) return;

    // Encode endpoints
    std::vector<float> zA = encodeToLatent(g_autoencoder, g_trainX[idxA]);
    std::vector<float> zB = encodeToLatent(g_autoencoder, g_trainX[idxB]);
    if (zA.empty() || zB.empty() || zA.size() != zB.size()) return;

    float latent_SDF[RES][RES];

    for (int s = 0; s <= numSteps; s++)
    {
        float t = (float)s / (float)numSteps;

        // 1) interpolate latent
        std::vector<float> z(zA.size());
        for (int i = 0; i < (int)z.size(); i++)
        {
            z[i] = (1.0f - t) * zA[i] + t * zB[i];
        }

        // 2) decode to normalized coeffs
        std::vector<float> yNorm = decodeFromLatent(g_autoencoder, z);
        if (yNorm.empty()) return;

        // 3) un-normalize
        std::vector<float> y(yNorm.size());
        int K = std::min((int)yNorm.size(), (int)g_featMeanVec.size());
        for (int i = 0; i < K; i++)
        {
            y[i] = yNorm[i] * g_featStdVec[i] + g_featMeanVec[i];
        }

        // 4) fill DCT block
        float dctTmp[RES][RES] = { 0 };
        int Kblock = std::min(TOP_K, (int)y.size());
        for (int i = 0; i < Kblock; i++)
        {
            int u = g_fixedU[i];
            int v = g_fixedV[i];
            dctTmp[u][v] = y[i];
        }

        // 5) inverse DCT → SDF
        computeInverseDCT(dctTmp, latent_SDF);

        // 6) draw
        float offsetX = (s - numSteps * 0.5f) * (RES + 10);
        float offsetY = -float(RES) * 2.75f;
        drawSDF(latent_SDF, offsetX * 0.5, offsetY, 0.5f);
    }
    drawString("interpolated latent vectors ", -float(RES) * 0.5f, -float(RES) * 2.75f - 10);
}


// ------------------------------------------------------------
// Setup + wiring
// ------------------------------------------------------------
void rebuildAll()
{
   // generateDataset();
    generateDatasetFromInShapes("data/inShapes.json");

    computeFixedBlockLayout();
    buildFixedLayoutFeatures();
    buildTrainingData();

    g_autoencoder.initialize
    (
        TOP_K,
        { 64, 16, LATENT_DIM, 16, 64},
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

    g_polygons = readPolygonsFromInShapes("data/inShapes.json");

}

void update(int value)
{
    if (!g_isTraining) return;
    if (g_trainX.empty()) return;

    if (g_trainMode == TRAIN_SGD)
    {
       // g_lastLoss = trainSGD(g_autoencoder, g_trainX, 20, 0.1f, 5);

        g_lastLoss = trainSGD(g_autoencoder, g_trainX, g_trainX, 20, 0.01f, 5);
        /*for (int e = 0; e < 20; e++)
        {
            for (auto& x : g_trainX)
            {
                g_lastLoss =  g_autoencoder.trainStep(x, 0.01f);
            }
        }*/
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
    drawGrid(RES*0.5);

    //

    glColor3f(0, 0, 0);

    if (!g_samples.empty() && g_currentShape >= 0 && g_currentShape < (int)g_samples.size())
    {
        DCTSample& sample = g_samples[g_currentShape];

        // draw all polygons (outer + islands) for this SDF layer
        for (int pId = 0; pId < (int)sample.polys.size(); pId++)
        {
            std::vector<zVector>& poly = sample.polys[pId];
            if (poly.size() < 2) continue;

            int n = (int)poly.size();
            for (int i = 0; i < n; i++)
            {
                int j = (i + 1) % n;
                drawLine(zVecToAliceVec(poly[i] * RES*0.5), zVecToAliceVec(poly[j] * RES*0.5));
            }
        }
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

    visualizeInterpolatedSDFs(1,4, 12);
    visualizeLatentInterpolatedSDFs_MLP (1, 4, 12);
    visualizeInterpolatedSDFValues(1, 4, 12);
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

    if (k == 'i')
    {
        std::vector<std::vector<zVector>> polygons = readPolygonsFromInShapes("data/inShapes.json");
        printf(" %i num polys\n", polygons.size());
    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif
