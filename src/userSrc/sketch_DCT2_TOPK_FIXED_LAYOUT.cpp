#define _MAIN_
#ifdef _MAIN_

#include "main.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;

#define RES 64
#define TOP_K 2025
#define NUM_SHAPES 5

// ------------------------------------------------------------
// Utility
// ------------------------------------------------------------
inline void getJetColor(float value, float& r, float& g, float& b)
{
    value = std::clamp(value, -1.0f, 1.0f);
    float x = (value + 1.0f) * 0.5f;
    float four = 4.0f * x;
    r = std::clamp(std::min(four - 1.5f, -four + 4.5f), 0.0f, 1.0f);
    g = std::clamp(std::min(four - 0.5f, -four + 3.5f), 0.0f, 1.0f);
    b = std::clamp(std::min(four + 0.5f, -four + 2.5f), 0.0f, 1.0f);
}

// ------------------------------------------------------------
// Polygon utilities
// ------------------------------------------------------------
float sdPolygon(float x, float y, const std::vector<zVector>& poly)
{
    int n = poly.size();
    float minDist = 1e9f;
    bool inside = false;

    for (int i = 0, j = n - 1; i < n; j = i++)
    {
        zVector a = poly[j];
        zVector b = poly[i];
        zVector pa(x - a.x, y - a.y, 0);
        zVector ba(b.x - a.x, b.y - a.y, 0);
        float h = std::clamp( (pa * ba) / (ba * ba), 0.0f, 1.0f);
        zVector d = pa - ba * h;
        minDist = std::min(minDist, d.length());

        bool cond1 = (a.y > y) != (b.y > y);
        bool cond2 = (x < (b.x - a.x) * (y - a.y) / (b.y - a.y + 1e-6f) + a.x);
        if (cond1 && cond2) inside = !inside;
    }
    return inside ? -minDist : minDist;
}

float sdfVoronoi(float x, float y)
{
    // Number of sites
    const int NUM_SITES = 64;

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


std::vector<zVector> randomPolygon(int n)
{
    std::vector<zVector> pts;
    float angle = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float r = 0.3f + 0.2f * ((float)rand() / RAND_MAX);
        angle = (2.0f * PI * i) / n + 0.1f * ((float)rand() / RAND_MAX);
        pts.push_back(zVector(r * cos(angle), r * sin(angle), 0));
    }
    return pts;
}

// ------------------------------------------------------------
// DCT / IDCT
// ------------------------------------------------------------
void computeDCT(float in[RES][RES], float out[RES][RES])
{
    for (int u = 0; u < RES; u++)
    {
        for (int v = 0; v < RES; v++)
        {
            float sum = 0.0f;
            for (int x = 0; x < RES; x++)
            {
                for (int y = 0; y < RES; y++)
                {
                    sum += in[x][y] *
                        cos((PI / RES) * (x + 0.5f) * u) *
                        cos((PI / RES) * (y + 0.5f) * v);
                }
            }
            float cu = (u == 0) ? sqrt(1.0f / RES) : sqrt(2.0f / RES);
            float cv = (v == 0) ? sqrt(1.0f / RES) : sqrt(2.0f / RES);
            out[u][v] = cu * cv * sum;
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
                for (int v = 0; v < RES; v++)
                {
                    float cu = (u == 0) ? sqrt(1.0f / RES) : sqrt(2.0f / RES);
                    float cv = (v == 0) ? sqrt(1.0f / RES) : sqrt(2.0f / RES);
                    sum += cu * cv * in[u][v] *
                        cos((PI / RES) * (x + 0.5f) * u) *
                        cos((PI / RES) * (y + 0.5f) * v);
                }
            }
            out[x][y] = sum;
        }
    }
}

inline void normaliseFIELD(float(&dctCoeffs)[RES][RES], float newMin = -1.0f, float newMax = 1.0f)
{
    float oldMin = 1e9f;
    float oldMax = -1e9f;

    // 1. Find min and max
    for (int i = 0; i < RES; i++)
    {
        for (int j = 0; j < RES; j++)
        {
            oldMin = std::min(oldMin, dctCoeffs[i][j]);
            oldMax = std::max(oldMax, dctCoeffs[i][j]);
        }
    }

    // 2. Rescale
    float oldRange = oldMax - oldMin;
    if (oldRange < 1e-6f) oldRange = 1.0f;
    float newRange = newMax - newMin;

    for (int i = 0; i < RES; i++)
    {
        for (int j = 0; j < RES; j++)
        {
            dctCoeffs[i][j] = newMin + ((dctCoeffs[i][j] - oldMin) / oldRange) * newRange;
        }
    }
}


// ------------------------------------------------------------
// Structures
// ------------------------------------------------------------
struct DCTCompressedSDF
{
    float sdf[RES][RES];
    float dct[RES][RES];
    std::vector<int> U, V;
    std::vector<float> values;
};

// ------------------------------------------------------------
// Globals
// ------------------------------------------------------------
std::vector<DCTCompressedSDF> shapes;
std::vector<int> fixedU, fixedV;
int currentShape = 0;

// ------------------------------------------------------------
// Draw helpers
// ------------------------------------------------------------
void drawField(float fld[RES][RES], float offsetX = 0)
{
    glPointSize(3);
    for (int i = 0; i < RES; i++)
    {
        for (int j = 0; j < RES; j++)
        {
            float val = fld[i][j];
            float r, g, b; getJetColor(val, r, g, b);
            glColor3f(r, g, b);
            float x = (float)i / RES * 100.0f - 50.0f + offsetX;
            float y = (float)j / RES * 100.0f - 50.0f;
            drawPoint(Alice::vec(x, y, 0));
        }
    }
    glPointSize(1);
}

// ------------------------------------------------------------
// Build dataset
// ------------------------------------------------------------
void generateShapes()
{
    shapes.clear();

    for (int s = 0; s < NUM_SHAPES; s++)
    {
        DCTCompressedSDF item;
        std::vector<zVector> poly = randomPolygon(5 + rand() % 4);

        // --- 1) compute SDF
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                float x = (float)i / RES * 2.0f - 1.0f;
                float y = (float)j / RES * 2.0f - 1.0f;
                item.sdf[i][j] = (s == 0) ? sdfVoronoi(x,y) : sdPolygon(x, y, poly);
                
            }
        }

        // --- 2) DCT
        normaliseFIELD(item.sdf);
        computeDCT(item.sdf, item.dct);
       

        // --- 3) extract top-K
        std::vector<std::pair<float, std::pair<int, int>>> coeffs;
        for (int u = 0; u < RES; u++)
        {
            for (int v = 0; v < RES; v++)
                coeffs.push_back({ fabs(item.dct[u][v]), {u, v} });
        }

        std::sort(coeffs.begin(), coeffs.end(),
            [](auto& a, auto& b) { return a.first > b.first; });

        int K = std::min(TOP_K, (int)coeffs.size());
        for (int i = 0; i < K; i++)
        {
            int u = coeffs[i].second.first;
            int v = coeffs[i].second.second;
            item.U.push_back(u);
            item.V.push_back(v);
            item.values.push_back(item.dct[u][v]);
        }

        shapes.push_back(item);
    }

    printf("Generated %d shapes\n", (int)shapes.size());
}

// ------------------------------------------------------------
// Compute global fixed U,V layout (energy-weighted union)
// ------------------------------------------------------------
void computeFixedLayout()
{
    std::map<std::pair<int, int>, float> energySum;

    for (auto& s : shapes)
    {
        for (int k = 0; k < s.U.size(); k++)
        {
            auto key = std::make_pair(s.U[k], s.V[k]);
            energySum[key] += fabs(s.values[k]);
        }
    }

    std::vector<std::pair<std::pair<int, int>, float>> ranked(energySum.begin(), energySum.end());
    std::sort(ranked.begin(), ranked.end(),
        [](auto& a, auto& b) { return a.second > b.second; });

    fixedU.clear(); fixedV.clear();
    for (int i = 0; i < TOP_K && i < ranked.size(); i++)
    {
        fixedU.push_back(ranked[i].first.first);
        fixedV.push_back(ranked[i].first.second);
    }

    printf("Computed global fixed UV layout (%d modes)\n", (int)fixedU.size());
}

// ------------------------------------------------------------
// Reconstruct one SDF using fixed UV layout
// ------------------------------------------------------------
void reconstructFixed(const DCTCompressedSDF& src, float out[RES][RES])
{
    float dct_temp[RES][RES] = { 0 };

    // map local coefficients to fixed UV
    for (int i = 0; i < fixedU.size(); i++)
    {
        int u = fixedU[i];
        int v = fixedV[i];

        // find if src had this mode
        float val = 0.0f;
        for (int k = 0; k < src.U.size(); k++)
        {
            if (src.U[k] == u && src.V[k] == v)
            {
                val = src.values[k];
                break;
            }
        }
        dct_temp[u][v] = val;
    }

    computeInverseDCT(dct_temp, out);
}

// ------------------------------------------------------------
bool recompute = false;
float reconField[RES][RES];

// ------------------------------------------------------------
void setup()
{
    srand(42);
    generateShapes();
    computeFixedLayout();
    reconstructFixed(shapes[currentShape], reconField);
}

void update(int value)
{
    if (recompute)
    {
        reconstructFixed(shapes[currentShape], reconField);
        recompute = false;
    }
}

void draw()
{
    backGround(0.8);
    drawGrid(50);

    drawString("Original SDF", -120, 75);
    drawField(shapes[currentShape].sdf, -120);

    drawString("Reconstructed (Fixed UV)", 120, 75);
    drawField(reconField, 120);

    glColor3f(1, 1, 1);
    char label[64];
    sprintf(label, "Shape %d / %d  (press 1-5 to switch)", currentShape + 1, NUM_SHAPES);
    drawString(label, -40, -70);
}

void keyPress(unsigned char k, int xm, int ym)
{
    if (k >= '1' && k <= '5')
    {
        int idx = k - '1';
        if (idx < shapes.size())
        {
            currentShape = idx;
            recompute = true;
        }
    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif
