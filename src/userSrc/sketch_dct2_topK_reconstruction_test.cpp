#define _MAIN_
#ifdef _MAIN_

#include "main.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;

// ---------------------------------------
// Utility conversion helpers
// ---------------------------------------
Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

zVector AliceVecToZvec(Alice::vec& in)
{
    return zVector(in.x, in.y, in.z);
}

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


// ---------------------------------------
#define RES 64
#define TOP_K 2025
// ---------------------------------------

float field_sdf[RES][RES];
float dctCoeffs[RES][RES];

std::vector<float> topKValues;
std::vector<int> topK_U;
std::vector<int> topK_V;

// ---------------------------------------
// Simple SDF for testing
// ---------------------------------------
//float sdfCircle(float x, float y, float cx, float cy, float r)
//{
//    float dx = x - cx;
//    float dy = y - cy;
//    return sqrt(dx * dx + dy * dy) - r;
//}

// ------------------------------------------------------------
// Complex shape SDF: combination of circle, box, and wedge
// ------------------------------------------------------------
// ------------------------------------------------------------
// Complex shape SDF: visible sharp features for testing DCT
// ------------------------------------------------------------

// ------------------------------------------------------------
// Voronoi SDF: distance to nearest site minus ridge pattern
// ------------------------------------------------------------
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



inline void normaliseDCT(float(&dctCoeffs)[RES][RES], float newMin = -1.0f, float newMax = 1.0f) 
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


// ---------------------------------------
// 2D DCT and Inverse DCT
// ---------------------------------------
void computeDCT(float input[RES][RES], float output[RES][RES])
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
                    sum += input[x][y] *
                        cos((PI / RES) * (x + 0.5f) * u) *
                        cos((PI / RES) * (y + 0.5f) * v);
                }
            }

            float cu = (u == 0) ? sqrt(1.0f / RES) : sqrt(2.0f / RES);
            float cv = (v == 0) ? sqrt(1.0f / RES) : sqrt(2.0f / RES);
            output[u][v] = cu * cv * sum;
        }
    }
}

void computeInverseDCT(float input[RES][RES], float output[RES][RES])
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
                    sum += cu * cv * input[u][v] *
                        cos((PI / RES) * (x + 0.5f) * u) *
                        cos((PI / RES) * (y + 0.5f) * v);
                }
            }
            output[x][y] = sum;
        }
    }
}

// ---------------------------------------
// Compute SDF + Adaptive Top-K DCT Features
// ---------------------------------------
void computeSDFandDCT()
{
    // 1) Compute SDF field for a circle
    for (int i = 0; i < RES; i++)
    {
        for (int j = 0; j < RES; j++)
        {
            float x = (float)i / RES * 2.0f - 1.0f;
            float y = (float)j / RES * 2.0f - 1.0f;
            field_sdf[i][j] = sdfVoronoi(x,y);// sdfCircle(x, y, 0.0f, 0.0f, 0.4f);
        }
    }

    // 2) Compute DCT
    computeDCT(field_sdf, dctCoeffs);

    // 3) Select Top-K coefficients by magnitude (like selectTopKBasis)
    std::vector<std::pair<float, std::pair<int, int>>> coeffs;
    coeffs.reserve(RES * RES);

    for (int u = 0; u < RES; u++)
    {
        for (int v = 0; v < RES; v++)
        {
            coeffs.push_back({ std::fabs(dctCoeffs[u][v]), {u, v} });
        }
    }

    std::sort(coeffs.begin(), coeffs.end(),
        [](auto& a, auto& b) { return a.first > b.first; });

    topKValues.clear();
    topK_U.clear();
    topK_V.clear();

    for (int i = 0; i < std::min(TOP_K, (int)coeffs.size()); i++)
    {
        int u = coeffs[i].second.first;
        int v = coeffs[i].second.second;
        topK_U.push_back(u);
        topK_V.push_back(v);
        topKValues.push_back(dctCoeffs[u][v]); // retain signed coefficient
    }

    printf("Extracted %d top DCT coefficients.\n", (int)topKValues.size());

    //normaliseDCT(dctCoeffs);

}

// ---------------------------------------
// Reconstruct field using only Top-K
// ---------------------------------------
void reconstructFromTopKDCT(float output[RES][RES])
{
    float dct_recon[RES][RES] = { 0 };

    for (int i = 0; i < topKValues.size(); i++)
    {
        int u = topK_U[i];
        int v = topK_V[i];
        dct_recon[u][v] = topKValues[i];
    }

    computeInverseDCT(dct_recon, output);
}

// ------------------------------------------------------------
// Reconstruct field from lowest N x N DCT frequency block
// ------------------------------------------------------------
void reconstructFromLowFreqBlock(float input[RES][RES], float output[RES][RES], int N)
{
    // clamp N
    if (N > RES) N = RES;

    // temporary buffer for truncated coefficients
    float dct_low[RES][RES] = { 0 };

    // copy only top-left N x N block
    for (int u = 0; u < N; u++)
    {
        for (int v = 0; v < N; v++)
        {
            dct_low[u][v] = input[u][v];
        }
    }

    // inverse DCT to reconstruct smooth field
    computeInverseDCT(dct_low, output);
}


// ---------------------------------------
// Visualization
// ---------------------------------------
void drawField(float fld[RES][RES], float offsetX = 0)
{
    glPointSize(3);
    for (int i = 0; i < RES; i++)
    {
        for (int j = 0; j < RES; j++)
        {
            float val = fld[i][j];
            float r, g, b;
            getJetColor(val, r, g, b);

            glColor3f(r, g, b);
            float x = (float)i / RES * 100.0f - 50.0f + offsetX;
            float y = (float)j / RES * 100.0f - 50.0f;
            drawPoint(Alice::vec(x, y, 0));
        }
    }
    glPointSize(1);
}

inline void drawDCTEnergyMap(float(&dctCoeffs)[RES][RES], float offsetX = 0)
{
    // 1. Compute absolute log-magnitude map
    static float mag[RES][RES];
    float minVal = 1e9f, maxVal = -1e9f;

    for (int i = 0; i < RES; i++)
    {
        for (int j = 0; j < RES; j++)
        {
            mag[i][j] = std::log10(1.0f + std::fabs(dctCoeffs[i][j]));
            minVal = std::min(minVal, mag[i][j]);
            maxVal = std::max(maxVal, mag[i][j]);
        }
    }

    // 2. Normalize to [0,1]
    float range = maxVal - minVal;
    if (range < 1e-6f) range = 1.0f;

    for (int i = 0; i < RES; i++)
    {
        for (int j = 0; j < RES; j++)
        {
            mag[i][j] = (mag[i][j] - minVal) / range;
        }
    }

    // 3. Draw as Jet-colored grid
    glPointSize(3);
    for (int i = 0; i < RES; i++)
    {
        for (int j = 0; j < RES; j++)
        {
            float r, g, b;
            getJetColor(2.0f * mag[i][j] - 1.0f, r, g, b); // remap to [-1,1]
            glColor3f(r, g, b);

            float x = (float)i / RES * 100.0f - 50.0f + offsetX;
            float y = (float)j / RES * 100.0f - 50.0f;
            drawPoint(Alice::vec(x, y, 0));
        }
    }
    glPointSize(1);

    glColor3f(1, 1, 1);
    drawString("DCT Energy Map", offsetX + 10, 20);
}


// ---------------------------------------
bool recompute = false;
bool showReconstruction = false;
float reconstructed[RES][RES];
float reconstructedLow[RES][RES];

// ---------------------------------------
void setup()
{
    computeSDFandDCT();
}

void update(int value)
{
    if (recompute)
    {
        computeSDFandDCT();
        recompute = false;
    }
}

void draw()
{
    backGround(0.8);
    drawGrid(50);

    if (!showReconstruction)
    {
        glColor3f(1, 1, 1);
        drawString("Original SDF Field", 10, 20);

        normaliseDCT(field_sdf);
        drawField(field_sdf, 0);
    }
    else
    {
        reconstructFromTopKDCT(reconstructed);
        drawString("Reconstructed from Top-K DCT", 10, 20);

        normaliseDCT(reconstructed);
        drawField(reconstructed, 0);

       
        reconstructFromLowFreqBlock(dctCoeffs, reconstructedLow, 32); // 8×8 low-frequency block
        normaliseDCT(reconstructedLow);
        drawField(reconstructedLow, 120);

    }

    //float viz[RES][RES];
    //for (int i = 0; i < RES; i++)
    //{
    //    for (int j = 0; j < RES; j++)
    //    {
    //        viz[i][j] = std::log10(1.0f + std::fabs(dctCoeffs[i][j]));
    //    }
    //}
    //normaliseDCT(viz, 0.0f, 1.0f);
    //drawField(viz,-120);

    drawDCTEnergyMap(dctCoeffs, -120); // draws on left
}

void keyPress(unsigned char k, int xm, int ym)
{
    if (k == 'r')
    {
        recompute = true;
    }
    if (k == 'k')
    {
        showReconstruction = !showReconstruction;
    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif
