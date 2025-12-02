
#ifndef _SCALAR_FIELD_
#define _SCALAR_FIELD_


//#pragma once


#include <vector>
#include <algorithm>
#include <cmath>
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;

//these two functiosn must be turned on for sketch_circleSDF_fitter.cpp
inline zVector zMax( zVector& a,  zVector& b)
{
    return zVector(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

inline float smin(float a, float b, float k)
{
    float h = std::max(k - fabs(a - b), 0.0f) / k;
    return std::min(a, b) - h * h * k * 0.25f;
}

inline zVector zMin( zVector& a,  zVector& b)
{
    return zVector(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

inline float zLerp(float a, float b, float t)
{
    return a + (b - a) * t;
}


inline void getJetColor(float value, float& r, float& g, float& b)
{
    // Clamp input to [-1, 1]
    value = std::clamp(value, -1.0f, 1.0f);

    // Normalize to [0, 1]
    float normalized = (value + 1.0f) * 0.5f;

    float fourValue = 4.0f * normalized;

    r = std::clamp(std::min(fourValue - 1.5f, -fourValue + 4.5f), 0.0f, 1.0f);
    g = std::clamp(std::min(fourValue - 0.5f, -fourValue + 3.5f), 0.0f, 1.0f);
    b = std::clamp(std::min(fourValue + 0.5f, -fourValue + 2.5f), 0.0f, 1.0f);
}


enum class SMinMode
{
    MIN,
    EXPONENTIAL,
    ROOT,
    SIGMOID,
    POLY_QUADRATIC,
    POLY_CUBIC,
    POLY_QUARTIC,
    CIRCULAR,
    CIRCULAR_GEOMETRIC
};

struct SkelNode
{
    zVector pos;
    int degree;
    int index;
};

struct SkelEdge
{
    int n0, n1;                    // node indices
    std::vector<zVector> polyline; // ordered points
};


#define OUT 1e6
#define SF_RES 200

class ScalarField2D
{
public:


  

    
    int div = 2; 

    zVector gridPoints[SF_RES][SF_RES];
    float field[SF_RES][SF_RES];
    float field_normalized[SF_RES][SF_RES];
    zVector gradient[SF_RES][SF_RES];
    std::vector<std::pair<zVector, zVector>> isolines;
    std::vector<std::vector<zVector>> allContours;

    ScalarField2D()
    {
        float span = 100.0f; // from -50 to +50
        float step = span / (SF_RES - 1); // spacing between grid points

        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                float x = -50.0f + i * step;
                float y = -50.0f + j * step;

                gridPoints[i][j] = zVector(x, y, 0);
                field[i][j] = 0;
                
            }
        }
    }

   

    void clearField()
    {
        for (int i = 0; i < SF_RES; i++)
            for (int j = 0; j < SF_RES; j++)
            {
                field[i][j] = field_normalized[i][j] = 0.0;
                gradient[i][j] = zVector(0, 0, 0);
            }
                              
        isolines.clear();
        allContours.clear();

    }

    //----------------------------------------

    float smin(float a, float b, float k)
    {
        float h = std::max(k - fabs(a - b), 0.0f) / k;
        return std::min(a, b) - h * h * k * 0.25f;
    }

    void addVoronoi( std::vector<zVector>& sites )
    {
        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                zVector pt = gridPoints[i][j];
                float minDist = 1e6f;
                float secondMinDist = 1e6f;

                for ( auto& site : sites)
                {
                    float d = pt.distanceTo(zVector(site));

                    // Track the two closest sites
                    if (d < minDist)
                    {
                        secondMinDist = minDist;
                        minDist = d;
                    }
                    else if (d < secondMinDist)
                    {
                        secondMinDist = d;
                    }
                }

                // Distance to the second closest site minus closest — sharpest at Voronoi edge
                field[i][j] = (secondMinDist - minDist);
            }
        }

        rescaleFieldToRange(-1, 1);
    }

    void addCircleSDF(zVector center, float radius , bool invertDistance = true)
    {
        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                zVector pt = gridPoints[i][j];
                float d = pt.distanceTo(center);

                // Signed distance: negative inside, zero on boundary, positive outside
                float val = (d > radius) ? d : d - radius;
                
                field[i][j] = val;// std::min(val, field[i][j]);
            }
        }

        rescaleFieldToRange(-1, 1);//closedfields rescale to -1,1
    }

    void addOrientedBoxSDF(zVector center, zVector halfSize, float angleRadians)
    {
        float c = cos(angleRadians);
        float s = sin(angleRadians);

        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                zVector p = gridPoints[i][j] - center;

                // Rotate point into box's local frame
                zVector pr(
                    c * p.x + s * p.y,
                    -s * p.x + c * p.y,
                    0.0f
                );

                zVector d = zVector(fabs(pr.x), fabs(pr.y), 0.0f) - halfSize;

                float outsideDist = zMax(d, zVector(0, 0, 0)).length();
                float insideDist = std::min(std::max(d.x, d.y), 0.0f);

                float signedDist = (outsideDist > 0.0f) ? outsideDist  :  insideDist;

                field[i][j] = signedDist ;// std::min(signedDist * scale, field[i][j]);
            }
        }

        rescaleFieldToRange(-1, 1);//closedfields rescale to -1,1
    }

    void addCircleSDFs(vector<zVector> rbfCenters, float radius = 2.0)
    {
        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                zVector p = gridPoints[i][j];
                float d = p.distanceTo(rbfCenters[0]);
                // Signed distance: negative inside, zero on boundary, positive outside
                //float val = (d > radius) ? d : d - radius;

                for (int i = 1; i < rbfCenters.size();  i++)
                {
                    float d_i = p.distanceTo(rbfCenters[i]);
                    //val = (d_i > radius) ? d_i : -d_i ;
                    d = std::min(d, d_i);
                }


                field[i][j] = (d < radius) ? (radius - d)*-1: d ;
            }
        }

        rescaleFieldToRange(-1, 1);
    }

    // ===============================================================
// COMPUTE SDF FROM A SET OF POLYLINES AND UNION THEM
// ===============================================================
// Put this inside ScalarField2D (public: or private:) — Allman style.
    inline float sdfSegment( zVector& p,  zVector& a,  zVector& b)
    {
        zVector ab = b - a;
        zVector ap = p - a;

        float denom = ab * ab;
        if (denom <= 1e-12f)
        {
            // Degenerate segment: treat as point
            return p.distanceTo(a);
        }

        float t = (ap * ab) / denom;
        t = std::max(0.0f, std::min(1.0f, t));

        zVector closest = a + ab * t;
        return p.distanceTo(closest);
    }

    // Union-of-polylines SDF; thickness = tube radius around polylines
    inline void addSDFfromPolylines
    (
        std::vector<std::vector<zVector>>& polylines,
        float thickness = 0.0f
    )
    {
        clearField(); // zero gradients/contours etc. (already in your class)

        // Initialize to +inf so mins work
        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                field[i][j] = OUT;
            }
        }

        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                zVector p = gridPoints[i][j];
                float dmin = OUT;

                for ( auto& poly : polylines)
                {
                    if (poly.size() < 2) continue;

                    for (int k = 0; k < (int)poly.size() - 1; k++)
                    {
                        dmin = std::min(dmin, sdfSegment(p, poly[k], poly[k + 1]));
                    }
                }

                // Signed: negative inside the tube (radius = thickness)
                field[i][j] = dmin - thickness;
            }
        }

        // Keep your usual normalization (you already have this helper)
        rescaleFieldToRange(-1.0f, 1.0f);
    }
    
    //------------
    inline bool pointInPolygon(const zVector& p, const std::vector<zVector>& poly)
    {
        bool inside = false;
        int n = poly.size();

        for (int i = 0, j = n - 1; i < n; j = i++)
        {
            const zVector& a = poly[i];
            const zVector& b = poly[j];

            bool intersect =
                ((a.y > p.y) != (b.y > p.y)) &&
                (p.x < (b.x - a.x) * (p.y - a.y) / ((b.y - a.y) + 1e-9f) + a.x);

            if (intersect)
                inside = !inside;
        }
        return inside;
    }


    inline void addSDFfromClosedPolygon( std::vector<zVector>& poly)
    {
        if (poly.size() < 3) return; // not a valid closed polygon

        clearField();

        // --- SDF helper: point-to-segment ---
        auto sdfSegment = [&]( zVector& p,  zVector& a,  zVector& b)
            {
                zVector ab = b - a;
                zVector ap = p - a;

                float denom = ab * ab;
                if (denom <= 1e-12f)
                {
                    return p.distanceTo(a);
                }

                float t = (ap * ab) / denom;
                t = std::max(0.0f, std::min(1.0f, t));
                zVector q = a + ab * t;
                return p.distanceTo(q);
            };

        // --- Loop over grid ---
        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                zVector p = gridPoints[i][j];

                // 1) Distance to boundary (unsigned)
                float dmin = OUT;
                for (int k = 0; k < (int)poly.size(); k++)
                {
                     zVector& a = poly[k];
                     zVector& b = poly[(k + 1) % poly.size()];
                    dmin = std::min(dmin, sdfSegment(p, a, b));
                }

                // 2) Sign of SDF: negative if inside
                bool inside = pointInPolygon(p, poly);

                field[i][j] = inside ? -dmin : dmin;
            }
        }

        // Optional normalization to [-1, 1]
        rescaleFieldToRange(-1.0f, 1.0f);
    }


    //----------------------------------------
    
    void unionWith( ScalarField2D& other)
    {
        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                field[i][j] = std::min(field[i][j], other.field[i][j]);
            }
        }
    }

    void intersectWith( ScalarField2D& other)
    {
        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                field[i][j] = std::max(field[i][j], other.field[i][j]);
            }
        }
    }

    void subtract( ScalarField2D& other)
    {
        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                field[i][j] = std::max(field[i][j], -other.field[i][j]);
            }
        }
    }

    void blendWith( ScalarField2D& other, float smooth_k, SMinMode mode = SMinMode::EXPONENTIAL)
    {
        auto smin_exponential = [](float a, float b, float k)
            {
                float r = exp2(-a / k) + exp2(-b / k);
                return -k * log2(r);
            };

        auto smin_root = [](float a, float b, float k)
            {
                k *= 2.0f;
                float x = b - a;
                return 0.5f * (a + b - sqrtf(x * x + k * k));
            };

        auto smin_sigmoid = [](float a, float b, float k)
            {
                k *= logf(2.0f);
                float x = b - a;
                return a + x / (1.0f - exp2(x / k));
            };

        auto smin_polyQuadratic = [](float a, float b, float k)
            {
                k *= 4.0f;
                float h = std::max(k - fabs(a - b), 0.0f) / k;
                return std::min(a, b) - h * h * k * 0.25f;
            };

        auto smin_polyCubic = [](float a, float b, float k)
            {
                k *= 6.0f;
                float h = std::max(k - fabs(a - b), 0.0f) / k;
                return std::min(a, b) - h * h * h * k / 6.0f;
            };

        auto smin_polyQuartic = [](float a, float b, float k)
            {
                k *= (16.0f / 3.0f);
                float h = std::max(k - fabs(a - b), 0.0f) / k;
                return std::min(a, b) - h * h * h * (4.0f - h) * k / 16.0f;
            };

        auto smin_circular = [](float a, float b, float k)
            {
                k *= 1.0f / (1.0f - sqrtf(0.5f));
                float h = std::max(k - fabs(a - b), 0.0f) / k;
                return std::min(a, b) - k * 0.5f * (1.0f + h - sqrtf(1.0f - h * (h - 2.0f)));
            };

        auto smin_circularGeometric = [](float a, float b, float k)
            {
                k *= 1.0f / (1.0f - sqrtf(0.5f));
                float dx = std::max(k - a, 0.0f);
                float dy = std::max(k - b, 0.0f);
                float l = sqrtf(dx * dx + dy * dy);
                return std::max(k, std::min(a, b)) - l;
            };

        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                float a = field[i][j];
                float b = other.field[i][j];

                switch (mode)
                {
                case SMinMode::MIN:
                    field[i][j] = std::min(a, b);
                    break;
                case SMinMode::EXPONENTIAL:
                    field[i][j] = smin_exponential(a, b, smooth_k);
                    break;
                case SMinMode::ROOT:
                    field[i][j] = smin_root(a, b, smooth_k);
                    break;
                case SMinMode::SIGMOID:
                    field[i][j] = smin_sigmoid(a, b, smooth_k);
                    break;
                case SMinMode::POLY_QUADRATIC:
                    field[i][j] = smin_polyQuadratic(a, b, smooth_k);
                    break;
                case SMinMode::POLY_CUBIC:
                    field[i][j] = smin_polyCubic(a, b, smooth_k);
                    break;
                case SMinMode::POLY_QUARTIC:
                    field[i][j] = smin_polyQuartic(a, b, smooth_k);
                    break;
                case SMinMode::CIRCULAR:
                    field[i][j] = smin_circular(a, b, smooth_k);
                    break;
                case SMinMode::CIRCULAR_GEOMETRIC:
                    field[i][j] = smin_circularGeometric(a, b, smooth_k);
                    break;
                default:
                    field[i][j] = std::min(a, b);
                    break;
                }
            }
        }

        rescaleFieldToRange(-1, 1); 
    }

    //---------------------------------------------

    void clampNeg()
    {
        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                field[i][j] = (field[i][j] < 0) ? field[i][j] * -1 : 0;
            }
        }
    }
    void clampPos()
    {
        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                field[i][j] = std::clamp(field[i][j], 0.f, 1.f);
            }
        }
    }

    void normalise()
    {
        float mn = 1e6f, mx = -1e6f;
        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                if (fabs(field[i][j] - OUT) < 1e-6)continue; //exclude outside

                mn = std::min(mn, field[i][j]);
                mx = std::max(mx, field[i][j]);
            }
        }

        float range = std::max(mx - mn, 1e-6f);
        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                if (fabs(field[i][j] - OUT) < 1e-6)continue; //exclude outside

                field_normalized[i][j] = ofMap(field[i][j], mn, mx, 0, 1) ;
            }
        }
    }
    void minMax( float &mn, float &mx)
    {
        mn = 1e6; mx = -mn;
        for (int i = 0; i < SF_RES; ++i)
            for (int j = 0; j < SF_RES; ++j)
            {
                float v = field[i][j];
                mn = min(v, mn);
                mx = max(v, mx);
            }
    }

    void rescaleFieldToRange(float targetMin = -1.0f, float targetMax = 1.0f)
    {
        float minVal[2] = { 1e6f,  1e6f };
        float maxVal[2] = { -1e6f, -1e6f };

        for (int i = 0; i < SF_RES; ++i)
            for (int j = 0; j < SF_RES; ++j)
            {
                float v = field[i][j];
                int idx = (v >= 0.0f) ? 0 : 1;
                minVal[idx] = std::min(minVal[idx], v);
                maxVal[idx] = std::max(maxVal[idx], v);
            }

        float range[2] = {
            std::max(maxVal[0] - minVal[0], 1e-6f),
            std::max(maxVal[1] - minVal[1], 1e-6f)
        };

        for (int i = 0; i < SF_RES; ++i)
            for (int j = 0; j < SF_RES; ++j)
                field[i][j] = (field[i][j] >= 0.0f)
                ? ofMap(field[i][j], minVal[0], maxVal[0], 0.0f, targetMax)
                : ofMap(field[i][j], minVal[1], maxVal[1], targetMin, 0.0f);

       

        //printf(" min max field values %.2f, %.2f,%.2f,%.2f \n", minVal[0], maxVal[0], minVal[1], maxVal[1]);
        minMax(minVal[0], maxVal[0]);
       // printf(" min max field values after rescale %.2f, %.2f \n", minVal[0], maxVal[0]);
       // printf(" range1 range2 %.2f, %.2f \n", range[0], range[1]);
    }


    // -----------------------------------------

    enum class PMVariant
    {
        Exp,        // c = exp(-(g/k)^2)
        Reciprocal  // c = 1 / (1 + (g/k)^2)
    };

    enum class DiffuseDir
    {
        None,           // classic Perona–Malik (edge-aware only)
        AlongGradient,  // more smoothing along ∇u direction
        AlongIsophote   // more smoothing along isophotes (⊥ ∇u)
    };

    void smoothDiffuseAnisotropic
    (
        float dt = 0.12f,
        int   iterations = 10,
        float k = 0.1f,          // contrast parameter (depends on your field scale)
        PMVariant variant = PMVariant::Exp,
        DiffuseDir dirMode = DiffuseDir::AlongIsophote,
        float dirBiasStrength = 2.0f,          // ≥1: stronger bias; 1 = neutral
        bool  ignoreOUT = true
    )
    {
        if (dt <= 0.0f || iterations <= 0)
        {
            return;
        }

        auto clampi = [](int v, int lo, int hi) -> int
            {
                return (v < lo) ? lo : (v > hi) ? hi : v;
            };

        auto conduct = [&](float g) -> float
            {
                // g = |gradient|
                float r = g / std::max(k, 1e-12f);
                if (variant == PMVariant::Exp)
                {
                    return std::exp(-r * r);
                }
                else
                {
                    return 1.0f / (1.0f + r * r);
                }
            };

        // small helpers
        auto safeSample = [&](int i, int j, int ii, int jj, float uC) -> float
            {
                float v = field[ii][jj];
                if (ignoreOUT && std::fabs(v - OUT) < 1e-6f) return uC;
                return v;
            };

        auto gradAt = [&](int i, int j, float& gx, float& gy)
            {
                 int iN = clampi(i - 1, 0, SF_RES - 1);
                 int iS = clampi(i + 1, 0, SF_RES - 1);
                 int jW = clampi(j - 1, 0, SF_RES - 1);
                 int jE = clampi(j + 1, 0, SF_RES - 1);

                 float uC = field[i][j];
                float uN = field[iN][j];
                float uS = field[iS][j];
                float uW = field[i][jW];
                float uE = field[i][jE];

                if (ignoreOUT)
                {
                    if (std::fabs(uN - OUT) < 1e-6f) uN = uC;
                    if (std::fabs(uS - OUT) < 1e-6f) uS = uC;
                    if (std::fabs(uW - OUT) < 1e-6f) uW = uC;
                    if (std::fabs(uE - OUT) < 1e-6f) uE = uC;
                }

                // central differences
                gx = 0.5f * (uE - uW);
                gy = 0.5f * (uS - uN);
            };

        std::vector<float> next(SF_RES * SF_RES, 0.0f);
        auto NX = [&](int i, int j) -> float& { return next[i * SF_RES + j]; };

        for (int it = 0; it < iterations; ++it)
        {
            for (int i = 0; i < SF_RES; ++i)
            {
                for (int j = 0; j < SF_RES; ++j)
                {
                     float uC = field[i][j];

                    if (ignoreOUT && std::fabs(uC - OUT) < 1e-6f)
                    {
                        NX(i, j) = OUT;
                        continue;
                    }

                     int iN = clampi(i - 1, 0, SF_RES - 1);
                     int iS = clampi(i + 1, 0, SF_RES - 1);
                     int jW = clampi(j - 1, 0, SF_RES - 1);
                     int jE = clampi(j + 1, 0, SF_RES - 1);

                    // neighbor samples (OUT-safe)
                    float uN = safeSample(i, j, iN, j, uC);
                    float uS = safeSample(i, j, iS, j, uC);
                    float uW = safeSample(i, j, i, jW, uC);
                    float uE = safeSample(i, j, i, jE, uC);

                    // Perona–Malik conductances using directional gradients
                    // (N,S,W,E approximate directional |∇u| via forward/backward diffs)
                    float gN = std::fabs(uN - uC);
                    float gS = std::fabs(uS - uC);
                    float gW = std::fabs(uW - uC);
                    float gE = std::fabs(uE - uC);

                    float cN = conduct(gN);
                    float cS = conduct(gS);
                    float cW = conduct(gW);
                    float cE = conduct(gE);

                    // Optional directional bias
                    if (dirMode != DiffuseDir::None)
                    {
                        float gx, gy;
                        gradAt(i, j, gx, gy);

                        // unit direction to bias along
                        float bx = 0.0f, by = 0.0f;
                        if (dirMode == DiffuseDir::AlongGradient)
                        {
                            bx = gx; by = gy;
                        }
                        else // AlongIsophote: tangent = perpendicular to gradient
                        {
                            bx = -gy; by = gx;
                        }
                         float bnorm = std::sqrt(bx * bx + by * by);
                        if (bnorm > 1e-12f)
                        {
                            bx /= bnorm; by /= bnorm;

                            // directions to neighbors
                            struct Dir { float dx, dy; float* c; };
                            Dir dirs[4] =
                            {
                                {  0.0f, -1.0f, &cN },
                                {  0.0f,  1.0f, &cS },
                                { -1.0f,  0.0f, &cW },
                                {  1.0f,  0.0f, &cE }
                            };

                            // cosine alignment (|dot|) ^ strength  -> scale conductance
                            for (auto& d : dirs)
                            {
                                float align = std::fabs(bx * d.dx + by * d.dy); // 0..1
                                float scale = std::pow(align, std::max(1.0f, dirBiasStrength));
                                *(d.c) *= scale;
                            }
                        }
                    }

                    // Divergence of c * ∇u (4-neighbour form)
                    float fluxN = cN * (uN - uC);
                    float fluxS = cS * (uS - uC);
                    float fluxW = cW * (uW - uC);
                    float fluxE = cE * (uE - uC);

                    float div = fluxN + fluxS + fluxW + fluxE;

                    NX(i, j) = uC + dt * div;
                }
            }

            // Commit
            for (int i = 0; i < SF_RES; ++i)
            {
                for (int j = 0; j < SF_RES; ++j)
                {
                    field[i][j] = NX(i, j);
                }
            }
        }
    }

    void smoothDiffuseIsotropic(float dt = 0.15f, int iterations = 10, bool ignoreOUT = true)
    {
        if (dt <= 0.0f || iterations <= 0)
        {
            return;
        }

        auto clampi = [](int v, int lo, int hi) -> int
            {
                return (v < lo) ? lo : (v > hi) ? hi : v;
            };

        std::vector<float> next(SF_RES * SF_RES, 0.0f);
        auto F = [&](int i, int j) -> float& { return field[i][j]; };
        auto NX = [&](int i, int j) -> float& { return next[i * SF_RES + j]; };

        for (int it = 0; it < iterations; ++it)
        {
            // Jacobi step
            for (int i = 0; i < SF_RES; ++i)
            {
                for (int j = 0; j < SF_RES; ++j)
                {
                     float uC = F(i, j);

                    if (ignoreOUT && std::fabs(uC - OUT) < 1e-6f)
                    {
                        NX(i, j) = OUT;
                        continue;
                    }

                    // Neumann-ish boundaries (clamped sampling)
                     int iN = clampi(i - 1, 0, SF_RES - 1);
                     int iS = clampi(i + 1, 0, SF_RES - 1);
                     int jW = clampi(j - 1, 0, SF_RES - 1);
                     int jE = clampi(j + 1, 0, SF_RES - 1);

                    float uN = F(iN, j);
                    float uS = F(iS, j);
                    float uW = F(i, jW);
                    float uE = F(i, jE);

                    if (ignoreOUT)
                    {
                        if (std::fabs(uN - OUT) < 1e-6f) uN = uC;
                        if (std::fabs(uS - OUT) < 1e-6f) uS = uC;
                        if (std::fabs(uW - OUT) < 1e-6f) uW = uC;
                        if (std::fabs(uE - OUT) < 1e-6f) uE = uC;
                    }

                    // 5-point Laplacian
                    float lap = (uN + uS + uW + uE - 4.0f * uC);
                    NX(i, j) = uC + dt * lap;
                }
            }

            // Commit
            for (int i = 0; i < SF_RES; ++i)
            {
                for (int j = 0; j < SF_RES; ++j)
                {
                    F(i, j) = NX(i, j);
                }
            }
        }
    }


    //---------------------------------------------
     zVector getGradient(int i, int j)
    {
        return gradient[i][j];
    }

     void computeGradient()
    {
        for (int i = 1; i < SF_RES - 1; i++)
        {
            for (int j = 1; j < SF_RES - 1; j++)
            {
                float dx = (field[i + 1][j] - field[i - 1][j]) * 0.5f;
                float dy = (field[i][j + 1] - field[i][j - 1]) * 0.5f;
                gradient[i][j] = zVector(dx, dy, 0) * -1;;//^ zVector(0, 0, 1);
                gradient[i][j].normalize();
            }
        }
    }

     float sampleAt(float x, float y)
     {
         float span = 100.0f;
         float step = span / (SF_RES - 1);

         float fx = (x + 50.0f) / step;
         float fy = (y + 50.0f) / step;

         int i = std::floor(fx);
         int j = std::floor(fy);

         float tx = fx - i;
         float ty = fy - j;

         if (i < 0 || j < 0 || i >= SF_RES - 1 || j >= SF_RES - 1)
             return 1e6f;

         float f00 = field[i][j];
         float f10 = field[i + 1][j];
         float f01 = field[i][j + 1];
         float f11 = field[i + 1][j + 1];

         float fx0 = (1 - tx) * f00 + tx * f10;
         float fx1 = (1 - tx) * f01 + tx * f11;
         return (1 - ty) * fx0 + ty * fx1;
     }

     zVector gradientAt( zVector& p)
     {
         float eps = 1.0f;
         float dx = sampleAt(p.x + eps, p.y) - sampleAt(p.x - eps, p.y);
         float dy = sampleAt(p.x, p.y + eps) - sampleAt(p.x, p.y - eps);
         return zVector(dx, dy, 0.0f) * 0.5f;
     }
     //---------------------------------------------
     
     vector<zVector> streamLine;
     void integrateStreamLine(zVector startPt)
     {
         zVector pt = startPt;
         streamLine.clear();
         zVector grad;
         for (int i = 0; i < 5000; i++)
         {
             grad = gradientAt(pt);
             grad.normalize();
             grad *= -1;
             pt += grad;
             streamLine.push_back(pt);
         }
     }

    //---------------------------------------------

    void processTriangle(zVector pts[3], float vals[3], float thresh, std::vector<std::pair<zVector, zVector>>& lines)
    {
        std::vector<zVector> crossings;
        for (int k = 0; k < 3; k++)
        {
            int nxt = (k + 1) % 3;
            if ((vals[k] < thresh && vals[nxt] >= thresh) || (vals[nxt] < thresh && vals[k] >= thresh))
            {
                float t = (thresh - vals[k]) / (vals[nxt] - vals[k]);
                zVector ip = pts[k] + (pts[nxt] - pts[k]) * t;
                crossings.push_back(ip);
            }
        }

        if (crossings.size() == 2)
        {
            lines.push_back({ crossings[0], crossings[1] });
        }
    }

    void computeIsocontours(float threshold)
    {
        isolines.clear();
        computeIsocontours(threshold, isolines);
    }

    void computeIsocontours(float threshold, std::vector<std::pair<zVector, zVector>>& output)
    {
        // Helper: Linear interpolation along an edge (correct for Bilinear patch boundaries)
        auto getT = [&](float v1, float v2) -> float {
            if (std::abs(v2 - v1) < 1e-6f) return 0.5f;
            return (threshold - v1) / (v2 - v1);
            };

        // Helper: Process a single quad (Marching Squares Logic)
        auto processSubQuad = [&](zVector pts[4], float vals[4])
            {
                int caseId = 0;
                if (vals[0] >= threshold) caseId |= 1; // BL
                if (vals[1] >= threshold) caseId |= 2; // BR
                if (vals[2] >= threshold) caseId |= 4; // TR
                if (vals[3] >= threshold) caseId |= 8; // TL

                if (caseId == 0 || caseId == 15) return;

                auto lerpPos = [&](int a, int b) {
                    float t = getT(vals[a], vals[b]);
                    return pts[a] + (pts[b] - pts[a]) * t;
                    };

                // Edges: 0:Bottom, 1:Right, 2:Top, 3:Left
                switch (caseId)
                {
                case 1:  output.emplace_back(lerpPos(0, 3), lerpPos(0, 1)); break;
                case 2:  output.emplace_back(lerpPos(0, 1), lerpPos(1, 2)); break;
                case 3:  output.emplace_back(lerpPos(0, 3), lerpPos(1, 2)); break;
                case 4:  output.emplace_back(lerpPos(1, 2), lerpPos(2, 3)); break;
                case 5:  output.emplace_back(lerpPos(0, 3), lerpPos(0, 1)); output.emplace_back(lerpPos(1, 2), lerpPos(2, 3)); break; // Saddle
                case 6:  output.emplace_back(lerpPos(0, 1), lerpPos(2, 3)); break;
                case 7:  output.emplace_back(lerpPos(0, 3), lerpPos(2, 3)); break;
                case 8:  output.emplace_back(lerpPos(0, 3), lerpPos(2, 3)); break;
                case 9:  output.emplace_back(lerpPos(0, 1), lerpPos(2, 3)); break;
                case 10: output.emplace_back(lerpPos(0, 3), lerpPos(2, 3)); output.emplace_back(lerpPos(0, 1), lerpPos(1, 2)); break; // Saddle
                case 11: output.emplace_back(lerpPos(1, 2), lerpPos(2, 3)); break;
                case 12: output.emplace_back(lerpPos(0, 3), lerpPos(1, 2)); break;
                case 13: output.emplace_back(lerpPos(0, 1), lerpPos(1, 2)); break;
                case 14: output.emplace_back(lerpPos(0, 3), lerpPos(0, 1)); break;
                }
            };

        for (int i = 0; i < SF_RES - 1; i++)
        {
            for (int j = 0; j < SF_RES - 1; j++)
            {
                // 1. Get main grid values
                float vBL = field[i][j];
                float vBR = field[i + 1][j];
                float vTR = field[i + 1][j + 1];
                float vTL = field[i][j + 1];

                // Optimization: Skip if cell is fully clear
                float mn = std::min({ vBL, vBR, vTR, vTL });
                float mx = std::max({ vBL, vBR, vTR, vTL });
                if (threshold < mn || threshold > mx) continue;

                // 2. Calculate Bilinear Sub-points
                // Center value of a bilinear patch is exactly the average of corners
                float vC = (vBL + vBR + vTR + vTL) * 0.25f;
                float vBot = (vBL + vBR) * 0.5f;
                float vTop = (vTL + vTR) * 0.5f;
                float vLeft = (vBL + vTL) * 0.5f;
                float vRight = (vBR + vTR) * 0.5f;

                zVector pBL = gridPoints[i][j];
                zVector pBR = gridPoints[i + 1][j];
                zVector pTR = gridPoints[i + 1][j + 1];
                zVector pTL = gridPoints[i][j + 1];

                zVector pC = (pBL + pBR + pTR + pTL) * 0.25f;
                zVector pBot = (pBL + pBR) * 0.5f;
                zVector pTop = (pTL + pTR) * 0.5f;
                zVector pLeft = (pBL + pTL) * 0.5f;
                zVector pRight = (pBR + pTR) * 0.5f;

                // 3. Process 4 sub-quadrants
                // Sub-quad 1: Bottom-Left
                zVector sq1_pts[4] = { pBL, pBot, pC, pLeft };
                float sq1_vals[4] = { vBL, vBot, vC, vLeft };
                processSubQuad(sq1_pts, sq1_vals);

                // Sub-quad 2: Bottom-Right
                zVector sq2_pts[4] = { pBot, pBR, pRight, pC };
                float sq2_vals[4] = { vBot, vBR, vRight, vC };
                processSubQuad(sq2_pts, sq2_vals);

                // Sub-quad 3: Top-Right
                zVector sq3_pts[4] = { pC, pRight, pTR, pTop };
                float sq3_vals[4] = { vC, vRight, vTR, vTop };
                processSubQuad(sq3_pts, sq3_vals);

                // Sub-quad 4: Top-Left
                zVector sq4_pts[4] = { pLeft, pC, pTop, pTL };
                float sq4_vals[4] = { vLeft, vC, vTop, vTL };
                processSubQuad(sq4_pts, sq4_vals);
            }
        }
    }

    vector< vector<zVector> > getOrderedContours(float tolerance = 1e-4f)
    {
        allContours.clear();
        if (isolines.empty()) return allContours;

        // Helper lambda for inexact match
        auto isClose = [tolerance](zVector& a, zVector& b)
            {
                return ((a - b) * (a - b)) < (tolerance * tolerance);
            };

        // Remaining unprocessed segments
        std::vector<std::pair<zVector, zVector>> remaining = isolines;

        while (!remaining.empty())
        {
            std::vector<zVector> contour;
            zVector start = remaining[0].first;
            zVector current = remaining[0].second;
            contour.push_back(start);
            contour.push_back(current);
            remaining.erase(remaining.begin());

            bool extended = true;
            while (extended)
            {
                extended = false;

                for (auto it = remaining.begin(); it != remaining.end(); ++it)
                {
                    if (isClose(current, it->first))
                    {
                        current = it->second;
                        contour.push_back(current);
                        remaining.erase(it);
                        extended = true;
                        break;
                    }
                    else if (isClose(current, it->second))
                    {
                        current = it->first;
                        contour.push_back(current);
                        remaining.erase(it);
                        extended = true;
                        break;
                    }
                    else if (isClose(contour.front(), it->first))
                    {
                        contour.insert(contour.begin(), it->second);
                        extended = true;
                        remaining.erase(it);
                        break;
                    }
                    else if (isClose(contour.front(), it->second))
                    {
                        contour.insert(contour.begin(), it->first);
                        extended = true;
                        remaining.erase(it);
                        break;
                    }
                }
            }

            allContours.push_back(contour);
        }

        return allContours;
    }

    float area_of_contour_island( vector<zVector> &island)
    {
        auto Mod = [](int a, int n) -> int {
            a = a % n;
            return (a < 0) ? a + n : a;
            };

        int n = island.size();
        if (n < 3) return 0.0f; // not a polygon

        float area = 0.0f;

        for (int i = 0; i < n; i++)
        {
            int j = Mod(i + 1, n);
            area += (island[i].x * island[j].y) - (island[j].x * island[i].y);
        }

        area = 0.5f * fabs(area);
        return area;
    }
    //---------------------------------------------

    // -----------------------------------------------------------------------------
    // High-Fidelity Medial Axis Extraction
    // Ported from JS 'isRidgeAtPoint(px, py)' logic
    // -----------------------------------------------------------------------------
    std::vector<zVector> medialPoints;
    

    std::vector<zVector> computeMedialAxis(
        float gradientStep = 1.0f,
        float gradientThreshold = 0.7f
    )
    {
        medialPoints.clear();
        medialPoints.reserve(SF_RES * SF_RES / 10);
        const float h = gradientStep;

        auto sample = [&](float x, float y)
            {
                return sampleAt(x, y);
            };

        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                zVector p = gridPoints[i][j];
                float px = p.x;
                float py = p.y;

                float d_c = field[i][j];
                if (d_c >= 0) continue;                // JS version: only check inside

                // central SDF value
                float d_px = sample(px + h, py);
                float d_nx = sample(px - h, py);
                float d_py = sample(px, py + h);
                float d_ny = sample(px, py - h);

                // Forward/backward finite differences
                float gx_pos = (d_px - d_c) / h;
                float gx_neg = (d_c - d_nx) / h;

                float gy_pos = (d_py - d_c) / h;
                float gy_neg = (d_c - d_ny) / h;

                bool ridge =
                    (fabs(gx_pos - gx_neg) > gradientThreshold) ||
                    (fabs(gy_pos - gy_neg) > gradientThreshold);

                if (ridge)
                {
                    medialPoints.push_back(p);
                }
            }
        }

        return medialPoints;
    }

    // ======================================================================
//  Topology-Preserving Skeletonization (Zhang–Suen Thinning)
//  Produces a 1-pixel-wide medial axis for the ScalarField's binary mask
// ======================================================================

    std::vector<zVector> computeSkeleton(float iso = 0.0f)
    {
        // Step 1: Build binary mask from SDF
        std::vector<uint8_t> mask(SF_RES * SF_RES, 0);
        auto IDX = [&](int i, int j) { return i * SF_RES + j; };

        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                mask[IDX(i, j)] = (field[i][j] <= iso) ? 1 : 0;
            }
        }

        // 8-neighbour offsets
        const int dx[8] = { 0,  1, 1, 1,  0, -1, -1, -1 };
        const int dy[8] = { 1,  1, 0, -1, -1, -1,  0,  1 };

        auto neighbourCount = [&](int x, int y)
            {
                int c = 0;
                for (int k = 0; k < 8; k++)
                {
                    int nx = x + dx[k], ny = y + dy[k];
                    if (nx >= 0 && nx < SF_RES && ny >= 0 && ny < SF_RES)
                        c += mask[IDX(nx, ny)];
                }
                return c;
            };

        auto transitions = [&](int x, int y)
            {
                // Count 0→1 transitions in ordered neighbour cycle
                int t = 0;
                for (int k = 0; k < 8; k++)
                {
                    int k2 = (k + 1) % 8;

                    int a = 0, b = 0;
                    int nx1 = x + dx[k], ny1 = y + dy[k];
                    int nx2 = x + dx[k2], ny2 = y + dy[k2];

                    if (nx1 >= 0 && nx1 < SF_RES && ny1 >= 0 && ny1 < SF_RES) a = mask[IDX(nx1, ny1)];
                    if (nx2 >= 0 && nx2 < SF_RES && ny2 >= 0 && ny2 < SF_RES) b = mask[IDX(nx2, ny2)];

                    if (a == 0 && b == 1) t++;
                }
                return t;
            };

        bool changed = true;

        // Temporary list of pixels to delete
        std::vector<std::pair<int, int>> toDelete;

        // ============================================================
        // Main thinning loop: two sub-iterations per Zhang–Suen step
        // ============================================================
        while (changed)
        {
            changed = false;
            toDelete.clear();

            // -------------------
            // Sub-iteration 1
            // -------------------
            for (int i = 1; i < SF_RES - 1; i++)
            {
                for (int j = 1; j < SF_RES - 1; j++)
                {
                    if (mask[IDX(i, j)] == 0) continue;

                    int N = neighbourCount(i, j);
                    if (N < 2 || N > 6) continue;

                    int T = transitions(i, j);
                    if (T != 1) continue;

                    // 4 specific neighbour constraints
                    int p2 = mask[IDX(i, j + 1)];
                    int p4 = mask[IDX(i + 1, j)];
                    int p6 = mask[IDX(i, j - 1)];
                    int p8 = mask[IDX(i - 1, j)];

                    if (p2 * p4 * p6 != 0) continue;
                    if (p4 * p6 * p8 != 0) continue;

                    toDelete.push_back({ i,j });
                }
            }

            if (!toDelete.empty())
            {
                changed = true;
                for (auto& p : toDelete) mask[IDX(p.first, p.second)] = 0;
            }

            toDelete.clear();

            // -------------------
            // Sub-iteration 2
            // -------------------
            for (int i = 1; i < SF_RES - 1; i++)
            {
                for (int j = 1; j < SF_RES - 1; j++)
                {
                    if (mask[IDX(i, j)] == 0) continue;

                    int N = neighbourCount(i, j);
                    if (N < 2 || N > 6) continue;

                    int T = transitions(i, j);
                    if (T != 1) continue;

                    int p2 = mask[IDX(i, j + 1)];
                    int p4 = mask[IDX(i + 1, j)];
                    int p6 = mask[IDX(i, j - 1)];
                    int p8 = mask[IDX(i - 1, j)];

                    if (p2 * p4 * p8 != 0) continue;
                    if (p2 * p6 * p8 != 0) continue;

                    toDelete.push_back({ i,j });
                }
            }

            if (!toDelete.empty())
            {
                changed = true;
                for (auto& p : toDelete) mask[IDX(p.first, p.second)] = 0;
            }
        }

        // ============================================================
        // Pack the final 1-pixel skeleton into world coordinates
        // ============================================================
        std::vector<zVector> skeleton;

        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                if (mask[IDX(i, j)] == 1)
                {
                    skeleton.push_back(gridPoints[i][j]);
                }
            }
        }

        return skeleton;
    }

    // ======================================================================
    // Convert skeleton pixels to graph structure
    // skeletonPts is output of computeSkeleton()
    // ======================================================================

    void buildSkeletonGraph(
        const std::vector<zVector>& skeletonPts,
        std::vector<SkelNode>& nodes,
        std::vector<SkelEdge>& edges)
    {
        const float eps = 0.01f;

        // ---- Build a fast lookup: map grid coordinate → index ----
        std::unordered_map<long long, int> lookup;

        auto hash = [&](int i, int j)
            {
                return (long long)i << 32 | (unsigned long long)j;
            };

        // Convert world coord → grid coord
        auto toGrid = [&](const zVector& p, int& i, int& j)
            {
                float span = 100.0f;
                float step = span / (SF_RES - 1);

                i = int((p.x + 50.0f) / step + 0.5f);
                j = int((p.y + 50.0f) / step + 0.5f);
                i = std::max(0, std::min(i, SF_RES - 1));
                j = std::max(0, std::min(j, SF_RES - 1));
            };

        int N = skeletonPts.size();
        std::vector<int> gi(N), gj(N);

        for (int k = 0; k < N; k++)
        {
            int i, j;
            toGrid(skeletonPts[k], i, j);
            gi[k] = i;
            gj[k] = j;
            lookup[hash(i, j)] = k;
        }

        // ---- Compute degree of each skeleton pixel ----
        std::vector<int> degree(N, 0);

        for (int k = 0; k < N; k++)
        {
            int i = gi[k];
            int j = gj[k];

            for (int di = -1; di <= 1; di++)
                for (int dj = -1; dj <= 1; dj++)
                {
                    if (di == 0 && dj == 0) continue;

                    long long h = hash(i + di, j + dj);
                    if (lookup.count(h))
                        degree[k]++;
                }
        }

        // ---- Identify node pixels: endpoints or junctions ----
        std::vector<bool> isNode(N, false);
        for (int k = 0; k < N; k++)
        {
            if (degree[k] != 2)  // endpoint OR branch
                isNode[k] = true;
        }

        // ---- Assign node indices ----
        nodes.clear();
        std::vector<int> nodeIndex(N, -1);

        for (int k = 0; k < N; k++)
        {
            if (isNode[k])
            {
                SkelNode nd;
                nd.pos = skeletonPts[k];
                nd.degree = degree[k];
                nd.index = nodes.size();

                nodes.push_back(nd);
                nodeIndex[k] = nd.index;
            }
        }

        // ---- Trace edges ----
        edges.clear();
        std::vector<bool> visited(N, false);

        auto neighbors = [&](int idx)
            {
                std::vector<int> out;
                int i = gi[idx], j = gj[idx];

                for (int di = -1; di <= 1; di++)
                    for (int dj = -1; dj <= 1; dj++)
                    {
                        if (di == 0 && dj == 0) continue;

                        long long h = hash(i + di, j + dj);
                        if (lookup.count(h))
                            out.push_back(lookup[h]);
                    }
                return out;
            };

        // For each node pixel, walk along degree-2 pixels until next node
        for (int k = 0; k < N; k++)
        {
            if (!isNode[k]) continue;

            auto nbrs = neighbors(k);

            for (int nb : nbrs)
            {
                if (visited[nb]) continue;

                // begin new edge
                SkelEdge e;
                e.n0 = nodeIndex[k];
                e.polyline.push_back(skeletonPts[k]);

                int cur = nb;
                int prev = k;

                while (true)
                {
                    visited[cur] = true;
                    e.polyline.push_back(skeletonPts[cur]);

                    if (isNode[cur])   // reached endpoint or junction
                    {
                        e.n1 = nodeIndex[cur];
                        break;
                    }

                    // continue along degree-2 chain
                    auto nbr2 = neighbors(cur);
                    int next = -1;
                    for (int x : nbr2)
                        if (x != prev) next = x;

                    if (next == -1) break; // dead end (should not occur)
                    prev = cur;
                    cur = next;
                }

                edges.push_back(e);
            }
        }
    }

    // --------------------------------------------

    // Add this helper function inside the ScalarField2D class

    // Implementation of Figure 4: Binary search for the medial point along the segment PQ.
    // It uses the SDF to approximate the 'maximal ball' check.
    zVector computeMedialPoint(zVector P, zVector Q, float EPSILON = 1e-4f)
    {
        // The basePoint P in Figure 4 is the starting point on the boundary (or the ray cast origin).
        // In this adaptive version, P and Q are points defining the segment to search.
        zVector basePoint = P;

        // Check if the segment is negligible from the start
        if ((Q - P).length() < EPSILON) return P;

        zVector midPoint = (P + Q) * 0.5f;

        // Radius is the distance from the midpoint (potential center) back to the boundary point P.
        // NOTE: This assumes P is on the boundary, which is the definition in the paper.
        float radius = (midPoint - P).length();

        // Binary search loop
        while ((Q - P).length() > EPSILON)
        {
            // Adaptation of mesh.containsBall(midPoint, radius) using SDF:
            // A ball is contained if the distance from the center (midPoint) to the boundary
            // is greater than or equal to the ball's radius R. The distance is -SDF(midPoint).
            // We use sampleAt to get the SDF value at the midpoint's location.
            // float sdf_at_midpoint = sampleAt(midPoint.x, midPoint.y);

            // --- NOTE: In the paper's context, the radius R must be the distance from M to P.
            //           For an inscribed ball, R = -SDF(M). We check if R_to_P is valid.
            // We will use the SDF value at the midpoint as the maximal possible radius at that center.

            float maximalRadius = -sampleAt(midPoint.x, midPoint.y);

            // The condition for the medial axis point search is: 
            // If the ball with radius R (distance to P) is contained, the true maximal ball center 
            // must lie further down the segment (toward Q). If not, the center is closer to P.

            // This is a simplified containment check. A true maximal ball center M is where 
            // R_M = -SDF(M). In a linear search, we check if R_to_P <= -SDF(M).
            if (radius <= maximalRadius) // If the ball is contained/smaller than maximal
            {
                // Grow: Move P towards Q (midPoint becomes the new lower bound)
                P = midPoint;
            }
            else
            {
                // Shrink: Move Q towards P (midPoint becomes the new upper bound)
                Q = midPoint;
            }

            // Update midPoint and radius for the next iteration
            midPoint = (P + Q) * 0.5f;
            radius = (midPoint - basePoint).length();
        }

        // The point P (which now approximates Q and midPoint) is the center of the maximal ball.
        // The radius is tracked for the Medial Axis Transform (MAT).
        // Since we don't have the internal structure for zVector/Point, we return the point.
        // midPoint.setRadius (radius)
        return midPoint;
    }

    // Add this high-level function inside the ScalarField2D class

// Implementation of Figure 3: Fast sampling-based algorithm (adapted for SDF).
// Since mesh sampling/ray intersection is unavailable, we must estimate P and Q.
// We approximate the ray-mesh-intersection-based P and Q with a simple ray test 
// from a boundary point (P) inward to a deep interior point (Q).
    std::vector<zVector> computeMedialAxisSampling(int numSamples, float step = 10.0f)
    {
        // Clear previous results
        medialPoints.clear();
        medialPoints.reserve(numSamples);

        // This is a very rough substitute for mesh boundary sampling and intersection.
        // We sample on the iso-contour and shoot a ray inward.
        // NOTE: This requires computeIsocontours(0.0) to be called first to get boundary lines.

        // We will use the boundary lines (isolines) found by computeIsocontours(0.0) as the 'mesh'.
        if (isolines.empty())
        {
            // Recompute the boundary if needed (assuming 0.0 is the boundary/iso-surface)
            computeIsocontours(0.0f);
        }

        if (isolines.empty()) return medialPoints;

        // Estimate the total length for proportional sampling
        float totalLength = 0.0f;
        for ( auto& segment : isolines)
        {
            totalLength += (segment.second - segment.first).length();
        }
        if (totalLength < 1e-6f) return medialPoints;

        // --- Adaptive Sampling Loop ---
        for (int i = 0; i < numSamples; i++)
        {
            // 1. Estimate mesh.getRandomFace() and f.getRandomPoint(): 
            //    Select a random point P on the boundary (a random isoline segment).
            float target = ((float)std::rand() / RAND_MAX) * totalLength;
            float currentLength = 0.0f;
            zVector P, inwardNormal;

            for ( auto& segment : isolines)
            {
                float segLen = (segment.second - segment.first).length();
                if (target < currentLength + segLen)
                {
                    float t = (target - currentLength) / segLen;
                    P = segment.first + (segment.second - segment.first) * t;

                    // Estimate inward normal (gradient points outward for SDF inside < 0)
                    inwardNormal = gradientAt(P) * -1;
                    inwardNormal.normalize();

                    break;
                }
                currentLength += segLen;
            }

            // If P is not found (shouldn't happen), skip
            if (P.length() == 0 && i > 0) continue;

            // 2. Estimate mesh.getIntersection(r):
            //    We simply shoot a fixed-length ray inward to a deep interior point Q.
            //    The maximal ball center MUST lie on the segment PQ.
            zVector Q = P + inwardNormal * step; // 'step' is a large distance inward

            // 3. Compute Medial Point using Binary Search
            zVector medialPoint = computeMedialPoint(P, Q, 1e-4f);

            medialPoints.push_back(medialPoint);
        }

        return medialPoints;
    }

    // --------------------------------------------

    void smoothContourAdaptive(std::vector<zVector>& contour, int iterations = 1, bool preserveEnds = true, float angleThreshold = 15.0f)
    {
        if (contour.size() < 3) return;

        auto angleBetween = []( zVector& a,  zVector& b,  zVector& c) -> float
            {
                zVector u = b - a;
                zVector v = c - b;
                u.normalize();
                v.normalize();
                float dot = std::clamp(u * v, -1.0f, 1.0f);
                return acos(dot) * RAD_TO_DEG;
            };

        for (int iter = 0; iter < iterations; iter++)
        {
            std::vector<zVector> smoothed = contour;

            for (size_t i = 1; i < contour.size() - 1; ++i)
            {
                if (preserveEnds && (i == 0 || i == contour.size() - 1)) continue;

                float angle = angleBetween(contour[i - 1], contour[i], contour[i + 1]);

                if (angle < angleThreshold)
                {
                    // High curvature → preserve
                    smoothed[i] = contour[i];
                }
                else
                {
                    // Low curvature → smooth
                    smoothed[i] =
                        contour[i - 1] * 0.25f +
                        contour[i] * 0.50f +
                        contour[i + 1] * 0.25f;
                }
            }

            contour = smoothed;
        }
    }

    void smoothContour(std::vector<zVector>& contour, int iterations = 1)
    {
        if (contour.size() < 3) return;

        for (int iter = 0; iter < iterations; iter++)
        {
            std::vector<zVector> smoothed = contour;

            for (size_t i = 1; i < contour.size() - 1; ++i)
            {
                smoothed[i] =
                    contour[i - 1] * 0.3f +
                    contour[i] * 0.4f +
                    contour[i + 1] * 0.3f;
            }

            contour = smoothed;
        }
    }

    //---------------------------------------------

    void drawStreamLine()
    {
        for (int i = 0; i < streamLine.size(); i++)
        {
            int nxt = (i + 1) % streamLine.size();

            drawLine(zVecToAliceVec(streamLine[i]), zVecToAliceVec(streamLine[nxt]));
        }

    }

    char s[20];
    void drawFieldPoints(bool drawGradient = false, bool debug = false)
    {
        Alice::vec pt;
        normalise();
        glPointSize(2);
        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                float f = field[i][j];
                    if (f > 1e2) continue;

                float r, g, b;
                getJetColor(f, r, g, b);

                //glColor3f(field[i][j], 0, 0);
                glColor3f(r, g, b);
                drawPoint( zVecToAliceVec(gridPoints[i][j]) );

                if (debug)
                {
                    sprintf(s, "%.2f", field[i][j]);
                    drawString(s, zVecToAliceVec(gridPoints[i][j]));
                }

                if (drawGradient)
                {
                    pt = zVecToAliceVec(gridPoints[i][j]);
                    zVector grad = gradient[i][j];
                    grad.normalize();

                    drawLine(pt , pt + zVecToAliceVec(grad) * 0.4);
                }
            }
        }
        glPointSize(1);
    }

    void drawIsocontours(float threshold, bool draw = true)
    {

        computeIsocontours(threshold);

        glColor3f(0, 0, 0);
        if (draw)
            for (auto& segment : isolines)
            {
                glLineWidth(3);
                drawLine(zVecToAliceVec(segment.first), zVecToAliceVec(segment.second));
                glLineWidth(1);
            }
        //glColor3f(0, 0, 0);


    }

    //---------------------------------------

    void printField()
    {
        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                float f = field[i][j];
                float r, g, b;
                getJetColor(f, r, g, b);

                glColor3f(r, g, b);
                drawPoint(zVecToAliceVec(gridPoints[i][j]));

                sprintf(s, "%.2f", field[i][j]);
                cout << s << endl;
            }
        }

    }

    void exportOrderedContoursAsCSV( std::string& filename, float tolerance = 1e-4f)
    {
        std::ofstream out(filename);
        if (!out.is_open())
        {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }

        std::vector<std::vector<zVector>> contours = getOrderedContours(tolerance);

        for (size_t i = 0; i < contours.size(); ++i)
        {
            out << "Contour_" << i << "\n";
            for ( auto& pt : contours[i])
            {
                out << std::fixed << std::setprecision(6)
                    << pt.x << "," << pt.y << "," << pt.z << "\n";
            }
            out << "\n";
        }

        out.close();
        std::cout << "Contours exported to: " << filename << std::endl;
    }

};


#endif // !_SCALAR_FIELD_