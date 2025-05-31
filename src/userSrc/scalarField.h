#pragma once
#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;

//these two functiosn must be turned on for sketch_circleSDF_fitter.cpp
inline zVector zMax(const zVector& a, const zVector& b)
{
    return zVector(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

inline float smin(float a, float b, float k)
{
    float h = std::max(k - fabs(a - b), 0.0f) / k;
    return std::min(a, b) - h * h * k * 0.25f;
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

#define OUT 1e6

class ScalarField2D
{
public:


  

    static const int RES = 100;
    int div = 2; 

    zVector gridPoints[RES][RES];
    float field[RES][RES];
    float field_normalized[RES][RES];
    zVector gradient[RES][RES];
    std::vector<std::pair<zVector, zVector>> isolines;
    std::vector<std::vector<zVector>> allContours;

    ScalarField2D()
    {
        float span = 100.0f; // from -50 to +50
        float step = span / (RES - 1); // spacing between grid points

        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
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
        for (int i = 0; i < RES; i++)
            for (int j = 0; j < RES; j++)
                              field[i][j] = 0;

    }

    //----------------------------------------

    float smin(float a, float b, float k)
    {
        float h = std::max(k - fabs(a - b), 0.0f) / k;
        return std::min(a, b) - h * h * k * 0.25f;
    }

    void addVoronoi(const std::vector<zVector>& sites )
    {
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                zVector pt = gridPoints[i][j];
                float minDist = 1e6f;
                float secondMinDist = 1e6f;

                for (const auto& site : sites)
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
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
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

        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
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
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                zVector p = gridPoints[i][j];
                float d = p.distanceTo(rbfCenters[0]);
                // Signed distance: negative inside, zero on boundary, positive outside
                float val = (d > radius) ? d : d - radius;

                for (int i = 1; i < rbfCenters.size();  i++)
                {
                    float d_i = p.distanceTo(rbfCenters[i]);
                    val = (d_i > radius) ? d_i : d_i - radius;
                    d = std::min(d, d_i);
                }

                field[i][j] = d;
            }
        }

        rescaleFieldToRange(-1, 1);
    }
    //----------------------------------------
    
    void unionWith(const ScalarField2D& other)
    {
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                field[i][j] = std::min(field[i][j], other.field[i][j]);
            }
        }
    }

    void intersectWith(const ScalarField2D& other)
    {
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                field[i][j] = std::max(field[i][j], other.field[i][j]);
            }
        }
    }

    void subtract(const ScalarField2D& other)
    {
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                field[i][j] = std::max(field[i][j], -other.field[i][j]);
            }
        }
    }

    void blendWith(const ScalarField2D& other, float smooth_k, SMinMode mode = SMinMode::EXPONENTIAL)
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

        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
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

    void normalise()
    {
        float mn = 1e6f, mx = -1e6f;
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                if (fabs(field[i][j] - OUT) < 1e-6)continue; //exclude outside

                mn = std::min(mn, field[i][j]);
                mx = std::max(mx, field[i][j]);
            }
        }

        float range = std::max(mx - mn, 1e-6f);
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                if (fabs(field[i][j] - OUT) < 1e-6)continue; //exclude outside

                field_normalized[i][j] = ofMap(field[i][j], mn, mx, 0, 1) ;
            }
        }
    }

    void rescaleFieldToRange(float targetMin = -1.0f, float targetMax = 1.0f)
    {
        float minVal[2] = { 1e6f,  1e6f };
        float maxVal[2] = { -1e6f, -1e6f };

        for (int i = 0; i < RES; ++i)
            for (int j = 0; j < RES; ++j)
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

        for (int i = 0; i < RES; ++i)
            for (int j = 0; j < RES; ++j)
                field[i][j] = (field[i][j] >= 0.0f)
                ? ofMap(field[i][j], minVal[0], maxVal[0], 0.0f, targetMax)
                : ofMap(field[i][j], minVal[1], maxVal[1], targetMin, 0.0f);
    }

    //---------------------------------------------


    void computeGradient()
    {
        for (int i = 1; i < RES - 1; i++)
        {
            for (int j = 1; j < RES - 1; j++)
            {
                float dx = (field[i + 1][j] - field[i - 1][j]) * 0.5f;
                float dy = (field[i][j + 1] - field[i][j - 1]) * 0.5f;
                gradient[i][j] = zVector(dx, dy, 0);
            }
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

    void computeIsocontours( float threshold)
    {
        isolines.clear();
        

        for (int i = 0; i < RES - 1; i++)
        {
            for (int j = 0; j < RES - 1; j++)
            {
                zVector quadPts[4] = {
                    gridPoints[i][j],
                    gridPoints[i + 1][j],
                    gridPoints[i][j + 1],
                    gridPoints[i + 1][j + 1]
                };

                float quadVals[4] = {
                    field[i][j],
                    field[i + 1][j],
                    field[i][j + 1],
                    field[i + 1][j + 1]
                };

                zVector tri1Pts[3] = { quadPts[0], quadPts[1], quadPts[2] };
                float tri1Vals[3] = { quadVals[0], quadVals[1], quadVals[2] };

                zVector tri2Pts[3] = { quadPts[1], quadPts[2], quadPts[3] };
                float tri2Vals[3] = { quadVals[1], quadVals[2], quadVals[3] };

                processTriangle(tri1Pts, tri1Vals, threshold, isolines);
                processTriangle(tri2Pts, tri2Vals, threshold, isolines);
            }
        }
    }

    void computeIsocontours(float threshold, std::vector<std::pair<zVector, zVector>>& output)
    {
        for (int i = 0; i < RES - 1; i++)
        {
            for (int j = 0; j < RES - 1; j++)
            {
                zVector p[4] = {
                    gridPoints[i][j],
                    gridPoints[i + 1][j],
                    gridPoints[i + 1][j + 1],
                    gridPoints[i][j + 1]
                };

                float v[4] = {
                    field[i][j],
                    field[i + 1][j],
                    field[i + 1][j + 1],
                    field[i][j + 1]
                };

                auto addLine = [&](zVector a, zVector b, float va, float vb)
                    {
                        if ((va < threshold && vb >= threshold) || (vb < threshold && va >= threshold))
                        {
                            float t = (threshold - va) / (vb - va);
                            output.emplace_back(a + (b - a) * t, zVector());  // start pt added
                        }
                    };

                std::vector<zVector> pts;
                for (int k = 0; k < 4; k++)
                {
                    int next = (k + 1) % 4;
                    if ((v[k] < threshold && v[next] >= threshold) ||
                        (v[next] < threshold && v[k] >= threshold))
                    {
                        float t = (threshold - v[k]) / (v[next] - v[k]);
                        pts.push_back(p[k] + (p[next] - p[k]) * t);
                    }
                }

                if (pts.size() == 2)
                {
                    output.emplace_back(pts[0], pts[1]);
                }
            }
        }
    }


    std::vector<std::vector<zVector>> getOrderedContours(float tolerance = 1e-4f)
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

 
    //---------------------------------------------

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

    char s[20];
    void drawFieldPoints( bool debug = false)
    {
        normalise();
        glPointSize(2);
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                float f = field[i][j];
                float r, g, b;
                getJetColor(f, r, g, b);

                glColor3f(r, g, b);
                drawPoint(zVecToAliceVec(gridPoints[i][j]));

                if (debug)
                {
                    sprintf(s, "%.2f", field[i][j]);
                    drawString(s, zVecToAliceVec(gridPoints[i][j]));
                }
            }
        }
        glPointSize(1);
    }

    void drawIsocontours(float threshold, bool draw = true)
    {

        computeIsocontours(threshold);

        glColor3f(1, 0, 0);
        if (draw)
            for (auto& segment : isolines)
            {
                glLineWidth(3);
                drawLine(zVecToAliceVec(segment.first), zVecToAliceVec(segment.second));
                glLineWidth(1);
            }
        glColor3f(0, 0, 0);


    }

    //---------------------------------------

    void printField()
    {
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
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

    void exportOrderedContoursAsCSV(const std::string& filename, float tolerance = 1e-4f)
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
            for (const auto& pt : contours[i])
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
