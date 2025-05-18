#pragma once
#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;

inline zVector zMax(const zVector& a, const zVector& b)
{
    return zVector(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

inline float smin(float a, float b, float k)
{
    float h = std::max(k - fabs(a - b), 0.0f) / k;
    return std::min(a, b) - h * h * k * 0.25f;
}


class ScalarField2D
{
public:

    static const int RES = 200;
    int div = 2; 

    zVector gridPoints[RES][RES];
    float field[RES][RES];
    float field_normalized[RES][RES];
    zVector gradient[RES][RES];
    std::vector<std::pair<zVector, zVector>> isolines;

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
                field[i][j] = 0.0f;
                field_normalized[i][j] = 0.0f;
            }
        }
    }

    void addRadialBasisFunctions(const std::vector<zVector>& centers)
    {
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                zVector pt = gridPoints[i][j];
                float value = 0.0f;

                for (const auto& c : centers)
                {
                    zVector tmp = c;
                    float d = pt.distanceTo(tmp);

                    float w = 1.0f / (1.0f + d * d); // inverse quadratic RBF
                    //float w = d * d; // or just 'd' for linear growth
                    value += w;
                }

                field[i][j] = value; // std::min(field[i][j], value);
            }
        }
    }

    void addBoxSDF(zVector boxCenter, zVector boxHalfSize)
    {
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                zVector p = gridPoints[i][j] - boxCenter;
                zVector q(fabs(p.x), fabs(p.y), 0);
                zVector d = q - boxHalfSize;

                float outsideDist = zMax(d, zVector(0, 0, 0)).length();
                float insideDist = std::min(std::max(d.x, d.y), 0.0f);

                field[i][j] = outsideDist + insideDist;
            }
        }
    }

    void addUnevenCapsuleSDF(zVector a, zVector b, float ra, float rb)
    {
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                zVector p = gridPoints[i][j];

                zVector pa = p - a;
                zVector ba = b - a;

                float hRaw = (pa * ba) / (ba * ba);
                float h = (hRaw < 0.0f) ? 0.0f : (hRaw > 1.0f) ? 1.0f : hRaw;

                float r = ra + (rb - ra) * h;

                float sdf = (pa - ba * h).length() - r;

                field[i][j] = sdf;
            }
        }
    }

    //void addOrientedBoxSDF(zVector center, zVector halfSize, float angleRadians)
    //{
    //    float c = cos(angleRadians);
    //    float s = sin(angleRadians);

    //    for (int i = 0; i < RES; i++)
    //    {
    //        for (int j = 0; j < RES; j++)
    //        {
    //            zVector p = gridPoints[i][j] - center;

    //            // Rotate point into box's local frame
    //            zVector pr(
    //                c * p.x + s * p.y,
    //                -s * p.x + c * p.y,
    //                0.0f
    //            );

    //            // Compute SDF to axis-aligned box in local frame
    //            zVector d = zVector(fabs(pr.x), fabs(pr.y), 0.0f) - halfSize;
    //            float outsideDist = zMax(d, zVector(0, 0, 0)).length();
    //            float insideDist = std::min(std::max(d.x, d.y), 0.0f);

    //            field[i][j] = std::min(outsideDist + insideDist, field[i][j]);
    //        }
    //    }
    //}

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

                // Signed distance to axis-aligned box in local frame
                zVector d = zVector(fabs(pr.x), fabs(pr.y), 0.0f) - halfSize;

                // Correct SDF: positive outside, negative inside
                float outsideDist = zMax(d, zVector(0, 0, 0)).length();
                float insideDist = std::min(std::max(d.x, d.y), 0.0f);

                float signedDist = (outsideDist > 0.0f) ? outsideDist : insideDist;

                field[i][j] = signedDist; // std::min(signedDist, field[i][j]);
            }
        }
    }


    void clearField()
    {
        for (int i = 0; i < RES; i++)
          for (int j = 0; j < RES; j++)
                field[i][j] = 0;

    }
    
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

    void blendWith(const ScalarField2D& other, float smooth_k)
    {
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                field[i][j] = smin(field[i][j], other.field[i][j], smooth_k);
            }
        }
    }


    void normalise()
    {
        float mn = 1e6f, mx = -1e6f;
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                mn = std::min(mn, field[i][j]);
                mx = std::max(mx, field[i][j]);
            }
        }

        float range = std::max(mx - mn, 1e-6f);
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                field_normalized[i][j] = ofMap(field[i][j], mn, mx, 0, 1);
            }
        }
    }

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

    void drawIsocontours(float threshold , bool draw = true)
    {
        isolines.clear();
        normalise();
        
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
                    field_normalized[i][j],
                    field_normalized[i + 1][j],
                    field_normalized[i][j + 1],
                    field_normalized[i + 1][j + 1]
                };

                zVector tri1Pts[3] = { quadPts[0], quadPts[1], quadPts[2] };
                float tri1Vals[3] = { quadVals[0], quadVals[1], quadVals[2] };

                zVector tri2Pts[3] = { quadPts[1], quadPts[2], quadPts[3] };
                float tri2Vals[3] = { quadVals[1], quadVals[2], quadVals[3] };

                processTriangle(tri1Pts, tri1Vals, threshold, isolines);
                processTriangle(tri2Pts, tri2Vals, threshold, isolines);
            }
        }

        glColor3f(0, 0, 0);
        if( draw)
        for (auto& segment : isolines)
        {
            glLineWidth(3);
             drawLine(zVecToAliceVec(segment.first), zVecToAliceVec(segment.second));
            glLineWidth(1);
        }

       

    }

    std::vector<std::vector<zVector>> getOrderedContours(float tolerance = 1e-4f)
    {
        std::vector<std::vector<zVector>> allContours;
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


    void drawFieldPoints()
    {
        normalise();
        glPointSize(1);
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                float f = field_normalized[i][j];
                glColor3f(f, 0.0f, 1.0f - f);
                drawPoint(zVecToAliceVec(gridPoints[i][j]));
            }
        }
        glPointSize(1);
    }
};
