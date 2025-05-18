#define _MAIN_
#ifdef _MAIN_

#include "main.h"
#include <vector>
#include <cmath>

#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;

Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

zVector AliceVecToZvec(Alice::vec& in)
{
    return zVector(in.x, in.y, in.z);
}

class ScalarField2D
{
public:

    static const int RES = 100;

    zVector gridPoints[RES][RES];
    float scalar[RES][RES];
    float scalarNorm[RES][RES];
    std::vector<std::pair<zVector, zVector>> isolines;

    ScalarField2D()
    {
        for (int i = 0; i < RES; ++i)
        {
            for (int j = 0; j < RES; ++j)
            {
                gridPoints[i][j] = zVector(i - RES / 2, j - RES / 2, 0);
                scalar[i][j] = 0.0f;
                scalarNorm[i][j] = 0.0f;
            }
        }
    }

    void normalize()
    {
        float minVal = 1e6, maxVal = -1e6;
        for (int i = 0; i < RES; ++i)
        {
            for (int j = 0; j < RES; ++j)
            {
                minVal = std::min(minVal, scalar[i][j]);
                maxVal = std::max(maxVal, scalar[i][j]);
            }
        }

        float range = maxVal - minVal;
        if (range == 0) range = 1.0f;

        for (int i = 0; i < RES; ++i)
        {
            for (int j = 0; j < RES; ++j)
            {
                scalarNorm[i][j] = (scalar[i][j] - minVal) / range;
            }
        }
    }

    void addRadialBasisFunctions(const std::vector<zVector>& centers, float radius = 10.0f, float weight = 1.0f, std::string type = "gaussian")
    {
        float r2 = radius * radius;

        for (int i = 0; i < RES; ++i)
        {
            for (int j = 0; j < RES; ++j)
            {
                zVector pt = gridPoints[i][j];
                float val = 0.0f;

                for (const auto& c : centers)
                {
                    zVector cen = c; 
                    float d2 = pt.squareDistanceTo(cen);

                    if (type == "gaussian")
                    {
                        val += weight * std::exp(-d2 / r2);
                    }
                    else if (type == "multiquadric")
                    {
                        val += weight * std::sqrt(d2 + r2);
                    }
                    else if (type == "inverse_quadric")
                    {
                        val += weight / (1.0f + d2 / r2);
                    }
                }

                scalar[i][j] += val;
            }
        }
    }

    void processTriangle(zVector pts[3], float values[3], float threshold, std::vector<std::pair<zVector, zVector>>& contour)
    {
        std::vector<zVector> edgePts;
        for (int i = 0; i < 3; i++)
        {
            int next = (i + 1) % 3;
            if ((values[i] < threshold && values[next] >= threshold) ||
                (values[next] < threshold && values[i] >= threshold))
            {
                float t = (threshold - values[i]) / (values[next] - values[i]);
                edgePts.push_back(pts[i] + (pts[next] - pts[i]) * t);
            }
        }

        if (edgePts.size() == 2)
        {
            contour.emplace_back(edgePts[0], edgePts[1]);
        }
    }

    void processAllTriangles(float threshold, std::vector<std::pair<zVector, zVector>>& contourLines)
    {
        normalize();

        contourLines.clear();
        for (int i = 0; i < RES - 1; i++)
        {
            for (int j = 0; j < RES - 1; j++)
            {
                zVector quadPts[4] = {
                    gridPoints[i][j],
                    gridPoints[i][j + 1],
                    gridPoints[i + 1][j],
                    gridPoints[i + 1][j + 1]
                };

                float quadVals[4] = {
                    scalarNorm[i][j],
                    scalarNorm[i][j + 1],
                    scalarNorm[i + 1][j],
                    scalarNorm[i + 1][j + 1]
                };

                zVector tri1[3] = { quadPts[0], quadPts[1], quadPts[2] };
                float val1[3] = { quadVals[0], quadVals[1], quadVals[2] };
                processTriangle(tri1, val1, threshold, contourLines);

                zVector tri2[3] = { quadPts[1], quadPts[2], quadPts[3] };
                float val2[3] = { quadVals[1], quadVals[2], quadVals[3] };
                processTriangle(tri2, val2, threshold, contourLines);
            }
        }
    }

    void draw()
    {
        glPointSize(2.0f);
        for (int i = 0; i < RES; ++i)
        {
            for (int j = 0; j < RES; ++j)
            {
                float c = scalarNorm[i][j];
                glColor3f(c, 0.0f, 0.0f);
                drawPoint(zVecToAliceVec(gridPoints[i][j]));
            }
        }
    }

    void drawIsoline(float threshold)
    {
        processAllTriangles(threshold, isolines);

        glColor3f(1, 1, 0);
        for (const auto& seg : isolines)
        {
            zVector a = seg.first;
            zVector b = seg.second;
            drawLine(zVecToAliceVec(a), zVecToAliceVec(b));
        }
    }
};


//--------------------------------------------------
// Global Instances
//--------------------------------------------------

ScalarField2D sField;
std::vector<zVector> rbfCenters = {
    zVector(0, 0, 0),
    zVector(25, 10, 0),
    zVector(-20, -15, 0)
};

double isolineThreshold = 0.5;

//--------------------------------------------------
void setup()
{
    S.addSlider(&isolineThreshold, "isovalue");
    S.sliders[0].minVal = 0.0;
    S.sliders[0].maxVal = 1.0;

    sField.addRadialBasisFunctions(rbfCenters, 15.0f, 1.0f, "gaussian");
    sField.normalize();
}

void update(int value)
{
}

void draw()
{
    backGround(0.85);
    drawGrid(50);

    sField.draw();
    sField.drawIsoline(isolineThreshold);
}

void keyPress(unsigned char k, int xm, int ym)
{
}

void mousePress(int b, int state, int x, int y)
{
}

void mouseMotion(int x, int y)
{
}

#endif // _MAIN_
