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

Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

zVector AliceVecToZvec(Alice::vec& in)
{
    return zVector(in.x, in.y, in.z);
}


#include "scalarField.h"

class ScalarField3DSlice
{
public:
    static const int RES = 100;
    zVector gridPoints[RES][RES];
    float field[RES][RES];
    std::vector<std::pair<zVector, zVector>> isolines;
    // --- in class ScalarField3DSlice ---

    std::vector<std::vector<zVector>> stackedContours;
    float contourZOffset = 0.25f;

    float sliceZ = 0.0f;
    float a1 = 1.0f, a2 = 1.0f, a3 = 1.0f, a4 = 1.0f, a5 = 1.0f, a6 = 1.0f;

    ScalarField3DSlice()
    {
        float span = 100.0f;
        float step = span / (RES - 1);
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                float x = -50.0f + i * step;
                float y = -50.0f + j * step;
                gridPoints[i][j] = zVector(x, y, 0);
                field[i][j] = 0.0f;
            }
        }
    }

    void update()
    {
        float span = PI; // Use PI/2 if you want even less repetition
        float step = (2.0f * span) / (RES - 1);

        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                float x = -span + i * step;
                float y = -span + j * step;
                float z = sliceZ;

                // Function sampling
                float fResult = (a1 * cos(1 * x) * cos(2 * y) * cos(3 * z)) +
                    (a3 * cos(2 * x) * cos(1 * y) * cos(3 * z)) +
                    (a4 * cos(2 * x) * cos(3 * y) * cos(1 * z)) +
                    (a5 * sin(3 * x) * sin(1 * y) * sin(2 * z)) +
                    (a2 * sin(1 * x) * sin(3 * y) * sin(2 * z)) +
                    (a6 * sin(3 * x) * sin(2 * y) * sin(1 * z));

                field[i][j] = fResult;
                gridPoints[i][j] = zVector(x * (50.0f / span), y * (50.0f / span), 0); // optional rescale to match visual range
            }
        }

    }

    void update(float z, float t)
    {
        float span = PI;
        float step = (2.0f * span) / (RES - 1);

        float rMin = PI * 0.5f, rMax = PI * 1.0f;
        float kMin = 0.2f, kMax = 0.6f;

        float r = ofLerp(rMin, rMax, 1.0f - t);
        float k = ofLerp(kMin, kMax, t);

        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                float x = -span + i * step;
                float y = -span + j * step;

                // slice at fixed height z
                float cylX = sqrt(x * x + z * z) - r;
                float cylY = sqrt(y * y + z * z) - r;

                // soft min = max for vault intersection
                float h = std::max(k - fabs(cylX - cylY), 0.0f) / k;
                float f = std::min(cylX, cylY) - h * h * k * 0.25f;

                field[i][j] = f;
                gridPoints[i][j] = zVector(x * (50.0f / span), y * (50.0f / span), 0);
            }
        }

        rescaleFieldToRange(-1, 1);
    }


    void rescaleFieldToRange(float targetMin = -1.0f, float targetMax = 1.0f)
    {
        float mn = 1e6f, mx = -1e6f;
        for (int i = 0; i < RES; i++)
            for (int j = 0; j < RES; j++)
            {
                mn = std::min(mn, field[i][j]);
                mx = std::max(mx, field[i][j]);
            }

        float range = std::max(mx - mn, 1e-6f);
        for (int i = 0; i < RES; i++)
            for (int j = 0; j < RES; j++)
                field[i][j] = ofMap(field[i][j], mn, mx, targetMin, targetMax);
    }

    void computeIsocontours(float threshold)
    {
        isolines.clear();
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
                    isolines.emplace_back(pts[0], pts[1]);
            }
        }
    }

    void drawFieldPoints()
    {
        glPointSize(2);
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                float r, g, b;
                getJetColor(field[i][j], r, g, b);
                glColor3f(r, g, b);
                drawPoint(zVecToAliceVec(gridPoints[i][j]));
            }
        }
        glPointSize(1);
    }

    void drawIsocontours(float threshold)
    {
        computeIsocontours(threshold);

        glColor3f(1, 0, 0);
        for (auto& seg : isolines)
        {
            drawLine(zVecToAliceVec(seg.first), zVecToAliceVec(seg.second));
        }
 
    }

    void copyAndStackCurrentContours(float threshold)
    {
        computeIsocontours(threshold);

        // Simple marching edges (not ordered), treat each segment as a contour for stacking
        std::vector<zVector> flatSegment;
        for (auto& seg : isolines)
        {
            flatSegment.push_back(seg.first);
            flatSegment.push_back(seg.second);
        }

        // Create a new contour layer and offset in Z
        for (auto& pt : flatSegment)
        {
            pt.z += stackedContours.size() * contourZOffset;
        }

        stackedContours.push_back(flatSegment);
    }



    void drawStackedContours()
    {
        glColor3f(0.2f, 0.2f, 0.2f);
        glLineWidth(2);

        for (auto& contour : stackedContours)
        {
            for (size_t i = 0; i < contour.size() - 1; i += 2)
            {

               // drawPoint(zVecToAliceVec(contour[i]));
                drawLine(zVecToAliceVec(contour[i]), zVecToAliceVec(contour[i + 1]));
            }
        }

        glLineWidth(1);
    }

};

//----------------------------------------------------------

ScalarField3DSlice myField;

double zSlice = -1.0;
double th = 0.0;
double a1 = 1.0, a2 = 1.0, a3 = 1.0, a4 = 1.0, a5 = 1.0, a6 = 1.0;

void setup()
{
    S.addSlider(&zSlice, "z");           S.sliders[0].minVal = -PI / 2.0; S.sliders[0].maxVal = PI / 2.0;
    S.addSlider(&th, "threshold");       S.sliders[1].minVal = -1.0;      S.sliders[1].maxVal = 1.0;
    S.addSlider(&a1, "a1");              S.sliders[2].minVal = -PI / 2.0; S.sliders[2].maxVal = PI / 2.0;
    S.addSlider(&a2, "a2");              S.sliders[3].minVal = -PI / 2.0; S.sliders[3].maxVal = PI / 2.0;
    S.addSlider(&a3, "a3");              S.sliders[4].minVal = -PI / 2.0; S.sliders[4].maxVal = PI / 2.0;
    S.addSlider(&a4, "a4");              S.sliders[5].minVal = -PI / 2.0; S.sliders[5].maxVal = PI / 2.0;
    S.addSlider(&a5, "a5");              S.sliders[6].minVal = -PI / 2.0; S.sliders[6].maxVal = PI / 2.0;
    S.addSlider(&a6, "a6");              S.sliders[7].minVal = -PI / 2.0; S.sliders[7].maxVal = PI / 2.0;

    myField.update();
}

void update(int value)
{
   


    myField.sliceZ = zSlice;
    myField.a1 = a1;
    myField.a2 = a2;
    myField.a3 = a3;
    myField.a4 = a4;
    myField.a5 = a5;
    myField.a6 = a6;

    myField.update();

    if (zSlice > 1.0)
        return;
    else
        zSlice += 0.01;

    keyPress('s', 0, 0);

    
}

void draw()
{
    backGround(0.9);
    drawGrid(50);

    myField.drawFieldPoints();
    
    myField.drawIsocontours(th);
    myField.drawStackedContours(); // <- new

}

void keyPress(unsigned char k, int xm, int ym) 
{

    if (k == 'c')
    {
        myField.stackedContours.clear();
    }
    if (k == 's')
    {
        myField.copyAndStackCurrentContours(th);

    }


}
void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_
