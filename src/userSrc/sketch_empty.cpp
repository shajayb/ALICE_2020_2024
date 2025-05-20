#define _MAIN_
#ifdef _MAIN_

#include "main.h"


//// zSpace Core Headers
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

// Derived class
class ScalarFieldWithCenters : public ScalarField2D
{
private:
    float animationPhase = 0.0f;
    float animationSpeed = 0.05f;

    std::vector<zVector> originalCenters;
    std::vector<zVector> targetCenters; // 🔸 NEW

    std::vector<std::vector<zVector>> stackedContours;
    float contourZOffset = 0.25f;

public:
    std::vector<zVector> centers;

    void initializeCentersOnCircle(int numPoints = 10, float radius = 20.0f)
    {
        centers.clear();
        originalCenters.clear();
        targetCenters.clear();

        float angleStep = TWO_PI / float(numPoints);
        for (int i = 0; i < numPoints; i++)
        {
            float x = radius * cos(i * angleStep);
            float y = radius * sin(i * angleStep);
            zVector c(x, y, 0);
            centers.push_back(c);
            originalCenters.push_back(c);
            targetCenters.push_back(zVector(0, 0, 0)); // default target at origin
        }
    }


    void addCirclesFromCenters(float sdfRadius = 5.0f)
    {
        clearField();

        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                zVector pt = gridPoints[i][j];
                float minDist = 1e6f;

                for (const auto& c : centers)
                {
                    float d = pt.distanceTo(zVector(c));  // copy to avoid const-ref issue
                    float sdf = d - sdfRadius;
                    minDist = std::min(minDist, sdf);
                }

                field[i][j] = minDist;
            }
        }

        rescaleFieldToRange(-1, 1);
    }

    void animateCentersToTargetsAndBack()
    {
        animationPhase += animationSpeed;
        float t = 0.5f * (1.0f + sin(animationPhase)); // [0,1]

        for (int i = 0; i < centers.size(); i++)
        {
            zVector orig = originalCenters[i];
            zVector target = targetCenters[i];

            orig *= (1.0f - t);
            target *= t;

            centers[i] = orig + target;
        }
    }


    void setAnimationSpeed(float s)
    {
        animationSpeed = s;
    }

    void setTargetForCenter(int index, const zVector& target)
    {
        if (index >= 0 && index < targetCenters.size())
        {
            targetCenters[index] = target;
        }
    }

    void setAllTargets(const std::vector<zVector>& targets)
    {
        int count = std::min(int(targetCenters.size()), int(targets.size()));
        for (int i = 0; i < count; i++)
        {
            targetCenters[i] = targets[i];
        }
    }

    void copyAndStackCurrentContours(float threshold)
    {
        computeIsocontours(threshold);
        auto contours = getOrderedContours();

        for (auto& contour : contours)
        {
            for (auto& pt : contour)
            {
                pt.z += stackedContours.size() * contourZOffset;
            }
            stackedContours.push_back(contour);
        }
    }

    void exportStackedContoursAsCSV(const std::string& filename = "stackedContours.csv")
    {
        std::ofstream out(filename);
        if (!out.is_open())
        {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }

        for (size_t layerIdx = 0; layerIdx < stackedContours.size(); ++layerIdx)
        {
            out << "Layer_" << layerIdx << "\n";
            for (const auto& pt : stackedContours[layerIdx])
            {
                out << std::fixed << std::setprecision(6)
                    << pt.x << "," << pt.y << "," << pt.z << "\n";
            }
            out << "\n";
        }

        out.close();
        std::cout << "Stacked contours exported to: " << filename << std::endl;
    }


    void draw(float threshold = 0.0f, bool showCenters = true)
    {
        drawFieldPoints();
        drawIsocontours(threshold);

        if (showCenters)
        {
            glPointSize(8);
            glColor3f(1, 0, 0);
            for (const auto& c : centers)
            {
                drawPoint(zVecToAliceVec(zVector(c)));
            }
            glPointSize(1);
        }
    }



    void drawStackedContours()
    {
        glColor3f(0.2f, 0.2f, 0.2f);
        glLineWidth(2);

        for ( auto contour : stackedContours)
        {
            smoothContour(contour, 15);
            for (size_t i = 0; i < contour.size() - 1; i++)
            {
                drawLine(zVecToAliceVec(zVector(contour[i])), zVecToAliceVec(zVector(contour[i + 1])));
            }
        }

        glLineWidth(1);
    }
};


//---------------------------------------------

ScalarFieldWithCenters myField;
double isoThreshold = 0.0f;
bool run = false;

void setup()
{
    S.addSlider(&isoThreshold, "thresh");
    S.sliders[0].minVal = -1.0;
    S.sliders[0].maxVal = 1.0;

    myField.initializeCentersOnCircle(10, 25.0f);
    myField.addCirclesFromCenters(6.0f);
}

void update(int value)
{
    if (!run) return;

    myField.animateCentersToTargetsAndBack();
    myField.addCirclesFromCenters(6.0f);

    keyPress('s', 0, 0);
}


void draw()
{
    backGround(0.8);
    drawGrid(50);

    myField.draw(isoThreshold, true);
    myField.drawStackedContours(); // ← draw all stacked layers
}


void keyPress(unsigned char k, int xm, int ym)
{
    if (k == '1')run = !run;

    if (k == 'e')
    {
        myField.exportStackedContoursAsCSV("data/stackedContours.csv");
    }

    
    if (k == 'r') // regenerate centers randomly on a circle
    {
        myField.initializeCentersOnCircle(10, 25.0f);
        myField.addCirclesFromCenters(6.0f);
    }

    if (k == 's') // Stack current contours
    {
        myField.copyAndStackCurrentContours(isoThreshold);
    }
}

void mousePress(int b, int state, int x, int y)
{
}

void mouseMotion(int x, int y)
{
}

#endif // _MAIN_
