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

#include "scalarField.h"

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

    //sField.addOrientedBoxSDF( zVector(0,0,0), zVector(24,12,0), PI * 0.25 );
    //sField.rescaleFieldToRange();

    sField.addCircleSDFs(rbfCenters, 20 );;


    sField.clearField();
    rbfCenters.clear();

    #define rx ofRandom(-50,50)

    for (int i = 0; i < 50; i++)
        rbfCenters.push_back(zVector(rx, rx, rx));

    sField.addVoronoi(rbfCenters);
    
}

void update(int value)
{
}

void draw()
{
    backGround(0.85);
    drawGrid(50);

    sField.drawFieldPoints();
    sField.drawIsocontours(isolineThreshold);

    //auto medial = sField.computeMedialAxis();
    //cout << medial.size() << endl;

    //// draw
    //glPointSize(5);
    //for (auto& m : medial)
    //{
    //    glColor3f(1, 1, 1);
    //    drawPoint(zVecToAliceVec(m));
    //}
    //glPointSize(1);

    auto skelPts =sField.computeMedialAxisSampling(100,2);// sField.computeSkeleton(isolineThreshold);   // medial axis for φ(x) ≤ 0

    glPointSize(5);
    glColor3f(1, 0, 0);

    for (auto& p : skelPts)
        drawPoint(zVecToAliceVec(p));

    glPointSize(1);


    // Build graph
    std::vector<SkelNode> nodes;
    std::vector<SkelEdge> edges;

    sField.buildSkeletonGraph(skelPts, nodes, edges);

    // Draw result
    for (auto& E : edges)
    {
        glColor3f(1, 0, 0);
        for (int i = 0; i + 1 < E.polyline.size(); i++)
            drawLine(zVecToAliceVec(E.polyline[i]),
                zVecToAliceVec(E.polyline[i + 1]));
    }

    for (auto& N : nodes)
    {
        glColor3f(0, 1, 0);
        drawPoint(zVecToAliceVec(N.pos));
    }
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
