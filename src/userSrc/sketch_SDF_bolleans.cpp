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

#include "scalarField.h" // Ensure scalarField.h is in userSrc or adjust include path

// global instances
ScalarField2D field1, field2;
double threshold = 0.15;

void setup()
{
    zVector center(0, 0, 0);
    zVector boxSz(10, 3, 0);

    field1.addBoxSDF(center, boxSz);
    field2.addOrientedBoxSDF(zVector(0, -5, 0), zVector(12, 4, 0), 3.14 * 0.25);

    S.addSlider(&threshold, "tv");
 
}

void update(int value)
{


}

void draw()
{
    backGround(0.8);
    drawGrid(50);

    field1.drawFieldPoints();
    field2.drawFieldPoints();


    
    bool draw = true;
    field1.drawIsocontours(threshold,draw);
    field2.drawIsocontours(threshold, draw);


}

void keyPress(unsigned char k, int xm, int ym)
{
  
 

    if (k == '=')
    {
        field1.unionWith(field2);
    }

    if (k == '-')
    {
        field1.subtract(field2);
    }

    if (k == '0')
    {
        field1.intersectWith(field2);
    }

    if (k == 'b')
    {
        field1.blendWith(field2, 4.0);
    }

    /// ------------------------------
    if (k == 'r')
    {
        field1.clearField();
        field2.clearField();
        setup();
        
    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_
