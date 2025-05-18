#define _MAIN_
#ifdef _MAIN_

#include "main.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>


using namespace zSpace;

Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

#include "scalarField.h"

//--------------------------------------------------
// Global Instance
//--------------------------------------------------

std::vector<zVector> rbfCenters;
double isoThresh = 0.5;
double smoothness = 4.0f;

ScalarField2D capsuleField, boxField;
static ScalarField2D field1, field2;

bool showField1 = true;
bool showField2 = true;
bool showBlended = true;


void setup()
{
    S.addSlider(&isoThresh, "threshold");// make a slider control for the variable called width;
    S.sliders[0].maxVal = 1;

    S.addSlider(&smoothness, "smooth");// make a slider control for the variable called width;
    S.sliders[1].maxVal = 100;

 

}

void update(int value)
{
}

bool initialized = false;

void draw()
{
    backGround(0.9);
    drawGrid(50);

 
   
    float angle = 30;
    float angleRadians = 30.0f * 3.1415926f / 180.0f;

    if (!initialized)
    {
        field1.clearField();
        field2.clearField();

      //  field1.addOrientedBoxSDF(zVector(-5, -5, 0), zVector(12,6, 0), angleRadians);
        
        //field1.addUnevenCapsuleSDF(zVector(-10, 0, 0), zVector(10, 0, 0), 2.0f, 5.0f);
       // field2.addBoxSDF(zVector(5, 5, 0), zVector(4, 2, 0));
       //field2.addOrientedBoxSDF(zVector( 8, -5, 0), zVector(12, 6, 0), -30.0f * 3.1415926f / 180.0f);

       field2.addOrientedBoxSDF(zVector(0, 0, 0), zVector(12, 6, 0), 90 * 3.1415926f / 180.0f);
       //
       
       rbfCenters.push_back(zVector(0, 0, 0));
       rbfCenters.push_back(zVector(0, 10, 0));

       field1.addRadialFunctions(rbfCenters);
       // 
       field1.normalise();
       field2.normalise();
      // field2.clearField();

       

       initialized = true;
    }

    if (showField1)
    {
        glColor3f(0.0, 0.0, 1.0);  // blue
        field1.drawFieldPoints();
        field1.drawIsocontours(isoThresh);
    }

    if (showField2)
    {
        glColor3f(1.0, 0.0, 0.0);  // red
        field2.drawFieldPoints();
        field2.drawIsocontours(isoThresh);
    }



}

void keyPress(unsigned char k, int xm, int ym)
{
     
    if (k == '=')
    {
       // field1.blendWith(field2, smoothness);  // adjust k to control smoothness
        field1.unionWith(field2);
    }

    if (k == '-')
    {
        // field1.blendWith(field2, smoothness);  // adjust k to control smoothness
        field1.subtract(field2);  // now performs clean SDF subtraction
        field1.normalise();
    }

    if (k == '0')
    {
        // field1.blendWith(field2, smoothness);  // adjust k to control smoothness
        field1.intersectWith(field2);
    }

    if (k == 'i')
    {
        initialized = false;
    }

   
    if (k == '1') showField1 = !showField1;
    if (k == '2') showField2 = !showField2;
   
}

void mousePress(int b, int state, int x, int y)
{
}

void mouseMotion(int x, int y)
{
}

#endif // _MAIN_
