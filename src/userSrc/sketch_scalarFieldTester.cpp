#define _MAIN_
#ifdef _MAIN_

#include "main.h"

//#include "alice/spatialBin.h"
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

#include "scalarField.h" // your extracted header

// -----------------------------------------------------------------------------
// Global Variables
// -----------------------------------------------------------------------------




class movingField
{
public:

    ScalarField2D field;
    zVector cen;
    zVector dir;
    int fieldTYP = 0;

    void moveTo( zVector &target)
    {
        zVector diff = (target - cen);
        diff.normalize();
        cen += diff * 0.1;
    }

    void update()
    {
        field.clearField();

        if( fieldTYP == 0)
            field.addOrientedBoxSDF(cen, zVector(12, 12, 0), dir.angle(zVector(0, 0, 0)));// addCircleSDF(zVector(-15, -15, 0), 25);
        
        if (fieldTYP == 1)
            field.addCircleSDF(cen, 10 , false);

    }

    void draw( float threshold)
    {
        field.drawFieldPoints();
        field.drawIsocontours(threshold);
    }
};

movingField fieldA, fieldB, fieldC;
double threshold;
bool showF1, showF2, showF3;

// -----------------------------------------------------------------------------
// Setup
// -----------------------------------------------------------------------------

void setup()
{
   

    fieldA = movingField();
    fieldB = movingField();
    fieldC = movingField();

    fieldA.fieldTYP = 1;
    fieldB.fieldTYP = 1;
    fieldC.fieldTYP = 1;

    fieldA.dir = zVector(1, 1, 0);
    fieldB.dir = zVector(0, 0, 0);

    fieldA.cen = zVector(-15, -15, 0) ;
    fieldB.cen = zVector(15, 15, 0);
    fieldC.cen = zVector(15, -15, 0);

    fieldA.update();
    fieldB.update();
    fieldC.update();


    vector< zVector> rbfCens;

   // for (int i = 0; i < 10; i++)rbfCens.push_back( zVector(ofRandom(-50, 50), ofRandom(-50, 50), 0)) ;
    for (float i = -25; i < 25; i+= 5.0)
    {
        for (float j = -25; j < 25; j += 5.0)
        {
            rbfCens.push_back(zVector(i,j, 0));
        }
    }

    //fieldC.field.addVoronoi(rbfCens);




    S.addSlider(&threshold, "tv");
    S.sliders[0].minVal = -1;
}

// -----------------------------------------------------------------------------
// Update
// -----------------------------------------------------------------------------

bool run = false;
vector< vector<zVector>> stackContours;

void update(int value)
{
    //if (run)
    //{
    //    threshold -= 0.01;
    //    if (threshold <= 0.2) threshold = 0.2;

    //    fieldA.computeIsocontours(threshold);
    //    fieldA.getOrderedContours();

    //    for (auto& c : fieldA.allContours)
    //    {
    //        fieldA.smoothContour(c,15);
    //        stackContours.push_back(c);
    //    }
    //}
}

// -----------------------------------------------------------------------------
// Draw
// -----------------------------------------------------------------------------


void draw()
{
    backGround(0.8);
    drawGrid(50);

    //float z = 0;
    //for (auto& c : stackContours)
    //{
    //    if (c.empty())continue;

    //    glPushMatrix();
    //    glTranslatef(0, 0, z);
    //    z += 1;

    //    for( int i =0 ; i < c.size() -1 ; i++)
    //    {
    //        drawLine(zVecToAliceVec(c[i]), zVecToAliceVec(c[i+1] ));
    //    }

    //    glPopMatrix();
    //}

   

   

    glPushMatrix();
    glTranslatef(-100, 0, 0);
        if (showF1)fieldA.draw(threshold);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(0, 0, 0);
        if (showF2) fieldB.draw(threshold);

    glPopMatrix();
    
    glPushMatrix();
    glTranslatef(100, 0, 0);
        if (showF3)fieldC.draw(threshold);

    glPopMatrix();
}


// -----------------------------------------------------------------------------
// Interaction (No-op)
// -----------------------------------------------------------------------------

void keyPress(unsigned char k, int xm, int ym) 
{

    if (k == 'r') run = !run;

    if (k == 'b')
    {
        fieldA.field.blendWith(fieldC.field,10);
        fieldA.field.blendWith(fieldB.field,10);

       // fieldA.normalise();
    }

    if (k == '-')
    {
        cout << k << endl;
       

        fieldA.field.subtract(fieldC.field);
        //fieldA.field.normalise();
        //fieldA.normalise();
        //fieldA.rescaleFieldToRange(-1, 1);
    }

    if (k == '1')showF1 = !showF1;
    if (k == '2')showF2 = !showF2;
    if (k == '3')showF3 = !showF3;


    if (k == 'w')
    {
        fieldA.field.printField();
    }
}
void mousePress(int b, int s, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif
