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

bool isPointInsideBB( zVector& pt,  zVector& minBB,  zVector& maxBB)
{
    return (pt.x >= minBB.x && pt.x <= maxBB.x) &&
        (pt.y >= minBB.y && pt.y <= maxBB.y) &&
        (pt.z >= minBB.z && pt.z <= maxBB.z);
}


void setup()
{
    S.addSlider(&isoThresh, "threshold");// make a slider control for the variable called width;
    S.sliders[0].minVal = -1;
    S.sliders[0].maxVal = 1;

    S.addSlider(&smoothness, "smooth");// make a slider control for the variable called width;
    S.sliders[1].maxVal = 100;

 

}

void update(int value)
{
    //keyPress('e', 0, 0);
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

       /* rbfCenters.push_back(zVector(10, 0, 0));
        rbfCenters.push_back(zVector(0, 0, 0));
        rbfCenters.push_back(zVector(-10, 0, 0));

        rbfCenters.push_back(zVector(10, 10, 0));
        rbfCenters.push_back(zVector(0, 10, 0));
        rbfCenters.push_back(zVector(-10, 10, 0));*/
        rbfCenters.clear();
        float span = 25; // from -50 to +50
        float step = span / (5 - 1); // spacing between grid points

        for (int i = -25; i < 25; i+=4)
        {
            for (int j = -25; j < 25; j += 4)
            {
                float x = -12 + i * step;
                float y = -12 + j * step;

               rbfCenters.push_back( zVector(i, j, 0) );
               // rbfCenters.push_back( zVector(ofRandom(-5, 5), ofRandom(-5, 5), 0) );
            }
        }


        field1.addVoronoi(rbfCenters);
      //  field1.addOrientedBoxSDF(zVector(-5, -5, 0), zVector(12,6, 0), angleRadians);

       /*field2.addCircleSDF({ zVector(-15,0,0) } , 25);
       field2.addCircleSDF({ zVector(15,0,0) }, 25);
       field2.addCircleSDF({ zVector(0,15,0) }, 25);
       field2.addCircleSDF({ zVector(0,-15,0) },25);*/
       // field2.addCircleSDFs(rbfCenters, 2);

      // field1.addOrientedBoxSDF(zVector(-0, 0, 0), zVector(12, 12, 0), 0 * 3.1415926f / 180.0f);
      // field2.addOrientedBoxSDF(zVector(-0, 0, 0), zVector(20, 10, 0), 30 * 3.1415926f / 180.0f);

       //
       
       
       // 
       //field1.normalise();
       //field2.normalise();
      // field2.clearField();
       field1.rescaleFieldToRange(-1.0f, 1.0f);
       field2.rescaleFieldToRange(-1.0f, 1.0f);

       

      
       initialized = true;
    }

    if (showField1)
    {
        glColor3f(0.0, 0.0, 1.0);  // blue
        field1.drawFieldPoints();
        field1.drawIsocontours(isoThresh);
       // field1.getOrderedContours();
    }

    if (showField2)
    {
        glColor3f(1.0, 0.0, 0.0);  // red
        field2.drawFieldPoints();
        field2.drawIsocontours(isoThresh);
    }


   /* for (auto& c : field1.allContours)
    {
        if (c.empty())continue;
        
        for (int i = 0; i < c.size() -1; i++)
        {
            drawLine(zVecToAliceVec(c[i]), zVecToAliceVec(c[i + 1]));
        }
    }*/

}

void keyPress(unsigned char k, int xm, int ym)
{

    if (k == 's')
    {
        //field1.getOrderedContours();
        for (auto& c : field1.allContours)
        {
            if (c.empty())continue;
            field1.smoothContour(c, 15);
        }
    }
     
    if (k == 'b')
    {
        field1.blendWith(field2, smoothness, SMinMode::CIRCULAR_GEOMETRIC);
        field1.normalise();
    }

    if (k == '=')
    {
       // field1.blendWith(field2, smoothness);  // adjust k to control smoothness
        field1.unionWith(field2);
        field1.normalise();
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
        field1.normalise();
    }



    if (k == 'e')
    {
       

        // reset forces
        int num = rbfCenters.size();
        if (num < 3)return;

        vector<zVector>forces;
        forces.reserve(num);

        for (int i = 0; i < num; i++)forces[i] = zVector(0, 0, 0);

        //calculate & store repulsive force per point
        for (int i = 0; i < num; i++)
        {
            for (int j = 0; j < num; j++)
            {
                if (i == j) continue;

                zVector e = rbfCenters[j] - rbfCenters[i];
                float d = rbfCenters[j].distanceTo(rbfCenters[i]);

                if (d > 1e-2)
                {
                    e.normalize();
                    e /= d * d;
                    forces[i] -= e;
                }

            }
        }

        // calculate the maximum and minimum magnitude of reuplisve force
        double force_max, force_min;
        force_min = 1e6; force_max = -force_min;

        for (int i = 0; i < num; i++)
        {
            float d = forces[i].length();
            force_max = MAX(force_max, d);
            force_min = MIN(force_min, d);
        }

        // re-scale all forces to be within 0 & 1
        for (int i = 0; i < num; i++)
        {
            float d = forces[i].length();
            forces[i].normalize();
            forces[i] *= ofMap(d, force_min, force_max, 0, 1);

        }

        // move each of the points, by applying their respective forces, if the magnitude of force is less than 1 and the point is whtin a radius of 10 from the origin;
        for (int i = 0; i < num; i++)
            //if (forces[i].length() < 100 /*&& rbfCenters[i].length() < 25*/)
            if(isPointInsideBB(rbfCenters[i], zVector(-50,-50,0), zVector(50, 50, 0)))
            rbfCenters[i] += forces[i] * 0.1;


        field1.clearField();
        field1.addVoronoi(rbfCenters);

        field2.clearField();
        field2.addCircleSDFs(rbfCenters, 2);
       
    }

    if( k == 'i') initialized = false;

   
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
