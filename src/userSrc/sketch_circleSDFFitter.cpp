#define _MAIN_
#ifdef _MAIN_

#include "main.h"

// zSpace Core
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>
using namespace zSpace;

#include <fstream>
#include <sstream>

Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

zVector AliceVecToZvec(Alice::vec& in)
{
    return zVector(in.x, in.y, in.z);
}


#include "scalarField.h"


//inline float smin(float a, float b, float k)
//{
//    float h = std::max(k - fabs(a - b), 0.0f) / k;
//    return std::min(a, b) - h * h * k * 0.25f;
//}
std::vector<zVector> polygon;
std::vector<zVector> sdfCenters;
ScalarField2D myField;

int numCircles = 12;
double thresholdValue = 0.0;
double radius = 1.0;
double smoothK = 3.0;


//-------------------------------
// Utility
//-------------------------------
void loadPolygonFromCSV(const std::string& filename)
{
    polygon.clear();
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string xStr, yStr;
        if (std::getline(ss, xStr, ',') && std::getline(ss, yStr))
        {
            float x = std::stof(xStr);
            float y = std::stof(yStr);
            polygon.emplace_back(x, y, 0);
        }
    }

    cout << polygon.size() << " polygon size" << endl;
}

//-------------------------------
// Circle SDF Blending
//-------------------------------
inline float circleSDF( zVector& pt, zVector& center, float r)
{
    return pt.distanceTo(zVector(center)) - r; // signed: negative inside, 0 on boundary, positive outside
}

inline float blendCircleSDFs( zVector& pt,  std::vector<zVector>& centers, float r, float k)
{
    if (centers.empty()) return 1e6f;

    float d = circleSDF(pt, centers[0], r);
    for (int i = 1; i < centers.size(); i++)
    {
        float d_i = circleSDF(pt, centers[i], r);
        d = smin(d, d_i, k);  // smooth union of signed distances
    }

    return d;
}

//bool isInsidePolygon( zVector& p,  std::vector<zVector>& poly)
//{
//    int crossings = 0;
//    for (int i = 0; i < poly.size(); i++)
//    {
//         zVector& a = poly[i];
//         zVector& b = poly[(i + 1) % poly.size()];
//        if ((a.y > p.y) != (b.y > p.y))
//        {
//            float atX = (b.x - a.x) * (p.y - a.y) / (b.y - a.y + 1e-6f) + a.x;
//            if (p.x < atX) crossings++;
//        }
//    }
//    return (crossings % 2 == 1);
//}

bool isInsidePolygon( zVector& p,  std::vector<zVector>& poly)
{
    int windingNumber = 0;

    for (int i = 0; i < poly.size(); i++)
    {
         zVector& a = poly[i];
         zVector& b = poly[(i + 1) % poly.size()];

        if (a.y <= p.y)
        {
            if (b.y > p.y && ((b - a) ^ (p - a)).z > 0)
                ++windingNumber;
        }
        else
        {
            if (b.y <= p.y && ((b - a) ^ (p - a)).z < 0)
                --windingNumber;
        }
    }

    return (windingNumber != 0);
}



float polygonSDF( zVector& pt,  std::vector<zVector>& poly)
{
    float minDist = 1e6f;
    for (int i = 0; i < poly.size(); i++)
    {
        zVector a = poly[i];
        zVector b = poly[(i + 1) % poly.size()];
        zVector ab = b - a;
        zVector ap = pt - a;

        float t = std::clamp((ap * ab) / (ab * ab), 0.0f, 1.0f);
        zVector closest = a + ab * t;
        float dist = pt.distanceTo(closest);
        minDist = std::min(minDist, dist);
    }

    bool inside = isInsidePolygon(pt, poly);
    return inside ? -minDist : minDist;
}


//-------------------------------
// Fit & Field
//-------------------------------

std::vector<zVector> candidatePts;
void fitSDFToPolygon()
{
    sdfCenters.clear();
    candidatePts.clear();
    

    // Sample a grid of points across bounding box
    for (float x = -50; x <= 50; x += 2.0f)
    {
        for (float y = -50; y <= 50; y += 2.0f)
        {
            zVector pt(x, y, 0);
            if (isInsidePolygon(pt, polygon))
            {
                candidatePts.push_back(pt);
            }
        }
    }

    cout << candidatePts.size() << " candidatePts size" << endl;

    for (int c = 0; c < numCircles; c++)
    {
        float maxResidual = -1e6;
        zVector bestPt;

     /*   for ( auto& pt : candidatePts)
        {
            float sdfVal = blendCircleSDFs(pt, sdfCenters, radius, smoothK);
            float absVal = fabs(sdfVal);
            if (absVal > maxResidual)
            {
                maxResidual = absVal;
                bestPt = pt;
            }
        }*/
        for (auto& pt : candidatePts)
        {
            float pred = blendCircleSDFs(pt, sdfCenters, radius, smoothK);
            float actual = polygonSDF(pt, polygon);

            float residual = fabs(pred - actual);

            if (residual > maxResidual)
            {
                maxResidual = residual;
                bestPt = pt;
            }
        }


        sdfCenters.push_back(bestPt);
    }

    printf(" NSDFC %i, NC %i \n", sdfCenters.size(), numCircles);
}

void buildScalarField()
{
    for (int i = 0; i < ScalarField2D::RES; i++)
    {
        for (int j = 0; j < ScalarField2D::RES; j++)
        {
            zVector pt = myField.gridPoints[i][j];
            float d = blendCircleSDFs(pt, sdfCenters, radius, smoothK);
            myField.field[i][j] = d; // signed SDF directly
        }
    }

    myField.rescaleFieldToRange(-1, 1);
}

std::vector<zVector> trainingSamples;

void samplePoints()
{
    trainingSamples.clear();
    for (float x = -50; x <= 50; x += 2.0f)
    {
        for (float y = -50; y <= 50; y += 2.0f)
        {
            zVector pt(x, y, 0);
            if (isInsidePolygon(pt, polygon)) trainingSamples.push_back(pt);
        }
    }

    cout << " training samples " << trainingSamples.size() << endl;
}

float computeTotalError()
{
    float err = 0.0f;
    for (auto& pt : trainingSamples)
    {
        float pred = blendCircleSDFs(pt, sdfCenters, radius, smoothK);
        float actual = polygonSDF(pt, polygon);
        float diff = pred - actual;
        err += diff * diff;
    }
    return err;
}

void optimiseCircleCenters(int iterations = 20, float step = 0.001f)
{
    const float eps = 1e-3f;

   //for (int it = 0; it < iterations; it++)
    {
       for (int c = 0; c < sdfCenters.size(); c++)
       {
           zVector center = sdfCenters[c];
           zVector grad(0, 0, 0);

           for (int d = 0; d < 2; d++)
           {
               zVector dir(0, 0, 0);
               if (d == 0) dir.x = eps;
               if (d == 1) dir.y = eps;

               std::vector<zVector> testCenters = sdfCenters;
               testCenters[c] = center + dir;
               float E_plus = 0;
               for (auto& pt : trainingSamples)
               {
                   float pred = blendCircleSDFs(pt, testCenters, radius, smoothK);
                   float actual = polygonSDF(pt, polygon);
                   
                   float diff = pred - actual;
                   E_plus += diff * diff;
               }

               testCenters[c] = center - dir;
               float E_minus = 0;
               for (auto& pt : trainingSamples)
               {
                   float pred = blendCircleSDFs(pt, testCenters, radius, smoothK);
                   float actual = polygonSDF(pt, polygon);
                   float diff = pred - actual;
                   E_minus += diff * diff;
               }

               float g = (E_plus - E_minus) / (2 * eps);
               if (d == 0) grad.x = g;
               if (d == 1) grad.y = g;
           }

           // Update
          
           sdfCenters[c] = center - grad * step;
            sdfCenters[c].x = std::clamp(sdfCenters[c].x, -50.0f, 50.0f);
           sdfCenters[c].y = std::clamp(sdfCenters[c].y, -50.0f, 50.0f);

       }


        std::cout << "Iteration " << " error: " << computeTotalError() << std::endl;
    }
}



//-------------------------------
// Visualisation
//-------------------------------
void drawPolygon()
{
    glColor3f(0, 0, 0);
    for (int i = 0; i < polygon.size(); i++)
    {
        int j = (i + 1) % polygon.size();
        drawLine(zVecToAliceVec(polygon[i]), zVecToAliceVec(polygon[j]));
    }
}

void drawCircles()
{
    glColor3f(0, 0, 1);
    for (auto& c : sdfCenters)
    {
        drawCircle(zVecToAliceVec(c), .5/*radius*/, 32);
    }
}

//-------------------------------
// MVC
//-------------------------------
void setup()
{
    loadPolygonFromCSV("data/polygon.txt");
    fitSDFToPolygon();       // initial greedy placement
    samplePoints();          // prepare training set
    optimiseCircleCenters(); // run gradient descent
    buildScalarField();      // update field


    S.addSlider(&thresholdValue, "iso");
    S.sliders[0].maxVal = 1;
    S.sliders[0].minVal = -1.0;

    S.addSlider(&radius, "r");
    S.sliders[1].maxVal = 20;

    S.addSlider(&smoothK, "k");
    S.sliders[2].maxVal = 10;
}

void update(int value)
{
    buildScalarField();
}

void draw()
{
    backGround(0.9);
    drawGrid(50);

    drawPolygon();
    drawCircles();

    glPointSize(5);
    for (auto& pt : candidatePts)drawPoint(zVecToAliceVec(pt));
    glPointSize(1);
  
    myField.drawFieldPoints();
    
    myField.drawIsocontours(thresholdValue, true);
   
}

void keyPress(unsigned char k, int xm, int ym)
{
    if (k == 'r')
    {
        /*loadPolygonFromCSV("data/polygon.csv");
        fitSDFToPolygon();
        buildScalarField();*/

        optimiseCircleCenters();
    }
    if (k == 'e')
    {
        myField.exportOrderedContoursAsCSV("data/contours.csv");
    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_
