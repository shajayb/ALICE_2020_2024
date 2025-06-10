#define _MAIN_
#ifdef _MAIN_

#include "main.h"

#include <vector>
#include <cmath>
#include <fstream>
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;

//------------------------------------------------------------------ Utility
Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

zVector AliceVecToZvec(Alice::vec& in)
{
    return zVector(in.x, in.y, in.z);
}

#include "scalarField.h" //// two functions must be turned on in scalarfIELD.H for sketch_circleSDF_fitter.cpp
#include "genericMLP.h" 

/// --------- sub class

//------------------------------------------------------------------ Utility

bool isInsidePolygon(zVector& p, std::vector<zVector>& poly)
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
float polygonSDF(zVector& p, std::vector<zVector>& poly)
{
    float minDist = 1e6;
    int n = poly.size();

    for (int i = 0; i < n; i++)
    {
        zVector a = poly[i];
        zVector b = poly[(i + 1) % n];

        zVector ab = b - a;
        zVector ap = p - a;

        float t = std::max(0.0f, std::min(1.0f, (ab * ap) / (ab * ab)));
        zVector proj = a + ab * t;
        float d = p.distanceTo(proj);
        minDist = std::min(minDist, d);
    }

    return minDist * (isInsidePolygon(p, poly) ? -1.0f : 1.0f);
}

void loadPolygonFromCSV(const std::string& filename, vector<zVector>& polygon)
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

void samplePoints(std::vector<zVector>& trainingSamples, std::vector<float>& sdfGT, vector<zVector>& polygon)
{
    // collect input-output pairs of information

    trainingSamples.clear();
    sdfGT.clear();

    for (float x = -50; x <= 50; x += 5.0f)
    {
        for (float y = -50; y <= 50; y += 5.0f)
        {
            zVector pt(x, y, 0);
            if (  isInsidePolygon(pt, polygon) )
            {
                trainingSamples.push_back(pt); // input exmaples
                sdfGT.push_back(polygonSDF(pt, polygon)); // known output expected for the input
            }
        }
    }

    std::cout << "Training samples: " << trainingSamples.size() << std::endl;
}

float blendCircleSDFs(zVector pt, std::vector<zVector>& centers, std::vector<float>& radii, float k)
{
    float d = 1e6;
    for (int i = 0; i < centers.size(); i++)
    {
        float dist = pt.distanceTo(centers[i]) - radii[i];
        d = smin(d, dist, k);
    }
    return d;
}

float orientedBoxSDF(zVector pt, zVector center, float width, float height, float angleRad)
{
    zVector d = pt - center;

    float cosA = cos(angleRad);
    float sinA = sin(angleRad);

    float localX = d.x * cosA + d.y * sinA;
    float localY = -d.x * sinA + d.y * cosA;

    float dx = fabs(localX) - width * 0.5f;
    float dy = fabs(localY) - height * 0.5f;

    float ax = std::max(dx, 0.0f);
    float ay = std::max(dy, 0.0f);

    float insideDist = std::min(std::max(dx, dy), 0.0f);
    return sqrtf(ax * ax + ay * ay) + insideDist;
}

float blendOrientedBoxSDFs(zVector pt, std::vector<zVector>& centers, std::vector<float>& angles, float width = 6.0f, float height = 12.0f, float k = 3.0f)
{
    float d = 1e6;
    for (int i = 0; i < centers.size(); i++)
    {
        float dist = orientedBoxSDF(pt, centers[i], width, height, angles[i]);
        d = min(d, dist);;// smin(d, dist, k);
    }
    return d;
}

zVector gradientAt(zVector pt, std::vector<zVector>& centers, std::vector<float>& angles, float h = 0.1f)
{
    float dx = blendOrientedBoxSDFs(pt + zVector(h, 0, 0), centers, angles) -
        blendOrientedBoxSDFs(pt - zVector(h, 0, 0), centers, angles);

    float dy = blendOrientedBoxSDFs(pt + zVector(0, h, 0), centers, angles) -
        blendOrientedBoxSDFs(pt - zVector(0, h, 0), centers, angles);

    zVector ret(dx, dy, 0);
    ret.normalize();
    return ret;
}

zVector gradientAt_polygonSDF(zVector pt, vector<zVector>&polygon, float h = 0.1f)
{
    float dx = polygonSDF(pt + zVector(h, 0, 0),polygon) -
        polygonSDF(pt - zVector(h, 0, 0), polygon);

    float dy = polygonSDF(pt + zVector(0, h, 0), polygon) -
        polygonSDF(pt - zVector(0, h, 0), polygon);

    zVector ret(dx, dy, 0);
    ret.normalize();
    return ret;
}


//------------------------------------------------------------------ MLP


class PolygonSDF_MLP : public MLP
{
public:
    using MLP::MLP;

    std::vector<zVector> polygon;
    std::vector<zVector> trainingSamples;
    std::vector<float> sdfGT;
    std::vector<float> losses;
    std::vector<float> losses_ang;

    std::vector<zVector> fittedCenters;
    std::vector<float> fittedRadii;

    int number_sdf;
    double radius = 8.;
    float smoothK = 3.0f;
    zVector sunDir = zVector(1, 1, 0);

    ScalarField2D generatedField;
    int epoch = 0;

    void decodeOutput(const std::vector<float>& out, std::vector<zVector>& centers, std::vector<float>& radii)
    {
        centers.resize(number_sdf);
        radii.resize(number_sdf);

        for (int i = 0; i < number_sdf; i++)
        {

            centers[i] = zVector(out[i * 3 + 0], out[i * 3 + 1],0);

            zVector grad_polygon = gradientAt_polygonSDF(centers[i], polygon);
            grad_polygon.normalize();
            float angle =  angleBetween(grad_polygon, zVector(1,0,0)) ;


            radii[i] = (out[i * 3 + 2] ) * 0.1 ;// radius * DEG_TO_RAD
        }
    }
    float angleBetween(zVector &a, zVector &b)
    {
        float dot = a.x * b.x + a.y * b.y;
        float det = a.x * b.y - a.y * b.x;
        return atan2(det, dot); // angle in radians
    }
    // method / action
    float evaluateLoss(std::vector<zVector>& centers, std::vector<float>& angles)
    {
        const int N = trainingSamples.size();
        const int numLossTypes = 2; // 0: coverage, 1: angular (add more as needed)

        std::vector< std::vector<float> > lossesByType(numLossTypes, std::vector<float>(N, 0.0f));

        zVector sunDir(-1, 0, 0);
        sunDir.normalize();

        // Step 1: compute raw losses
        for (int i = 0; i < N; i++)
        {
            zVector pt = trainingSamples[i];

            // Loss 0: coverage (MSE)
            float pred = blendCircleSDFs(pt, centers, angles, smoothK);
            float err = pred - sdfGT[i];
            lossesByType[0][i] = err * err;

            // Loss 1: angular alignment (squared angle)
            zVector grad = gradientAt(pt, centers, angles); // gradient of blendedSDF
            zVector grad_polygon = gradientAt_polygonSDF(pt, polygon); // gradient of polygonSDF;
            grad.normalize();
            grad = grad ^ zVector(0, 0, 1);
            grad_polygon.normalize();

            float angleErr = angleBetween(grad, grad_polygon) ;
            lossesByType[1][i] = angleErr * angleErr;
        }

        // Step 2: normalize each loss type to [0,1]
        std::vector<bool> normalizeLoss = { false, true }; // match number of loss types

        for (int t = 0; t < numLossTypes; t++)
        {
            if (!normalizeLoss[t]) continue;

            float minVal = 1e6f, maxVal = -1e6f;
            for (float v : lossesByType[t])
            {
                minVal = std::min(minVal, v);
                maxVal = std::max(maxVal, v);
            }

            float range = std::max(maxVal - minVal, 1e-6f);
            for (float& v : lossesByType[t])
            {
                v = (v - minVal) / range;
            }
        }


        // Step 3: weighted sum of all loss types
        std::vector<float> weights = { 1.f,15 }; // must match numLossTypes
        float totalLoss = 0.0f;
        for (int i = 0; i < N; i++)
        {
            float combined = 0.0f;
            for (int t = 0; t < numLossTypes; t++)
            {
                combined += weights[t] * lossesByType[t][i];
            }
            totalLoss += combined;
        }

        // Optional debug access: you may assign lossesByType[0] → `losses`, lossesByType[1] → `losses_ang`
        losses = lossesByType[0];
        losses_ang = lossesByType[1];

        return totalLoss / trainingSamples.size();
    }


    float computeLoss(std::vector<float>& x, std::vector<float>& dummy) override
    {
        auto out = forward(x);
        std::vector<zVector> centers;
        std::vector<float> angles;
        decodeOutput(out, centers, angles);

        epoch++;
        return evaluateLoss(centers, angles);
    }

    void computeGradient(std::vector<float>& x, std::vector<float>& dummy, std::vector<float>& gradOut) override
    {
        auto out = forward(x);
        float eps = 0.01f;

        std::vector<zVector> baseCenters;
        std::vector<float> baseAngles;
        decodeOutput(out, baseCenters, baseAngles);

        float baseLoss = evaluateLoss(baseCenters, baseAngles);

        gradOut.assign(out.size(), 0.0f);

        for (int i = 0; i < out.size(); ++i)
        {
            std::vector<float> outPerturbed = out;
            outPerturbed[i] += eps;

            std::vector<zVector> centers;
            std::vector<float> angles;
            decodeOutput(outPerturbed, centers, angles);

            float lossPerturbed = evaluateLoss(centers, angles);
            gradOut[i] = (lossPerturbed - baseLoss) / eps;
        }
    }


    ///

    void GenerateField(std::vector<float>& x)
    {
        auto out = forward(x);

        std::vector<zVector> centers(number_sdf);
        std::vector<float> radii(number_sdf);

        decodeOutput(out,centers, radii);

        GenerateField(centers, radii);

        
    }

    void GenerateField(std::vector<zVector>& centers, std::vector<float> &radii)
    {

       // for (auto& r : radii)r = radius;

        generatedField.clearField();
       // generatedField.addVoronoi(trainingSamples);

        for (int i = 0; i < generatedField.RES; i++)
        {
            for (int j = 0; j < generatedField.RES; j++)
            {

                zVector pt = generatedField.gridPoints[i][j];
                
              //  float d_v = generatedField.field[i][j];
                float d_c =  blendOrientedBoxSDFs(pt, centers, radii);;// blendCircleSDFs(pt, centers, radii, smoothK);//
               // float d_p = polygonSDF(pt, polygon);


                generatedField.field[i][j] = d_c;// min(-d_v, d_c);// min(min(-d_v, d_c), -d_p);
            }
        }

        generatedField.rescaleFieldToRange(-1, 1);


    }

    void drawLossText(float startY = 150)
    {
        setup2d();
        
        glColor3f(0, 0, 0);
        char s[100];

        float lossSum = 0;
        float loss_A_Sum = 0;

        for (int i = 0; i < losses_ang.size(); i++)
        {
            lossSum += losses[i];
            loss_A_Sum += losses_ang[i];  
        }


        sprintf(s, " loss %1.2f", lossSum / trainingSamples.size());
        drawText(string(s), 50, startY);

        sprintf(s, " loss_ang %1.2f", loss_A_Sum);
        drawText(string(s), 50, startY + 15);

        restore3d();
    }
    
    void drawLossBarGraph(const std::vector<float>& losses, float startPtX, float startPtY, float screenWidth = 800, float barHeight = 50)
    {
        if (losses.empty()) return;

        setup2d(); // Switch to orthographic 2D mode

            int N = losses.size();
            float barSpacing = screenWidth / (float)N;

            // --- Normalize losses to [0, 1]
            float minVal = 1e6f, maxVal = -1e6f;
            for (float v : losses)
            {
                minVal = std::min(minVal, v);
                maxVal = std::max(maxVal, v);
            }
            float range = std::max(maxVal - minVal, 1e-6f);  // avoid divide by zero

            float loss_A_Sum = 0;
            float lossSum = 0;

            for (int i = 0; i < N; i++)
            {
                float normalized = (losses[i] - minVal) / range;
                float x = startPtX + i * barSpacing;
                float h = barHeight * normalized;

                float r, g, b;
                getJetColor(normalized, r, g, b);

                glColor3f(r, g, b);
                drawLine(Alice::vec(x, startPtY, 0), Alice::vec(x, startPtY + h, 0));

                lossSum += losses[i];
                loss_A_Sum += losses_ang[i];  // unnormalized angular loss sum
            }
        
            


       

        restore3d(); // Restore to 3D mode
    }

    void drawText(string &str , float x = 50, float y=100)
    {
        unsigned int i;
        glRasterPos2f(x, y);

        
        for (i = 0; i < str.length(); i++)
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, str[i]);
    }

    void visualiseField( float threshold = 0.01)
    {
        generatedField.drawFieldPoints();
        generatedField.drawIsocontours(threshold);

  
        for (int i = 0; i < trainingSamples.size(); i++)
        {
            //glColor3f(0, 1, 0);
            Alice::vec a, b;
            a = zVecToAliceVec(trainingSamples[i]);
            //b = zVecToAliceVec(zVector(0,0, losses[i]));
            ////drawLine(a, a + b * 0.2);

            zVector grad_polygon = gradientAt_polygonSDF(trainingSamples[i], polygon);
            grad_polygon.normalize();

            glColor3f(0, 0, 0);
            drawLine(a, a + zVecToAliceVec(grad_polygon));

            glColor3f(0, 0, 0);
            drawLine(a, a + zVecToAliceVec(grad_polygon));

        }

        
    }

    void visualiseGradients( vector<float> &x)
    {
       
       
        auto out = forward(x);

        std::vector<zVector> centers(number_sdf);
        std::vector<float> radii(number_sdf);

        decodeOutput(out, centers, radii);


        for (int i = 0; i < number_sdf; i++)
        {
            zVector grad_polygon = gradientAt_polygonSDF(centers[i], polygon);
            grad_polygon.normalize();

            Alice::vec a = zVecToAliceVec(centers[i]);

            glColor3f(0, 0, 0);
            drawLine(a, a + zVecToAliceVec(grad_polygon)*3);

            ///

            float cosA = cos(radii[i]);
            float sinA = sin(radii[i]);

             zVector axisX(cosA, -sinA, 0); // local X direction
             zVector axisY(sinA, cosA, 0); // local Y direction

            zVector grad = axisY;// gradientAt(centers[i], centers, radii);
            grad.normalize();

            glColor3f(1, 0, 0);
            drawLine(a, a + zVecToAliceVec(grad) * 4);

            
        }


        for (int i = 0; i < trainingSamples.size(); i++)
        {
            //glColor3f(0, 1, 0);
            Alice::vec a, b;
            a = zVecToAliceVec(trainingSamples[i]);
            //b = zVecToAliceVec(zVector(0,0, losses[i]));
            ////drawLine(a, a + b * 0.2);

            zVector grad_polygon = gradientAt_polygonSDF(trainingSamples[i], polygon);
            grad_polygon.normalize();

            glColor3f(0, 0, 0);
            drawLine(a, a + zVecToAliceVec(grad_polygon));

            zVector grad = gradientAt(trainingSamples[i], centers,radii);
            grad.normalize();

            glColor3f(1, 0, 0);
            drawLine(a, a + zVecToAliceVec(grad));

        }
    }

};




//------------------------------------------------------------------ MVC test for subClassMLP
std::vector<zVector> polygon;
std::vector<zVector> trainingSamples;
std::vector<float> sdfGT;


#define NUM_CENTERS 15

PolygonSDF_MLP mlp;
std::vector<float> grads;
std::vector<float> mlp_input_data;

double lr = 0.0001;
double tv = -0.005;
// function
void initializeMLP()
{
    int input_dim = NUM_CENTERS * 3;
    int output_dim = NUM_CENTERS * 3;
    std::vector<int> hidden_dims = { 16 };

    mlp = PolygonSDF_MLP(input_dim, hidden_dims, output_dim); // assumes MLP constructor initializes weights/biases
    mlp_input_data.assign(input_dim, 1.0f); // or use 0.0f for strict zero-input
    mlp.number_sdf = NUM_CENTERS;

}


void setup()
{


    initializeMLP();  // create MLP

    // load boudnary polygon from a text file;
    loadPolygonFromCSV("data/polygon.txt", polygon);

    //calculate training set
    samplePoints(trainingSamples, sdfGT, polygon);
    mlp.polygon = polygon;
    mlp.trainingSamples = trainingSamples;
    mlp.sdfGT = sdfGT;
    mlp.losses.resize(sdfGT.size());
    mlp.losses_ang.resize(sdfGT.size());

    lr = 0.01;
    S.numSliders = 0;
    S.addSlider(&lr, "LR");
    S.sliders[0].minVal = lr;
    S.sliders[0].maxVal = lr*3;

    //

    S.addSlider(&tv, "TV");
    S.sliders[1].minVal = -1;
    S.sliders[1].maxVal = 1;



}


bool run = false;
void update(int value)
{
    if (run)
    {
        keyPress('t', 0, 0);
    }
}

void draw()
{
    backGround(1);
    drawGrid(50);

    glColor3f(1, 0, 0);
    for (auto& c : mlp.fittedCenters)
    {
        drawCircle(zVecToAliceVec(c), 3, 32);
    }

    glColor3f(0, 0, 1);
    for (auto& p : trainingSamples)
    {
        drawPoint(zVecToAliceVec(p));
    }


    // --------------- polygon
    glColor3f(0, 0, 0);
    for (int i = 0; i < polygon.size(); i++)
    {
        int j = (i + 1) % polygon.size();
        drawLine(zVecToAliceVec(polygon[i]), zVecToAliceVec(polygon[j]));
    }


    /// MLP Viz
    mlp.visualize(zVector(50, 450, 0), 200, 250);
    mlp.drawLossBarGraph(mlp.losses, 50, 650, 200, 40);  // bottom-left start, 800px width, 40px bar height
    mlp.drawLossBarGraph(mlp.losses_ang, 50, 725, 200, 40);
    mlp.drawLossText(800);
  

    mlp.visualiseField(tv);

    mlp.visualiseGradients(mlp_input_data);

    

}





void keyPress(unsigned char k, int xm, int ym)
{
    if (k == 't')
    {
        grads.clear();
        std::vector<float> dummy;
        float loss = mlp.computeLoss(mlp_input_data, dummy);
        mlp.computeGradient(mlp_input_data, dummy, grads);

        mlp.backward(grads, lr);

        cout << "loss :" << loss << endl;

        keyPress('u', 0, 0);
        
    }



    if (k == 'r')run = !run;

    if (k == 'u')
    {
        vector<zVector>CENS;
        vector<float>angles;

        CENS.push_back(zVector(0, 10, 0));
        angles.push_back(45 * DEG_TO_RAD);

        CENS.push_back(zVector(10, 10, 0));
        angles.push_back(135 * DEG_TO_RAD);


       // mlp.GenerateField(CENS,angles);
        mlp.GenerateField(mlp_input_data);
    }

}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_
