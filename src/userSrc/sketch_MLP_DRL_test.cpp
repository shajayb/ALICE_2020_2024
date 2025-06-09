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
            if (isInsidePolygon(pt, polygon))
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

float blendOrientedBoxSDFs(zVector pt, std::vector<zVector>& centers, std::vector<float>& angles, float width = 8.0f, float height = 4.0f, float k = 3.0f)
{
    float d = 1e6;
    for (int i = 0; i < centers.size(); i++)
    {
        float dist = orientedBoxSDF(pt, centers[i], width, height, angles[i]);
        d = smin(d, dist, k);
    }
    return d;
}


//------------------------------------------------------------------ MLP


class PolygonSDF_MLP : public MLP
{
public:
    using MLP::MLP;

    std::vector<zVector> polygon;
    std::vector<zVector> trainingSamples;
    std::vector<float> sdfGT;

    std::vector<zVector> fittedCenters;
    std::vector<float> fittedRadii;

    int number_sdf;
    double radius = 60;
    float smoothK = 3.0f;

    ScalarField2D generatedField;

    void decodeOutput(const std::vector<float>& out, std::vector<zVector>& centers, std::vector<float>& radii)
    {
        centers.resize(number_sdf);
        radii.resize(number_sdf);

        for (int i = 0; i < number_sdf; i++)
        {
            centers[i] = zVector(out[i * 3 + 0], out[i * 3 + 1], 0);
            radii[i] = radius * DEG_TO_RAD;
        }
    }
    // method / action
    float computeLoss(std::vector<float>& x, std::vector<float>& dummy) override
    {
        auto out = forward(x);
        std::vector<zVector> centers(number_sdf);
        std::vector<float> radii(number_sdf);

        decodeOutput(out, centers, radii);

        float loss = 0.0f;
        for (int i = 0; i < trainingSamples.size(); i++)
        {
            float pred = blendOrientedBoxSDFs(trainingSamples[i], centers, radii);// blendCircleSDFs(trainingSamples[i], centers, radii, smoothK);
            float err = pred - sdfGT[i];
            loss += err * err;
        }

        return loss / trainingSamples.size();
    }

    void computeGradient(std::vector<float>& x, std::vector<float>& dummy, std::vector<float>& gradOut) override
    {
        auto out = forward(x);
        std::vector<zVector> centers(number_sdf);
        std::vector<float> radii(number_sdf);

        decodeOutput(out, centers, radii);

        fittedCenters = centers;
        fittedRadii = radii;

        float eps = 0.01f;
        gradOut.assign(out.size(), 0.0f);

        for (int s = 0; s < trainingSamples.size(); ++s)
        {
            zVector pt = trainingSamples[s];
            float gt = sdfGT[s];
            float pred = blendOrientedBoxSDFs(trainingSamples[s], centers, radii);// blendCircleSDFs(pt, centers, radii, smoothK);
            float err = pred - gt;

            for (int i = 0; i < number_sdf; i++)
            {
                std::vector<zVector> cx = centers, cy = centers;
                cx[i].x += eps;
                cy[i].y += eps;
                float gx = (blendOrientedBoxSDFs(pt, cx, radii) - pred) / eps;
                float gy = (blendOrientedBoxSDFs(pt, cy, radii) - pred) / eps;
                gradOut[i * 3 + 0] += 2 * err * gx;
                gradOut[i * 3 + 1] += 2 * err * gy;
            }
        }

        for (float& g : gradOut) g /= trainingSamples.size();
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


        for (int i = 0; i < generatedField.RES; i++)
        {
            for (int j = 0; j < generatedField.RES; j++)
            {

                zVector pt = generatedField.gridPoints[i][j];
                generatedField.field[i][j] = blendOrientedBoxSDFs(pt, centers, radii);;// blendCircleSDFs(pt, centers, radii, smoothK);
            }
        }

        generatedField.rescaleFieldToRange(-1, 1);


    }

    void visualiseField()
    {
        generatedField.drawFieldPoints();
        generatedField.drawIsocontours(0.01f);
    }
};

//----------------------- Unit test for generic MLP



//------------------------------------------------------------------ MVC test for subClassMLP
std::vector<zVector> polygon;
std::vector<zVector> trainingSamples;
std::vector<float> sdfGT;


#define NUM_CIRCLES 16

PolygonSDF_MLP mlp;
std::vector<float> grads;
std::vector<float> mlp_input_data;


// function
void initializeMLP()
{
    int input_dim = NUM_CIRCLES * 3;
    int output_dim = NUM_CIRCLES * 3;
    std::vector<int> hidden_dims = { 48,48 };

    mlp = PolygonSDF_MLP(input_dim, hidden_dims, output_dim); // assumes MLP constructor initializes weights/biases
    mlp_input_data.assign(input_dim, 1.0f); // or use 0.0f for strict zero-input
    mlp.number_sdf = NUM_CIRCLES;

}


void setup()
{


    initializeMLP();  // create MLP

    // load boudnary polygon from a text file;
    loadPolygonFromCSV("data/polygon.txt", polygon);

    //calculate training set
    samplePoints(trainingSamples, sdfGT, polygon);
    mlp.trainingSamples = trainingSamples;
    mlp.sdfGT = sdfGT;
}

void update(int value)
{
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


    glColor3f(0, 0, 0);
    for (int i = 0; i < polygon.size(); i++)
    {
        int j = (i + 1) % polygon.size();
        drawLine(zVecToAliceVec(polygon[i]), zVecToAliceVec(polygon[j]));
    }

    mlp.visualize(zVector(50, 450, 0), 400, 300);

    mlp.visualiseField();
}

void keyPress(unsigned char k, int xm, int ym)
{
    if (k == 't')
    {
        grads.clear();
        std::vector<float> dummy;  // unused placeholder
        float loss = mlp.computeLoss(mlp_input_data, dummy);
        mlp.computeGradient(mlp_input_data, dummy, grads);

        mlp.backward(grads, 0.1f);

        std::cout << "Loss: " << loss << std::endl;


    }


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
