﻿#define _MAIN_
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

void loadPolygonFromCSV(const std::string& filename, vector<zVector> &polygon)
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

void samplePoints(std::vector<zVector> &trainingSamples, std::vector<float> &sdfGT, vector<zVector>& polygon)
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
    double radius = 10.0f;
    float smoothK = 3.0f;

    // method / action
    float computeLoss(std::vector<float>& x, std::vector<float>& dummy) override
    {
        auto out = forward(x);
        std::vector<zVector> centers(number_sdf);
        std::vector<float> radii(number_sdf);

        for (int i = 0; i < number_sdf; i++)
        {
            centers[i] = zVector(out[i * 2 + 0], out[i * 2 + 1], 0);
            radii[i] = radius;
        }

        float loss = 0.0f;
        for (int i = 0; i < trainingSamples.size(); i++)
        {
            float pred = blendCircleSDFs(trainingSamples[i], centers, radii, smoothK);
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

        for (int i = 0; i < number_sdf; i++)
        {
            centers[i] = zVector(out[i * 2 + 0], out[i * 2 + 1], 0);
            radii[i] = radius;
        }

        fittedCenters = centers;
        fittedRadii = radii;

        float eps = 0.01f;
        gradOut.assign(out.size(), 0.0f);

        for (int s = 0; s < trainingSamples.size(); ++s)
        {
            zVector pt = trainingSamples[s];
            float gt = sdfGT[s];
            float pred = blendCircleSDFs(pt, centers, radii, smoothK);
            float err = pred - gt;

            for (int i = 0; i < number_sdf; i++)
            {
                std::vector<zVector> cx = centers, cy = centers;
                cx[i].x += eps;
                cy[i].y += eps;
                float gx = (blendCircleSDFs(pt, cx, radii, smoothK) - pred) / eps;
                float gy = (blendCircleSDFs(pt, cy, radii, smoothK) - pred) / eps;
                gradOut[i * 2 + 0] += 2 * err * gx;
                gradOut[i * 2 + 1] += 2 * err * gy;
            }
        }

        for (float& g : gradOut) g /= trainingSamples.size();
    }


};

//----------------------- Unit test for generic MLP

void runUnitTest()
{
    MLP net(2, { 8, 8 }, 1);

    std::vector<std::vector<float>> X, Y;
    for (int i = 0; i < 100; ++i)
    {
        float x0 = ((float)rand() / RAND_MAX) * 6.28f - 3.14f;
        float x1 = ((float)rand() / RAND_MAX) * 6.28f - 3.14f;
        float y = std::sin(x0) + std::cos(x1);
        X.push_back({ x0, x1 });
        Y.push_back({ y });
    }

    float lr = 0.01f;
    for (int epoch = 0; epoch < 1200; ++epoch)
    {
        float totalLoss = 0.0f;
        for (int i = 0; i < X.size(); ++i)
        {
            std::vector<float> grads;
            net.computeGradient(X[i], Y[i], grads);
            net.backward(grads, lr);
            auto out = net.forward(X[i]);
            totalLoss += net.computeLoss(out, Y[i]);
        }
        totalLoss /= X.size();
        if (epoch % 50 == 0)
            std::cout << "Epoch " << epoch << " Avg Loss: " << totalLoss << std::endl;
    }

    std::cout << "Test prediction:\n";
    for (int i = 0; i < 5; ++i)
    {
        auto out = net.forward(X[i]);
        std::cout << "Input: (" << X[i][0] << ", " << X[i][1] << ") Target: " << Y[i][0] << " Pred: " << out[0] << std::endl;
    }
}


//------------------------------------------------------------------ MVC test for subClassMLP
std::vector<zVector> polygon;
std::vector<zVector> trainingSamples;
std::vector<float> sdfGT;


#define NUM_CENTERS 16

PolygonSDF_MLP mlp;
std::vector<float> grads;
std::vector<float> mlp_input_data;


// function
void initializeMLP()
{
    int input_dim = NUM_CENTERS * 2;
    int output_dim = NUM_CENTERS * 2;
    std::vector<int> hidden_dims = { 8 };

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
        runUnitTest();
    }


}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_
