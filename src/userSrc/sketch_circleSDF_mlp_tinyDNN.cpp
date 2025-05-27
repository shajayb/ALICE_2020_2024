


#define _MAIN_
#ifdef _MAIN_



#include "main.h"
#include <vector>
#include <cmath>
#include <tiny_dnn/config.h>
#include <tiny_dnn/tiny_dnn.h>

#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;
using namespace tiny_dnn;

Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

zVector AliceVecToZvec(Alice::vec& in)
{
    return zVector(in.x, in.y, in.z);
}

inline zVector zMax( zVector& a,  zVector& b)
{
    return zVector(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

inline zVector zMin( zVector& a,  zVector& b)
{
    return zVector(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

vector<zVector> loadPolygonFromCSV( std::string& filename)
{
    vector<zVector> poly;
    poly.clear();

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
            poly.emplace_back(x, y, 0);
        }
    }

    cout << poly.size() << " polygon size" << endl;
    return poly;
}


inline float smin(float a, float b, float k)
{
    float h = std::max(k - fabs(a - b), 0.0f) / k;
    return std::min(a, b) - h * h * k * 0.25f;
}

zModel model;
std::vector<zVector> polygonPoints;  // GT polygon
std::vector<float> sdfGT;            // GT SDF values
std::vector<zVector> sdfCenters;
std::vector<float> predictedRadii;

network<sequential> mlp;
adagrad optimizer;

tensor_t inBatch = { {1.0f} };
tensor_t gradOutput(1);

int N_CIRCLES = 4;
bool train = false;

//network<sequential> buildMLP()
//{
//    network<sequential> net;
//    net << fully_connected_layer(1, 64) << relu()
//        << fully_connected_layer(64, 64) << relu()
//        << fully_connected_layer(64, N_CIRCLES * 3);
//    return net;
//}

//void decodeOutput( vec_t& out)
//{
//    sdfCenters.clear();
//    predictedRadii.clear();
//    for (int i = 0; i < N_CIRCLES; ++i)
//    {
//        sdfCenters.push_back(zVector(out[i * 3 + 0], out[i * 3 + 1], 0));
//        predictedRadii.push_back(fabs(out[i * 3 + 2]));
//    }
//}
//
//void trainStep()
//{
//    vec_t input = { 1.0f };
//    vec_t output = mlp.forward(input);
//    decodeOutput(output);
//
//    float loss = 0.0f;
//    float eps = 1e-3f;
//
//    gradOutput[0] = vec_t(output.size(), 0.0f);
//
//    for (int i = 0; i < polygonPoints.size(); ++i)
//    {
//        float basePred = blendCircleSDFs(polygonPoints[i]);
//        float error = basePred - sdfGT[i];
//        loss += error * error;
//
//        for (int j = 0; j < output.size(); ++j)
//        {
//            vec_t perturbed = output;
//            perturbed[j] += eps;
//            decodeOutput(perturbed);
//            float newPred = blendCircleSDFs(polygonPoints[i]);
//
//            float dL = (newPred - basePred) / eps;
//            gradOutput[0][j] += 2.0f * error * dL;
//        }
//
//        decodeOutput(output); // reset
//    }
//
//    std::cout << "Loss: " << loss << std::endl;
//
//    mlp.backward(inBatch, gradOutput);
//    optimizer.update(mlp, 1);
//}
//
//float blendCircleSDFs( zVector& pt)
//{
//    float d = (pt - sdfCenters[0]).length() - predictedRadii[0];
//    for (int i = 1; i < sdfCenters.size(); i++)
//    {
//        float d_i = (pt - sdfCenters[i]).length() - predictedRadii[i];
//        d = smin(d, d_i, 3.0f);
//    }
//    return d;
//}
//
//void drawCircles()
//{
//    glColor3f(1, 0, 0);
//    for (int i = 0; i < sdfCenters.size(); i++)
//    {
//        drawCircle(zVecToAliceVec(sdfCenters[i]), predictedRadii[i], 32);
//    }
//}


// -----------------------------------


void setup()
{
    mlp = buildMLP();

    for (int i = 0; i < 64; i++)
    {
        float theta = i * TWO_PI / 64.0;
        zVector pt(cos(theta) * 15, sin(theta) * 10, 0);
        polygonPoints.push_back(pt);
        sdfGT.push_back(0.0f);
    }
}

void update(int value)
{
    if (train)
    {
        trainStep();
    }
}

void draw()
{
    backGround(0.9);
    drawGrid(50);

    glColor3f(0, 0, 1);
    for (auto& p : polygonPoints)
        drawPoint(zVecToAliceVec(p));

    drawCircles();
}

void keyPress(unsigned char k, int xm, int ym)
{
    if (k == 't')
    {
        train = !train;
    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_
