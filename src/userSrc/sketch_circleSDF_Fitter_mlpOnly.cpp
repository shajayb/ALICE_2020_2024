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

#include "scalarField.h" //// two functiosn must be turned on in scalarfIELD.H for sketch_circleSDF_fitter.cpp

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
float polygonSDF( zVector& p,  std::vector<zVector>& poly)
{
    float minDist = 1e6;
    int n = poly.size();

    for (int i = 0; i < n; i++)
    {
        zVector a = poly[i];
        zVector b = poly[(i + 1) % n];

        zVector ab = b - a;
        zVector ap = p - a;

        float t = std::max(0.0f, std::min(1.0f, (ab * ap) / (ab*ab)));
        zVector proj = a + ab * t;
        float d = p.distanceTo(proj);
        minDist = std::min(minDist, d);
    }

    return minDist * (isInsidePolygon(p, poly) ? -1.0f : 1.0f);
}

//------------------------------------------------------------------ Globals

std::vector<zVector> polygon;
std::vector<zVector> trainingSamples;
std::vector<float> sdfGT;

float smoothK = 3.0f;
float radius = 10.0f;
#define NUM_SDF 16

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

void samplePoints()
{
    trainingSamples.clear();
    sdfGT.clear();

    for (float x = -50; x <= 50; x += 5.0f)
    {
        for (float y = -50; y <= 50; y += 5.0f)
        {
            zVector pt(x, y, 0);
            if (isInsidePolygon(pt, polygon))
            {
                trainingSamples.push_back(pt);
                sdfGT.push_back(polygonSDF(pt, polygon));
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

class MLP
{
public:
    int inputDim = 1;
    int outputDim = 6;
    std::vector<int> hiddenLayerDims = { 8, 8 };
    std::vector<std::vector<std::vector<float>>> W;
    std::vector<std::vector<float>> b;

    std::vector<std::vector<float>> activations;

    std::vector<zVector> fittedCenters;
    std::vector<float> fittedRadii;


    MLP()
    {
        std::vector<int> layerDims = { inputDim };
        layerDims.insert(layerDims.end(), hiddenLayerDims.begin(), hiddenLayerDims.end());
        layerDims.push_back(outputDim);

        for (int l = 0; l < layerDims.size() - 1; l++)
        {
            int inSize = layerDims[l];
            int outSize = layerDims[l + 1];
            W.push_back(std::vector<std::vector<float>>(outSize, std::vector<float>(inSize, 0.01f)));
            b.push_back(std::vector<float>(outSize, 0.0f));
        }
    }

    MLP(int inDim, std::vector<int>& hiddenDims, int outDim)
    {
        inputDim = inDim;
        outputDim = outDim;
        hiddenLayerDims = hiddenDims;

        std::vector<int> layerDims = { inputDim };
        layerDims.insert(layerDims.end(), hiddenLayerDims.begin(), hiddenLayerDims.end());
        layerDims.push_back(outputDim);

        W.clear();
        b.clear();

        for (int l = 0; l < layerDims.size() - 1; l++)
        {
            int inSize = layerDims[l];
            int outSize = layerDims[l + 1];

            W.push_back(std::vector<std::vector<float>>(outSize, std::vector<float>(inSize, 0.01f)));
            b.push_back(std::vector<float>(outSize, 0.0f));
        }
    }


    std::vector<float> forward( std::vector<float>& x)
    {
        activations.clear();
        activations.push_back(x); // input

        std::vector<float> a = x;
        for (int l = 0; l < W.size(); l++)
        {
            std::vector<float> z = b[l];
            for (int i = 0; i < W[l].size(); i++)
            {
                for (int j = 0; j < W[l][i].size(); j++)
                {
                    z[i] += W[l][i][j] * a[j];
                }
                if (l < W.size() - 1) z[i] = tanh(z[i]);
            }
            activations.push_back(z);
            a = z;
        }

        return a;
    }

    float computeLoss( std::vector<float>& x, std::vector<zVector>& polygon)
    {
        auto out = forward(x);

        std::vector<zVector> centers(NUM_SDF);
        std::vector<float> radii(NUM_SDF);
        for (int i = 0; i < NUM_SDF; i++)
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

    void computeGradient( std::vector<float>& x, std::vector<zVector>& polygon, std::vector<float>& gradOut)
    {
        auto out = forward(x);
        std::vector<zVector> centers(NUM_SDF);
        std::vector<float> radii(NUM_SDF);

        for (int i = 0; i < NUM_SDF; i++)
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

            for (int i = 0; i < NUM_SDF; i++)
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

    void backward( std::vector<float>& gradOut, float lr)
    {
        int L = hiddenLayerDims.size();
        int current = L;

        std::vector<float> grad = gradOut;
        std::vector<float> prevGrad;

        for (int l = W.size() - 1; l >= 0; --l)
        {
             std::vector<float>& prev = activations[l];
            std::vector<float> delta(prev.size(), 0.0f);

            for (int i = 0; i < W[l].size(); ++i)
            {
                for (int j = 0; j < W[l][i].size(); ++j)
                {
                    W[l][i][j] -= lr * grad[i] * prev[j];
                    delta[j] += grad[i] * W[l][i][j];
                }
                b[l][i] -= lr * grad[i];
            }

            if (l > 0)
            {
                grad.resize(prev.size());
                for (int i = 0; i < delta.size(); ++i)
                {
                    float act = activations[l][i];
                    grad[i] = delta[i] * (1 - act * act);
                }
            }
        }
    }
};

//------------------------------------------------------------------ MVC

MLP mlp;
std::vector<float> grads;
std::vector<float> mlp_input_data;





void initializeMLP()
{
    int input_dim = NUM_SDF * 2;
    int output_dim = NUM_SDF * 2;
    std::vector<int> hidden_dims = { 32, 32 };

    mlp = MLP(input_dim, hidden_dims, output_dim); // assumes MLP constructor initializes weights/biases
    mlp_input_data.assign(input_dim, 1.0f); // or use 0.0f for strict zero-input
}


void setup()
{
    loadPolygonFromCSV("data/polygon.txt");
    samplePoints();

    initializeMLP();  // <-- add this line
}

void update(int value) {}

void draw()
{
    backGround(0.8);
    drawGrid(50);

    glColor3f(1, 0, 0);
    for (auto& c : mlp.fittedCenters)
    {
        drawCircle(zVecToAliceVec(c), radius * 0.25, 32);
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
}

void keyPress(unsigned char k, int xm, int ym)
{
    if (k == 't')
    {
        grads.clear();
        float loss = mlp.computeLoss(mlp_input_data, polygon);
        mlp.computeGradient(mlp_input_data, polygon, grads);
        mlp.backward(grads, 0.1f);

        std::cout << "Loss: " << loss << std::endl;
    }

}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_
