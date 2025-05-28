#define _MAIN_
#ifdef _MAIN_

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>

#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;

inline float circleSDF(zVector p, zVector c, float r)
{
    return (p - c).length() - r;
}

inline float blendCircleSDFs(zVector pt, const std::vector<zVector>& centers, const std::vector<float>& radii)
{
    float result = 1e6;
    for (int i = 0; i < centers.size(); i++)
    {
        float d = circleSDF(pt, centers[i], radii[i]);
        result = std::min(result, d);
    }
    return result;
}

std::vector<zVector> samplePts;
std::vector<float> sdfGT;
std::vector<zVector> fittedCenters;
std::vector<float> fittedRadii;

std::vector<zVector> loadPolygonFromCSV(const std::string& filename)
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

class MLP
{
public:
    int inputDim, hiddenDim, outputDim;
    std::vector<std::vector<float>> W1, W2;
    std::vector<float> b1, b2;
    std::vector<float> input, hidden, output;

    MLP(int inDim, int hDim, int outDim)
        : inputDim(inDim), hiddenDim(hDim), outputDim(outDim)
    {
        W1.resize(hiddenDim, std::vector<float>(inputDim));
        W2.resize(outputDim, std::vector<float>(hiddenDim));
        b1.resize(hiddenDim);
        b2.resize(outputDim);
        input.resize(inputDim);
        hidden.resize(hiddenDim);
        output.resize(outputDim);

        std::default_random_engine eng;
        std::normal_distribution<float> dist(0.0, 0.1);

        for (auto& row : W1) for (auto& val : row) val = dist(eng);
        for (auto& row : W2) for (auto& val : row) val = dist(eng);
    }

    std::vector<float> forward(const std::vector<float>& x)
    {
        input = x;
        for (int i = 0; i < hiddenDim; ++i)
        {
            hidden[i] = b1[i];
            for (int j = 0; j < inputDim; ++j)
                hidden[i] += W1[i][j] * input[j];
            hidden[i] = std::tanh(hidden[i]);
        }
        for (int i = 0; i < outputDim; ++i)
        {
            output[i] = b2[i];
            for (int j = 0; j < hiddenDim; ++j)
                output[i] += W2[i][j] * hidden[j];
        }
        return output;
    }

    float computeLossAndGradient(const std::vector<float>& x, const std::vector<zVector>& polygon, std::vector<float>& gradOut)
    {
        std::vector<zVector> centers(outputDim / 3);
        std::vector<float> radii(outputDim / 3);

        auto out = forward(x);
        for (int i = 0; i < centers.size(); i++)
        {
            centers[i] = zVector(out[i * 3 + 0], out[i * 3 + 1], 0);
            radii[i] = std::abs(out[i * 3 + 2]);
        }

        // Save for visualization
        fittedCenters = centers;
        fittedRadii = radii;

        float loss = 0;
        gradOut.assign(outputDim, 0.0f);

        for (int j = 0; j < outputDim; ++j)
        {
            float eps = 0.01f;
            std::vector<float> perturbedInput = x;
            perturbedInput[j] += eps;

            auto perturbedOut = forward(perturbedInput);
            std::vector<zVector> cPert(centers.size());
            std::vector<float> rPert(centers.size());

            for (int i = 0; i < centers.size(); i++)
            {
                cPert[i] = zVector(perturbedOut[i * 3 + 0], perturbedOut[i * 3 + 1], 0);
                rPert[i] = std::abs(perturbedOut[i * 3 + 2]);
            }

            float gradLoss = 0;
            for (int s = 0; s < samplePts.size(); ++s)
            {
                float f = blendCircleSDFs(samplePts[s], centers, radii);
                float fPert = blendCircleSDFs(samplePts[s], cPert, rPert);
                float gt = sdfGT[s];
                float err = f - gt;
                loss += err * err;
                gradLoss += 2 * err * (fPert - f) / eps;
            }
            gradOut[j] = gradLoss;
        }

        return loss;
    }

    void backward(const std::vector<float>& gradOut, float lr)
    {
        std::vector<float> gradHidden(hiddenDim);

        for (int i = 0; i < hiddenDim; ++i)
        {
            gradHidden[i] = 0;
            for (int j = 0; j < outputDim; ++j)
            {
                gradHidden[i] += gradOut[j] * W2[j][i];
                W2[j][i] -= lr * gradOut[j] * hidden[i];
            }
            b2[i] -= lr * gradOut[i];
        }

        for (int i = 0; i < hiddenDim; ++i)
        {
            float dtanh = 1.0f - hidden[i] * hidden[i];
            for (int j = 0; j < inputDim; ++j)
            {
                W1[i][j] -= lr * gradHidden[i] * dtanh * input[j];
            }
            b1[i] -= lr * gradHidden[i] * dtanh;
        }
    }
};

#define NUM_SDF 3
std::vector<zVector> polygon;

float polygonSDF(std::vector<zVector>& poly, zVector& p)
{
    float minDist = 1e6;
    int N = poly.size();
    for (int i = 0; i < N; i++)
    {
        zVector a = poly[i];
        zVector b = poly[(i + 1) % N];
        zVector pa = p - a;
        zVector ba = b - a;
        float h = std::clamp((pa * ba) / (ba * ba), 0.0f, 1.0f);
        zVector proj = a + ba * h;
        minDist = std::min(minDist, p.distanceTo(proj));
    }

    int crossings = 0;
    for (int i = 0; i < N; i++)
    {
        zVector a = poly[i];
        zVector b = poly[(i + 1) % N];
        if (((a.y > p.y) != (b.y > p.y)) &&
            (p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y + 1e-6) + a.x))
        {
            crossings++;
        }
    }

    float sign = (crossings % 2 == 0) ? 1.0f : -1.0f;
    return minDist * sign;
}

void computeSampleData()
{
    samplePts.clear();
    sdfGT.clear();
    for (int i = -50; i <= 50; i += 5)
    {
        for (int j = -50; j <= 50; j += 5)
        {
            zVector pt(i, j, 0);
            samplePts.push_back(pt);
            sdfGT.push_back(polygonSDF(polygon, pt));
        }
    }
}

void trainCircleSDF()
{
    MLP mlp(NUM_SDF * 3, 32, 3 * NUM_SDF);
    std::vector<float> input(NUM_SDF * 3, 0.0f);
    std::vector<float> gradOut;

    for (int epoch = 0; epoch < 100; ++epoch)
    {
        float loss = mlp.computeLossAndGradient(input, polygon, gradOut);
        mlp.backward(gradOut, 0.001);
        std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
    }
}

void setup()
{
    polygon.clear();
    polygon = loadPolygonFromCSV("data/polygon.txt");
    computeSampleData();
    trainCircleSDF();
}

void draw()
{
    backGround(0.9);
    drawGrid(50);

    glColor3f(1, 0, 0);
    glBegin(GL_LINE_LOOP);
    for (auto& pt : polygon)
    {
        drawVertex(pt);
    }
    glEnd();

    glColor3f(0, 0, 1);
    for (int i = 0; i < fittedCenters.size(); i++)
    {
        drawCircle(fittedCenters[i], fittedRadii[i], 32);
    }
}

#endif // _MAIN_
