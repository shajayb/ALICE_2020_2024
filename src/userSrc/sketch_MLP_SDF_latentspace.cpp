#define _MAIN_
#ifdef _MAIN_

#include <tiny_dnn/config.h>
#include <tiny_dnn/tiny_dnn.h>

#include "main.h"

#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;

inline zVector zMax(zVector& a, zVector& b)
{
    return zVector(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

inline void getJetColor(float value, float& r, float& g, float& b)
{
    value = std::clamp(value, -1.0f, 1.0f);
    float normalized = (value + 1.0f) * 0.5f;
    float fourValue = 4.0f * normalized;

    r = std::clamp(std::min(fourValue - 1.5f, -fourValue + 4.5f), 0.0f, 1.0f);
    g = std::clamp(std::min(fourValue - 0.5f, -fourValue + 3.5f), 0.0f, 1.0f);
    b = std::clamp(std::min(fourValue + 0.5f, -fourValue + 2.5f), 0.0f, 1.0f);
}

Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

constexpr int RES = 128;
constexpr int latentDim = 8;
constexpr int inputDim = RES * RES;

std::vector<vec_t> sdfStack;
vec_t reconstructedSDF;
int curSample = 0;

network<sequential> autoencoder;

zModel model;

//------------------------------------------
// Generate synthetic circular SDF
//------------------------------------------
vec_t generateCircleSDF(float radius)
{
    vec_t sdf;
    float cx = RES * 0.5f, cy = RES * 0.5f;

    for (int y = 0; y < RES; y++)
    {
        for (int x = 0; x < RES; x++)
        {
            float dx = x - cx;
            float dy = y - cy;
            float dist = sqrt(dx * dx + dy * dy) - radius;
            sdf.push_back(static_cast<float_t>(dist / RES));
        }
    }
    return sdf;
}

void generateTrainingSet()
{
    sdfStack.clear();
    for (float r = 4.0f; r <= 12.0f; r += 1.5f)
    {
        sdfStack.push_back(generateCircleSDF(r));
    }
}

void buildNetwork()
{
    autoencoder = network<sequential>(); // clear old layers

    autoencoder
        << fully_connected_layer(inputDim, 64) << relu()
        << fully_connected_layer(64, latentDim)
        << fully_connected_layer(latentDim, 64) << relu()
        << fully_connected_layer(64, inputDim);
}


void decodeCurrent()
{
    reconstructedSDF = autoencoder.predict(sdfStack[curSample]);
}

void drawSDFGrid(const vec_t& sdf, zVector offset)
{
    for (int y = 0; y < RES; ++y)
    {
        for (int x = 0; x < RES; ++x)
        {
            float val = sdf[y * RES + x];
            float r, g, b;
            getJetColor(val, r, g, b);
            glColor3f(r, g, b);

            zVector pt = zVector(x, -y, 0) + offset;
            drawPoint(zVecToAliceVec(pt));
        }
    }
}

void trainOneEpoch()
{
    adagrad optimizer;
    std::vector<vec_t> inputs, targets;

    for (auto& sdf : sdfStack)
    {
        vec_t safeInput;
        for (auto v : sdf) safeInput.push_back(static_cast<float_t>(v));
        inputs.push_back(safeInput);
        targets.push_back(safeInput);
    }

    autoencoder.fit<mse>(optimizer, inputs, targets, 5, 1);

    // ---- compute loss manually ----
    float totalLoss = 0.0f;
    for (int i = 0; i < inputs.size(); ++i)
    {
        vec_t output = autoencoder.predict(inputs[i]);
        float sampleLoss = 0.0f;
        for (int j = 0; j < output.size(); ++j)
        {
            float diff = output[j] - targets[i][j];
            sampleLoss += diff * diff;
        }
        totalLoss += sampleLoss / output.size();
    }

    totalLoss /= inputs.size();
    std::cout << "Mean Squared Loss: " << totalLoss << std::endl;
}


//------------------------------------------
void setup()
{
    generateTrainingSet();
    buildNetwork();
    decodeCurrent();
}

void update(int value) {}

void draw()
{
    backGround(0.9);
    drawGrid(50);

    drawSDFGrid(sdfStack[curSample], zVector(0, 0, 0));            // Input SDF
    drawSDFGrid(reconstructedSDF, zVector(RES + 4, 0, 0));         // Reconstruction
}

void keyPress(unsigned char k, int xm, int ym)
{
    if (k == 'n')
    {
        curSample = (curSample + 1) % sdfStack.size();
        decodeCurrent();
    }

    if (k == 't')
    {
        trainOneEpoch();
        decodeCurrent();
    }

    if (k == 'e')
    {
        vec_t latent = autoencoder.predict(sdfStack[curSample]);

        std::cout << "Latent: ";
        for (auto& v : latent) printf(" %.3f", v);
        std::cout << "\n";
    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_
