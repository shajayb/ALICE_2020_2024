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

constexpr int RES = 32;
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

//void generateTrainingSet()
//{
//    sdfStack.clear();
//    for (float r = 4.0f; r <= 12.0f; r += 1.5f)
//    {
//        sdfStack.push_back(generateCircleSDF(r));
//    }
//}

#include "scalarField.h"

void generateTrainingSet()
{
    sdfStack.clear();

    for (int i = 0; i < 150; ++i)
    {
        ScalarField2D F;

        int choice = rand() % 5;

        if (choice == 0) // Single circle
        {
            zVector c(ofRandom(-20, 20), ofRandom(-20, 20), 0);
            float r = ofRandom(5, 15);
            F.addCircleSDF(c, r);
        }

        else if (choice == 1) // Two blended circles
        {
            ScalarField2D A, B;
            A.addCircleSDF(zVector(ofRandom(-25, 0), ofRandom(-10, 10), 0), ofRandom(5, 10));
            B.addCircleSDF(zVector(ofRandom(0, 25), ofRandom(-10, 10), 0), ofRandom(5, 10));
            A.blendWith(B, 10.0f, SMinMode::EXPONENTIAL);
            F = A;
        }

        else if (choice == 2) // Box subtracting circle
        {
            ScalarField2D box, circle;
            box.addOrientedBoxSDF(zVector(0, 0, 0), zVector(10, 15, 0), ofRandom(-PI / 4.0, PI / 4.0));
            circle.addCircleSDF(zVector(ofRandom(-10, 10), ofRandom(-10, 10), 0), ofRandom(5, 10));
            box.subtract(circle);
            F = box;
        }

        else if (choice == 3) // Voronoi
        {
            std::vector<zVector> pts;
            for (int k = 0; k < 6; k++)
            {
                pts.push_back(zVector(ofRandom(-30, 30), ofRandom(-30, 30), 0));
            }
            F.addVoronoi(pts);
        }

        else if (choice == 4) // Three-way union (mixed types)
        {
            ScalarField2D A, B, C;
            A.addOrientedBoxSDF(zVector(ofRandom(-15, 0), ofRandom(-15, 15), 0), zVector(5, 8, 0), ofRandom(0, PI / 3.0));
            B.addCircleSDF(zVector(ofRandom(0, 15), ofRandom(-15, 15), 0), ofRandom(5, 10));
            C.addOrientedBoxSDF(zVector(ofRandom(-5, 5), ofRandom(-15, 15), 0), zVector(3, 12, 0), ofRandom(-PI / 6.0, PI / 6.0));
            A.unionWith(B);
            A.unionWith(C);
            F = A;
        }

        // Convert to vec_t for training
        vec_t sdf;
        float dx = static_cast<float>(F.RES - 1) / (RES - 1);
        float dy = static_cast<float>(F.RES - 1) / (RES - 1);

        for (int j = 0; j < RES; ++j)
        {
            for (int i = 0; i < RES; ++i)
            {
                int sx = static_cast<int>(i * dx);
                int sy = static_cast<int>(j * dy);

                sx = std::clamp(sx, 0, F.RES - 1);
                sy = std::clamp(sy, 0, F.RES - 1);

                sdf.push_back(static_cast<float_t>(F.field[sx][sy]));
            }
        }


        sdfStack.push_back(sdf);
    }
}


void buildNetwork()
{
    autoencoder = network<sequential>(); // clear old layers

    autoencoder
        << fully_connected_layer(inputDim, 256) << relu()
        << fully_connected_layer(256, 128) << relu()
        << fully_connected_layer(128, latentDim)
        << fully_connected_layer(latentDim, 128) << relu()
        << fully_connected_layer(128, 256) << relu()
        << fully_connected_layer(256, inputDim);



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
    adam optimizer;
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

ScalarField2D F;

void draw()
{
    backGround(0.9);
    drawGrid(50);

    drawSDFGrid(sdfStack[curSample], zVector(0, 0, 0));            // Input SDF
    drawSDFGrid(reconstructedSDF, zVector(RES + 4, 0, 0));         // Reconstruction
}

vec_t latentVec(latentDim, 0.0f); // starts at origin
int latentStepIndex = 0;
float latentStepSize = 0.1f;


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

    if (k == 'l') // walk through latent dimensions
    {
        latentVec[latentStepIndex] += latentStepSize;
        reconstructedSDF = autoencoder.predict(latentVec);

        std::cout << "Latent dim " << latentStepIndex << " += " << latentStepSize << "\n";
        latentStepIndex = (latentStepIndex + 1) % latentDim;  // move to next dim next time
    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_
