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

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;

class FeedForwardNN
{
public:
    struct Layer
    {
        std::vector<std::vector<float>> weights; // [out][in]
        std::vector<float> biases;               // [out]
        std::string activation;                  // "tanh", "relu", "linear"
    };

    std::vector<Layer> layers;

    FeedForwardNN() {}

    void addLayer(const std::vector<std::vector<float>>& W,
        const std::vector<float>& b,
        const std::string& activation)
    {
        Layer layer;
        layer.weights = W;
        layer.biases = b;
        layer.activation = activation;
        layers.push_back(layer);
    }

    vec_t forward(const vec_t& input) const
    {
        vec_t x = input;

        for (const Layer& layer : layers)
        {
            vec_t y(layer.biases.size());

            for (size_t i = 0; i < layer.biases.size(); ++i)
            {
                float sum = layer.biases[i];
                for (size_t j = 0; j < x.size(); ++j)
                {
                    sum += layer.weights[i][j] * x[j];
                }

                if (layer.activation == "tanh")      y[i] = std::tanh(sum);
                else if (layer.activation == "relu") y[i] = std::max(0.0f, sum);
                else                                 y[i] = sum; // linear
            }

            x = y; // propagate
        }

        return x;
    }
};




FeedForwardNN convertTinyDNNToFFNN(network<sequential>& net, int startIndex, int endIndex)
{
    FeedForwardNN ffnn;

    for (int i = startIndex; i <= endIndex; ++i)
    {
        layer* l = net[i];

        if (l->layer_type() == "fully-connected")
        {
            auto* fc = dynamic_cast<fully_connected_layer*>(l);
            if (!fc) continue;

            auto wts = fc->weights();
            vec_t& flatWeights = *wts[0];
            vec_t& biases = *wts[1];

            int outDim = fc->out_size();
            int inDim = fc->in_size();

            std::vector<std::vector<float>> W(outDim, std::vector<float>(inDim));
            std::vector<float> b(outDim);

            for (int o = 0; o < outDim; ++o)
            {
                for (int j = 0; j < inDim; ++j)
                {
                    W[o][j] = flatWeights[o * inDim + j];
                }
                b[o] = biases[o];
            }

        


            // Default to linear; activation will be applied in next loop if present
            ffnn.addLayer(W, b, "linear");
        }
        else if (l->layer_type() == "tanh-activation" || l->layer_type() == "relu-activation")
        {

            if (ffnn.layers.empty())
            {
                std::cerr << "⚠️ Activation layer found without preceding FC.\n";
                continue;
            }

            cout << l->layer_type() << endl;
            ffnn.layers.back().activation = (l->layer_type() == "tanh-activation") ? "tanh" : "relu";
        }
        else
        {
            std::cerr << "⚠️ Skipping unsupported layer type: " << l->layer_type() << "\n";
        }
    }

    return ffnn;
}


void generateTrainingSet()
{
    sdfStack.clear();

    for (int i = 0; i < 50; ++i)
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
        << fully_connected_layer(inputDim, 256) 
        << relu()
        << fully_connected_layer(256, 128) 
        << relu()
        << fully_connected_layer(128, latentDim)
        << fully_connected_layer(latentDim, 128) 
        << relu()
        << fully_connected_layer(128, 256) 
        << relu()
        << fully_connected_layer(256, inputDim);

    //tiny_dnn::activation::tanh()

}


void decodeCurrent()
{
    reconstructedSDF = autoencoder.predict(sdfStack[curSample]);
}

network<sequential> buildEncoderFromAutoencoder(network<sequential>& autoencoder)
{
    network<sequential> encoder;

    for (size_t i = 0; i <= 4 && i < autoencoder.depth(); ++i)
    {
        layer* orig = autoencoder[i];

        if (orig->layer_type() == "fully-connected")
        {
            auto* fc = dynamic_cast<fully_connected_layer*>(orig);
            if (!fc) continue;

            auto* new_fc = new fully_connected_layer(fc->in_size(), fc->out_size());

            // Copy weights and biases
            auto orig_w = fc->weights();
            auto new_w = new_fc->weights();
            for (size_t k = 0; k < orig_w.size(); ++k)
            {
                if (orig_w[k] && new_w[k])
                    *new_w[k] = *orig_w[k];

                printf(" %.2f , %.2f \n", *orig_w[k], *new_w[k]);
            }

            encoder << *new_fc;
        }
        else if (orig->layer_type() == "tanh-activation")
        {
            encoder << tanh_layer();
        }
        else if (orig->layer_type() == "relu-activation")
        {
            encoder << relu_layer();
        }
        else if (orig->layer_type() == "sigmoid-activation")
        {
            encoder << sigmoid_layer();
        }
        else
        {
            std::cerr << "Skipping unsupported encoder layer type: " << orig->layer_type() << std::endl;
        }
    }

    return encoder;
}

network<sequential> buildDecoderFromAutoencoder(network<sequential>& autoencoder)
{
    network<sequential> decoder;

    layer* orig = autoencoder[5];
    if (!(orig->in_size() == latentDim))return decoder;

    for (size_t i = 5; i < autoencoder.depth(); ++i)
    {
        layer* orig = autoencoder[i];

        // Identify layer type
        if (orig->layer_type() == "fully-connected")
        {
            auto* fc = dynamic_cast<fully_connected_layer*>(orig);
            if (!fc) continue;

            // Clone layer
            auto* new_fc = new fully_connected_layer(fc->in_size(), fc->out_size());

            // Copy weights
            auto orig_w = fc->weights();
            auto new_w = new_fc->weights();

            for (size_t k = 0; k < orig_w.size(); ++k)
            {
                if (orig_w[k] && new_w[k])
                {
                    *new_w[k] = *orig_w[k];
                }
            }
          

           decoder << *new_fc;
        
        }
        else if (orig->layer_type() == "tanh-activation")
        {
            decoder << tanh_layer();
        }
        else if (orig->layer_type() == "relu-activation")
        {
            decoder << relu_layer();
        }
        else if (orig->layer_type() == "sigmoid-activation")
        {
            decoder << sigmoid_layer();
        }
        else
        {
            std::cerr << "Skipping unsupported decoder layer type: " << orig->layer_type() << std::endl;
        }
    }

    return decoder;
}

void printNetworkSummary(network<sequential>& autoencoder, const FeedForwardNN& encoderNN, const FeedForwardNN& decoderNN)
{
    std::cout << "\n========== NETWORK STRUCTURE SUMMARY ==========\n";

    std::cout << "\n--- TinyDNN Autoencoder Layers ---\n";
    for (size_t i = 0; i < autoencoder.depth(); ++i)
    {
        layer* l = autoencoder[i];
        std::cout << "  [Layer " << i << "] type: " << l->layer_type()
            << " | in: " << l->in_data_size()
            << " | out: " << l->out_data_size() << "\n";
    }

    std::cout << "\n--- FeedForwardNN Encoder ---\n";
    for (size_t i = 0; i < encoderNN.layers.size(); ++i)
    {
        const auto& layer = encoderNN.layers[i];
        std::cout << "  [Layer " << i << "] in: " << layer.weights[0].size()
            << " | out: " << layer.weights.size()
            << " | activation: " << layer.activation << "\n";
    }

    std::cout << "\n--- FeedForwardNN Decoder ---\n";
    for (size_t i = 0; i < decoderNN.layers.size(); ++i)
    {
        const auto& layer = decoderNN.layers[i];
        std::cout << "  [Layer " << i << "] in: " << layer.weights[0].size()
            << " | out: " << layer.weights.size()
            << " | activation: " << layer.activation << "\n";
    }

    std::cout << "===============================================\n";
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

        FeedForwardNN encoderNN = convertTinyDNNToFFNN(autoencoder, 0, 4);
        FeedForwardNN decoderNN = convertTinyDNNToFFNN(autoencoder, 5, autoencoder.depth() - 1);

        vec_t inputVec = sdfStack[curSample];
        vec_t latent = encoderNN.forward(inputVec);
        vec_t output = decoderNN.forward(latent);
        
        // autoencoder.predict(sdfStack[curSample]); versus decoderNN.forward(latent);

        vec_t ref = autoencoder.predict(inputVec);

        float mse = 0.0f;
        for (int i = 0; i < output.size(); ++i)
        {
            float diff = output[i] - ref[i];
            mse += diff * diff;
        }
        mse /= output.size();

        std::cout << "🔍 MSE between decoderNN and autoencoder: " << mse << std::endl;

        //FeedForwardNN encoderNN = convertTinyDNNToFFNN(autoencoder, 0, 4);
       // FeedForwardNN decoderNN = convertTinyDNNToFFNN(autoencoder, 5, autoencoder.depth() - 1);

        printNetworkSummary(autoencoder, encoderNN, decoderNN);


       // 
        //layer* orig = autoencoder[0];
        //std::cout << "input: " << orig->in_data_size() << " (" << orig->in_data_shape() << ")\n";
        //std::cout << "output: " << orig->out_data_size() << " (" << orig->out_data_shape() << ")\n";

        //if (orig->layer_type() == "fully-connected")
        //{
        //    auto orig_w = orig->weights();

        //    if (orig_w.size() >= 2)
        //    {
        //        vec_t& weights = *orig_w[0];  // weights matrix (flattened)
        //        vec_t& biases = *orig_w[1];  // bias vector

        //        cout << weights.size() << ":" << biases.size() << endl;

        //        std::cout << "Weights:\n";
        //        /*for (size_t i = 0; i < weights.size(); ++i)
        //        {
        //            printf("  w[%03zu] = %.8f\n", i, weights[i]);
        //        }

        //        std::cout << "Biases:\n";
        //        for (size_t i = 0; i < biases.size(); ++i)
        //        {
        //            printf("  b[%03zu] = %.8f\n", i, biases[i]);
        //        }*/
        //    }
        //    else
        //    {
        //        std::cerr << "⚠️ Layer has fewer than 2 trainable parameters.\n";
        //    }
        //}
        //else
        //{
        //    cout << orig->layer_type() << endl;
        //}

     
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