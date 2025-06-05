#define _MAIN_
#ifdef _MAIN_

#include "main.h"

// zSpace Core
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>
using namespace zSpace;

#include <algorithm>
#include <vector>
#include <cmath>     // For std::tanh, std::sqrt, std::clamp
#include <random>    // For std::default_random_engine, std::normal_distribution
#include <iostream>  // For std::cout
#include <numeric>   // For std::iota

// Assuming tiny_dnn's vec_t is std::vector<float_t> with a custom allocator.
// For compatibility with the new MLP, we'll convert to std::vector<float>.
// If tiny_dnn is completely removed, this alias might change.
// For now, let's assume float_t is float.
using float_t = float;
using vec_t = std::vector<float_t>; // Standard vector for MLP compatibility

Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

constexpr int RES = 32;
constexpr int latentDim = 8;
constexpr int inputDim = RES * RES; // Input and output dimension for SDF

std::vector<vec_t> sdfStack;
vec_t reconstructedSDF;
int curSample = 0;

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

#include "scalarField.h" // Assuming this defines ScalarField2D and related functions

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

//------------------------------------------
// MLP Autoencoder Class
//------------------------------------------
class MLPAutoencoder
{
public:
    int inputDim;
    int latentDim;
    std::vector<int> encoderHiddenDims;
    std::vector<int> decoderHiddenDims;

    // Weights and biases for encoder layers (input -> h1 -> ... -> latent)
    // W_enc[0] connects input to first encoder hidden.
    // W_enc[i] connects enc_h_i to enc_h_{i+1} or to latent.
    std::vector<std::vector<std::vector<float>>> W_enc;
    std::vector<std::vector<float>> b_enc;

    // Weights and biases for decoder layers (latent -> h1 -> ... -> output)
    // W_dec[0] connects latent to first decoder hidden.
    // W_dec[i] connects dec_h_i to dec_h_{i+1} or to output.
    std::vector<std::vector<std::vector<float>>> W_dec;
    std::vector<std::vector<float>> b_dec;

    // Cached activations for backpropagation
    std::vector<std::vector<float>> enc_activations; // Stores activated outputs of each encoder layer (including input and latent)
    std::vector<std::vector<float>> dec_activations; // Stores activated outputs of each decoder layer (including latent and output)

    MLPAutoencoder() {} // Default constructor

    MLPAutoencoder(int inDim, const std::vector<int>& encHDims, int latDim, const std::vector<int>& decHDims)
        : inputDim(inDim), latentDim(latDim), encoderHiddenDims(encHDims), decoderHiddenDims(decHDims)
    {
        std::default_random_engine eng;
        std::normal_distribution<float> dist(0.0, 0.1); // For weight initialization

        // Initialize Encoder
        int current_enc_input_dim = inputDim;
        enc_activations.resize(encoderHiddenDims.size() + 2); // Input, hidden layers, latent layer
        enc_activations[0].resize(inputDim); // Input layer

        W_enc.resize(encoderHiddenDims.size() + 1); // input->h1, h1->h2, ..., hN->latent
        b_enc.resize(encoderHiddenDims.size() + 1);

        for (size_t l = 0; l < encoderHiddenDims.size(); ++l) {
            int current_enc_output_dim = encoderHiddenDims[l];
            W_enc[l].resize(current_enc_output_dim, std::vector<float>(current_enc_input_dim));
            b_enc[l].resize(current_enc_output_dim);
            enc_activations[l + 1].resize(current_enc_output_dim);

            for (auto& row : W_enc[l]) for (auto& val : row) val = dist(eng);
            for (auto& val : b_enc[l]) val = dist(eng);

            current_enc_input_dim = current_enc_output_dim;
        }
        // Latent layer connection (last encoder layer)
        W_enc[encoderHiddenDims.size()].resize(latentDim, std::vector<float>(current_enc_input_dim));
        b_enc[encoderHiddenDims.size()].resize(latentDim);
        enc_activations[encoderHiddenDims.size() + 1].resize(latentDim); // Latent layer activation
        for (auto& row : W_enc[encoderHiddenDims.size()]) for (auto& val : row) val = dist(eng);
        for (auto& val : b_enc[encoderHiddenDims.size()]) val = dist(eng);

        // Initialize Decoder
        int current_dec_input_dim = latentDim;
        dec_activations.resize(decoderHiddenDims.size() + 2); // Latent, hidden layers, output layer
        dec_activations[0].resize(latentDim); // Latent layer (input to decoder)

        W_dec.resize(decoderHiddenDims.size() + 1); // latent->h1, h1->h2, ..., hN->output
        b_dec.resize(decoderHiddenDims.size() + 1);

        for (size_t l = 0; l < decoderHiddenDims.size(); ++l) {
            int current_dec_output_dim = decoderHiddenDims[l];
            W_dec[l].resize(current_dec_output_dim, std::vector<float>(current_dec_input_dim));
            b_dec[l].resize(current_dec_output_dim);
            dec_activations[l + 1].resize(current_dec_output_dim);

            for (auto& row : W_dec[l]) for (auto& val : row) val = dist(eng);
            for (auto& val : b_dec[l]) val = dist(eng);

            current_dec_input_dim = current_dec_output_dim;
        }
        // Output layer connection (last decoder layer)
        W_dec[decoderHiddenDims.size()].resize(inputDim, std::vector<float>(current_dec_input_dim));
        b_dec[decoderHiddenDims.size()].resize(inputDim);
        dec_activations[decoderHiddenDims.size() + 1].resize(inputDim); // Output layer activation
        for (auto& row : W_dec[decoderHiddenDims.size()]) for (auto& val : row) val = dist(eng);
        for (auto& val : b_dec[decoderHiddenDims.size()]) val = dist(eng);
    }

    // Tanh activation function
    float tanh_activation(float x) { return std::tanh(x); }
    // Derivative of tanh(x) with respect to x, given activated output y = tanh(x)
    float tanh_derivative(float y) { return 1.0f - y * y; }

    // Encoder forward pass: input -> latent
    std::vector<float> encode(const std::vector<float>& input)
    {
        enc_activations[0] = input; // Store input

        // Propagate through encoder hidden layers
        for (size_t l = 0; l < encoderHiddenDims.size(); ++l) {
            const std::vector<float>& prev_layer_activations = enc_activations[l];
            std::vector<float>& current_layer_activations = enc_activations[l + 1];
            const std::vector<std::vector<float>>& weights = W_enc[l];
            const std::vector<float>& biases = b_enc[l];

            for (size_t i = 0; i < current_layer_activations.size(); ++i) {
                float sum = biases[i];
                for (size_t j = 0; j < prev_layer_activations.size(); ++j) {
                    sum += weights[i][j] * prev_layer_activations[j];
                }
                current_layer_activations[i] = tanh_activation(sum); // Tanh activation
            }
        }
        // Latent layer (last encoder layer, no activation here, or can be tanh)
        const std::vector<float>& prev_layer_activations = enc_activations[encoderHiddenDims.size()];
        std::vector<float>& latent_output = enc_activations[encoderHiddenDims.size() + 1];
        const std::vector<std::vector<float>>& weights = W_enc[encoderHiddenDims.size()];
        const std::vector<float>& biases = b_enc[encoderHiddenDims.size()];

        for (size_t i = 0; i < latent_output.size(); ++i) {
            float sum = biases[i];
            for (size_t j = 0; j < prev_layer_activations.size(); ++j) {
                sum += weights[i][j] * prev_layer_activations[j];
            }
            latent_output[i] = sum; // Linear activation for latent space
        }
        return latent_output;
    }

    // Decoder forward pass: latent -> output
    std::vector<float> decode(const std::vector<float>& latent_vec)
    {
        dec_activations[0] = latent_vec; // Store latent vector as input to decoder

        // Propagate through decoder hidden layers
        for (size_t l = 0; l < decoderHiddenDims.size(); ++l) {
            const std::vector<float>& prev_layer_activations = dec_activations[l];
            std::vector<float>& current_layer_activations = dec_activations[l + 1];
            const std::vector<std::vector<float>>& weights = W_dec[l];
            const std::vector<float>& biases = b_dec[l];

            for (size_t i = 0; i < current_layer_activations.size(); ++i) {
                float sum = biases[i];
                for (size_t j = 0; j < prev_layer_activations.size(); ++j) {
                    sum += weights[i][j] * prev_layer_activations[j];
                }
                current_layer_activations[i] = tanh_activation(sum); // Tanh activation
            }
        }
        // Output layer (last decoder layer, linear activation for SDF)
        const std::vector<float>& prev_layer_activations = dec_activations[decoderHiddenDims.size()];
        std::vector<float>& output = dec_activations[decoderHiddenDims.size() + 1];
        const std::vector<std::vector<float>>& weights = W_dec[decoderHiddenDims.size()];
        const std::vector<float>& biases = b_dec[decoderHiddenDims.size()];

        for (size_t i = 0; i < output.size(); ++i) {
            float sum = biases[i];
            for (size_t j = 0; j < prev_layer_activations.size(); ++j) {
                sum += weights[i][j] * prev_layer_activations[j];
            }
            output[i] = sum; // Linear activation for SDF output
        }
        return output;
    }

    // Full autoencoder prediction (encode then decode)
    std::vector<float> predict(const std::vector<float>& input) {
        std::vector<float> latent = encode(input);
        return decode(latent);
    }

    // Compute Mean Squared Error loss and gradients for the output layer
    float computeLossAndGradient(const std::vector<float>& input_sdf, std::vector<float>& grad_output_sdf)
    {
        // 1. Forward pass (encode then decode)
        std::vector<float> reconstructed_sdf = predict(input_sdf);

        // 2. Compute MSE Loss
        float totalLoss = 0.0f;
        grad_output_sdf.assign(inputDim, 0.0f); // Initialize gradients for output layer

        for (size_t i = 0; i < inputDim; ++i) {
            float error = reconstructed_sdf[i] - input_sdf[i];
            totalLoss += error * error; // Sum of squared errors

            // Gradient of MSE w.r.t. output: dLoss/d_output_i = 2 * (output_i - target_i)
            grad_output_sdf[i] = 2.0f * error;
        }

        return totalLoss / inputDim; // Average MSE loss
    }

    // Backpropagation for the entire autoencoder
    void backward(const std::vector<float>& grad_output_sdf, float learning_rate)
    {
        // --- Decoder Backpropagation ---
        std::vector<float> current_grad_dec = grad_output_sdf; // Gradient w.r.t. decoder's output (reconstructed SDF)

        // Iterate backward through decoder layers (from output to latent)
        for (int l = decoderHiddenDims.size(); l >= 0; --l) {
            const std::vector<float>& current_layer_activations = dec_activations[l + 1]; // Activated output of current layer
            const std::vector<float>& prev_layer_activations = dec_activations[l];        // Activated output of previous layer (input to current layer)

            std::vector<float> grad_prev_layer_raw(prev_layer_activations.size(), 0.0f); // Gradient w.r.t. raw sums of previous layer

            // If not the output layer, apply activation derivative
            if (l < decoderHiddenDims.size()) { // For hidden layers of decoder
                for (size_t i = 0; i < current_grad_dec.size(); ++i) {
                    current_grad_dec[i] *= tanh_derivative(current_layer_activations[i]); // Apply derivative of tanh
                }
            }
            // else: for output layer, activation is linear, derivative is 1, so current_grad_dec remains as is.

            // Update weights and biases for current decoder layer
            for (size_t i = 0; i < current_grad_dec.size(); ++i) { // Iterate through neurons in current layer
                for (size_t j = 0; j < prev_layer_activations.size(); ++j) {
                    W_dec[l][i][j] -= learning_rate * current_grad_dec[i] * prev_layer_activations[j];
                    grad_prev_layer_raw[j] += current_grad_dec[i] * W_dec[l][i][j]; // Accumulate for previous layer's raw sums
                }
                b_dec[l][i] -= learning_rate * current_grad_dec[i];
            }
            current_grad_dec = grad_prev_layer_raw; // Pass gradients backward
        }

        // After decoder backprop, current_grad_dec now holds gradients w.r.t. the latent vector.
        // This is the input to the decoder, and the output of the encoder.

        // --- Encoder Backpropagation ---
        std::vector<float> current_grad_enc = current_grad_dec; // Gradients w.r.t. latent vector (output of encoder)

        // Iterate backward through encoder layers (from latent to input)
        for (int l = encoderHiddenDims.size(); l >= 0; --l) {
            const std::vector<float>& current_layer_activations = enc_activations[l + 1]; // Activated output of current layer
            const std::vector<float>& prev_layer_activations = enc_activations[l];        // Activated output of previous layer (input to current layer)

            std::vector<float> grad_prev_layer_raw(prev_layer_activations.size(), 0.0f); // Gradient w.r.t. raw sums of previous layer

            // If not the latent layer, apply activation derivative
            if (l < encoderHiddenDims.size()) { // For hidden layers of encoder
                for (size_t i = 0; i < current_grad_enc.size(); ++i) {
                    current_grad_enc[i] *= tanh_derivative(current_layer_activations[i]); // Apply derivative of tanh
                }
            }
            // else: for latent layer, activation is linear, derivative is 1, so current_grad_enc remains as is.

            // Update weights and biases for current encoder layer
            for (size_t i = 0; i < current_grad_enc.size(); ++i) { // Iterate through neurons in current layer
                for (size_t j = 0; j < prev_layer_activations.size(); ++j) {
                    W_enc[l][i][j] -= learning_rate * current_grad_enc[i] * prev_layer_activations[j];
                    grad_prev_layer_raw[j] += current_grad_enc[i] * W_enc[l][i][j]; // Accumulate for previous layer's raw sums
                }
                b_enc[l][i] -= learning_rate * current_grad_enc[i];
            }
            current_grad_enc = grad_prev_layer_raw; // Pass gradients backward
        }
        // After this loop, current_grad_enc holds gradients w.r.t. the input layer,
        // which we typically don't update in an autoencoder.
    }

    // Visualization function (adapted from your original MLP)
    void visualize(zVector topLeft = zVector(50, 500, 0), float bboxWidth = 300.0f, float bboxHeight = 600.0f)
    {
        setup2d();

        // Calculate total number of layers for visualization
        // Input (1) + EncHidden + Latent (1) + DecHidden + Output (1)
        int totalLayers = 1 + encoderHiddenDims.size() + 1 + decoderHiddenDims.size() + 1;
        float nodeRadius = 5.0f;

        // Determine max nodes per layer for vertical spacing
        int maxNodesPerLayer = inputDim; // Start with input/output size
        for (int dim : encoderHiddenDims) maxNodesPerLayer = std::max(maxNodesPerLayer, dim);
        maxNodesPerLayer = std::max(maxNodesPerLayer, latentDim);
        for (int dim : decoderHiddenDims) maxNodesPerLayer = std::max(maxNodesPerLayer, dim);

        float layerSpacing = (totalLayers > 1) ? bboxWidth / (totalLayers - 1) : 0.0f;
        float verticalSpacing = (maxNodesPerLayer > 1) ? bboxHeight / (maxNodesPerLayer - 1) : 0.0f;

        std::vector<std::vector<zVector>> nodePositions(totalLayers);

        // Compute node positions for Encoder part (Input -> EncHidden -> Latent)
        // Layer 0: Input
        int current_layer_idx = 0;
        int numNodes = inputDim;
        float yStart = topLeft.y - 0.5f * (numNodes - 1) * verticalSpacing;
        for (int n = 0; n < numNodes; n++) {
            nodePositions[current_layer_idx].push_back(zVector(topLeft.x, yStart + n * verticalSpacing, 0));
        }

        // Encoder Hidden Layers
        for (size_t l = 0; l < encoderHiddenDims.size(); ++l) {
            current_layer_idx++;
            numNodes = encoderHiddenDims[l];
            yStart = topLeft.y - 0.5f * (numNodes - 1) * verticalSpacing;
            for (int n = 0; n < numNodes; n++) {
                nodePositions[current_layer_idx].push_back(zVector(topLeft.x + current_layer_idx * layerSpacing, yStart + n * verticalSpacing, 0));
            }
        }

        // Latent Layer
        current_layer_idx++;
        numNodes = latentDim;
        yStart = topLeft.y - 0.5f * (numNodes - 1) * verticalSpacing;
        for (int n = 0; n < numNodes; n++) {
            nodePositions[current_layer_idx].push_back(zVector(topLeft.x + current_layer_idx * layerSpacing, yStart + n * verticalSpacing, 0));
        }

        // Decoder Hidden Layers
        for (size_t l = 0; l < decoderHiddenDims.size(); ++l) {
            current_layer_idx++;
            numNodes = decoderHiddenDims[l];
            yStart = topLeft.y - 0.5f * (numNodes - 1) * verticalSpacing;
            for (int n = 0; n < numNodes; n++) {
                nodePositions[current_layer_idx].push_back(zVector(topLeft.x + current_layer_idx * layerSpacing, yStart + n * verticalSpacing, 0));
            }
        }

        // Output Layer
        current_layer_idx++;
        numNodes = inputDim; // Output is same dimension as input
        yStart = topLeft.y - 0.5f * (numNodes - 1) * verticalSpacing;
        for (int n = 0; n < numNodes; n++) {
            nodePositions[current_layer_idx].push_back(zVector(topLeft.x + current_layer_idx * layerSpacing, yStart + n * verticalSpacing, 0));
        }


        // Draw weight connections
        // Encoder connections
        for (size_t l = 0; l < W_enc.size(); ++l) { // W_enc.size() is encoderHiddenDims.size() + 1
            const std::vector<std::vector<float>>& weights = W_enc[l];
            const std::vector<float>& prev_activations = enc_activations[l]; // Use cached activations

            for (size_t i = 0; i < weights.size(); ++i) { // To neurons in current layer
                for (size_t j = 0; j < prev_activations.size(); ++j) { // From neurons in previous layer
                    float w = weights[i][j];
                    float val = std::clamp(w * 5.0f, -1.0f, 1.0f); // Amplify for visualization
                    float r, g, b;
                    getJetColor(val, r, g, b);

                    (std::abs(val) > 0.05) ? glColor3f(r, g, b) : glColor3f(0.8, 0.8, 0.8); // Dim inactive connections

                    drawLine(zVecToAliceVec(nodePositions[l][j]), zVecToAliceVec(nodePositions[l + 1][i]));
                }
            }
        }

        // Decoder connections
        for (size_t l = 0; l < W_dec.size(); ++l) { // W_dec.size() is decoderHiddenDims.size() + 1
            const std::vector<std::vector<float>>& weights = W_dec[l];
            const std::vector<float>& prev_activations = dec_activations[l]; // Use cached activations

            // Offset for nodePositions to align with the full autoencoder visualization
            int node_pos_offset = 1 + encoderHiddenDims.size(); // Input + EncHidden layers

            for (size_t i = 0; i < weights.size(); ++i) { // To neurons in current decoder layer
                for (size_t j = 0; j < prev_activations.size(); ++j) { // From neurons in previous decoder layer
                    float w = weights[i][j];
                    float val = std::clamp(w * 5.0f, -1.0f, 1.0f);
                    float r, g, b;
                    getJetColor(val, r, g, b);

                    (std::abs(val) > 0.05) ? glColor3f(r, g, b) : glColor3f(0.8, 0.8, 0.8);

                    drawLine(zVecToAliceVec(nodePositions[node_pos_offset + l][j]), zVecToAliceVec(nodePositions[node_pos_offset + l + 1][i]));
                }
            }
        }


        // Draw neuron activations (circles)
        // Encoder activations
        for (size_t l = 0; l < enc_activations.size(); ++l) {
            for (size_t i = 0; i < enc_activations[l].size(); ++i) {
                float act = enc_activations[l][i];
                // For visualization, clamp and normalize activation values
                float normalized_act = std::clamp(act, -1.0f, 1.0f);
                float r, g, b;
                getJetColor(normalized_act, r, g, b);
                glColor3f(r, g, b); // Color based on activation
                drawCircle(zVecToAliceVec(nodePositions[l][i]), nodeRadius, 12);
            }
        }
        // Decoder activations (excluding the latent layer, as it's covered by encoder activations)
        for (size_t l = 1; l < dec_activations.size(); ++l) { // Start from index 1 (first decoder hidden layer)
            int node_pos_offset = 1 + encoderHiddenDims.size(); // Input + EncHidden layers
            for (size_t i = 0; i < dec_activations[l].size(); ++i) {
                float act = dec_activations[l][i];
                float normalized_act = std::clamp(act, -1.0f, 1.0f);
                float r, g, b;
                getJetColor(normalized_act, r, g, b);
                glColor3f(r, g, b); // Color based on activation
                drawCircle(zVecToAliceVec(nodePositions[node_pos_offset + l][i]), nodeRadius, 12);
            }
        }

        restore3d();
    }
};


MLPAutoencoder mlp_autoencoder; // Global instance of the new autoencoder
std::vector<float> gradOutputSDF; // Gradient for the output SDF

// Global flags
bool train_mlp_autoencoder = false; // Renamed from train_mlp for clarity
bool opt_gd = false; // Kept for potential future use if GD is separate


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


//-------------------------------
// MVC
//-------------------------------
void setup()
{
    generateTrainingSet(); // Generate the SDF training data

    // --- MLP Autoencoder Initialization ---
    std::vector<int> encoder_hidden_dims = { 256, 128 }; // Example encoder hidden layers
    std::vector<int> decoder_hidden_dims = { 128, 256 }; // Example decoder hidden layers

    mlp_autoencoder = MLPAutoencoder(inputDim, encoder_hidden_dims, latentDim, decoder_hidden_dims);
    // --- End MLP Autoencoder Initialization ---

    // Initial decode to show something at start
    if (!sdfStack.empty()) {
        reconstructedSDF = mlp_autoencoder.predict(sdfStack[curSample]);
    }
    else {
        reconstructedSDF.assign(inputDim, 0.0f); // Assign zeros if no data
    }

    glLineWidth(1.5f);
}

void update(int value)
{
    if (train_mlp_autoencoder) {
        // Convert vec_t from sdfStack to std::vector<float> for MLP input
        std::vector<float> current_input_sdf = sdfStack[curSample];

        float loss = mlp_autoencoder.computeLossAndGradient(current_input_sdf, gradOutputSDF);
        mlp_autoencoder.backward(gradOutputSDF, 0.001); // Use a learning rate

        // Update reconstructed SDF for visualization
        reconstructedSDF = mlp_autoencoder.predict(current_input_sdf);

        std::cout << "MLP Autoencoder Training Loss: " << loss << std::endl;

        // Move to the next sample for continuous training
        curSample = (curSample + 1) % sdfStack.size();
    }
}

void draw()
{
    backGround(0.9);
    drawGrid(50);

    drawSDFGrid(sdfStack[curSample], zVector(0, 0, 0));            // Input SDF
    drawSDFGrid(reconstructedSDF, zVector(RES + 4, 0, 0));         // Reconstruction

    // Visualize the MLP Autoencoder
    mlp_autoencoder.visualize(zVector(50, 500, 0), 300, 600);
}

vec_t latentVec(latentDim, 0.0f); // starts at origin
int latentStepIndex = 0;
float latentStepSize = 0.1f;

void keyPress(unsigned char k, int xm, int ym)
{
    if (k == 'n') // 'n' key to move to the next sample
    {
        curSample = (curSample + 1) % sdfStack.size();
        reconstructedSDF = mlp_autoencoder.predict(sdfStack[curSample]);
    }

    if (k == 't') // 't' key to toggle MLP autoencoder training
    {
        train_mlp_autoencoder = !train_mlp_autoencoder;
        opt_gd = false; // Ensure GD is off if MLP is on
        std::cout << "MLP Autoencoder Training Toggled: " << (train_mlp_autoencoder ? "ON" : "OFF") << std::endl;
    }

    if (k == 'e') // 'e' key to encode the current SDF sample and print its latent vector
    {
        if (!sdfStack.empty()) {
            std::vector<float> current_input_sdf = sdfStack[curSample];
            std::vector<float> latent = mlp_autoencoder.encode(current_input_sdf);

            std::cout << "Latent: ";
            for (auto& v : latent) printf(" %.3f", v);
            std::cout << "\n";
        }
    }

    if (k == 'l') // 'l' key to walk through latent dimensions
    {
        // Increment the current latent dimension by latentStepSize
        latentVec[latentStepIndex] += latentStepSize;

        // Use the decoder part of the MLP Autoencoder to reconstruct from the modified latent vector
        reconstructedSDF = mlp_autoencoder.decode(latentVec);

        std::cout << "Latent dim " << latentStepIndex << " += " << latentStepSize << "\n";
        latentStepIndex = (latentStepIndex + 1) % latentDim;  // Move to the next dimension for the next 'l' press
    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_
