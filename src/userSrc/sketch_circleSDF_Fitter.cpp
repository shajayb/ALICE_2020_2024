#define _MAIN_
#ifdef _MAIN_

#include "main.h"

// zSpace Core
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>
using namespace zSpace;

#include <fstream>
#include <sstream>

Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

zVector AliceVecToZvec(Alice::vec& in)
{
    return zVector(in.x, in.y, in.z);
}

inline zVector zMax(zVector& a, zVector& b)
{
    return zVector(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

inline zVector zMin(zVector& a, zVector& b)
{
    return zVector(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

#include "scalarField.h" //// two functiosn must be turned on in scalarfIELD.H for sketch_circleSDF_fitter.cpp


//inline float smin(float a, float b, float k)
//{
//    float h = std::max(k - fabs(a - b), 0.0f) / k;
//    return std::min(a, b) - h * h * k * 0.25f;
//}
std::vector<zVector> polygon;
std::vector<zVector> sdfCenters;
std::vector<float>predictedRadii;


ScalarField2D myField;

int numCircles = 16;
double thresholdValue = 0.1;
double radius = 8.0;
double smoothK = 8.0;
bool vizField = false;

// ----------------- MLP 
//std::vector<zVector> samplePts;
std::vector<float> sdfGT;
std::vector<zVector> fittedCenters;
std::vector<float> fittedRadii;
#define NUM_CENTERS 16
//std::vector<zVector> polygon;
double threshold;


//-------------------------------
// Utility
//-------------------------------
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

//-------------------------------
// Circle SDF Blending
//-------------------------------
inline float circleSDF(zVector& pt, zVector& center, float r)
{
    return pt.distanceTo(zVector(center)) - r; // signed: negative inside, 0 on boundary, positive outside
}

inline float blendCircleSDFs(zVector& pt, std::vector<zVector>& centers, float r, float k)
{
    if (centers.empty()) return 1e6f;

    float d = circleSDF(pt, centers[0], r);
    for (int i = 1; i < centers.size(); i++)
    {
        float d_i = circleSDF(pt, centers[i], r);
        d = smin(d, d_i, k);  //std::min(d,d_i) smooth union of signed distances
    }

    return d;
}

inline float blendCircleSDFs(zVector& pt, std::vector<zVector>& centers, vector<float>& radii, float k)
{
    if (centers.empty()) return 1e6f;

    float d = circleSDF(pt, centers[0], radii[0]);
    for (int i = 1; i < centers.size(); i++)
    {
        float d_i = circleSDF(pt, centers[i], radii[i]);
        d = smin(d, d_i, k);  // std::min(d, d_i);//smooth union of signed distances
    }

    return d;
}

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

float polygonSDF(zVector& pt, std::vector<zVector>& poly)
{
    float minDist = 1e6f;
    for (int i = 0; i < poly.size(); i++)
    {
        zVector a = poly[i];
        zVector b = poly[(i + 1) % poly.size()];
        zVector ab = b - a;
        zVector ap = pt - a;

        float t = std::clamp((ap * ab) / (ab * ab), 0.0f, 1.0f);
        zVector closest = a + ab * t;
        float dist = pt.distanceTo(closest);
        minDist = std::min(minDist, dist);
    }

    bool inside = isInsidePolygon(pt, poly);
    return inside ? -minDist : minDist;
}


//-------------------------------
// Fit & Field
//-------------------------------

void initilaiseCircleCenters()
{
    sdfCenters.clear();
    for (int c = 0; c < numCircles; c++) sdfCenters.push_back(zVector(0, 0, 0));
}

void buildScalarField(int opt = 1)
{
    myField.clearField();
    for (int i = 0; i < ScalarField2D::RES; i++)
    {
        for (int j = 0; j < ScalarField2D::RES; j++)
        {
            zVector pt = myField.gridPoints[i][j];
            float d = (opt == 1) ? polygonSDF(pt, polygon) : blendCircleSDFs(pt, sdfCenters, radius, smoothK);// polygonSDF(pt,polygon);
            myField.field[i][j] = d; // signed SDF directly
        }
    }

    myField.rescaleFieldToRange(-1, 1);
}

std::vector<zVector> trainingSamples;

void samplePoints()
{
    trainingSamples.clear();

    for (float x = -50; x <= 50; x += 5.0f)
    {
        for (float y = -50; y <= 50; y += 5.0f)
        {
            zVector pt(x, y, 0);
            if (isInsidePolygon(pt, polygon))
            {
                trainingSamples.push_back(pt);
                sdfGT.push_back(polygonSDF(pt, polygon));//mlp
            }
        }
    }

    cout << " training samples " << trainingSamples.size() << endl;
}

float computeTotalError()
{
    float err = 0.0f;
    for (auto& pt : trainingSamples)
    {
        float pred = blendCircleSDFs(pt, sdfCenters, radius, smoothK);
        float actual = polygonSDF(pt, polygon);
        float diff = pred - actual;
        err += diff * diff;
    }
    return err / trainingSamples.size();
}

void optimiseCircleCenters(int iterations = 20, float step = 0.001f)
{
    const float eps = 1e-3f;

    //for (int it = 0; it < iterations; it++)
    {
        for (int c = 0; c < sdfCenters.size(); c++)
        {
            zVector center = sdfCenters[c];
            zVector grad(0, 0, 0);

            for (int d = 0; d < 2; d++)
            {
                zVector dir(0, 0, 0);
                if (d == 0) dir.x = eps;
                if (d == 1) dir.y = eps;

                std::vector<zVector> testCenters = sdfCenters;
                testCenters[c] = center + dir;
                float E_plus = 0;
                int i = 0;
                for (auto& pt : trainingSamples)
                {
                    float pred = blendCircleSDFs(pt, testCenters, radius, smoothK);
                    float actual = sdfGT[i++]; //polygonSDF(pt, polygon);

                    float diff = pred - actual;
                    E_plus += diff * diff;
                }

                testCenters[c] = center - dir;
                float E_minus = 0;
                i = 0;
                for (auto& pt : trainingSamples)
                {
                    float pred = blendCircleSDFs(pt, testCenters, radius, smoothK);
                    float actual = sdfGT[i++];// polygonSDF(pt, polygon);
                    float diff = pred - actual;
                    E_minus += diff * diff;
                }

                float g = (E_plus - E_minus) / (2 * eps);
                if (d == 0) grad.x = g;
                if (d == 1) grad.y = g;
            }

            // Update

            sdfCenters[c] = center - grad * step;
            sdfCenters[c].x = std::clamp(sdfCenters[c].x, -50.0f, 50.0f);
            sdfCenters[c].y = std::clamp(sdfCenters[c].y, -50.0f, 50.0f);

        }


        std::cout << "Iteration " << " error: " << computeTotalError() << std::endl;
    }
}

// ---------------- MLP

#include <vector>
#include <cmath> // For std::tanh
#include <random> // For std::default_random_engine, std::normal_distribution
#include <iostream> // For std::cout (for debugging, if needed)

// Assuming these are defined elsewhere in your project
// zVector is likely a 3D vector class
// zVector pt.distanceTo(zVector)
// isInsidePolygon(zVector&, std::vector<zVector>&)
// blendCircleSDFs(zVector&, std::vector<zVector>&, std::vector<float>&, float)
// sdfGT, samplePts, trainingSamples, NUM_SDF, fittedCenters, fittedRadii, radius, smoothK, polygon
// are global or passed in as needed.
// For this MLP class, I'll assume they are accessible from the global scope or passed as parameters.

// Dummy definitions for compilation purposes if not available:
// #define NUM_SDF 16
// zVector class needs to be defined if not already.
// For example:
// struct zVector {
//     float x, y, z;
//     zVector(float x_ = 0, float y_ = 0, float z_ = 0) : x(x_), y(y_), z(z_) {}
//     float distanceTo(const zVector& other) const {
//         return std::sqrt(std::pow(x - other.x, 2) + std::pow(y - other.y, 2) + std::pow(z - other.z, 2));
//     }
//     zVector operator+(const zVector& other) const { return zVector(x + other.x, y + other.y, z + other.z); }
//     zVector operator-(const zVector& other) const { return zVector(x - other.x, y - other.y, z - other.z); }
//     zVector operator*(float s) const { return zVector(x * s, y * s, z * s); }
// };
// std::vector<zVector> fittedCenters;
// std::vector<float> fittedRadii;
// float radius = 8.0f;
// float smoothK = 3.0f;
// std::vector<float> sdfGT;
// std::vector<zVector> samplePts; // Assuming this is used for MLP's internal sample points
// std::vector<zVector> trainingSamples; // Assuming this is used for MLP's internal sample points
// std::vector<zVector> polygon;
// bool isInsidePolygon(zVector& p, std::vector<zVector>& poly) { return true; } // Dummy
// float blendCircleSDFs(zVector& pt, std::vector<zVector>& centers, std::vector<float>& radii, float k) { return 0.0f; } // Dummy


class MLP
{
public:
    int inputDim, outputDim;
    std::vector<int> hiddenLayerDims; // Stores dimensions of each hidden layer
    int numHiddenLayers;

    // Weights and biases for each layer
    // W[0] connects input to first hidden layer
    // W[i] connects hidden layer i to hidden layer i+1 (or to output if i is last hidden)
    std::vector<std::vector<std::vector<float>>> W; // W[layer_idx][neuron_idx_current_layer][neuron_idx_prev_layer]
    std::vector<std::vector<float>> b;              // b[layer_idx][neuron_idx_current_layer]

    // Activations for each layer (including input and output)
    std::vector<std::vector<float>> activations; // activations[layer_idx][neuron_idx]
    // activations[0] = input
    // activations[1...numHiddenLayers] = hidden layer activations
    // activations[numHiddenLayers + 1] = output layer activations

    float prevLoss = 0; // Not directly used in core MLP logic but kept as per original

    MLP() {} // Default constructor

    MLP(int inDim, const std::vector<int>& hDims, int outDim)
        : inputDim(inDim), outputDim(outDim), hiddenLayerDims(hDims), numHiddenLayers(hDims.size())
    {
        // Resize activations vector to hold input, hidden layers, and output layer
        // Layer 0: input layer
        // Layer 1 to numHiddenLayers: hidden layers
        // Layer numHiddenLayers + 1: output layer
        activations.resize(numHiddenLayers + 2);

        // Initialize input layer activation size
        activations[0].resize(inputDim);

        // Initialize weights and biases vectors
        W.resize(numHiddenLayers + 1); // numHiddenLayers + 1 sets of weights (input->h1, h1->h2, ..., hN->output)
        b.resize(numHiddenLayers + 1); // numHiddenLayers + 1 sets of biases (for h1, h2, ..., output)

        std::default_random_engine eng;
        std::normal_distribution<float> dist(0.0, 0.1);

        // Initialize Input to First Hidden Layer
        int prevLayerDim = inputDim;
        if (numHiddenLayers > 0)
        {
            int currentLayerDim = hiddenLayerDims[0];
            W[0].resize(currentLayerDim, std::vector<float>(prevLayerDim));
            b[0].resize(currentLayerDim);
            activations[1].resize(currentLayerDim); // Size for first hidden layer

            for (auto& row : W[0])
                for (auto& val : row)
                    val = dist(eng);
            for (auto& val : b[0])
                val = dist(eng);

            prevLayerDim = currentLayerDim;
        }


        // Initialize Hidden Layers
        for (int l = 1; l < numHiddenLayers; ++l)
        {
            int currentLayerDim = hiddenLayerDims[l];
            W[l].resize(currentLayerDim, std::vector<float>(prevLayerDim));
            b[l].resize(currentLayerDim);
            activations[l + 1].resize(currentLayerDim); // Size for current hidden layer

            for (auto& row : W[l])
                for (auto& val : row)
                    val = dist(eng);
            for (auto& val : b[l])
                val = dist(eng);

            prevLayerDim = currentLayerDim;
        }

        // Initialize Last Hidden Layer to Output Layer (if hidden layers exist)
        if (numHiddenLayers > 0) {
            W[numHiddenLayers].resize(outputDim, std::vector<float>(prevLayerDim));
            b[numHiddenLayers].resize(outputDim);
            activations[numHiddenLayers + 1].resize(outputDim); // Size for output layer

            for (auto& row : W[numHiddenLayers])
                for (auto& val : row)
                    val = dist(eng);
            for (auto& val : b[numHiddenLayers])
                val = dist(eng);
        }
        else { // No hidden layers, direct input to output
            W[0].resize(outputDim, std::vector<float>(inputDim)); // W[0] connects input to output
            b[0].resize(outputDim);
            activations[1].resize(outputDim); // activations[1] is now the output layer

            for (auto& row : W[0])
                for (auto& val : row)
                    val = dist(eng);
            for (auto& val : b[0])
                val = dist(eng);
        }
    }

    std::vector<float> forward(const std::vector<float>& x)
    {
        // Store input in the first activation layer
        activations[0] = x;

        // Propagate through hidden layers
        for (int l = 0; l < numHiddenLayers; ++l)
        {
            const std::vector<float>& prev_layer_activations = activations[l];
            std::vector<float>& current_layer_activations = activations[l + 1];
            const std::vector<std::vector<float>>& weights = W[l];
            const std::vector<float>& biases = b[l];
            int current_layer_dim = (l == numHiddenLayers - 1) ? outputDim : hiddenLayerDims[l]; // Last hidden layer to output

            for (int i = 0; i < current_layer_dim; ++i)
            {
                float sum = biases[i];
                for (int j = 0; j < prev_layer_activations.size(); ++j)
                {
                    sum += weights[i][j] * prev_layer_activations[j];
                }
                current_layer_activations[i] = std::tanh(sum); // Tanh activation for hidden layers
            }
        }

        // Output layer (no activation, or specific activation if needed for task)
        // If numHiddenLayers is 0, this calculates output directly from input
        const std::vector<float>& prev_layer_activations = activations[numHiddenLayers];
        std::vector<float>& output_layer_activations = activations[numHiddenLayers + 1];
        const std::vector<std::vector<float>>& weights = W[numHiddenLayers]; // W[numHiddenLayers] connects last hidden to output
        const std::vector<float>& biases = b[numHiddenLayers];

        for (int i = 0; i < outputDim; ++i)
        {
            float sum = biases[i];
            for (int j = 0; j < prev_layer_activations.size(); ++j)
            {
                sum += weights[i][j] * prev_layer_activations[j];
            }
            output_layer_activations[i] = sum; // Linear activation for output layer
        }

        return output_layer_activations;
    }

    // see : https://g.co/gemini/share/4882f6e2724c for the explanation of the correction
    float computeLossAndGradient(const std::vector<float>& x, std::vector<zVector>& polygon, std::vector<float>& gradOut)
    {
        // Ensure gradOut is correctly sized
        gradOut.assign(outputDim, 0.0f);

        // 1. Forward pass to get MLP's predicted parameters
        // This 'output' vector contains the raw (unclamped/untransformed) values
        // that the MLP predicts for x, y, and potentially radius for each circle.
        auto raw_mlp_output = forward(x);

        std::vector<zVector> centers(NUM_CENTERS);
        std::vector<float> radii(NUM_CENTERS);

        // Interpret MLP output as circle parameters.
        // Apply any necessary transformations (e.g., clamping for robustness, but ideally learned)
        // For now, let's keep the radius fixed as in your problem statement
        for (int i = 0; i < NUM_CENTERS; i++)
        {
            // For x, y, consider if they need to be clamped or scaled.
            // For now, using raw output for gradient calculation, but apply insidePolygon check for the actual usage.
            // Note: Your original code had i * 3 + 0/1/2 for output, but only two outputs (x,y) are used per circle.
            // If outputDim is NUM_SDF * 2, then use i * 2. If it's NUM_SDF * 3 (for radius), use i * 3.
            // Assuming outputDim is NUM_SDF * 2 for x,y coordinates only.
            centers[i] = zVector(raw_mlp_output[i * 2 + 0], raw_mlp_output[i * 2 + 1], 0);

            // This check is for visualization/usage, the gradient should ideally flow through if possible.
            // For now, we'll keep it as is, but it can create discontinuous gradients.
            // fittedCenters should ideally be initialized better and perhaps be a member of the MLP class or passed directly
            // centers[i] = (isInsidePolygon(pt, polygon)) ? pt : fittedCenters[i]; // This line is problematic for gradients
            radii[i] = radius; // Fixed radius as per your current setup. If MLP should learn, remove this.
        }

        // Save for visualization (fittedCenters and fittedRadii must be accessible globally or as class members)
        fittedCenters = centers;
        fittedRadii = radii;

        float totalLoss = 0.0f;

        // 2. Calculate the loss based on the predicted circle parameters
        // This part iterates through sample points and calculates the SDF loss.
        // We also need to compute the gradient of this loss with respect to each circle parameter (center.x, center.y, radius).

        // Numerical gradient of the *SDF blending function* with respect to circle parameters
        // This is the crucial part for the MLP's backprop.
        float eps_param = 0.01f; // Epsilon for numerical gradient of SDF w.r.t. circle params

        for (int s = 0; s < trainingSamples.size(); ++s)
        {
            zVector current_sample_pt = trainingSamples[s];
            float gt_sdf = sdfGT[s];

            // Calculate the predicted blended SDF with current parameters
            float predicted_sdf = blendCircleSDFs(current_sample_pt, centers, radii, smoothK);
            float error = predicted_sdf - gt_sdf;
            totalLoss += error * error; // Sum of squared errors

            // Now, compute the gradient of `error*error` (or `predicted_sdf`) with respect to each
            // of the *output parameters* (x, y for centers, and radius) that come from the MLP.

            zVector gradient;
            for (int i = 0; i < NUM_CENTERS; ++i) // For each circle
            {
                // Gradient with respect to center.x of circle i
                std::vector<zVector> centers_plus_x = centers;
                centers_plus_x[i].x += eps_param;
                float predicted_sdf_plus_x = blendCircleSDFs(current_sample_pt, centers_plus_x, radii, smoothK);
                float grad_sdf_cx = (predicted_sdf_plus_x - predicted_sdf) / eps_param;
                // Chain rule: dLoss/d_param = dLoss/d_predicted_sdf * d_predicted_sdf/d_param
                // dLoss/d_predicted_sdf = 2 * (predicted_sdf - gt_sdf)
                gradOut[i * 2 + 0] += 2 * error * grad_sdf_cx; // Accumulate gradient for output x

                // Gradient with respect to center.y of circle i
                std::vector<zVector> centers_plus_y = centers;
                centers_plus_y[i].y += eps_param;
                float predicted_sdf_plus_y = blendCircleSDFs(current_sample_pt, centers_plus_y, radii, smoothK);
                float grad_sdf_cy = (predicted_sdf_plus_y - predicted_sdf) / eps_param;
                gradOut[i * 2 + 1] += 2 * error * grad_sdf_cy; // Accumulate gradient for output y

                // Gradient with respect to radius of circle i (if it were learned by MLP)
                // Currently, radius is fixed. If MLP were to learn radius, uncomment and adapt:
                /*
                std::vector<float> radii_plus_r = radii;
                radii_plus_r[i] += eps_param;
                float predicted_sdf_plus_r = blendCircleSDFs(current_sample_pt, centers, radii_plus_r, smoothK);
                float grad_sdf_r = (predicted_sdf_plus_r - predicted_sdf) / eps_param;
                gradOut[i * 3 + 2] += 2 * error * grad_sdf_r; // Accumulate gradient for output radius
                */

                gradient.x = grad_sdf_cx;
                gradient.y = grad_sdf_cy;

                glColor3f(0.5, 0, 0.5);
                drawLine(zVecToAliceVec(current_sample_pt), zVecToAliceVec(current_sample_pt + gradient * 2));
            }

            glColor3f(0, 0.8, 0);
            zVector zPoint = current_sample_pt + zVector(0, 0, 1) * (error * error) * 0.05;

            drawLine(zVecToAliceVec(current_sample_pt), zVecToAliceVec(zPoint));

        }

        // Normalize gradients by sample count for average gradient
        for (float& g : gradOut)
        {
            g /= trainingSamples.size();
        }

        return totalLoss / trainingSamples.size(); // Return average loss
    }

    // see : https://g.co/gemini/share/4882f6e2724c for the explanation of the correction
    void backward(const std::vector<float>& gradOut, float lr)
    {
        // raw_grad_outputs corresponds to dLoss/d_raw_output_values
        std::vector<float> current_grad_output = gradOut; // This will be propagated backward

        // Gradients for Output Layer weights (W[numHiddenLayers]) and biases (b[numHiddenLayers])
        // Iterate backward from the output layer
        int current_layer_idx = numHiddenLayers; // Index for W and b for the last connection (hidden_N -> output)
        int prev_layer_idx = numHiddenLayers;    // Index for activations of the layer before the output (hidden_N)

        std::vector<float>& output_layer_activations_raw = activations[numHiddenLayers + 1]; // Raw outputs
        const std::vector<float>& prev_layer_activations = activations[prev_layer_idx]; // Activations from last hidden layer (or input if no hidden layers)

        std::vector<float> grad_prev_layer_raw(prev_layer_activations.size(), 0.0f); // Accumulate gradients for previous layer's raw sums

        for (int i = 0; i < outputDim; ++i) // Iterate through output neurons
        {
            // Update weights connecting previous layer to this (output) layer
            // dLoss/dW_ji = dLoss/d_current_output_i * d_current_output_i/dW_ji = current_grad_output[i] * prev_layer_activations[j]
            for (int j = 0; j < prev_layer_activations.size(); ++j)
            {
                W[current_layer_idx][i][j] -= lr * current_grad_output[i] * prev_layer_activations[j];
            }

            // Update biases for this (output) layer
            // dLoss/db_i = dLoss/d_current_output_i * 1 = current_grad_output[i]
            b[current_layer_idx][i] -= lr * current_grad_output[i];

            // Accumulate gradients for the previous layer's raw sums (before activation)
            // dLoss/d_raw_prev_layer_j contribution from current_output_i
            // = dLoss/d_current_output_i * d_current_output_i/d_raw_prev_layer_j
            // = current_grad_output[i] * W[current_layer_idx][i][j]
            for (int j = 0; j < prev_layer_activations.size(); ++j)
            {
                grad_prev_layer_raw[j] += current_grad_output[i] * W[current_layer_idx][i][j];
            }
        }

        // Backpropagate through hidden layers
        for (int l = numHiddenLayers - 1; l >= 0; --l) // Iterate from last hidden layer backward to first
        {
            // Apply activation function derivative for the layer we just backpropagated *from*
            // The activations for this layer are at activations[l+1] (unless it's the input layer)
            const std::vector<float>& current_layer_activations = activations[l + 1]; // These are the activated outputs of the current layer

            // dLoss/d_activated_current_layer_i = dLoss/d_raw_current_layer_i * d(tanh(raw_current_layer_i))/d_raw_current_layer_i
            std::vector<float> grad_current_layer_activated(current_layer_activations.size());
            for (int i = 0; i < current_layer_activations.size(); ++i)
            {
                // Use the grad_prev_layer_raw from the *next* layer's calculation, which is the gradient w.r.t. the raw sums of *this* layer
                grad_current_layer_activated[i] = grad_prev_layer_raw[i] * (1.0f - current_layer_activations[i] * current_layer_activations[i]);
            }

            // Now update weights and biases for current layer
            current_layer_idx = l;
            prev_layer_idx = l; // Activations of the layer *before* this one (activations[l])

            const std::vector<float>& prev_layer_activations_for_weights = activations[prev_layer_idx];

            // If this is the first hidden layer (l=0), prev_layer_activations_for_weights is the input
            // If it's a subsequent hidden layer, it's the previous hidden layer's activations

            grad_prev_layer_raw.assign(prev_layer_activations_for_weights.size(), 0.0f);
            //if (l > 0) { // For hidden layers
            //    grad_prev_layer_raw.assign(prev_layer_activations_for_weights.size(), 0.0f); // Reset for next iteration
            //}
            //else { 
            //    /* For input layer (l=0), we don't need grad_prev_layer_raw for previous layer (input)
            //     This is the last step of backprop, no more layers to backpropagate to.
            //     We're calculating gradients for W[0] and b[0]*/
            //    grad_prev_layer_raw.assign(prev_layer_activations_for_weights.size(), 0.0f);
            //}


            for (int i = 0; i < grad_current_layer_activated.size(); ++i) // Iterate through neurons in current layer
            {
                // Update weights connecting previous layer to current layer
                // dLoss/dW_ji = dLoss/d_activated_current_layer_i * prev_layer_activations_for_weights[j]
                for (int j = 0; j < prev_layer_activations_for_weights.size(); ++j)
                {
                    W[current_layer_idx][i][j] -= lr * grad_current_layer_activated[i] * prev_layer_activations_for_weights[j];
                }

                // Update biases for current layer
                // dLoss/db_i = dLoss/d_activated_current_layer_i
                b[current_layer_idx][i] -= lr * grad_current_layer_activated[i];

                // Accumulate gradients for the *previous* layer's raw sums (for the next iteration of backprop)
                if (l > 0) // Only if there's a previous layer to backpropagate to
                {
                    for (int j = 0; j < prev_layer_activations_for_weights.size(); ++j)
                    {
                        grad_prev_layer_raw[j] += grad_current_layer_activated[i] * W[current_layer_idx][i][j];
                    }
                }
            }
        }
    }

    //-----------

    void visualize(zVector topLeft = zVector(50, 450, 0), float bboxWidth = 400.0f, float bboxHeight = 300.0f)
    {
        setup2d();

        int numLayers = activations.size();
        float nodeRadius = 5.0f;

        // Get max nodes per layer to compute spacing
        int maxNodesPerLayer = 0;
        for (auto& layer : activations)
        {
            maxNodesPerLayer = std::max(maxNodesPerLayer, (int)layer.size());
        }

        float layerSpacing = (numLayers > 1) ? bboxWidth / (numLayers - 1) : 0.0f;
        float verticalSpacing = (maxNodesPerLayer > 1) ? bboxHeight / (maxNodesPerLayer - 1) : 0.0f;

        std::vector<std::vector<zVector>> nodePositions(numLayers);

        // Compute node positions
        for (int l = 0; l < numLayers; l++)
        {
            int numNodes = activations[l].size();
            float yStart = topLeft.y - 0.5f * (numNodes - 1) * verticalSpacing;

            for (int n = 0; n < numNodes; n++)
            {
                float x = topLeft.x + l * layerSpacing;
                float y = yStart + n * verticalSpacing;
                nodePositions[l].push_back(zVector(x, y, 0));
            }
        }

        // Draw weight connections (only color lines from active neurons)
        for (int l = 0; l < numLayers - 1; l++)
        {
            int fromSize = activations[l].size();
            int toSize = activations[l + 1].size();

            for (int i = 0; i < fromSize; i++)
            {
                float activation = activations[l][i];

                for (int j = 0; j < toSize; j++)
                {
                    float w = W[l][j][i];  // correct index order
                    float val = std::clamp(w * 5.0f, -1.0f, 1.0f);  // amplify small weights
                    float r, g, b;
                    getJetColor(val, r, g, b);

                    (val > 0.9) ? glColor3f(r, g, b) : glColor3f(0.8, 0.8, 0.8);

                    drawLine(zVecToAliceVec(nodePositions[l][i]), zVecToAliceVec(nodePositions[l + 1][j]));
                }
            }
        }


        // Draw neuron activations
        for (int l = 0; l < numLayers; l++)
        {
            for (int i = 0; i < activations[l].size(); i++)
            {
                float act = std::tanh(activations[l][i]);
                float r, g, b;
                getJetColor(act, r, g, b);
                //glColor3f(r, g, b);
                glColor3f(0, 0, 0);

                drawCircle(zVecToAliceVec(nodePositions[l][i]), nodeRadius, 12);
            }
        }

        restore3d();
    }



};






bool train = false;
std::vector<float> input(numCircles * 2, 0.0f);
std::vector<float> gradOut;
MLP mlp;
ScalarField2D sf;

std::vector<float> mlp_input_data;
//std::vector<float> gradOut;

// Global flags
bool train_mlp = false;
bool opt_gd = false;


//-------------------------------
// Visualisation
//-------------------------------
void drawPolygon()
{
    glColor3f(0, 0, 0);
    for (int i = 0; i < polygon.size(); i++)
    {
        int j = (i + 1) % polygon.size();
        drawLine(zVecToAliceVec(polygon[i]), zVecToAliceVec(polygon[j]));
    }
}

void drawCircles()
{
    glColor3f(0, 0, 1);
    for (auto& c : sdfCenters)
    {
        drawCircle(zVecToAliceVec(c), radius, 32);
        drawPoint(zVecToAliceVec(c));
    }
}

//-------------------------------
// MVC
//-------------------------------
void setup()
{
    loadPolygonFromCSV("data/polygon.txt");
    initilaiseCircleCenters();       // initial greedy placement (for blue circles)
    samplePoints();          // prepare training set (trainingSamples and sdfGT)
    optimiseCircleCenters(); // run initial gradient descent (for blue circles)
    buildScalarField();      // update field based on current sdfCenters

    S.numSliders = 0;
    S.addSlider(&thresholdValue, "iso");
    S.sliders[0].maxVal = 1;
    S.sliders[0].minVal = -1.0;

    S.addSlider(&radius, "r");
    S.sliders[1].maxVal = 20;

    S.addSlider(&smoothK, "k");
    S.sliders[2].maxVal = 10;

    B = *new ButtonGroup(Alice::vec(50, 800, 0));
    B.addButton(&vizField, "field");

    // --- MLP Initialization (Properly done ONCE here) ---
    int input_dim = NUM_CENTERS * 2; // Input is a fixed vector of zeros, matching mlp_input_data below
    int output_dim = NUM_CENTERS * 2; // Output is x,y for each of NUM_SDF circles
    std::vector<int> hidden_dims = { 32, 32 }; // Example hidden layers

    // Call the initialize method on the global 'mlp' object.
    // This correctly resizes and populates its internal vectors.
    mlp = MLP(input_dim, hidden_dims, output_dim); // Assuming MLP has an `initialize` method

    // Resize the global input data vector for the MLP
    mlp_input_data.assign(input_dim, 1.0f); // Initialize with zeros, size must match input_dim
    // --- End MLP Initialization ---


    glLineWidth(1.5f);
    //glDisable(GL_LINE_SMOOTH); // Avoid over-blending
    //glDisable(GL_LINE_STIPPLE); // Avoid over-blending

}

bool opt = false;
void update(int value)
{
    if (opt_gd) optimiseCircleCenters(); // GD optimization for blue circles

    //if (train_mlp) // Check the MLP training flag
    //{
    //    //for (auto& val : mlp_input_data)
    //    //    val = ofRandom(-1.0f, 1.0f);  // Inject noise

    //    float loss = mlp.computeLossAndGradient(mlp_input_data, polygon, gradOut);
    //    mlp.backward(gradOut, 0.01); // Use a learning rate
    //    // fittedCenters is updated within computeLossAndGradient, so it's ready for draw()
    //    std::cout << "MLP Training Loss: " << loss << std::endl;

    //    sdfCenters.clear();
    //    sdfCenters = fittedCenters;
    //}

    buildScalarField(); // Rebuild scalar field based on current `sdfCenters` (GD)
    // Note: `fittedCenters` (MLP) are drawn separately, but not used for `myField` directly.
    // You might want to build the field based on `fittedCenters` if MLP is active.
}
void draw()
{
    backGround(1.0);
    //drawGrid(50);

    if (train_mlp) // Check the MLP training flag
    {
        //for (auto& val : mlp_input_data)
        //    val = ofRandom(-1.0f, 1.0f);  // Inject noise

        float loss = mlp.computeLossAndGradient(mlp_input_data, polygon, gradOut);
        mlp.backward(gradOut, 0.01); // Use a learning rate
        // fittedCenters is updated within computeLossAndGradient, so it's ready for draw()
        std::cout << "MLP Training Loss: " << loss << std::endl;

        sdfCenters.clear();
        sdfCenters = fittedCenters;
    }


    drawPolygon();
    //drawCircles();

    /*glPointSize(5);
    for (auto& pt : candidatePts)drawPoint(zVecToAliceVec(pt));
    glPointSize(1);*/
    //mlp
    glColor3f(1, 0, 0);
    for (auto& pt : fittedCenters)drawCircle(zVecToAliceVec(pt), radius * 0.5, 32);

    glPushMatrix();
    //glTranslatef(100, 100, 0);
    if (vizField)myField.drawFieldPoints();
    glPopMatrix();

    mlp.visualize(zVector(50, 500, 0), 300, 600);

    //
    buildScalarField(0);

    glPushMatrix();
    glTranslatef(100, 0, 0);
    drawPolygon();
    myField.drawIsocontours(thresholdValue, true);
    if (vizField)myField.drawFieldPoints();
    drawCircles();
    glPopMatrix();


}

void keyPress(unsigned char k, int xm, int ym)
{
    if (k == 'r')
    {
        opt_gd = !opt_gd; // Toggle GD optimizer
        train_mlp = false; // Turn off MLP training
    }

    if (k == 't')
    {
        // --- CRITICAL CHANGE: NO MLP RE-INITIALIZATION HERE ---
        // Just toggle the training flag. MLP is already initialized ONCE in setup().
        train_mlp = !train_mlp; // Toggle MLP training
        opt_gd = false; // Turn off GD optimizer
        // --- END CRITICAL CHANGE ---
    }

    if (k == 'n')
    {
        // Manual single step for MLP training (if not in continuous 't' mode)
        if (!train_mlp) {
            float loss = mlp.computeLossAndGradient(mlp_input_data, polygon, gradOut);
            mlp.backward(gradOut, 0.01);
            std::cout << "Manual MLP Step Loss: " << loss << std::endl;
        }
    }
}


void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_