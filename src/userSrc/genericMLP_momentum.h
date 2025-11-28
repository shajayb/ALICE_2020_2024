#ifndef _GENERIC_MLP_
#define _GENERIC_MLP_

#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>

//------------------------------------------------------------------ MLP base class

class MLP
{
public:
    int inputDim = 2;
    int outputDim = 1;
    std::vector<int> hiddenDims = { 8, 8 };

    // Weights and Biases
    std::vector<std::vector<std::vector<float>>> W;
    std::vector<std::vector<float>> b;

    // Momentum / Velocity (New for better convergence)
    std::vector<std::vector<std::vector<float>>> vW;
    std::vector<std::vector<float>> vb;
    float momentum = 0.9f;

    // Activations storage for backprop
    std::vector<std::vector<float>> activations;

    // Optimization State
    bool isGradientPass = false; // Flag for derived classes to lock batching
    int batchOffset = 0;         // Counter to cycle through mini-batches

    MLP()
    {}

    MLP(int inDim, std::vector<int> hidden, int outDim)
    {
        initialize(inDim, hidden, outDim);
    }

    void initialize(int inDim, std::vector<int> hidden, int outDim)
    {
        inputDim = inDim;
        hiddenDims = hidden;
        outputDim = outDim;

        std::vector<int> layerDims = { inputDim };
        layerDims.insert(layerDims.end(), hiddenDims.begin(), hiddenDims.end());
        layerDims.push_back(outputDim);

        W.clear(); b.clear();
        vW.clear(); vb.clear();

        // Use a fixed seed for reproducibility, or allow random
        // srand(time(NULL)); 

        for (int l = 0; l < layerDims.size() - 1; ++l)
        {
            int inSize = layerDims[l];
            int outSize = layerDims[l + 1];

            // --- Xavier / Glorot Initialization ---
            // Keeps variance consistent across layers to prevent vanishing gradients
            float limit = std::sqrt(6.0f / (float)(inSize + outSize));

            std::vector<std::vector<float>> wLayer(outSize, std::vector<float>(inSize));
            std::vector<std::vector<float>> vwLayer(outSize, std::vector<float>(inSize, 0.0f)); // Init velocity to 0
            std::vector<float> bLayer(outSize, 0.0f); // Biases init to 0
            std::vector<float> vbLayer(outSize, 0.0f);

            for (int i = 0; i < outSize; ++i) {
                for (int j = 0; j < inSize; ++j) {
                    // Random value between [-limit, limit]
                    float r = (float)rand() / RAND_MAX;
                    wLayer[i][j] = (r * 2.0f * limit) - limit;
                }
            }

            W.push_back(wLayer);
            vW.push_back(vwLayer);
            b.push_back(bLayer);
            vb.push_back(vbLayer);
        }
    }

    std::vector<float> forward(std::vector<float>& x)
    {
        activations.clear();
        activations.push_back(x);
        std::vector<float> a = x;

        for (int l = 0; l < W.size(); ++l)
        {
            std::vector<float> z(b[l]); // Copy biases
            for (int i = 0; i < W[l].size(); ++i)
                for (int j = 0; j < W[l][i].size(); ++j)
                    z[i] += W[l][i][j] * a[j];

            // Apply tanh to all layers except potentially the last one
            // (Standard behavior for regression is linear output, but your original code 
            // applied tanh to hidden layers only. Preserving that logic:)
            if (l < W.size() - 1)
                for (auto& val : z) val = std::tanh(val);

            activations.push_back(z);
            a = z;
        }
        return a;
    }

    virtual float computeLoss(std::vector<float>& y_pred, std::vector<float>& y_true)
    {
        // Standard MSE (Override this in derived class for coverage/SDF loss)
        float loss = 0.0f;
        for (int i = 0; i < y_pred.size(); ++i)
        {
            float err = y_pred[i] - y_true[i];
            loss += err * err;
        }
        return loss / y_pred.size();
    }

    // Standard Analytical Gradient (for MSE)
    virtual void computeGradient(std::vector<float>& x, std::vector<float>& y_true, std::vector<float>& gradOut)
    {
        std::vector<float> y_pred = forward(x);
        gradOut.assign(outputDim, 0.0f);
        for (int i = 0; i < outputDim; ++i)
        {
            gradOut[i] = 2.0f * (y_pred[i] - y_true[i]) / outputDim;
        }
    }

    // --- NEW: Numerical Gradient Helper ---
    // Derived classes should call this inside their computeGradient() override
    void computeGradientNumerical(std::vector<float>& x, std::vector<float>& y_dummy, std::vector<float>& gradOut)
    {
        // 1. Lock the Batch
        // This tells the loss function (in derived class) NOT to shuffle points
        isGradientPass = true;

        // 2. Base Pass
        std::vector<float> y0 = forward(x);
        float eps = 1e-2f;

        gradOut.assign(outputDim, 0.0f);

        // 3. Perturbation Loop (Central Difference)
        for (int i = 0; i < outputDim; ++i)
        {
            std::vector<float> y_plus = y0;
            y_plus[i] += eps;
            float L_plus = computeLoss(y_plus, y_dummy); // Virtual call uses derived loss

            std::vector<float> y_minus = y0;
            y_minus[i] -= eps;
            float L_minus = computeLoss(y_minus, y_dummy);

            gradOut[i] = (L_plus - L_minus) / (2.0f * eps);
        }

        // 4. Unlock Batch & Advance
        isGradientPass = false;
        batchOffset++; // Increment offset so next frame uses different random samples
    }

    void backward(std::vector<float>& gradOut, float lr)
    {
        std::vector<float> delta = gradOut;

        for (int l = W.size() - 1; l >= 0; --l)
        {
            std::vector<float> prev = activations[l];
            std::vector<float> newDelta(prev.size(), 0.0f);

            for (int i = 0; i < W[l].size(); ++i)
            {
                for (int j = 0; j < W[l][i].size(); ++j)
                {
                    // 1. Compute Gradient
                    float grad = delta[i] * prev[j];

                    // 2. Update Velocity (Momentum)
                    vW[l][i][j] = momentum * vW[l][i][j] - lr * grad;

                    // 3. Update Weight
                    W[l][i][j] += vW[l][i][j];

                    newDelta[j] += delta[i] * W[l][i][j];
                }

                // Bias Update
                vb[l][i] = momentum * vb[l][i] - lr * delta[i];
                b[l][i] += vb[l][i];
            }

            if (l > 0)
            {
                for (int i = 0; i < newDelta.size(); ++i)
                {
                    float a = activations[l][i];
                    newDelta[i] *= (1 - a * a); // tanh derivative
                }
                delta = newDelta;
            }
        }
    }

    // ---------------- VISUALIZATION HELPERS (Retained from Original) ----------------

    void drawSolidCircle(Alice::vec center, float radius, int numSegments = 32)
    {
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(center.x, center.y);  // center
        for (int i = 0; i <= numSegments; i++)
        {
            float angle = TWO_PI * i / numSegments;
            float x = center.x + radius * cos(angle);
            float y = center.y + radius * sin(angle);
            glVertex2f(x, y);
        }
        glEnd();
    }

    void visualize(zVector topLeft = zVector(50, 450, 0), float bboxWidth = 400.0f, float bboxHeight = 300.0f)
    {
        setup2d();  // 2D drawing

        int numLayers = activations.size();
        float nodeRadius = 3.0f;

        // Compute max nodes per layer for vertical spacing
        int maxNodesPerLayer = 0;
        for (auto& layer : activations)
            maxNodesPerLayer = std::max(maxNodesPerLayer, (int)layer.size());

        float layerSpacing = (numLayers > 1) ? bboxWidth / (numLayers - 1) : 150.0f;
        float verticalSpacing = (maxNodesPerLayer > 1) ? bboxHeight / (maxNodesPerLayer - 1) : 30.0f;

        std::vector<std::vector<zVector>> nodePositions(numLayers);

        // Compute node positions
        for (int l = 0; l < numLayers; l++)
        {
            int numNodes = activations[l].size();
            float yStart = topLeft.y - 0.5f * (numNodes - 1) * verticalSpacing;

            for (int i = 0; i < numNodes; i++)
            {
                float x = topLeft.x + l * layerSpacing;
                float y = yStart + i * verticalSpacing;
                nodePositions[l].push_back(zVector(x, y, 0));
            }
        }

        // --- Draw weight connections
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        for (int l = 0; l < numLayers - 1; l++)
        {
            int fromSize = activations[l].size();
            int toSize = activations[l + 1].size();

            for (int i = 0; i < fromSize; i++)
            {
                for (int j = 0; j < toSize; j++)
                {
                    float w = W[l][j][i];
                    float absW = fabs(w);

                    if (absW < 0.02f) continue;  // skip very weak connections

                    float val = std::clamp(w * 5.0f, -1.0f, 1.0f);
                    float r, g, b;
                    getJetColor(val, r, g, b);

                    glColor4f(r, g, b, 0.4f);  // faded connection
                    glLineWidth(std::clamp(absW * 5.0f, 0.5f, 3.0f));
                    drawLine(zVecToAliceVec(nodePositions[l][i]), zVecToAliceVec(nodePositions[l + 1][j]));
                }
            }
        }

        glDisable(GL_BLEND);
        glLineWidth(1.0f);

        // --- Draw nodes
        for (int l = 0; l < numLayers; l++)
        {
            for (int i = 0; i < activations[l].size(); i++)
            {
                float act = activations[l][i];
                float r, g, b;
                getJetColor(act, r, g, b);

                glColor3f(r, g, b);
                drawSolidCircle(zVecToAliceVec(nodePositions[l][i]), nodeRadius, 12);
            }
        }

        restore3d();
    }
};

#endif