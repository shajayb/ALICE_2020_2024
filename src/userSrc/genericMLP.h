#pragma once
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdio>

#include <vector>
#include <cmath>
#include <fstream>





//------------------------------------------------------------------ MLP base class

class MLP
{
public:
    int inputDim = 2;
    int outputDim = 1;
    std::vector<int> hiddenDims = { 8, 8 };

    std::vector<std::vector<std::vector<float>>> W;
    std::vector<std::vector<float>> b;
    std::vector<std::vector<float>> activations;


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
        for (int l = 0; l < layerDims.size() - 1; ++l)
        {
            int inSize = layerDims[l];
            int outSize = layerDims[l + 1];
            W.push_back(std::vector<std::vector<float>>(outSize, std::vector<float>(inSize)));
            b.push_back(std::vector<float>(outSize));
            for (auto& w_row : W[l])
                for (auto& w : w_row)
                    w = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }

    std::vector<float> forward(std::vector<float>& x)
    {
        activations.clear();
        activations.push_back(x);
        std::vector<float> a = x;

        for (int l = 0; l < W.size(); ++l)
        {
            std::vector<float> z(b[l]);
            for (int i = 0; i < W[l].size(); ++i)
                for (int j = 0; j < W[l][i].size(); ++j)
                    z[i] += W[l][i][j] * a[j];

            if (l < W.size() - 1)
                for (auto& val : z) val = std::tanh(val);

            activations.push_back(z);
            a = z;
        }
        return a;
    }

    virtual float computeLoss(std::vector<float>& y_pred, std::vector<float>& y_true)
    {
        float loss = 0.0f;
        for (int i = 0; i < y_pred.size(); ++i)
        {
            float err = y_pred[i] - y_true[i];
            loss += err * err;
        }
        return loss / y_pred.size();
    }

    virtual void computeGradient(std::vector<float>& x, std::vector<float>& y_true, std::vector<float>& gradOut)
    {
        std::vector<float> y_pred = forward(x);
        gradOut.assign(outputDim, 0.0f);
        for (int i = 0; i < outputDim; ++i)
        {
            gradOut[i] = 2.0f * (y_pred[i] - y_true[i]) / outputDim;
        }
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
                    newDelta[j] += delta[i] * W[l][i][j];
                    W[l][i][j] -= lr * delta[i] * prev[j];
                }
                b[l][i] -= lr * delta[i];
            }

            if (l > 0)
            {
                for (int i = 0; i < newDelta.size(); ++i)
                {
                    float a = activations[l][i];
                    newDelta[i] *= (1 - a * a); // tanh'
                }
                delta = newDelta;
            }
        }
    }

    //

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




    //void visualize(zVector topLeft = zVector(50, 450, 0), float bboxWidth = 400.0f, float bboxHeight = 300.0f)
    //{
    //    setup2d();

    //    int numLayers = activations.size();
    //    float nodeRadius = 5.0f;

    //    // Get max nodes per layer to compute spacing
    //    int maxNodesPerLayer = 0;
    //    for (auto& layer : activations)
    //    {
    //        maxNodesPerLayer = std::max(maxNodesPerLayer, (int)layer.size());
    //    }

    //    float layerSpacing = (numLayers > 1) ? bboxWidth / (numLayers - 1) : 0.0f;
    //    float verticalSpacing = (maxNodesPerLayer > 1) ? bboxHeight / (maxNodesPerLayer - 1) : 0.0f;

    //    std::vector<std::vector<zVector>> nodePositions(numLayers);

    //    // Compute node positions
    //    for (int l = 0; l < numLayers; l++)
    //    {
    //        int numNodes = activations[l].size();
    //        float yStart = topLeft.y - 0.5f * (numNodes - 1) * verticalSpacing;

    //        for (int n = 0; n < numNodes; n++)
    //        {
    //            float x = topLeft.x + l * layerSpacing;
    //            float y = yStart + n * verticalSpacing;
    //            nodePositions[l].push_back(zVector(x, y, 0));
    //        }
    //    }

    //    // Draw weight connections (color strong weights)
    //    for (int l = 0; l < numLayers - 1; l++)
    //    {
    //        int fromSize = activations[l].size();
    //        int toSize = activations[l + 1].size();

    //        for (int i = 0; i < fromSize; i++)
    //        {
    //            float activation = activations[l][i];

    //            for (int j = 0; j < toSize; j++)
    //            {
    //                float w = W[l][j][i];
    //                float val = std::clamp(w * 5.0f, -1.0f, 1.0f);

    //                float r, g, b;
    //                getJetColor(val, r, g, b);
    //                (val > 0.9) ? glColor3f(r, g, b) : glColor3f(0.8, 0.8, 0.8);

    //                drawLine(zVecToAliceVec(nodePositions[l][i]), zVecToAliceVec(nodePositions[l + 1][j]));
    //            }
    //        }
    //    }

    //    // Draw neuron activations
    //    for (int l = 0; l < numLayers; l++)
    //    {
    //        for (int i = 0; i < activations[l].size(); i++)
    //        {
    //            float act = std::tanh(activations[l][i]);
    //            float r, g, b;
    //            getJetColor(act, r, g, b);
    //            glColor3f(0, 0, 0); // black outline

    //            drawCircle(zVecToAliceVec(nodePositions[l][i]), nodeRadius, 12);
    //        }
    //    }

    //    restore3d();
    //}

};