#pragma once
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdio>

class MLP
{
public:
    int inputDim, outputDim;
    std::vector<int> hiddenDims;

    std::vector<std::vector<std::vector<float>>> W; // W[layer][neuron][input]
    std::vector<std::vector<float>> B;              // B[layer][neuron]

    std::function<float(float)> activation = relu;
    std::function<float(float)> activation_deriv = relu_deriv;

    std::function<float(float, float)> loss = mse_loss;
    std::function<float(float, float)> dloss = mse_dloss;

    MLP(int inDim, int outDim, const std::vector<int>& hidden)
        : inputDim(inDim), outputDim(outDim), hiddenDims(hidden)
    {
        initializeWeights();
    }

    void initializeWeights()
    {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        std::vector<int> layers = { inputDim };
        layers.insert(layers.end(), hiddenDims.begin(), hiddenDims.end());
        layers.push_back(outputDim);

        W.clear();
        B.clear();

        for (size_t l = 0; l < layers.size() - 1; ++l)
        {
            int inL = layers[l];
            int outL = layers[l + 1];

            std::vector<std::vector<float>> wLayer(outL, std::vector<float>(inL));
            std::vector<float> bLayer(outL);

            for (int j = 0; j < outL; ++j)
            {
                for (int i = 0; i < inL; ++i)
                {
                    wLayer[j][i] = dist(gen);
                }
                bLayer[j] = dist(gen);
            }

            W.push_back(wLayer);
            B.push_back(bLayer);
        }
    }

    std::vector<float> forward(const std::vector<float>& input)
    {
        std::vector<float> layerIn = input;

        for (size_t l = 0; l < W.size(); ++l)
        {
            std::vector<float> layerOut(W[l].size());

            for (size_t j = 0; j < W[l].size(); ++j)
            {
                float sum = B[l][j];
                for (size_t i = 0; i < W[l][j].size(); ++i)
                {
                    sum += W[l][j][i] * layerIn[i];
                }

                layerOut[j] = (l == W.size() - 1) ? sum : activation(sum);
            }

            layerIn = layerOut;
        }

        return layerIn;
    }

    void backward(const std::vector<float>& input,
        const std::vector<float>& target,
        float learningRate)
    {
        std::vector<std::vector<float>> A, Z;
        A.push_back(input);

        for (size_t l = 0; l < W.size(); ++l)
        {
            std::vector<float> z(W[l].size());
            std::vector<float> a(W[l].size());

            for (size_t j = 0; j < W[l].size(); ++j)
            {
                float sum = B[l][j];
                for (size_t i = 0; i < W[l][j].size(); ++i)
                {
                    sum += W[l][j][i] * A.back()[i];
                }

                z[j] = sum;
                a[j] = (l == W.size() - 1) ? sum : activation(sum);
            }

            Z.push_back(z);
            A.push_back(a);
        }

        std::vector<std::vector<float>> dA(W.size());
        dA.back() = std::vector<float>(outputDim);

        for (int j = 0; j < outputDim; ++j)
        {
            dA.back()[j] = dloss(A.back()[j], target[j]);
        }

        for (int l = (int)W.size() - 1; l >= 0; --l)
        {
            int outDim = W[l].size();
            int inDim = W[l][0].size();

            std::vector<float> dZ(outDim);
            for (int j = 0; j < outDim; ++j)
            {
                dZ[j] = (l == W.size() - 1) ? dA[l][j]
                    : dA[l][j] * activation_deriv(Z[l][j]);
            }

            std::vector<float> dAprev(inDim, 0.0f);
            for (int j = 0; j < outDim; ++j)
            {
                for (int i = 0; i < inDim; ++i)
                {
                    dAprev[i] += W[l][j][i] * dZ[j];
                    W[l][j][i] -= learningRate * dZ[j] * A[l][i];
                }
                B[l][j] -= learningRate * dZ[j];
            }

            if (l > 0)
            {
                dA[l - 1] = dAprev;
            }
        }
    }

    float computeLoss(const std::vector<float>& output, const std::vector<float>& target)
    {
        float total = 0.0f;
        for (size_t i = 0; i < output.size(); ++i)
        {
            total += loss(output[i], target[i]);
        }
        return total / output.size();
    }

    void train(const std::vector<std::vector<float>>& inputs,
        const std::vector<std::vector<float>>& targets,
        int epochs,
        float learningRate,
        bool verbose = false)
    {
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            float epochLoss = 0.0f;

            for (size_t i = 0; i < inputs.size(); ++i)
            {
                std::vector<float> out = forward(inputs[i]);
                epochLoss += computeLoss(out, targets[i]);
                backward(inputs[i], targets[i], learningRate);
            }

            epochLoss /= inputs.size();

            if (verbose && (epoch % 100 == 0))
            {
                printf("Epoch %d | Loss: %.6f\n", epoch, epochLoss);
            }
        }
    }

    // --- Default ReLU + MSE ---
    static float relu(float x)
    {
        return std::max(0.0f, x);
    }

    static float relu_deriv(float x)
    {
        return x > 0 ? 1.0f : 0.0f;
    }

    static float mse_loss(float pred, float target)
    {
        float e = pred - target;
        return e * e;
    }

    static float mse_dloss(float pred, float target)
    {
        return 2.0f * (pred - target);
    }
};
