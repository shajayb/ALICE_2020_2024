#include <iostream>
#include "genericMLP.h"

int main() {
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;

    for (float x = 0.0f; x <= 1.0f; x += 0.01f) {
        for (float y = 0.0f; y <= 1.0f; y += 0.01f) {
            inputs.push_back({ x, y });
            float logic = (x > 0.5f) != (y > 0.5f) ? 1.0f : 0.0f;
            targets.push_back({ logic });
        }
    }

    MLP net(2, 1, { 6, 6 });
    net.train(inputs, targets, 1000, 0.01f, true);

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto out = net.forward(inputs[i]);
        printf("Input: %.2f %.2f => Output: %.4f | Target: %.1f\n",
            inputs[i][0], inputs[i][1], out[0], targets[i][0]);
    }

    return 0;
}

