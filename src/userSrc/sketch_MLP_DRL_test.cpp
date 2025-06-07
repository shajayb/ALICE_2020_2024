#define _MAIN_
#ifdef _MAIN_

#include "main.h"
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>
#include <vector>
#include <cmath>

using namespace zSpace;

Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

zVector AliceVecToZvec(Alice::vec& in)
{
    return zVector(in.x, in.y, in.z);
}



#include "scalarField.h"
#include "genericMLP.h"

//----------------------------------------------------------
class TangencyMLP : public MLP
{
public:
    float targetDistance = 10.0f; // 2 * radius if radius = 5

    TangencyMLP(int inDim, std::vector<int> hidden, int outDim)
        : MLP(inDim, hidden, outDim) {}

    float computeLoss(std::vector<float>& y_pred, std::vector<float>& y_true) override
    {
        zVector c0(y_pred[0], y_pred[1], 0);
        zVector c1(y_pred[2], y_pred[3], 0);
        zVector c2(y_pred[4], y_pred[5], 0);

        float d01 = c0.distanceTo(c1);
        float d02 = c0.distanceTo(c2);
        float d12 = c1.distanceTo(c2);

        float loss = powf(d01 - targetDistance, 2) +
            powf(d02 - targetDistance, 2) +
            powf(d12 - targetDistance, 2);

        return loss;
    }

    void computeGradient(std::vector<float>& x, std::vector<float>& y_true, std::vector<float>& gradOut) override
    {
        std::vector<float> y_pred = forward(x);
        gradOut.assign(outputDim, 0.0f);

        zVector c0(y_pred[0], y_pred[1], 0);
        zVector c1(y_pred[2], y_pred[3], 0);
        zVector c2(y_pred[4], y_pred[5], 0);

        auto grad = [](zVector& a, zVector& b, float target) -> zVector
            {
                zVector dir = a - b;
                float d = dir.length();
                if (d < 1e-4f) return zVector();
                return dir * (2.0f * (d - target) / d);

            };

        zVector g01 = grad(c0, c1, targetDistance);
        zVector g02 = grad(c0, c2, targetDistance);
        zVector g12 = grad(c1, c2, targetDistance);

        gradOut[0] = g01.x + g02.x;
        gradOut[1] = g01.y + g02.y;

        gradOut[2] = -g01.x + g12.x;
        gradOut[3] = -g01.y + g12.y;

        gradOut[4] = -g02.x - g12.x;
        gradOut[5] = -g02.y - g12.y;
    }
};

TangencyMLP mlp(6, { 32, 32 }, 6); // << SYS-DES#1 = NN ARCHITECTURE
std::vector<float> input = { 10, 10, 30, 10, 20, 30 };
std::vector<float> grad;

void setup()
{
    mlp.forward(input);
}

void update(int value)
{

}

void draw()
{
    backGround(0.9);
    drawGrid(50);

    std::vector<float> out = mlp.forward(input);
    zVector c0(out[0], out[1], 0);
    zVector c1(out[2], out[3], 0);
    zVector c2(out[4], out[5], 0);

    glColor3f(1, 0, 0);
    drawCircle(zVecToAliceVec(c0), 5, 32);
    drawCircle(zVecToAliceVec(c1), 5, 32);
    drawCircle(zVecToAliceVec(c2), 5, 32);

    glColor3f(0, 0, 0);
    drawLine(zVecToAliceVec(c0), zVecToAliceVec(c1));
    drawLine(zVecToAliceVec(c0), zVecToAliceVec(c2));
    drawLine(zVecToAliceVec(c1), zVecToAliceVec(c2));
}

void keyPress(unsigned char k, int x, int y)
{
    if (k == 't')
    {
        for (int i = 0; i < 1; i++)
        {
            mlp.computeGradient(input, input, grad);
            mlp.backward(grad, 0.01f);
        }
    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_
