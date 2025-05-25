#define _MAIN_
#ifdef _MAIN_

#include "main.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;

Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

zVector AliceVecToZvec(Alice::vec& in)
{
    return zVector(in.x, in.y, in.z);
}

using namespace zSpace;

#define RES 100
#define EPSILON 1e-6
#define SINKHORN_ITER 100
#define LAMBDA 0.1f

float sourceField[RES][RES];
float targetField[RES][RES];
float advectedField[RES][RES];

zVector velocityField[RES][RES];
zVector flowField[RES][RES];

//--------------------------------------------------
// Utility Functions
//--------------------------------------------------

void initializeFields()
{
    for (int i = 0; i < RES; i++)
    {
        for (int j = 0; j < RES; j++)
        {
            float dx = (i - RES / 3);
            float dy = (j - RES / 3);
            sourceField[i][j] = expf(-(dx * dx + dy * dy) / 200.0f);
            targetField[i][j] = 0.0f;
        }
    }

    int numBlobs = 8;
    float radius = RES * 0.35f;
    zVector center(RES / 2.0f, RES / 2.0f, 0);

    for (int b = 0; b < numBlobs; b++)
    {
        float angle = b * TWO_PI / numBlobs;
        zVector blobCenter = center + zVector(cos(angle), sin(angle), 0) * radius;

        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                float dx = i - blobCenter.x;
                float dy = j - blobCenter.y;
                targetField[i][j] += 0.15f * expf(-(dx * dx + dy * dy) / 40.0f);
            }
        }
    }
}

void computeVelocityFieldFromSinkhorn()
{
    const int N = RES * RES;
    std::vector<float> mu(N), nu(N);
    std::vector<float> K(N * N);
    std::vector<float> u(N, 1.0f), v(N, 1.0f);

    for (int i = 0; i < RES; i++)
        for (int j = 0; j < RES; j++)
        {
            int id = i * RES + j;
            mu[id] = sourceField[i][j];
            nu[id] = targetField[i][j];
        }

    for (int i = 0; i < RES; i++)
    {
        for (int j = 0; j < RES; j++)
        {
            int a = i * RES + j;
            for (int m = 0; m < RES; m++)
            {
                for (int n = 0; n < RES; n++)
                {
                    int b = m * RES + n;
                    float dx = i - m;
                    float dy = j - n;
                    float dist2 = dx * dx + dy * dy;
                    K[a * N + b] = expf(-dist2 * LAMBDA);
                }
            }
        }
    }

    for (int iter = 0; iter < SINKHORN_ITER; iter++)
    {
        for (int i = 0; i < N; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) sum += K[i * N + j] * v[j];
            u[i] = (sum > EPSILON) ? mu[i] / sum : 0.0f;
        }

        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int i = 0; i < N; i++) sum += K[i * N + j] * u[i];
            v[j] = (sum > EPSILON) ? nu[j] / sum : 0.0f;
        }
    }

    // Compute flow direction from transport plan
    for (int i = 0; i < RES; i++)
    {
        for (int j = 0; j < RES; j++)
        {
            int a = i * RES + j;
            zVector weightedSum(0, 0, 0);
            float total = 0.0f;

            for (int m = 0; m < RES; m++)
            {
                for (int n = 0; n < RES; n++)
                {
                    int b = m * RES + n;
                    float t = u[a] * K[a * N + b] * v[b];
                    weightedSum += zVector(m - i, n - j, 0) * t;
                    total += t;
                }
            }

            flowField[i][j] = (total > EPSILON) ? weightedSum / total : zVector(0, 0, 0);
            flowField[i][j].normalize();
            velocityField[i][j] = flowField[i][j];
        }
    }
}

void advectField(float dt = 1.0f)
{
    for (int i = 1; i < RES - 1; i++)
    {
        for (int j = 1; j < RES - 1; j++)
        {
            float x = std::clamp(i - dt * velocityField[i][j].x, 0.0f, float(RES - 1));
            float y = std::clamp(j - dt * velocityField[i][j].y, 0.0f, float(RES - 1));

            int i0 = std::clamp(int(floor(x)), 0, RES - 1);
            int i1 = std::clamp(i0 + 1, 0, RES - 1);
            int j0 = std::clamp(int(floor(y)), 0, RES - 1);
            int j1 = std::clamp(j0 + 1, 0, RES - 1);

            float sx = x - i0;
            float sy = y - j0;

            float val = (1 - sx) * (1 - sy) * sourceField[i0][j0] +
                sx * (1 - sy) * sourceField[i1][j0] +
                (1 - sx) * sy * sourceField[i0][j1] +
                sx * sy * sourceField[i1][j1];

            advectedField[i][j] = val;
        }
    }
}

//--------------------------------------------------
// MVC Callbacks
//--------------------------------------------------

void setup()
{
    initializeFields();
    //computeVelocityFieldFromSinkhorn();
   // advectField(0.5f);
}

void update(int value) {}

void draw()
{
    backGround(0.8);
    drawGrid(50);

    glPointSize(2);
    for (int i = 1; i < RES; i++)
    {
        for (int j = 1; j < RES; j++)
        {
            float r = advectedField[i][j];
            glColor3f(r, 0, 0);
            drawPoint(Alice::vec(i, j, 0));

            // Visualize flow vectors
            if (i % 10 == 0 && j % 10 == 0)
            {
                glColor3f(0, 1, 0);
                drawLine(Alice::vec(i, j, 0), Alice::vec(i, j, 0) + zVecToAliceVec(flowField[i][j] * 2));
            }
        }
    }
    glPointSize(1);
}

void keyPress(unsigned char k, int xm, int ym)
{
    if (k == 'r')
    {
        computeVelocityFieldFromSinkhorn();
        advectField(1);
        for (int i = 0; i < RES; i++)
            for (int j = 0; j < RES; j++)
                sourceField[i][j] = advectedField[i][j];
    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}


#endif // _MAIN_