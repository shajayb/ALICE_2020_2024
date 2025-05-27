#define _MAIN_
#ifdef _MAIN_

#include "main.h"

#include <vector>
#include <cmath>
#include <iostream>

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

inline zVector zMax(const zVector& a, const zVector& b)
{
    return zVector(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

inline zVector zMin(const zVector& a, const zVector& b)
{
    return zVector(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

vector<zVector> loadPolygonFromCSV(const std::string& filename)
{
    vector<zVector> poly;
    poly.clear();

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
            poly.emplace_back(x, y, 0);
        }
    }

    cout << poly.size() << " polygon size" << endl;
    return poly;
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


inline float smin(float a, float b, float k)
{
    float h = std::max(k - fabs(a - b), 0.0f) / k;
    return std::min(a, b) - h * h * k * 0.25f;
}

float sdfCircle( zVector& p,  zVector& center, float r)
{
    return (p - center).length() - r;
}

float blendCircleSDFs( zVector& p,  std::vector<zVector>& centers,  std::vector<float>& radii, float k)
{
    float result = 1e6f;
    float d = sdfCircle(p, centers[0], radii[0]);
    for (int i = 1; i < centers.size(); i++)
    {
        float d_i = sdfCircle(p, centers[i], radii[i]);
        d = smin(d, d_i, k);  // smooth union of signed distances
    }

    return d;
}

float SDF_polygon( std::vector<zVector>& polyline,  zVector& p)
{
    float minDist = 1e6f;
    for (int i = 0; i < polyline.size() - 1; ++i)
    {
        zVector a = polyline[i];
        zVector b = polyline[i + 1];
        zVector pa = p - a;
        zVector ba = b - a;
        float h = std::clamp((pa * ba) / (ba*ba), 0.0f, 1.0f);
        zVector proj = a + ba * h;
        float d = (p - proj).length();
        minDist = std::min(minDist, d);
    }
    return minDist;
}



class MLP
{
public:
    int inputDim = 1;
    int hiddenDim = 16;
    int outputDim;

    std::vector<float> W1, b1;
    std::vector<float> W2, b2;
    std::vector<float> h, h_relu, out;

    MLP(int outDim)
    {
        outputDim = outDim;
        W1.resize(hiddenDim * inputDim);
        b1.resize(hiddenDim);
        W2.resize(hiddenDim * outputDim);
        b2.resize(outputDim);
        h.resize(hiddenDim);
        h_relu.resize(hiddenDim);
        out.resize(outputDim);



        for (auto& w : W1) w = ofRandom(-0.05f, 0.05f);
        for (auto& w : W2) w = ofRandom(-0.05f, 0.05f);
        std::fill(b1.begin(), b1.end(), 0.0f);
        std::fill(b2.begin(), b2.end(), 0.0f);
    }

    std::vector<float> forward( std::vector<float>& in)
    {
        for (int i = 0; i < hiddenDim; ++i)
        {
            h[i] = b1[i];
            for (int j = 0; j < inputDim; ++j)
                h[i] += W1[i * inputDim + j] * in[j];
            h_relu[i] = std::max(0.0f, h[i]);
        }

        for (int i = 0; i < outputDim; ++i)
        {
            out[i] = b2[i];
            for (int j = 0; j < hiddenDim; ++j)
                out[i] += W2[i * hiddenDim + j] * h_relu[j];
        }

        return out;
    }

    void backward( std::vector<float>& dL_dout,  std::vector<float>& input, float lr = 0.01f)
    {
        std::vector<float> dL_dh(hiddenDim, 0.0f);
        for (int i = 0; i < outputDim; ++i)
        {
            for (int j = 0; j < hiddenDim; ++j)
            {
                dL_dh[j] += dL_dout[i] * W2[i * hiddenDim + j];
                W2[i * hiddenDim + j] -= lr * dL_dout[i] * h_relu[j];
            }
            b2[i] -= lr * dL_dout[i];
        }

        for (int j = 0; j < hiddenDim; ++j)
        {
            float grad = (h[j] > 0) ? dL_dh[j] : 0.0f;
            for (int k = 0; k < inputDim; ++k)
                W1[j * inputDim + k] -= lr * grad * input[k];
            b1[j] -= lr * grad;
        }
    }
};

zVector bboxCenter;
float bboxScale;

void decodeMLPOutput(std::vector<float>& out, std::vector<zVector>& centers, std::vector<float>& radii, zVector bboxCenter, float bboxScale)
{
    int N = out.size() / 3;
    centers.resize(N);
    radii.resize(N);

    for (int i = 0; i < N; ++i)
    {
        float x = out[i * 3 + 0];
        float y = out[i * 3 + 1];
        float r = std::abs(out[i * 3 + 2]);

        // Rescale and shift to bounding box
        x *= bboxScale;
        y *= bboxScale;
        r *= bboxScale;

        centers[i] = zVector(x, y, 0) + bboxCenter;
        radii[i] = std::clamp(r, 1.0f, bboxScale * 0.5f);
    }
}

void computeGradients(
     zVector& p,
     std::vector<zVector>& centers,
     std::vector<float>& radii,
    float alpha,
    std::vector<float>& dL_dout,
    float error
)
{
    int N = centers.size();
    std::vector<float> d_i(N), w_i(N), dPred_dDi(N);
    float w_sum = 0.0f, pred = 0.0f;

    for (int i = 0; i < N; i++)
    {
        d_i[i] = sdfCircle(p, centers[i], radii[i]);
        w_i[i] = expf(-alpha * d_i[i]);
        w_sum += w_i[i];
    }

    for (int i = 0; i < N; i++)
    {
        pred += (w_i[i] * d_i[i]);
    }

    pred /= w_sum;

    for (int i = 0; i < N; i++)
    {
        dPred_dDi[i] = (w_i[i] * (1 - alpha * (d_i[i] - pred))) / w_sum;

        zVector dp_dc = (centers[i] - p);
        float len = dp_dc.length();
        if (len < 1e-6) len = 1e-6;
        dp_dc /= len;

        int j = i * 3;
        //dL_dout[j + 0] += error * dPred_dDi[i] * dp_dc.x; // ∂L/∂cx
        //dL_dout[j + 1] += error * dPred_dDi[i] * dp_dc.y; // ∂L/∂cy
        ////dL_dout[j + 2] += error * dPred_dDi[i] * (-1.0f) * dPred_dDi[i]; // ∂L/∂r
        //dL_dout[j + 2] += error * dPred_dDi[i] * (-1.0f);  // correct ∂d/∂r is -1

        dL_dout[j + 0] += error * dPred_dDi[i] * dp_dc.x * bboxScale;
        dL_dout[j + 1] += error * dPred_dDi[i] * dp_dc.y * bboxScale;
        dL_dout[j + 2] += error * dPred_dDi[i] * (-1.0f) * bboxScale;


    }

    for (float& g : dL_dout)
    {
        if (!std::isfinite(g)) g = 0.0f;
        g = std::clamp(g, -10.0f, 10.0f);
    }
}


// ----------------------------
// Globals
// ----------------------------

MLP mlp(3 * 5); // 5 circles
std::vector<zVector> polygon = { zVector(-10, -10, 0), zVector(0, 10, 0), zVector(10, -10, 0), zVector(-10, -10, 0) };
std::vector<zVector> samplePts;
std::vector<float> sdfGT;

std::vector<zVector> sdfCenters;
std::vector<float> predictedRadii;

bool showTraining = false;

// ----------------------------
// MVC
// ----------------------------



void setup()
{
    polygon.clear();
    polygon = loadPolygonFromCSV("data/polygon.txt");

    // --- Compute bounding box ---
    zVector minBB(1e6, 1e6, 0), maxBB(-1e6, -1e6, 0);
    for (auto& p : polygon)
    {
        minBB = zMin(minBB, p);
        maxBB = zMax(maxBB, p);
    }

    bboxCenter = (minBB + maxBB) * 0.5f;
    bboxScale = (maxBB - minBB).length() * 0.5f;

    // --- Sample points inside polygon ---
    samplePts.clear();
    sdfGT.clear();

    for (float x = -50; x <= 50; x += 2)
    {
        for (float y = -50; y <= 50; y += 2)
        {
            zVector p(x, y, 0);
            if (!isInsidePolygon(p, polygon)) continue;

            samplePts.push_back(p);
            sdfGT.push_back(SDF_polygon(polygon, p));
        }
    }

    printf(" %i, %i \n", samplePts.size(), sdfGT.size());
}


void update(int value)
{
    if (!showTraining) return;

    std::vector<float> input = { 1.0f };
    std::vector<float> out = mlp.forward(input);

    decodeMLPOutput(out, sdfCenters, predictedRadii, bboxCenter, bboxScale);

    float loss = 0.0f;
    std::vector<float> dL_dout(out.size(), 0.0f);
    float alpha = 2.0f; // softness control for softmin

    for (int i = 0; i < samplePts.size(); i++)
    {
        float pred = blendCircleSDFs(samplePts[i], sdfCenters, predictedRadii, alpha);
        float error = pred - sdfGT[i];
        loss += error * error;

        computeGradients(samplePts[i], sdfCenters, predictedRadii, alpha, dL_dout, 2.0f * error / samplePts.size());
    }

    std::cout << "Loss: " << loss / samplePts.size() << std::endl;

    mlp.backward(dL_dout, input, 0.1f);  // try 0.005 or 0.001 if unstable

    //

    for (int i = 0; i < sdfCenters.size(); i++)
    {
        printf("%.2f %.2f %.2f\n", sdfCenters[i].x, sdfCenters[i].y, predictedRadii[i]);
    }
}

void draw()
{
    backGround(0.9);
    drawGrid(50);

    glColor3f(0, 0, 0);
    for (int i = 0; i < polygon.size() - 1; i++)
        drawLine( zVecToAliceVec(polygon[i]), zVecToAliceVec(polygon[i + 1]));

    glColor3f(0.2, 0.2, 1);
    for (auto& p : samplePts)
        drawPoint(zVecToAliceVec(p));

    glColor3f(1, 0, 0);
    for (int i = 0; i < sdfCenters.size(); i++)
        drawCircle(zVecToAliceVec(sdfCenters[i]), predictedRadii[i], 32);
}

void keyPress(unsigned char k, int xm, int ym)
{
    if (k == 'm')
    {
        showTraining = true;
    }

    if (k == 't')
    {
        showTraining = true;
        update(0); // single step
        showTraining = false;
    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_
