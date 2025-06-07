#define _MAIN_
#ifdef _MAIN_



#include "main.h"
#include <vector>
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

inline zVector zMax(zVector& a, zVector& b)
{
    return zVector(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

inline zVector zMin(zVector& a, zVector& b)
{
    return zVector(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
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

#include "scalarField.h"

inline float circleSDF(zVector p, zVector c, float r)
{
    return (p - c).length() - r;
}

inline float blendCircleSDFs(zVector pt, const std::vector<zVector>& centers, const std::vector<float>& radii)
{

    float result = 1e6;
    if (centers.empty())return result;

    float d = circleSDF(pt, centers[0], radii[0]);

    for (int i = 1; i < centers.size(); i++)
    {
        d = circleSDF(pt, centers[i], radii[i]);
        result = std::min(result, d);
    }
    return result;
}

std::vector<zVector> samplePts;
std::vector<float> sdfGT;
std::vector<zVector> fittedCenters;
std::vector<float> fittedRadii;
#define NUM_CIRCLES 8
std::vector<zVector> polygon;
double threshold;

std::vector<zVector> loadPolygonFromCSV(const std::string& filename)
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

class MLP
{
public:
    int inputDim, hiddenDim, outputDim;
    std::vector<std::vector<float>> W1, W2;
    std::vector<float> b1, b2;
    std::vector<float> input, hidden, output;

    MLP()
    {

    }
    MLP(int inDim, int hDim, int outDim)
        : inputDim(inDim), hiddenDim(hDim), outputDim(outDim)
    {
        W1.resize(hiddenDim, std::vector<float>(inputDim));
        W2.resize(outputDim, std::vector<float>(hiddenDim));
        b1.resize(hiddenDim);
        b2.resize(outputDim);
        input.resize(inputDim);
        hidden.resize(hiddenDim);
        output.resize(outputDim);

        std::default_random_engine eng;
        std::normal_distribution<float> dist(0.0, 0.1);

        for (auto& row : W1) for (auto& val : row) val = dist(eng);
        for (auto& row : W2) for (auto& val : row) val = dist(eng);
    }

    std::vector<float> forward(const std::vector<float>& x)
    {
        input = x;
        for (int i = 0; i < hiddenDim; ++i)
        {
            hidden[i] = b1[i];
            for (int j = 0; j < inputDim; ++j)
                hidden[i] += W1[i][j] * input[j];
            hidden[i] = std::tanh(hidden[i]);
        }
        for (int i = 0; i < outputDim; ++i)
        {
            output[i] = b2[i];
            for (int j = 0; j < hiddenDim; ++j)
                output[i] += W2[i][j] * hidden[j];
        }
        return output;
    }

    float computeLossAndGradient(const std::vector<float>& x, std::vector<zVector>& polygon, std::vector<float>& gradOut)
    {
        std::vector<zVector> centers(outputDim / 3);
        std::vector<float> radii(outputDim / 3);

       
        auto out = forward(x);
        for (int i = 0; i < centers.size(); i++)
        {
            zVector pt(out[i * 3 + 0], out[i * 3 + 1], 0);
            centers[i] = (isInsidePolygon(pt, polygon)) ? pt : fittedCenters[i];
            radii[i] = 8; // std::clamp(std::abs(out[i * 3 + 2]), 8.f, 8.f);
        }

        // Save for visualization
        fittedCenters = centers;
        fittedRadii = radii;

        float loss = 0;
        gradOut.assign(outputDim, 0.0f);

        for (int j = 0; j < outputDim; ++j)
        {
            float eps = 0.01f;
            std::vector<float> perturbedInput = x;
            perturbedInput[j] += eps;

            auto perturbedOut = forward(perturbedInput);
            std::vector<zVector> cPert(centers.size());
            std::vector<float> rPert(centers.size());

            for (int i = 0; i < centers.size(); i++)
            {
                zVector pt(perturbedOut[i * 3 + 0], perturbedOut[i * 3 + 1], 0);

                cPert[i] = (isInsidePolygon(pt, polygon)) ? pt : fittedCenters[i];

                rPert[i] = 8;//  std::clamp(std::abs(perturbedOut[i * 3 + 2]), 8.0f, 8.f);
            }

            float gradLoss = 0;
            /*for (int s = 0; s < samplePts.size(); ++s)
            {
                float f = blendCircleSDFs(samplePts[s], centers, radii);
                float fPert = blendCircleSDFs(samplePts[s], cPert, rPert);
                float gt = sdfGT[s];
                float err = f - gt;
                loss += err * err;
                gradLoss += 2 * err * (fPert - f) / eps;
            }*/

            // Compute original loss
            float baseError = 0.0f;
            for (int s = 0; s < samplePts.size(); ++s)
            {
                float f = blendCircleSDFs(samplePts[s], centers, radii);
                float gt = sdfGT[s];
                float err = f - gt;
                baseError += err * err;
            }
            loss += baseError;

            // Compute perturbed loss (x_j + eps)
            float errorPertPlus = 0.0f;
            for (int s = 0; s < samplePts.size(); ++s)
            {
                float fPert = blendCircleSDFs(samplePts[s], cPert, rPert);
                float gt = sdfGT[s];
                float errPert = fPert - gt;
                errorPertPlus += errPert * errPert;
            }

            // Compute perturbed loss (x_j - eps)
            std::vector<float> perturbedInputMinus = x;
            perturbedInputMinus[j] -= eps;
            auto perturbedOutMinus = forward(perturbedInputMinus);
            std::vector<zVector> cPertMinus(centers.size());
            std::vector<float> rPertMinus(centers.size());
            for (int i = 0; i < centers.size(); i++)
            {
                cPertMinus[i] = zVector(perturbedOutMinus[i * 3 + 0], perturbedOutMinus[i * 3 + 1], 0);
                cPertMinus[i] = (isInsidePolygon(cPertMinus[i], polygon)) ? cPertMinus[i] : fittedCenters[i];
                rPertMinus[i] = 8;// std::abs(perturbedOutMinus[i * 3 + 2]);
            }
            float errorPertMinus = 0.0f;
            for (int s = 0; s < samplePts.size(); ++s)
            {
                float fPert = blendCircleSDFs(samplePts[s], cPertMinus, rPertMinus);
                float gt = sdfGT[s];
                float errPert = fPert - gt;
                errorPertMinus += errPert * errPert;
            }

            // Central difference gradient
            gradLoss = (errorPertPlus - errorPertMinus) / (2.0f * eps);


            gradOut[j] = gradLoss;
        }

     

        return loss;
    }

    void backward(const std::vector<float>& gradOut, float lr)
    {
        std::vector<float> gradHidden(hiddenDim);

        for (int i = 0; i < hiddenDim; ++i)
        {
            gradHidden[i] = 0;
            for (int j = 0; j < outputDim; ++j)
            {
                gradHidden[i] += gradOut[j] * W2[j][i];
                W2[j][i] -= lr * gradOut[j] * hidden[i];
            }
        }

        for (int j = 0; j < outputDim; ++j)
        {
            b2[j] -= lr * gradOut[j];
        }

    }
};



float polygonSDF(std::vector<zVector>& poly, zVector& p)
{
   /* float minDist = 1e6;
    int N = poly.size();
    for (int i = 0; i < N; i++)
    {
        zVector a = poly[i];
        zVector b = poly[(i + 1) % N];
        zVector pa = p - a;
        zVector ba = b - a;
        float h = std::clamp((pa * ba) / (ba * ba), 0.0f, 1.0f);
        zVector proj = a + ba * h;
        minDist = std::min(minDist, p.distanceTo(proj));
    }

    int crossings = 0;
    for (int i = 0; i < N; i++)
    {
        zVector a = poly[i];
        zVector b = poly[(i + 1) % N];
        if (((a.y > p.y) != (b.y > p.y)) &&
            (p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y + 1e-6) + a.x))
        {
            crossings++;
        }
    }

    float sign = (crossings % 2 == 0) ? 1.0f : -1.0f;
    return minDist * sign;*/

    float minDist = 1e6f;
    for (int i = 0; i < poly.size(); i++)
    {
        zVector a = poly[i];
        zVector b = poly[(i + 1) % poly.size()];
        zVector ab = b - a;
        zVector ap = p - a;

        float t = std::clamp((ap * ab) / (ab * ab), 0.0f, 1.0f);
        zVector closest = a + ab * t;
        float dist = p.distanceTo(closest);
        minDist = std::min(minDist, dist);
    }

    bool inside = isInsidePolygon(p, poly);
    return inside ? -minDist : minDist;
}

void computeSampleData()
{
    samplePts.clear();
    sdfGT.clear();
    for (int i = -50; i <= 50; i += 5)
    {
        for (int j = -50; j <= 50; j += 5)
        {

            zVector pt(i, j, 0);

            if (!isInsidePolygon(pt, polygon))continue;

            samplePts.push_back(pt);
            sdfGT.push_back(polygonSDF(polygon, pt));
        }
    }
}

bool train = false;
std::vector<float> input(NUM_CIRCLES * 3, 0.0f);
std::vector<float> gradOut;
MLP mlp;
ScalarField2D sf;
bool drawSF = false;

void trainCircleSDF()
{

    mlp = MLP(NUM_CIRCLES * 3, 32, 3 * NUM_CIRCLES);

}

//---------------------------------

void setup()
{
    polygon.clear();
    polygon = loadPolygonFromCSV("data/polygon.txt");
    computeSampleData();

    S.addSlider(&threshold, "iso");
    S.sliders[0].minVal = -1;


    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
}

void draw()
{
    backGround(0.9);
    drawGrid(50);

    glDisable(GL_LINE_STIPPLE);
    glColor3f(1, 0, 0);
    glLineWidth(5);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < polygon.size() - 1; i++)
    {

        drawLine(zVecToAliceVec(polygon[i]), zVecToAliceVec(polygon[i + 1]));
    }
    glEnd();
    glLineWidth(1);

    glColor3f(0, 0, 1);
    for (int i = 0; i < fittedCenters.size(); i++)
    {
        drawCircle(zVecToAliceVec(fittedCenters[i]), fittedRadii[i], 32);
    }

    for (int i = 0; i < samplePts.size() - 1; i++)
    {

        drawPoint(zVecToAliceVec(samplePts[i]));
    }

    if (drawSF)
    {
        sf.drawFieldPoints();
        sf.drawIsocontours(threshold);
    }
}


void update(int val)
{
    if (!train)return;

    keyPress('n', 0, 0);
}

void keyPress(unsigned char k, int xm, int ym)
{
    if (k == 't')
    {
        trainCircleSDF();
        train = true;
    }

    if (k == 'n')
    {
        for (int epoch = 0; epoch < 150; ++epoch)
        {
            float loss = mlp.computeLossAndGradient(input, polygon, gradOut);
            mlp.backward(gradOut, 0.001);
            std::cout << "Epoch " << ", Loss: " << loss << std::endl;
        }
    }

    if (k == 'd')
    {
        train = false;
        drawSF = true;
        float span = 100.0f; // from -50 to +50
        float step = span / (sf.RES - 1); // spacing between grid points

        //fittedCenters.clear();
        ///fittedRadii.clear();

        /*for (int i = 0; i < 5; i++)
        {
            int n = ofRandom(0, samplePts.size() - 1);
            fittedCenters.push_back(samplePts[n]);
            fittedRadii.push_back(15);
        }*/

        for (int i = 0, n = 0; i < sf.RES; i++, n++)
        {
            for (int j = 0, m = 0; j < sf.RES; j++, m++)
            {
                float x = -50.0f + i * step;
                float y = -50.0f + j * step;
                float f = //blendCircleSDFs(zVector(x, y, 0), fittedCenters, fittedRadii);
                    polygonSDF(polygon, zVector(x, y, 0));
                sf.field[n][m] = f;

                //if (n == 1 )cout << f << endl;
            }

        }

        sf.rescaleFieldToRange(-1, 1);
    }
}

void mousePress(int b, int s, int x, int y)
{

}
void mouseMotion(int x, int y)
{

}

#endif