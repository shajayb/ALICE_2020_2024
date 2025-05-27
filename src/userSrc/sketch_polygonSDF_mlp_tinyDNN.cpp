


#define _MAIN_
#ifdef _MAIN_

#include <tiny_dnn/config.h>
#include <tiny_dnn/tiny_dnn.h>

#include "main.h"
#include <vector>
#include <cmath>


#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>


using namespace zSpace;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;

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

#include "scalarField.h" ;

vector<zVector> loadPolygonFromCSV( const std::string& filename)
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

//--------------------------------------------------------------
inline float polygonSDF( zVector& p,  std::vector<zVector>& poly)
{
    float minDist = 1e6;
    int n = poly.size();
    for (int i = 0; i < n; ++i)
    {
        zVector a = poly[i];
        zVector b = poly[(i + 1) % n];
        zVector ab = b - a;
        zVector ap = p - a;
        float t = std::clamp((ab * ap) / (ab*ab), 0.0f, 1.0f);
        zVector proj = a + ab * t;
        minDist = std::min(minDist, (p - proj).length());
    }

    // simple point-in-polygon (ray-casting)
    int count = 0;
    for (int i = 0; i < n; ++i)
    {
        zVector a = poly[i];
        zVector b = poly[(i + 1) % n];
        if (((a.y > p.y) != (b.y > p.y)) &&
            (p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x))
        {
            count++;
        }
    }

    return (count % 2 == 0) ? minDist : -minDist;
}

//--------------------------------------------------------------
std::vector<zVector> polygon;
std::vector<tiny_dnn::vec_t> inputs;
std::vector<tiny_dnn::vec_t> outputs;

tiny_dnn::network<tiny_dnn::sequential> net;
bool trained = false;

//--------------------------------------------------------------
void setup()
{
    polygon.clear();
    polygon = loadPolygonFromCSV("data/polygon.txt");
}

//--------------------------------------------------------------
void update(int value) {}

//--------------------------------------------------------------
void draw()
{
    backGround(0.9);
    drawGrid(20);

    // draw polygon
    glColor3f(0, 0, 0);
    for (int i = 0; i < polygon.size(); ++i)
    {
        drawLine(zVecToAliceVec(polygon[i]), zVecToAliceVec(polygon[(i + 1) % polygon.size()]));
    }

    if (trained)
    {
        for (float x = -50; x <= 50; x += 1.0f)
        {
            for (float y = -50; y <= 50; y += 1.0f)
            {
                tiny_dnn::vec_t in = { x, y };
                float sdf = net.predict(in)[0];
                float r, g, b;
                getJetColor(sdf / 10.0f, r, g, b);
                glColor3f(r, g, b);
                drawPoint(zVecToAliceVec(zVector(x, y, 0)));
            }
        }
    }
}

//--------------------------------------------------------------
void keyPress(unsigned char k, int xm, int ym)
{
    if (k == 'm')
    {
        inputs.clear();
        outputs.clear();

        for (float x = -50; x <= 50; x += 5.0f)
        {
            for (float y = -50; y <= 50; y += 5.0f)
            {
                zVector pt(x, y, 0);
                float sdf = polygonSDF(pt, polygon);

                inputs.push_back({ x, y });
                outputs.push_back({ sdf });
            }
        }

        std::cout << "Generated " << inputs.size() << " training samples.\n";
    }

    if (k == 't')
    {
        net << tiny_dnn::fully_connected_layer(2, 32)
            << tiny_dnn::activation::relu()
            << tiny_dnn::fully_connected_layer(32, 32)
            << tiny_dnn::activation::relu()
            << tiny_dnn::fully_connected_layer(32, 1);


        tiny_dnn::adagrad optimizer;
        net.train<tiny_dnn::mse>(optimizer, inputs, outputs, 16, 50);

        trained = true;
        std::cout << "MLP training complete.\n";
    }

    if (k == 'e')
    {
        std::ofstream out("data/sdf_samples.csv");
        for (int i = 0; i < inputs.size(); i++)
        {
            out << inputs[i][0] << "," << inputs[i][1] << "," << outputs[i][0] << "\n";
        }
        out.close();
    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_


