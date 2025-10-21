#define _MAIN_
#ifdef _MAIN_

#include "main.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>


using namespace zSpace;

Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

#include "scalarField.h"

#pragma once

#include "scalarField.h"
#include <fstream>
#include <sstream>
#include <map>

class HeightField2D : public ScalarField2D
{
public:

    std::vector<zVector> samples;
    double zMin = 0.0f, zMax = 1.0f;
    double zScale = 5;

    void readSamplesAndInterpolate(const std::string& filename)
    {
        samples.clear();
        zMin = 0.0f, zMax = 1.0f;
        zScale = 5;

        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Failed to open " << filename << std::endl;
            return;
        }

        std::string line;
        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string xStr, yStr, zStr;

            if (std::getline(ss, xStr, ',') &&
                std::getline(ss, yStr, ',') &&
                std::getline(ss, zStr))
            {
                float x = std::stof(xStr);
                float y = std::stof(yStr);
                float z = std::stof(zStr);
                samples.emplace_back(x, y, z);
            }
        }

        file.close();

        rescaleSamplesToBoundingBox(zVector(-50, -50, -50), zVector(50, 50, 50));

        interpolateToGrid();
    }

    //void rescaleSamplesToBoundingBox(zVector& targetMin, zVector& targetMax)
    //{
    //    if (samples.empty()) return;

    //    zVector bmin = samples[0];
    //    zVector bmax = samples[0];

    //    for (const auto& s : samples)
    //    {
    //        bmin.x = std::min(bmin.x, s.x);
    //        bmin.y = std::min(bmin.y, s.y);
    //        bmax.x = std::max(bmax.x, s.x);
    //        bmax.y = std::max(bmax.y, s.y);
    //    }

    //    zVector scale = targetMax - targetMin;
    //    zVector dataRange = bmax - bmin;

    //    for (auto& s : samples)
    //    {
    //        s.x = targetMin.x + (s.x - bmin.x) / dataRange.x * scale.x;
    //        s.y = targetMin.y + (s.y - bmin.y) / dataRange.y * scale.y;
    //        // z remains untouched for now
    //    }

    //    // compute zMin/zMax for later use
    //    zMin = samples[0].z;
    //    zMax = samples[0].z;
    //    for (auto& s : samples)
    //    {
    //        zMin = std::min(float(zMin), s.z);
    //        zMax = std::max(float(zMax), s.z);
    //    }

    //    printf("original z-coordinate / height ragnge %.2f,%.2f,\n", zMin, zMax);
    //}

    void rescaleSamplesToBoundingBox(zVector& targetMin, zVector& targetMax)
    {
        if (samples.empty())
        {
            return;
        }

        // --- 1) Source 2D bbox (x,y only)
        zVector bmin = samples[0];
        zVector bmax = samples[0];

        for (const auto& s : samples)
        {
            bmin.x = std::min(bmin.x, s.x);
            bmin.y = std::min(bmin.y, s.y);
            bmax.x = std::max(bmax.x, s.x);
            bmax.y = std::max(bmax.y, s.y);
        }

        zVector src = bmax - bmin;                 // source width/height
        zVector dst = targetMax - targetMin;       // target width/height

        if (src.x < 1e-9f || src.y < 1e-9f)
        {
            printf("rescaleSamplesToBoundingBox: degenerate source bbox.\n");
            return;
        }

        // --- 2) Uniform "contain" scale (never exceeds target on any axis)
        float scale = std::min(dst.x / src.x, dst.y / src.y);

        // --- 3) Centered placement: scale about source center, move to target center
        zVector cSrc = (bmin + bmax) * 0.5f;
        zVector cDst = (targetMin + targetMax) * 0.5f;

        for (auto& s : samples)
        {
            s.x = cDst.x + (s.x - cSrc.x) * scale;
            s.y = cDst.y + (s.y - cSrc.y) * scale;
            // z remains untouched
        }

        // --- 4) z-range (diagnostic)
        zMin = 1e-6;;// samples[0].z;
        zMax = -zMin;// samples[0].z;
        for (const auto& s : samples)
        {
            zMin = std::min(float(zMin), s.z);
            zMax = std::max(float(zMax), s.z);
        }

        printf("rescaleSamplesToBoundingBox: contain scale=%.6f  src(%.3f,%.3f) -> dst(%.3f,%.3f)  z[%.3f,%.3f]\n",
            scale, src.x, src.y, dst.x, dst.y, zMin, zMax);
    }




    void interpolateToGrid()
    {
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                zVector gp = gridPoints[i][j];
                float num = 0.0f;
                float den = 0.0f;

                for (const auto& s : samples)
                {
                    float d = gp.distanceTo(zVector(s.x, s.y, 0));
                    if (d < 1e-3f) d = 1e-3f;

                    float w = 1.0f / (d * d);
                    num += w * s.z;
                    den += w;
                }

                field[i][j] = (den > 0.0f) ? num / den : 0.0f;
            }
        }

        // normalise doesnt affect field values.. normalised values are stored in a separate array for visualisation purposes
        normalise(); 
        rescaleFieldToRange(-1, 1);
    }

    void setGridPointHeights()
    {
        if (samples.empty()) return;

        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                gridPoints[i][j].z = ofMap(field[i][j], -1, 1, -zScale, zScale); ;// ofMap(field_normalized[i][j], 0, 1, -zScale, zScale);
            }
        }
    }


    void drawSamplePoints()
    {
        if (samples.empty()) return;

        glPointSize(1);
        glBegin(GL_POINTS);
        for (const auto& ptRaw : samples)
        {
            zVector pt = ptRaw;
            pt.z = ofMap(ptRaw.z, zMin, zMax, -1.0f, 1.0f) * zScale; // normalized and scaled

            float color = ofMap(pt.z, -zScale, zScale, 0.0f, 1.0f);
            glColor3f(color, 0.0f, 1.0f - color);

            Alice::vec av = zVecToAliceVec(pt);
            glVertex3f(av.x, av.y, av.z);
        }
        glEnd();
        glPointSize(1);
    }

};


HeightField2D myHeightField , myHeightField1;
double threshold;

void setup()
{
    S.numSliders = 0;
    S.addSlider(&threshold, "tv");// make a slider control for the variable called width;
    S.sliders[0].minVal = myHeightField.zScale * -1;
    S.sliders[0].maxVal = myHeightField.zScale;


    myHeightField = *new HeightField2D();
    myHeightField1 = *new HeightField2D();

    myHeightField.clearField();
    myHeightField.readSamplesAndInterpolate("data/cabins_site.txt");
    myHeightField.setGridPointHeights();

    myHeightField1.readSamplesAndInterpolate("data/cabins_site.txt");
    myHeightField1.setGridPointHeights();
}

void update(int value)
{
}

void draw()
{
    backGround(0.9);
    drawGrid(50);


   myHeightField.drawSamplePoints();

    {
        myHeightField.drawFieldPoints(false, false);

        glColor3f(0, 0, 0);
        for (double tv = -myHeightField.zScale; tv < threshold; tv += 0.1) myHeightField.drawIsocontours(tv);
        glLineWidth(1);
    }

   // glTranslatef(120, 0, 0);
    {
       // myHeightField1.drawFieldPoints(false, false);

        glColor3f(1, 0, 0);
       // for (double tv = -myHeightField.zScale; tv < threshold; tv += 0.05) myHeightField1.drawIsocontours(tv);
      
    }


}

void keyPress(unsigned char k, int xm, int ym)
{

    if (k == 's')
    {
        myHeightField.smoothDiffuseIsotropic(0.15, 1, true);
        myHeightField1.smoothDiffuseAnisotropic(0.2, 1, 0.1, ScalarField2D::PMVariant::Exp, ScalarField2D::DiffuseDir::AlongIsophote, 2, true);
    }



    
}

void mousePress(int b, int state, int x, int y)
{
}

void mouseMotion(int x, int y)
{
}

#endif // _MAIN_
