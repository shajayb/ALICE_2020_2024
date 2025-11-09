#define _MAIN_
#ifdef _MAIN_

#include "main.h"

#include <vector>
#include <cmath>
#include <fstream>
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;



//------------------------------------------------------------------ Utility
Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

zVector AliceVecToZvec(Alice::vec& in)
{
    return zVector(in.x, in.y, in.z);
}

#include "scalarField.h" //// two functions must be turned on in scalarfIELD.H for sketch_circleSDF_fitter.cpp
#include "genericMLP.h" 

//#include "OT/OptimalTransport_proximal.h" 

/// --------- sub class

ScalarField2D OT_field;

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cstdint>

using namespace std;

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <algorithm>

using namespace std;

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <algorithm>

using namespace std;

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <algorithm>

using namespace std;

namespace ScalarFieldIO
{

#pragma pack(push, 1)
    struct MyBMPFileHeader
    {
        uint16_t bfType;
        uint32_t bfSize;
        uint16_t bfReserved1;
        uint16_t bfReserved2;
        uint32_t bfOffBits;
    };

    struct MyBMPInfoHeader
    {
        uint32_t biSize;
        int32_t  biWidth;
        int32_t  biHeight;
        uint16_t biPlanes;
        uint16_t biBitCount;
        uint32_t biCompression;
        uint32_t biSizeImage;
        int32_t  biXPelsPerMeter;
        int32_t  biYPelsPerMeter;
        uint32_t biClrUsed;
        uint32_t biClrImportant;
    };
#pragma pack(pop)

    template <int RES>
    void loadBMPToScalarFieldRGB( string& filename, float field[RES][RES])
    {
        ifstream file(filename, ios::binary);
        assert(file.is_open());

        MyBMPFileHeader fileHeader;
        MyBMPInfoHeader infoHeader;

        file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
        assert(fileHeader.bfType == 0x4D42);  // 'BM'

        file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));
        assert(infoHeader.biBitCount == 24);  // Expect 24-bit RGB

        int width = infoHeader.biWidth;
        int height = infoHeader.biHeight;

        int rowSize = ((width * 3 + 3) & ~3); // 3 bytes per pixel, 4-byte aligned rows

        file.seekg(fileHeader.bfOffBits, ios::beg);

        vector<unsigned char> row(rowSize);

        for (int y = height - 1; y >= 0; --y)
        {
            file.read(reinterpret_cast<char*>(row.data()), rowSize);
            for (int x = 0; x < width; ++x)
            {
                int idx = x * 3;
                unsigned char B = row[idx + 0];
                unsigned char G = row[idx + 1];
                unsigned char R = row[idx + 2];

                float gray = (R + G + B) / (3.0f * 255.0f);

                // Clamp to your RES
                if (x < RES && y < RES)
                {
                    field[y][x] = gray;
                }
            }
        }

        file.close();
    }

    template <int RES>
    void saveScalarFieldToBMP_RGB(
         string& templateBMP,
         string& outputBMP,
        float field[RES][RES])
    {
        // Read header from template BMP
        ifstream file(templateBMP, ios::binary);
        assert(file.is_open());

        MyBMPFileHeader fileHeader;
        MyBMPInfoHeader infoHeader;

        file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
        assert(fileHeader.bfType == 0x4D42);

        file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));
        assert(infoHeader.biBitCount == 24);

        int width = infoHeader.biWidth;
        int height = infoHeader.biHeight;
        int rowSize = ((width * 3 + 3) & ~3);

        file.close();

        // Open output BMP
        ofstream out(outputBMP, ios::binary);
        assert(out.is_open());

        // Write headers exactly as template
        out.write(reinterpret_cast< char*>(&fileHeader), sizeof(fileHeader));
        out.write(reinterpret_cast< char*>(&infoHeader), sizeof(infoHeader));

        // If template has a palette: none for 24-bit so we skip

        // Write pixel data — bottom up
        vector<unsigned char> row(rowSize, 0);

        for (int y = height - 1; y >= 0; --y)
        {
            for (int x = 0; x < width; ++x)
            {
                // Clamp i,j to RES grid
                int i = min(x, RES - 1);
                int j = min(y, RES - 1);

                float gray = field[j][i];
                gray = std::clamp(gray, 0.0f, 1.0f);
                unsigned char g = static_cast<unsigned char>(gray * 255.0f);

                row[x * 3 + 0] = g; // B
                row[x * 3 + 1] = g; // G
                row[x * 3 + 2] = g; // R
            }

            out.write(reinterpret_cast<char*>(row.data()), rowSize);
        }

        out.close();
    }

} // namespace ScalarFieldIO

using namespace ScalarFieldIO;

double tv = 0;

void setup()
{

    S.addSlider(&tv, "tv");
    S.sliders[0].minVal = -1;

    // ----- 

    OT_field.addCircleSDF(zVector(0, 0, 0), 15);
}

bool run = false;
void update(int value)
{
    
}

void draw()
{
    backGround(0.9);
    drawGrid(50);

   
    OT_field.drawFieldPoints();
    OT_field.drawIsocontours(tv);
}

int n = 0;
void keyPress(unsigned char k, int xm, int ym)
{

    if (k == 'i')
    {
        expr int RES = 128;

        char s[200];
        sprintf(s, "data/out%i.bmp", n);

        cout << s << endl;
        loadBMPToScalarFieldRGB<RES>(s, OT_field.field);

        OT_field.rescaleFieldToRange(-1, 1);

        n++;
        if (n >= 30)n = 0;
    }

    if (k == '1')
    {

        OT_field.clearField();


        int n = 2;
        float inc = TWO_PI / float(n);
        float r = 15;
        vector<zVector> centers;

        for (int i = 0; i < n; i++)
        {
            float x = r * sin(inc * i);
            float y = r * cos(inc * i);

            centers.push_back(zVector(x, y, 0));
        }

        OT_field.addCircleSDFs(centers, 8); 
        OT_field.clampNeg();
        OT_field.normalise();
       

        expr int RES = 128;
        saveScalarFieldToBMP_RGB<RES>(
            "scalarField1_gray.bmp",      // template BMP for header
            "data/scalarField1_gray.bmp",  // output filename
            OT_field.field    // your scalar field
        );

        cout << "scalarField1_gray.bmp" << endl;
    }

    if (k == '2')
    {

        OT_field.clearField();


        int n = 4;
        float inc = TWO_PI / float(n);
        float r = 25;
        vector<zVector> centers;

        for (int i = 0; i < n; i++)
        {
            float x = r * sin(inc * i);
            float y = r * cos(inc * i);

            centers.push_back(zVector(x, y, 0));
        }

        OT_field.addCircleSDFs(centers, 8);
        OT_field.clampNeg();
        OT_field.normalise();

        expr int RES = 128;
        saveScalarFieldToBMP_RGB<RES>(
            "scalarField1_gray.bmp",      // template BMP for header
            "data/scalarField2_gray.bmp",  // output filename
            OT_field.field    // your scalar field
        );

        cout << "scalarField2_gray.bmp" << endl;
    }

}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_
