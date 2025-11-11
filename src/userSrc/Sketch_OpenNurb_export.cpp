#define _MAIN_
#ifdef _MAIN_

#include "main.h"

// - ----------- SYSTEM -----------
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <map>

// - ----------- ZSPACE -----------

#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;

// - ----------- UTLITIES  -----------

Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

bool pointInsidePolygon( zVector& pt,  std::vector<zVector>& poly)
{
    int crossings = 0;
    int N = poly.size();

    for (int i = 0; i < N; ++i)
    {
         zVector& a = poly[i];
         zVector& b = poly[(i + 1) % N];

        // Only consider edges crossing the horizontal line
        if (((a.y > pt.y) != (b.y > pt.y)))
        {
            float t = (pt.y - a.y) / (b.y - a.y);
            float xCross = a.x + t * (b.x - a.x);

            if (pt.x < xCross)
                crossings++;
        }
    }

    return (crossings % 2 == 1); // inside if odd
}

bool loadPolygonFromFile( std::string& filePath, std::vector<zVector>& polygon)
{
    polygon.clear();
    std::ifstream file(filePath);

    if (!file.is_open())
    {
        std::cerr << "Failed to open polygon file: " << filePath << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string xStr, yStr, zStr;

        if (std::getline(ss, xStr, ',') && std::getline(ss, yStr, ',') && std::getline(ss, zStr, ','))
        {
            float x = std::stof(xStr);
            float y = std::stof(yStr);
            float z = std::stof(zStr);

            polygon.push_back(zVector(x, y, z));
        }
    }

    file.close();
    return !polygon.empty();
}

// - ----------- OPEN NURBS  -----------


// defining OPENNURBS_PUBLIC_INSTALL_DIR enables automatic linking using pragmas
#define OPENNURBS_PUBLIC_INSTALL_DIR "C:/Users/shajay.b/source/repos/opennurbs"
// uncomment the next line if you want to use opennurbs as a DLL
#define OPENNURBS_IMPORTS
#include "C:/Users/shajay.b/source/repos/opennurbs/opennurbs_public.h"


// - ----------- OTHER CUSTOM CLASSES  -----------

#include "scalarField.h"
#include "genericMLP.h"
#include "parcel.h"

// - ----------- ----------- MVC APPLICATION  ---------------------- -----------


// dummy test of non-ON class;
parcel plot;

void setup()
{
    // Init OpenNURBS
    ON::Begin();


    ON_3dPointArray pts;
    pts.Append(ON_3dPoint(0, 0, 0));
    pts.Append(ON_3dPoint(10, 0, 0));
    pts.Append(ON_3dPoint(10, 10, 0));
    pts.Append(ON_3dPoint(0, 10, 0));
    pts.Append(ON_3dPoint(5, 5, 0));
    pts.Append(ON_3dPoint(0, 0, 0)); // close

    ON_PolylineCurve* plc = new ON_PolylineCurve(pts);

    // Model + attributes
    ONX_Model model;
    model.m_properties.m_Notes.m_notes = L"Simple polyline via OpenNURBS";
    model.m_properties.m_Notes.m_bVisible = true;

    ON_3dmObjectAttributes attr;
    attr.m_name = L"SimplePolyline";

    model.AddModelGeometryComponent(plc, &attr);

    // --- Easiest: write via filename overload (avoids archive signature issues)
     wchar_t* outfile = L"data/simplest_polyline.3dm";
    bool ok = model.Write(outfile);             // writes using default/latest version
    // If you want a specific version (e.g., Rhino 6 = 60, 7 = 70) and your SDK supports it:
    // bool ok = model.Write(outfile, 60);


    ok ? printf("Wrote 3dm: simple_polyline.3dm\n") : printf("Failed to write 3dm file.\n");

    ON::End();

    // dummy test of non-ON class, forcing lazy loading by calling function;
    plot.addCenter();
}

void update(int value)
{
   
}

void draw()
{
    backGround(0.45);
    drawGrid(50);

}




void keyPress(unsigned char k, int xm, int ym)
{
}

void mousePress(int b, int state, int x, int y)
{
}

void mouseMotion(int x, int y)
{
}

#endif _MAIN_