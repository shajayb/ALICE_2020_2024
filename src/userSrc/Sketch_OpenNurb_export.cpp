#define _MAIN_
#ifdef _MAIN_

#include "main.h"

// ----------- SYSTEM -----------
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>

// ----------- ZSPACE -----------
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;


// ------------------------------------------------------------
// Utility
// ------------------------------------------------------------

Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

// ----------- OPENNURBS WRAPPER -----------
#include "RhinoIO.h"

// ----------- OTHER CLASSES -----------
#include "scalarField.h"
#include "genericMLP.h"
#include "parcel.h"


// ----------- MVC / GLOBALS -----------
RhinoIO IO_R;     // our one permanent OpenNURBS model manager
parcel plot;

// ------------------------------------------------------------
// SETUP
// ------------------------------------------------------------

void setup()
{
    printf("=== RhinoIO Example Sketch ===\n");

    // --------------------------------------------------------
    // 1. Read a Rhino 3DM file
    // --------------------------------------------------------
    bool ok = IO_R.Read3dm(L"data/rhino_test_read_multi.3dm");

    if (!ok)
    {
        printf("Failed to read file.\n");
        return;
    }
    printf("Successfully read: data/rhino_test_read_multi.3dm\n");


    // --------------------------------------------------------
    // 2. Print layer list from the file
    // --------------------------------------------------------
    IO_R.PrintLayerList();


    // --------------------------------------------------------
    // 3. Group curves by Z elevation (horizontal slices)
    // --------------------------------------------------------
    std::map<int, std::vector<const ON_Curve*>> zGroups;
    IO_R.GroupPlanarHorizontalCurves(zGroups);

    for (auto& kv : zGroups)
    {
        double z = kv.first / 1000.0;
        printf("Z = %.3f → %zu curves\n", z, kv.second.size());
    }


    // --------------------------------------------------------
    // 4. Compute lengths of all curves in the model
    // --------------------------------------------------------
    std::vector<double> lengths;
    IO_R.ComputeAllCurveLengths(lengths);

    printf("\n=== Curve Lengths ===\n");
    for (int i = 0; i < lengths.size(); i++)
    {
        printf("Curve %d length = %.6f\n", i, lengths[i]);
    }


    // --------------------------------------------------------
    // 5. Example: Add a simple polyline into the model
    // --------------------------------------------------------
    std::vector<std::vector<zVector>> examplePoly;

    examplePoly.push_back({
        zVector(0, 0, 0),
        zVector(10, 0, 0),
        zVector(10, 10, 0),
        zVector(0, 10, 0)
        });

    IO_R.addCurves(examplePoly, 1.0f);

    // --------------------------------------------------------
    // 6. Write the updated file back out
    // --------------------------------------------------------
    IO_R.Write3dm(L"data/output_from_RhinoIO.3dm");

    printf("\nWrote: data/output_from_RhinoIO.3dm\n");


    // Force parcel class lazy loading
    plot.addCenter();
}


// ------------------------------------------------------------
// UPDATE
// ------------------------------------------------------------

void update(int value)
{
}


// ------------------------------------------------------------
// DRAW
// ------------------------------------------------------------

void draw()
{
    backGround(0.45);
    drawGrid(50);
}


// ------------------------------------------------------------
// INPUT
// ------------------------------------------------------------

void keyPress(unsigned char k, int xm, int ym)
{
}

void mousePress(int b, int state, int x, int y)
{
}

void mouseMotion(int x, int y)
{
}

#endif // _MAIN_
