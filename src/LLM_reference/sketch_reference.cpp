#define _MAIN_
#ifdef _MAIN_

#include "main.h"

#include <vector>
#include <algorithm>
#include <cmath>


#include "RhinoIO.h"

// ------------------- Utility / helper -------------------

// MVC - 
 // Model - 2D/3D world / scene into which you put objects like points, lines, triangles
 // View - refers to the 3D view that we manipulate by cliking left, right, and middle mouse button
 // Controller - refers to key board and mouse

// green run button : builds or compiles the code into a spoftware called ALice.exe and then launches the program (Alice.exe)

// ------------------- MARCHING SQUARES DATA -------------------



float isoValue = 0.0f;   // the true batwing contour
float sliceHeight = 0.0f;   // z value used in sin(z)
int sliceIndex = 0;
float freq = 1.0f;


// ---------------------- FUNCTIONS

float convert_d_to_t(float d)
{

    float maxD = 29.0f * 1.41421356f; // 29 * sqrt(2)
    float t = d / maxD;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    return t;
}

// Proper RGB rainbow color map (HSV-style)
inline void getRGBColor(float t, float& r, float& g, float& b)
{
    t = std::clamp(t, 0.0f, 1.0f);

    float h = t * 6.0f;  // range: 0..6
    int   i = int(h);    // integer segment
    float f = h - i;     // fractional part

    switch (i % 6)
    {
    case 0: r = 1.0f;     g = f;         b = 0.0f;       break; // Red → Yellow
    case 1: r = 1.0f - f; g = 1.0f;      b = 0.0f;       break; // Yellow → Green
    case 2: r = 0.0f;     g = 1.0f;      b = f;         break; // Green → Cyan
    case 3: r = 0.0f;     g = 1.0f - f;  b = 1.0f;       break; // Cyan → Blue
    case 4: r = f;        g = 0.0f;      b = 1.0f;       break; // Blue → Magenta
    case 5: r = 1.0f;     g = 0.0f;      b = 1.0f - f;   break; // Magenta → Red
    }
}

void draw_point_in_color(float x, Alice::vec pt)
{
    float r, g, b;
    getRGBColor(x, r, g, b);   // <-- use new RGB color function

    glColor3f(r, g, b);
    drawPoint(pt);

    // direction tick (unchanged)
    Alice::vec ptcopy = pt;
    ptcopy.normalise();
    Alice::vec pt_c = pt + ptcopy;
    glColor3f(0, 0, 0);
    drawLine(pt, pt_c);
}



///  -------------------------------------- marching square helper functions

struct Segment {
    Alice::vec a;
    Alice::vec b;
};



// ---------------------------------------  CLASS FUNCTIONS


class scalarField2D
{

public:

    //class variables
    static const int RES = 50;
    Alice::vec gridPts[RES][RES];
    float      distVals[RES][RES];

    std::vector<Segment> g_contours;    // global container
    std::vector<std::vector<Segment>> contourStack;   // stores all slices


    //class methods()

    void build_grid_of_points()
    {

        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                float x = -25 + i;   // match your draw-for-loop
                float y = -25 + j;

                gridPts[i][j] = Alice::vec(x, y, 0);
            }
        }
    }

    void calculate_distance_values()
    {
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                // convert Alice::vec → zVector
                Alice::vec ap = gridPts[i][j];

                float x = ap.x * 0.1f;   // scale so sinh() behaves nicely
                float y = ap.y * 0.1f;

                float batwing = sin(sliceHeight) - sinh(x) * sinh(y);

                distVals[i][j] = batwing;

            }
        }
    }


    Alice::vec interp(Alice::vec p1, Alice::vec p2, float v1, float v2, float iso)
    {
        float t = (iso - v1) / ((v2 - v1) + 1e-9f);
        return p1 + (p2 - p1) * t;
    }

    void marchCell(
        Alice::vec p00, Alice::vec p10,
        Alice::vec p01, Alice::vec p11,
        float v00, float v10,
        float v01, float v11,
        float iso)
    {
        int caseID = 0;
        if (v00 > iso) caseID |= 1;
        if (v10 > iso) caseID |= 2;
        if (v11 > iso) caseID |= 4;
        if (v01 > iso) caseID |= 8;

        // list of intersection points
        Alice::vec e[4];
        bool edgeOn[4] = { false,false,false,false };

        // bottom (p00->p10)
        if ((v00 > iso) != (v10 > iso)) {
            e[0] = interp(p00, p10, v00, v10, iso);
            edgeOn[0] = true;
        }

        // right (p10->p11)
        if ((v10 > iso) != (v11 > iso)) {
            e[1] = interp(p10, p11, v10, v11, iso);
            edgeOn[1] = true;
        }

        // top (p11->p01)
        if ((v11 > iso) != (v01 > iso)) {
            e[2] = interp(p11, p01, v11, v01, iso);
            edgeOn[2] = true;
        }

        // left (p01->p00)
        if ((v01 > iso) != (v00 > iso)) {
            e[3] = interp(p01, p00, v01, v00, iso);
            edgeOn[3] = true;
        }

        // Build line segments depending on case
        // Each marching squares case has 0, 1, or 2 segments
        static const int lookup[16][4] = {
            {-1,-1,-1,-1},  // 0
            { 3,0,-1,-1 },  // 1
            { 0,1,-1,-1 },  // 2
            { 3,1,-1,-1 },  // 3
            { 1,2,-1,-1 },  // 4
            { 3,0,1,2 },    // 5 (two segments)
            { 0,2,-1,-1 },  // 6
            { 3,2,-1,-1 },  // 7
            { 2,3,-1,-1 },  // 8
            { 2,0,-1,-1 },  // 9
            { 0,1,2,3 },    // 10 (two segments)
            { 2,1,-1,-1 },  // 11
            { 1,3,-1,-1 },  // 12
            { 1,0,-1,-1 },  // 13
            { 0,3,-1,-1 },  // 14
            {-1,-1,-1,-1}   // 15
        };

        const int* L = lookup[caseID];

        if (L[0] != -1 && L[1] != -1) {
            g_contours.push_back({ e[L[0]], e[L[1]] });
        }
        if (L[2] != -1 && L[3] != -1) {
            g_contours.push_back({ e[L[2]], e[L[3]] });
        }
    }


    void computeContours(float iso)
    {
        g_contours.clear();

        for (int i = 0; i < 49; i++)
        {
            for (int j = 0; j < 49; j++)
            {
                Alice::vec p00 = gridPts[i][j];
                Alice::vec p10 = gridPts[i + 1][j];
                Alice::vec p01 = gridPts[i][j + 1];
                Alice::vec p11 = gridPts[i + 1][j + 1];

                float v00 = distVals[i][j];
                float v10 = distVals[i + 1][j];
                float v01 = distVals[i][j + 1];
                float v11 = distVals[i + 1][j + 1];

                marchCell(p00, p10, p01, p11, v00, v10, v01, v11, iso);
            }
        }

        // after computing g_contours:
        float zOffset = sliceIndex * 0.2f;   // adjust spacing as needed

        // apply z-offset to all segments in this slice
        for (auto& seg : g_contours)
        {
            seg.a.z = zOffset;
            seg.b.z = zOffset;
        }

        contourStack.push_back(g_contours);
        sliceIndex++;


    }

    void drawContours()
    {
        glLineWidth(2);
        glColor3f(0, 0, 0);

        for (auto& s : g_contours)
            drawLine(s.a, s.b);

        glLineWidth(1);
    }

    void drawContourStack()
    {
        glLineWidth(1.5f);

        for (auto& slice : contourStack)
        {
            for (auto& s : slice)
            {
                glColor3f(0, 0, 0);
                drawLine(s.a, s.b);
            }
        }
    }

    void drawField()
    {
        float mn = 1e9;
        float mx = -1e9;

        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                float v = distVals[i][j];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
        }

        // Avoid division by zero
        float denom = (mx - mn);
        if (denom < 1e-9) denom = 1.0f;

        // --------------------------------------------------
        // 2. Draw points using normalized values
        // --------------------------------------------------
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                Alice::vec pt = gridPts[i][j];

                float v = distVals[i][j];
                float t = (v - mn) / denom;        // normalized to [0,1]

                t = std::clamp(t, 0.0f, 1.0f);

                draw_point_in_color(t, pt);
            }
        }
    }
};

///  -------------------------------------- marching square helper functions

// ------------------- APPLICATION CODE  -------------------

Alice::vec a(5, 5, 0);
scalarField2D myField; // myField is an (global) instance of the class scalarField2D; // gl9obal instacce
RhinoIO myRIO;

// ---------- rhino export helper function

void exportContourStackToRhino()
{
    // Container for RhinoIO
    std::vector<std::vector<zVector>> rhinoCurves;

    // Convert each slice into polylines
    for (auto& slice : myField.contourStack) // for each contour
    {
        for (auto& seg : slice) // for each segment in each contour
        {
            std::vector<zVector> crv;

            zVector a(seg.a.x, seg.a.y, seg.a.z);
            zVector b(seg.b.x, seg.b.y, seg.b.z);

            crv.push_back(a);
            crv.push_back(b);

            rhinoCurves.push_back(crv);
        }
    }

    // Export using RhinoIO
    myRIO.addCurves(
        rhinoCurves,
        1.0f,
        ON_3dPoint(0, 0, 0),
        ON_3dPoint(0, 0, 0));




    bool written = myRIO.Write3dm(L"data/contour_stack.3dm");

    if (written)
        printf("Wrote data/contour_stack.3dm successfully.\n");
    else
        printf("FAILED to write data/contour_stack.3dm\n");
}


void setup()
{
    printf("helllo --- setup \n");

    // 1. build a grid of points
    // to call a function inside a class
    // step 1 : define an instance of the class
    // step 2 : use instance.method() notation

    myField.build_grid_of_points();
    // 2. compute scalar values for each point (distance from origin)
   
    myField.calculate_distance_values();

    // 3. build initial isolines
    myField.computeContours(isoValue);

}

void update(int value) // update somethings in the model / scene
{

}

void draw()
{
    backGround(0.9);
    drawGrid(50);

    glPointSize(5);
    drawPoint(a);


    myField.drawField();
    // ===== Draw contour overlay =====
   
    glColor3f(1, 0, 0);
    myField.drawContours();       // draw the newest slice on top

    glColor3f(0, 0, 0);
    myField.drawContourStack();   // draw ALL slices

    //////////////---------------------------------



}

// ----------------------- 

void keyPress(unsigned char k, int xm, int ym)
{
    if (k == '1') { isoValue -= 1.0f; myField.computeContours( isoValue); }
    if (k == '2') { isoValue += 1.0f; myField.computeContours( isoValue); }

    if (k == 'z')  // go “down" in z
    {
        sliceHeight -= 0.1f;
        printf("sliceHeight = %.2f\n", sliceHeight);

        for (int i = 0; i < myField.RES; i++)
            for (int j = 0; j < myField.RES; j++)
            {
                Alice::vec ap = myField.gridPts[i][j];
                float x = ap.x * 0.1f;
                float y = ap.y * 0.1f;
                myField.distVals[i][j] = sin(sliceHeight) - sinh(x) * sinh(y);
            }

        myField.computeContours(isoValue);
    }

    if (k == 'x')   // go “up” in z
    {
        sliceHeight += 0.1f;
        printf("sliceHeight = %.2f\n", sliceHeight);

        for (int i = 0; i < myField.RES; i++)
            for (int j = 0; j < myField.RES; j++)
            {
                Alice::vec ap = myField.gridPts[i][j];
                float x = ap.x * 0.1f;
                float y = ap.y * 0.1f;
                myField.distVals[i][j] = sin(sliceHeight) - sinh(x) * sinh(y);
            }

        myField.computeContours(isoValue);
    }

    //if (k == '=')  // increase frequency
    //{
    //    freq += 0.1;
    //    printf("freq = %.2f\n", freq);

    //    for (int i = 0; i < RES; i++)
    //        for (int j = 0; j < RES; j++)
    //        {
    //            Alice::vec ap = gridPts[i][j];
    //            float x = ap.x * 0.1f;
    //            float y = ap.y * 0.1f;

    //            distVals[i][j] =
    //                sin(freq * x) * sin(freq * y) - sin(freq * sliceHeight);

    //        }

    //    computeContours(gridPts, distVals, isoValue);
    //}


    if (k == 'e')
    {
        exportContourStackToRhino();
    }


}

void mousePress(int b, int state, int x, int y)
{

}

void mouseMotion(int x, int y)
{

}

#endif // _MAIN_
