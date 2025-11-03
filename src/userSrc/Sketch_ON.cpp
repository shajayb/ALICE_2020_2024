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

bool pointInsidePolygon(const zVector& pt, const std::vector<zVector>& poly)
{
    int crossings = 0;
    int N = poly.size();

    for (int i = 0; i < N; ++i)
    {
        const zVector& a = poly[i];
        const zVector& b = poly[(i + 1) % N];

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


bool loadPolygonFromFile(const std::string& filePath, std::vector<zVector>& polygon)
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

// defining OPENNURBS_PUBLIC_INSTALL_DIR enables automatic linking using pragmas
#define OPENNURBS_PUBLIC_INSTALL_DIR "C:/Users/shajay.b/source/repos/opennurbs"
// uncomment the next line if you want to use opennurbs as a DLL
#define OPENNURBS_IMPORTS
#include "C:/Users/shajay.b/source/repos/opennurbs/opennurbs_public.h"


#define INTERNAL_INITIALIZE_MODEL(model) Internal_SetExampleModelProperties(model,OPENNURBS__FUNCTION__,__FILE__)

static void Internal_SetExampleModelProperties(
    ONX_Model& model,
    const char* function_name,
    const char* source_file_name
)
{
    const bool bHaveFunctionName = (nullptr != function_name && 0 != function_name[0]);
    if (!bHaveFunctionName)
        function_name = "";

    const bool bHaveFileName = (nullptr != source_file_name && 0 != source_file_name[0]);
    if (!bHaveFileName)
        source_file_name = "";

    model.m_sStartSectionComments = "This was file created by OpenNURBS toolkit example code.";

    // set application information
    const ON_wString wide_function_name(function_name);
    const ON_wString wide_source_file_name(source_file_name);
    model.m_properties.m_Application.m_application_name
        = bHaveFunctionName
        ? ON_wString::FormatToString(L"OpenNURBS toolkit Example: %ls() function", static_cast<const wchar_t*>(wide_function_name))
        : ON_wString(L"OpenNURBS Examples");

    model.m_properties.m_Application.m_application_URL = L"http://www.opennurbs.org";
    model.m_properties.m_Application.m_application_details
        = bHaveFileName
        ? ON_wString::FormatToString(L"Opennurbs examples are in the file %ls.", static_cast<const wchar_t*>(wide_source_file_name))
        : ON_wString::FormatToString(L"Opennurbs examples are example_*.cpp files.");

    // some notes
    if (bHaveFunctionName && bHaveFileName)
    {
        model.m_properties.m_Notes.m_notes
            = ON_wString::FormatToString(
                L"This .3dm file was made with the OpenNURBS toolkit example function %s() defined in source code file %ls.",
                static_cast<const wchar_t*>(wide_function_name),
                static_cast<const wchar_t*>(wide_source_file_name));
        model.m_properties.m_Notes.m_bVisible = model.m_properties.m_Notes.m_notes.IsNotEmpty();
    }

    // set revision history information
    model.m_properties.m_RevisionHistory.NewRevision();
}

static bool Internal_WriteExampleModel(
    const ONX_Model& model,
    const wchar_t* filename,
    ON_TextLog& error_log
)
{
    int version = 0;

    // writes model to archive
    return model.Write(filename, version, &error_log);
}

ON_3dmObjectAttributes* Internal_CreateManagedAttributes(
    int layer_index,
    const wchar_t* name
)
{
    ON_3dmObjectAttributes* attributes = new ON_3dmObjectAttributes();
    attributes->m_layer_index = layer_index;
    attributes->m_name = name;
    return attributes;
}

static bool write_curves_example(const wchar_t* filename, ON_TextLog& error_log)
{
    // example demonstrates how to write a NURBS curve, line, and circle
    ONX_Model model;
    INTERNAL_INITIALIZE_MODEL(model);

    // file settings (units, tolerances, views, ...)
    model.m_settings.m_ModelUnitsAndTolerances.m_unit_system = ON::LengthUnitSystem::Inches;
    model.m_settings.m_ModelUnitsAndTolerances.m_absolute_tolerance = 0.001;
    model.m_settings.m_ModelUnitsAndTolerances.m_angle_tolerance = ON_PI / 180.0; // radians
    model.m_settings.m_ModelUnitsAndTolerances.m_relative_tolerance = 0.01; // 1%

    // add some layers
    
    model.AddDefaultLayer(nullptr, ON_Color::UnsetColor);
    const int line_layer_index = model.AddLayer(L"line layer", ON_Color::Black);
    const int wiggle_layer_index = model.AddLayer(L"green NURBS wiggle", ON_Color::SaturatedGreen);
    const int circles_layer_index = model.AddLayer(L"blue circles", ON_Color::SaturatedBlue);

    {
        // add a line
        ON_Object* managed_line = new ON_LineCurve(ON_Line(ON_3dPoint(1.0, 2.0, -1.5), ON_3dPoint(5.0, 3.0, 2.0)));
        model.AddManagedModelGeometryComponent(
            managed_line,
            Internal_CreateManagedAttributes(line_layer_index, L"straight line curve")
        );
    }

    {
        //// add a wiggly cubic curve
        //ON_NurbsCurve* wiggle = new ON_NurbsCurve(
        //    3, // dimension
        //    false, // true if rational
        //    4,     // order = degree+1
        //    6      // number of control vertices
        //);
        //int i;
        //for (i = 0; i < wiggle->CVCount(); i++) {
        //    ON_3dPoint pt(2 * i, -i, (i - 3) * (i - 3)); // pt = some 3d point
        //    wiggle->SetCV(i, pt);
        //}

        //// ON_NurbsCurve's have order+cv_count-2 knots.
        //wiggle->SetKnot(0, 0.0);
        //wiggle->SetKnot(1, 0.0);
        //wiggle->SetKnot(2, 0.0);
        //wiggle->SetKnot(3, 1.5);
        //wiggle->SetKnot(4, 2.3);
        //wiggle->SetKnot(5, 4.0);
        //wiggle->SetKnot(6, 4.0);
        //wiggle->SetKnot(7, 4.0);

        // Create a polycurve that passes through a sequence of 3D points
        std::vector<ON_3dPoint> pts =
        {
            ON_3dPoint(0, 0, 0),
            ON_3dPoint(2, 1, 0),
            ON_3dPoint(4, 0, 1),
            ON_3dPoint(6, -1, 0),
            ON_3dPoint(8, 0, -1)
        };

        // Create an empty polycurve
        ON_PolyCurve* poly = new ON_PolyCurve();

        // Add line segments connecting consecutive points
        for (int i = 0; i < (int)pts.size() - 1; i++)
        {
            // Create a line segment between consecutive points
            ON_Line line(pts[i], pts[i + 1]);

            // Wrap it in an ON_LineCurve (inherits ON_Curve)
            ON_LineCurve* segment = new ON_LineCurve(line);

            // Append the segment to the polycurve
            poly->Append(segment);
        }

        // Optionally close the curve (make it periodic)
        // poly->SetClosedCurve(true);

        // Verify the domain and segment count
        poly->SetDomain(0.0, (double)(pts.size() - 1));
        //printf("PolyCurve with %d segments created.\n", poly->Count());


        model.AddManagedModelGeometryComponent(
            poly,
            Internal_CreateManagedAttributes(wiggle_layer_index, L"wiggly cubic curve")
        );
    }

    {
        // add two circles
        ON_ArcCurve* circle1 = new ON_ArcCurve(ON_Circle(ON_3dPoint(1.0, 2.0, -1.5), 3.0));
        model.AddManagedModelGeometryComponent(
            circle1,
            Internal_CreateManagedAttributes(circles_layer_index, L"radius 3 circle")
        );

        ON_ArcCurve* circle2 = new ON_ArcCurve(ON_Circle(ON_3dPoint(1.0, 2.0, -1.5), 5.0));
        model.AddManagedModelGeometryComponent(
            circle2,
            Internal_CreateManagedAttributes(circles_layer_index, L"radius 5 circle")
        );
    }

    return Internal_WriteExampleModel(model, filename, error_log);
}



void setup()
{
    bool rc = false;
    const wchar_t* filename;

    ON::Begin();
    // If you want to learn to write b-rep models, first work through
    // this example paying close attention to write_trimmed_surface_example(),
    // then examime example_brep.cpp.

    // errors printed to stdout
    ON_TextLog error_log;

    // messages printed to stdout
    ON_TextLog message_log;

    filename = L"alice_curves.3dm";
    rc = write_curves_example(filename, error_log);
    if (rc)
        message_log.Print(L"Successfully wrote %ls.\n", filename);
    else
        message_log.Print(L"Errors while writing %ls.\n", filename);

    ON::End();
  
}


bool compute = false;
void update(int value)
{
   
}

void draw()
{
    backGround(0.45);
    drawGrid(50);

}



double prevLoss = 0.0;
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