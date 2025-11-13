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

#include "RhinoIO.h"

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
//#define OPENNURBS_PUBLIC_INSTALL_DIR "C:/Users/shajay.b/source/repos/ALICE_2020_2024/src/openNurbs" ; //C:/Users/shajay.b/source/repos/opennurbs"
// uncomment the next line if you want to use opennurbs as a DLL
#define OPENNURBS_IMPORTS
#include "opennurbs_public.h"



// - ----------- OTHER CUSTOM CLASSES  -----------

#include "scalarField.h"
#include "genericMLP.h"
#include "parcel.h"

// - ----------- ----------- MVC APPLICATION  ---------------------- -----------


// dummy test of non-ON class;
parcel plot;

RhinoIO IO_R;

double ComputeCurveLength(const ON_Curve* crv)
{
    if (!crv) return 0.0;

    // Convert ANY curve type to a NURBS curve
    ON_NurbsCurve nurbs;
    double length = 0.0;

    const int N = 256;


    ON_Interval dom = crv->Domain();
    double t0 = dom.Min(), t1 = dom.Max();

    ON_3dPoint p_prev = crv->PointAt(t0);

    for (int i = 1; i <= N; i++)
    {
        double t = t0 + (t1 - t0) * (double(i) / double(N));
        ON_3dPoint p = crv->PointAt(t);
        length += p.DistanceTo(p_prev);
        p_prev = p;
    }


    return length;
}


void setup()
{
    // Init OpenNURBS
   
    // ------------------------
    ON::Begin();
    // ------------------------

    // --------------- write ----------

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

    // --------------- read ----------

    ONX_Model read_model;
    ON_TextLog dump;

    //dump.Print("\nOpenNURBS Archive File:  %ls\n", sFileName);

    // open file containing opennurbs archive
    wchar_t* infile = L"data/rhino_test_read_multi.3dm";
    FILE* archive_fp = ON::OpenFile(infile, L"rb");
    if (!archive_fp)
    {
        printf("  Unable to open file.\n");

        ON::End();
        return ;
    }

    // create achive object from file pointer
    ON_BinaryFile archive(ON::archive_mode::read3dm, archive_fp);

    // read the contents of the file into "model"
    bool rc = read_model.Read(archive, &dump);

    // close the file
    ON::CloseFile(archive_fp);

    // -------------------------------- iterate through model

    ONX_ModelComponentIterator global_layer_it(read_model, ON_ModelComponent::Type::Layer);
    for (ON_ModelComponentReference mcr = global_layer_it.FirstComponentReference();
        mcr.IsEmpty() == false;
        mcr = global_layer_it.NextComponentReference())
    {
        const ON_Layer* layer = ON_Layer::Cast(mcr.ModelComponent());
        if (!layer)
            continue;

        std::wcout << L"  [" << layer->Index() << L"] "
            << layer->Name().Array() << std::endl;
    }

    //---------------

    std::wstring searchStr = L"Layer1"; // change as needed
    std::wcout << L"\n🔍 Searching for layers containing: " << searchStr << std::endl;
    const ON_Layer* layer;

    ONX_ModelComponentIterator layer_it(read_model, ON_ModelComponent::Type::Layer);
    for (ON_ModelComponentReference mcr = layer_it.FirstComponentReference();
        mcr.IsEmpty() == false;
        mcr = layer_it.NextComponentReference())
    {
        layer = ON_Layer::Cast(mcr.ModelComponent());

        if (!layer)
            continue;

        std::wstring lname = layer->Name().Array();
        if (lname.find(searchStr) == std::wstring::npos)
            continue;

        std::wcout << L"\n-- Matching Layer: " << lname
            << L" (Index: " << layer->Index() << L") --" << std::endl;
 
    }

    //-----------------------------------------------------------------
  // Iterate model geometry
  //-----------------------------------------------------------------
    //ONX_ModelComponentIterator geo_it(read_model, ON_ModelComponent::Type::ModelGeometry);
    //for (ON_ModelComponentReference gref = geo_it.FirstComponentReference();
    //    gref.IsEmpty() == false;
    //    gref = geo_it.NextComponentReference())
    //{
    //    const ON_ModelGeometryComponent* geo_comp = ON_ModelGeometryComponent::Cast(gref.ModelComponent());
    //    if (!geo_comp)
    //        continue;

    //    //const ON_3dmObjectAttributes* attr = geo_comp->Attributes(nullptr);
    //    //if (!attr || attr->m_layer_index != layer->Index())
    //    //    continue;

    //    const ON_Geometry* geom = geo_comp->Geometry(nullptr);
    //    if (!geom)
    //        continue;

    //   

    //    if (const ON_Curve* crv = ON_Curve::Cast(geom))
    //    {

    //        ON_NurbsCurve nurbs;
    //        if (crv->GetNurbForm(nurbs))
    //        {
    //            printf("    Degree: %d, CV count: %d\n",
    //                nurbs.Degree(), nurbs.CVCount());
    //        }

    //    }
    //    else if (const ON_Surface* srf = ON_Surface::Cast(geom))
    //        std::wcout << L"  • Surface object\n";
    //    else if (const ON_Brep* brep = ON_Brep::Cast(geom))
    //        std::wcout << L"  • Brep object\n";
    //    else if (const ON_Mesh* mesh = ON_Mesh::Cast(geom))
    //        std::wcout << L"  • Mesh object\n";
    //    else
    //        std::wcout << L"  • Other geometry type\n";
    //}

    // tolerance for comparing Z heights
    const double zTolerance = 1e-3;

    // map from rounded Z value → vector of curves at that Z
    std::map<int, std::vector<const ON_Curve*>> zGroups;

    ONX_ModelComponentIterator geo_it(read_model, ON_ModelComponent::Type::ModelGeometry);
    for (ON_ModelComponentReference gref = geo_it.FirstComponentReference();
        !gref.IsEmpty();
        gref = geo_it.NextComponentReference())
    {
        const ON_ModelGeometryComponent* geo_comp =
            ON_ModelGeometryComponent::Cast(gref.ModelComponent());
        if (!geo_comp)
            continue;

        const ON_Geometry* geom = geo_comp->Geometry(nullptr);
        if (!geom)
            continue;

        const ON_Curve* crv = ON_Curve::Cast(geom);
        if (!crv)
            continue; // skip non-curves

        // --- Check if the curve is planar ---
        ON_Plane plane;
        if (!crv->IsPlanar(&plane, zTolerance))
            continue; // skip non-planar curves

        // --- Check if plane is (approximately) parallel to world XY ---
        // Normal near Z axis → consider "horizontal"
        ON_3dVector n = plane.Normal();
        n.Unitize();
        double dotZ = fabs(n * ON_3dVector::ZAxis);
        if (dotZ < 0.999) // ~ within 1 degree of horizontal
            continue;

        // --- Extract the Z height of the plane ---
        double zVal = plane.Origin().z;

        // Quantize Z to avoid floating-point mismatch
        int zKey = (int)std::round(zVal * 1000.0); // precision 1e-3

        // --- Store curve pointer ---
        zGroups[zKey].push_back(crv);

        // Optional: print debug info
        printf("Found planar curve at Z = %.3f\n", zVal);
    }

    // --- Convert map to vector-of-vectors ---
    std::vector<std::vector<const ON_Curve*>> groupedCurvesByZ;
    for (auto& kv : zGroups)
    {
        double zVal = kv.first / 1000.0;      // recover Z height
        const auto& curvesAtZ = kv.second;

        printf("\nZ group %.3f has %zu curves\n", zVal, curvesAtZ.size());

        for (size_t i = 0; i < curvesAtZ.size(); i++)
        {
            const ON_Curve* crv = curvesAtZ[i];
            if (!crv) continue;

            //------------------------------------------------------------
            // 1) BOUNDING BOX
            //------------------------------------------------------------
            ON_BoundingBox bbox;
            if (crv->GetBoundingBox(bbox))
            {
                printf("  Curve %zu BBox:\n", i);
                printf("    Min = (%.3f, %.3f, %.3f)\n",
                    bbox.m_min.x, bbox.m_min.y, bbox.m_min.z);
                printf("    Max = (%.3f, %.3f, %.3f)\n",
                    bbox.m_max.x, bbox.m_max.y, bbox.m_max.z);
            }
            else
            {
                printf("  Curve %zu: Failed to compute BBox\n", i);
            }

            //------------------------------------------------------------
            // 2) CURVE LENGTH
            //------------------------------------------------------------

            double length = ComputeCurveLength(crv);
            printf("  Curve %zu length: %.6f\n", i, length);



        }
    }


    



    // ------------------------
    ON::End();
    // ------------------------

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