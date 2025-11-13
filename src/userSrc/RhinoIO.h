#pragma once

// -------------------- OPENNURBS --------------------
#define OPENNURBS_IMPORTS
#include "opennurbs_public.h"

// -------------------- SYSTEM ------------------------
#include <string>
#include <vector>
#include <map>
#include <cstdio>
#include <cmath>

#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;

class RhinoIO
{
public:

    // Stored model in memory
    ONX_Model model;

public:

    // ----------------------------------------------------
    // Constructor / Destructor
    // ----------------------------------------------------

    RhinoIO()
    {
        ON::Begin();
    }

    ~RhinoIO()
    {
        ON::End();
    }

    // ----------------------------------------------------
    // Read 3DM into internal model
    // ----------------------------------------------------

    inline bool Read3dm(const std::wstring& filePath)
    {
        FILE* archive_fp = ON::OpenFile(filePath.c_str(), L"rb");
        if (!archive_fp)
        {
            wprintf(L"[RhinoIO] ERROR: Cannot open file %ls\n", filePath.c_str());
            return false;
        }

        ON_TextLog dump;
        ON_BinaryFile archive(ON::archive_mode::read3dm, archive_fp);

        bool rc = model.Read(archive, &dump);

        ON::CloseFile(archive_fp);
        return rc;
    }

    // ----------------------------------------------------
    // Write internal model to file
    // ----------------------------------------------------

    inline bool Write3dm(const std::wstring& filePath)
    {
        return model.Write(filePath.c_str());
    }

    // ----------------------------------------------------
    // Compute curve length (sampling, curve-type independent)
    // ----------------------------------------------------

    inline double ComputeCurveLength(const ON_Curve* crv)
    {
        if (!crv)
        {
            return 0.0;
        }

        ON_Interval dom = crv->Domain();
        double t0 = dom.Min();
        double t1 = dom.Max();

        const int N = 256;
        double length = 0.0;

        ON_3dPoint prev = crv->PointAt(t0);

        for (int i = 1; i <= N; i++)
        {
            double t = t0 + (t1 - t0) * (double(i) / double(N));
            ON_3dPoint p = crv->PointAt(t);

            length += p.DistanceTo(prev);
            prev = p;
        }

        return length;
    }

    // ----------------------------------------------------
    // Compute all curve lengths in internal model
    // ----------------------------------------------------

    inline void ComputeAllCurveLengths(std::vector<double>& lengths)
    {
        lengths.clear();

        ONX_ModelComponentIterator it(model, ON_ModelComponent::Type::ModelGeometry);

        for (ON_ModelComponentReference ref = it.FirstComponentReference();
            !ref.IsEmpty();
            ref = it.NextComponentReference())
        {
            const ON_ModelGeometryComponent* geoComp =
                ON_ModelGeometryComponent::Cast(ref.ModelComponent());

            if (!geoComp)
            {
                continue;
            }

            const ON_Geometry* geom = geoComp->Geometry(nullptr);
            if (!geom)
            {
                continue;
            }

            const ON_Curve* crv = ON_Curve::Cast(geom);
            if (!crv)
            {
                continue;
            }

            lengths.push_back(ComputeCurveLength(crv));
        }
    }

    // ----------------------------------------------------
    // Group planar horizontal curves by Z height
    // ----------------------------------------------------

    inline void GroupPlanarHorizontalCurves(
        std::map<int, std::vector<const ON_Curve*>>& zGroups)
    {
        zGroups.clear();

        const double zTolerance = 1e-3;

        ONX_ModelComponentIterator it(model, ON_ModelComponent::Type::ModelGeometry);

        for (ON_ModelComponentReference ref = it.FirstComponentReference();
            !ref.IsEmpty();
            ref = it.NextComponentReference())
        {
            const ON_ModelGeometryComponent* geoComp =
                ON_ModelGeometryComponent::Cast(ref.ModelComponent());

            if (!geoComp)
            {
                continue;
            }

            const ON_Geometry* geom = geoComp->Geometry(nullptr);
            if (!geom)
            {
                continue;
            }

            const ON_Curve* crv = ON_Curve::Cast(geom);
            if (!crv)
            {
                continue;
            }

            // Check if planar
            ON_Plane plane;
            if (!crv->IsPlanar(&plane, zTolerance))
            {
                continue;
            }

            // Check if parallel to world XY
            ON_3dVector n = plane.Normal();
            n.Unitize();

            if (fabs(n * ON_3dVector::ZAxis) < 0.999)
            {
                continue;
            }

            // Extract Z
            double z = plane.Origin().z;

            // Quantize to avoid FP noise
            int zKey = (int)std::round(z * 1000.0);

            zGroups[zKey].push_back(crv);
        }
    }

    // ----------------------------------------------------
    // Print layer list
    // ----------------------------------------------------

    inline void PrintLayerList()
    {
        ONX_ModelComponentIterator it(model, ON_ModelComponent::Type::Layer);

        for (ON_ModelComponentReference ref = it.FirstComponentReference();
            !ref.IsEmpty();
            ref = it.NextComponentReference())
        {
            const ON_Layer* layer = ON_Layer::Cast(ref.ModelComponent());
            if (!layer)
            {
                continue;
            }

            std::wstring name = layer->Name().Array();
            wprintf(L"[Layer %d] %ls\n", layer->Index(), name.c_str());
        }
    }

    inline void addCurves(const std::vector<std::vector<zVector>>& polygons,
        float scale = 1.0f,
        const ON_3dPoint& cDst = ON_3dPoint(0, 0, 0),
        const ON_3dPoint& cSrc = ON_3dPoint(0, 0, 0))
    {
        for (const auto& poly : polygons)
        {
            if (poly.size() < 2)
            {
                continue;
            }

            ON_3dPointArray pts;

            for (size_t i = 0; i < poly.size(); i++)
            {
                const zVector& p = poly[i];

                // reverse mapping WITHOUT *1000 
                double x = cSrc.x + (p.x - cDst.x) / scale;
                double y = cSrc.y + (p.y - cDst.y) / scale;
                double z = 0.0;

                pts.Append(ON_3dPoint(x, y, z));
            }

            // auto-close if needed
            const zVector& p0 = poly.front();
            const zVector& pN = poly.back();

            if (!(p0.x == pN.x && p0.y == pN.y && p0.z == pN.z))
            {
                double x = cSrc.x + (p0.x - cDst.x) / scale;
                double y = cSrc.y + (p0.y - cDst.y) / scale;
                pts.Append(ON_3dPoint(x, y, 0.0));
            }

            ON_PolylineCurve* plc = new ON_PolylineCurve(pts);

            ON_3dmObjectAttributes attr;
            attr.m_name = L"CurvePolygon";

            model.AddModelGeometryComponent(plc, &attr);
        }
    }

};
