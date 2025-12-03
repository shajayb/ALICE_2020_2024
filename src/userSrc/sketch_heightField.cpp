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
#include <random>

// - ----------- OPEN NURBS -----------
#include "RhinoIO.h"

// - ----------- ZSPACE -----------
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;

// - ----------- LOCAL MODULES -----------
#include "scalarField.h"
#include "HeightField.h"
#include "heightField_NN.h"
#include "parcel_vector.h"
#include "SSAO.h"

// - ----------- UTILITIES -----------

//Alice::vec zVecToAliceVec(zVector& in)
//{
//    return Alice::vec(in.x, in.y, in.z);
//}
//
//// Helper to check polygon containment (Odd-even rule)
//bool pointInsidePolygon(zVector& pt, std::vector<zVector>& poly)
//{
//    int crossings = 0;
//    int N = poly.size();
//
//    for (int i = 0; i < N; ++i)
//    {
//        zVector& a = poly[i];
//        zVector& b = poly[(i + 1) % N];
//
//        if (((a.y > pt.y) != (b.y > pt.y)))
//        {
//            float t = (pt.y - a.y) / (b.y - a.y);
//            float xCross = a.x + t * (b.x - a.x);
//
//            if (pt.x < xCross)
//                crossings++;
//        }
//    }
//
//    return (crossings % 2 == 1);
//}

// Convert OpenNurbs Mesh to flat arrays for OpenGL
inline void convert_ONMesh_to_tri_arrays
(
    const ON_Mesh* msh,
    std::vector<float>& vertices,
    std::vector<float>& normals,
    std::vector<unsigned int>& indices
)
{
    vertices.clear();
    normals.clear();
    indices.clear();

    if (!msh) return;

    const int Vcount = msh->VertexCount();
    const int Fcount = msh->m_F.Count();

    // Ensure vertex normals exist
    const ON_Mesh* srcMesh = msh;
    std::unique_ptr<ON_Mesh> tempMesh;

    if (!msh->HasVertexNormals())
    {
        tempMesh.reset(msh->Duplicate());
        tempMesh->ComputeVertexNormals();
        srcMesh = tempMesh.get();
    }

    // Copy vertices
    vertices.reserve(Vcount * 3);
    for (int i = 0; i < Vcount; i++)
    {
        const ON_3dPoint& p = srcMesh->m_V[i];
        vertices.push_back((float)p.x);
        vertices.push_back((float)p.y);
        vertices.push_back((float)p.z);
    }

    // Copy normals
    normals.reserve(Vcount * 3);
    for (int i = 0; i < srcMesh->m_N.Count(); i++)
    {
        const ON_3fVector& n = srcMesh->m_N[i];
        normals.push_back(n.x);
        normals.push_back(n.y);
        normals.push_back(n.z);
    }

    // Build indices
    indices.reserve(Fcount * 6);
    for (int f = 0; f < Fcount; f++)
    {
        const ON_MeshFace& face = srcMesh->m_F[f];
        int v0 = face.vi[0];
        int v1 = face.vi[1];
        int v2 = face.vi[2];
        int v3 = face.vi[3];

        bool isTriangle = (v2 == v3);

        indices.push_back((unsigned int)v0);
        indices.push_back((unsigned int)v1);
        indices.push_back((unsigned int)v2);

        if (!isTriangle)
        {
            indices.push_back((unsigned int)v0);
            indices.push_back((unsigned int)v2);
            indices.push_back((unsigned int)v3);
        }
    }
}

// ------------------------------------------------------------
// DATA FUNCTIONS
// ------------------------------------------------------------

void write_3DM(vector<parcel> plots, float scale, zVector cDst, zVector cSrc)
{
    RhinoIO rio;

    // --------- add parcel polygons to RIO
    vector< vector<zVector> > all_polygons;
    vector<zVector> poly;

    for (auto plot : plots)
    {
        poly.clear();
        for (int i = 0; i < plot.nPoints; i++) poly.push_back(plot.polyPoints[i] * 1000);

        all_polygons.push_back(poly);

        // -- add curve representing oriented 12 x 5 box ;
        poly.clear();
        plot.setDefaultBox_OrientedRectangle(12 * 0.5, 5.5 * 0.5, plot.directionOfBox);
        plot.invertBox(plot.directionOfBox);
        plot.transformBox();
        plot.flipNormals();

        for (int i = 0; i < plot.nPoints; i++) poly.push_back(plot.polyPoints[i] * 1000);
        all_polygons.push_back(poly);
    }

    rio.addCurves(all_polygons, scale, ON_3dPoint(cDst.x, cDst.y, cDst.z), ON_3dPoint(cSrc.x, cSrc.y, cSrc.z));

    // --------- add BBOX polygons to RIO
    {
        ON_3dPointArray pts;
        zVector z_pts[5] =
        {
            zVector(-50,-50,0),
            zVector(-50, 50, 0),
            zVector(+50,+50,0),
            zVector(+50,-50,0),
            zVector(-50,-50,0) // close
        };

        for (int i = 0; i < 5; i++)
        {
            zVector pt = (z_pts[i] / scale) * 1000;
            pts.Append(ON_3dPoint(pt.x, pt.y, pt.z));
        }

        rio.addPolyCurve(pts);
    }

    // --------------- write file  ---
    rio.Write3dm(L"data/plots.3dm");
}

void shuffleSDFSamplePoints(std::vector<sdfSamples>& pts)
{
    static std::random_device rd;
    static std::mt19937 g(rd());
    std::shuffle(pts.begin(), pts.end(), g);
}

void trainSGD(heightfieldNN& nn, vector<float>& dummyInput, vector<float>& dummyTarget, vector<float>& output, double& prevLoss, float& learningRate)
{
    if (nn.sdfSamplePoints.empty())
    {
        printf("ERROR: sdfSamplePoints is empty. Call generateSDFSamplePointsFromPolygon() first.\n");
        return;
    }

    // 1. Shuffle the SDF sample points
    shuffleSDFSamplePoints(nn.sdfSamplePoints);

    // 2. Forward pass using dummyInput
    if (dummyInput.size() != nn.inputDim)
    {
        dummyInput.clear();
        dummyInput.resize(nn.inputDim, 0.0f);
    }

    std::vector<float> y_pred = nn.forward(dummyInput);

    // 3. Compute loss
    float loss = nn.computeLoss(y_pred, dummyTarget);

    // 4. Compute gradient
    std::vector<float> grad;
    nn.computeGradient(dummyInput, dummyTarget, grad);

    // 5. Learning-rate adaptation + backward update
    if (fabs(loss - prevLoss) < 1e-2) learningRate *= 1.1;

    learningRate = ofClamp(learningRate, 1e-2, 0.95);

    nn.backward(grad, learningRate);

    // 6. Update output
    output = y_pred;
    prevLoss = loss;
}

void shortest_paths_N_x_M
(
    vector<zVector>& Source, vector<zVector>& Sinks, HeightField2D& sf_field,
    vector< vector<zVector> >& paths, vector<zVector>& clippingPolygon
)
{
    paths.clear();
    for (int n = 0; n < Source.size(); n++)
    {
        for (int m = 0; m < Sinks.size(); m++)
        {
            zVector str = Source[n];
            zVector end = Sinks[m];

            if (clippingPolygon.size() > 3)
            {
                if (!pointInsidePolygon(str, clippingPolygon)) continue;
                if (!pointInsidePolygon(end, clippingPolygon)) continue;
            }

            sf_field.findShortestPath(str, end);

            for (int i = 0; i < 5; i++)
                sf_field.smoothPath();

            paths.push_back(sf_field.lastShortestPath);
        }
    }
}

// ------------------------------------------------------------
// MATRIX HELPERS
// ------------------------------------------------------------

// Builds a TRS matrix for the cabins (Translation, Rotation, Scale)
mat4f computeBoxTransform(vec3f pos, vec3f dir, float ls = 12.0f * 0.5, float ws = 6.0f * 0.5, float hs = 3.0f * 0.5)
{
    // 1. Rotation (Align X-axis to dir)
    mat4f R = alignToDir(dir);

    // 2. Scale (Cabins are roughly 12x6x4)
    mat4f S = identity4f();
    S.m[0] = ls; S.m[5] = ws; S.m[10] = hs;

    // 3. Translation
    mat4f T = identity4f();
    T.m[12] = pos.x; T.m[13] = pos.y; T.m[14] = pos.z;

    // Order: T * (R * S) -> Scale local, Rotate, Translate
    mat4f RS = multiply(R, S);
    mat4f TRS = multiply(T, RS);

    return TRS;
}

// - ----------- APPLICATION STATE -----------

struct SimulationState
{
    // Height Fields
    HeightField2D terrainField;
    HeightField2D terrainOriginal;
    HeightField2D costField;
    HeightField2D existingPathsField;

    // Site Data
    double isoThreshold = -0.05f;
    float zRangeMin = 0.0f;
    std::vector<zVector> boundaryPolygon;
    std::vector<std::vector<zVector>> existingPaths;
    std::vector<std::vector<zVector>> calculatedPaths;
    std::vector<std::vector<zVector>> contours;

    // Neural Network
    heightfieldNN nn;
    std::vector<float> nnOutput;
    std::vector<float> nnDummyInput = { 0.0f };
    std::vector<float> nnDummyTarget = { 1.0f };
    float learningRate = 0.5f;
    double prevLoss = 0.0;
    bool isTraining = false;

    // Parcels
    std::vector<parcel> plots;
    spaceGrid grid;

    // Rendering / Shader
    SimpleSSAO ssao;
    SSAOMesh orientedCubeMesh;
    SSAOMesh terrainMesh;

    // Init
    void initialize()
    {
        terrainField = HeightField2D();
        costField = HeightField2D();
        grid = spaceGrid();

        // NN Init
        nn = heightfieldNN(25);
        nnDummyInput.assign(nn.inputDim, 0.0f);
        nnOutput = nn.forward(nnDummyInput);

        // SSAO Init
        ssao.setup();
        ssao.samples = 1024;
        ssao.bias = 0.1;
        ssao.radius = 30;
        ssao.mode = 2; // AO_BLUR default

        createUnitCube(orientedCubeMesh);
    }

    void createUnitCube(SSAOMesh& m)
    {
        ssao_createUnitCube(m);
    }

    // Static helper for cube creation
    static void ssao_createUnitCube(SSAOMesh& m)
    {
        m.vertices.clear(); m.normals.clear(); m.indices.clear();
        float x = 0.5f, y = 0.5f, z = 0.5f;

        std::vector<float> v = {
            -x,-y,-z, x,-y,-z, x, y,-z, -x, y,-z, // Back
            -x,-y, z, x,-y, z, x, y, z, -x, y, z, // Front
            -x,-y,-z, -x,-y, z, -x, y, z, -x, y,-z, // Left
            x,-y,-z, x,-y, z, x, y, z, x, y,-z, // Right
            -x, y,-z, x, y,-z, x, y, z, -x, y, z, // Top
            -x,-y,-z, x,-y,-z, x,-y, z, -x,-y, z  // Bottom
        };

        std::vector<float> n = {
            0, 0,-1, 0, 0,-1, 0, 0,-1, 0, 0,-1,
            0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
            -1, 0, 0,-1, 0, 0,-1, 0, 0,-1, 0, 0,
            1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
            0,-1, 0, 0,-1, 0, 0,-1, 0, 0,-1, 0
        };

        m.vertices = v;
        m.normals = n;

        unsigned int idx[] = { 0,2,1, 0,3,2, 4,5,6, 4,6,7, 8,9,10, 8,10,11, 12,14,13, 12,15,14, 16,17,18, 16,18,19, 20,22,21, 20,23,22 };
        for (int i = 0; i < 36; i++) m.indices.push_back(idx[i]);
        m.dirty = true;
    }
};

SimulationState sim;

// - ----------- LOGIC & ACTIONS -----------

void Action_ImportRhinoData()
{
    RhinoIO in_RIO;
    if (!in_RIO.Read3dm(L"data/CF_beta_village_extract.3dm"))
    {
        std::cerr << "Failed to read 3DM file." << std::endl;
        return;
    }

    auto names = in_RIO.GetObjectInfo();
    std::vector<RhinoObjectInfo> curves, meshes, pclouds;
    in_RIO.SeparateGeometryTypes(names, curves, meshes, pclouds);

    // 1. Process Point Clouds -> HeightField
    for (const auto& obj : pclouds)
    {
        const ON_PointCloud* PC = ON_PointCloud::Cast(obj.geometry);
        if (PC)
        {
            int count = PC->m_P.Count();
            sim.terrainField.samples.clear();

            for (int i = 0; i < count; i++)
            {
                const ON_3dPoint& p = PC->m_P[i];
                sim.terrainField.samples.emplace_back(zVector(p.x, p.y, p.z));
            }

            sim.terrainField.rescaleSamplesToBoundingBox(zVector(-50, -50, -50), zVector(50, 50, 50));
            sim.terrainField.clearField();
            sim.terrainField.interpolateToGrid_MLS();
            sim.zRangeMin = sim.terrainField.zMin;

            // Backup original
            sim.terrainOriginal.clearField();
            for (int i = 0; i < SF_RES; i++)
                for (int j = 0; j < SF_RES; j++)
                    sim.terrainOriginal.field[i][j] = sim.terrainField.field[i][j];
            sim.terrainOriginal.scale = sim.terrainField.scale;
        }
    }

    // 2. Process Curves -> Existing Paths & Boundary
    sim.existingPaths.clear();
    std::vector<zVector> poly;

    for (const auto& obj : curves)
    {
        const ON_Curve* crv = ON_Curve::Cast(obj.geometry);
        if (!crv) continue;

        if (obj.name == L"EXIST_PATH")
        {
            poly.clear();
            in_RIO.sample_curve_unifrom(crv, poly);
            sim.existingPaths.push_back(poly);
        }
        else if (obj.name == L"BND")
        {
            sim.boundaryPolygon.clear();
            in_RIO.sample_curve_unifrom(crv, sim.boundaryPolygon);
        }
    }

    // Scale paths
    for (auto& path : sim.existingPaths)
        for (auto& p : path)
            p *= sim.terrainField.scale;

    // Create Path SDF
    sim.existingPathsField.clearField();
    sim.existingPathsField.addSDFfromPolylines(sim.existingPaths, 3);

    // Project paths to terrain Z
    for (auto& path : sim.existingPaths)
    {
        for (auto& p : path)
        {
            p.z = sim.terrainField.getFieldValue(p);
            p.z = sim.terrainField.mapIsoToActualHeight(p.z);
        }
    }

    // Trim Terrain
    sim.terrainField.rescalePoints(sim.boundaryPolygon);
    sim.terrainField.trimFieldWithPolygon(sim.boundaryPolygon);
    sim.terrainField.subtract(sim.existingPathsField);

    sim.nn.correspondingHeightField = &sim.terrainField;

    // 3. Process Meshes -> SSAO
    for (const auto& obj : meshes)
    {
        const ON_Mesh* msh = ON_Mesh::Cast(obj.geometry);
        if (msh && obj.name == L"TERRAIN")
        {
            std::cout << "[App] Found Terrain Mesh! Converting..." << std::endl;
            convert_ONMesh_to_tri_arrays
            (
                msh,
                sim.terrainMesh.vertices,
                sim.terrainMesh.normals,
                sim.terrainMesh.indices
            );
            sim.terrainMesh.dirty = true;

            // Apply scale
            for (auto& x : sim.terrainMesh.vertices) x *= sim.terrainField.scale;

            sim.ssao.addObject(&sim.terrainMesh, identity4f());
            break;
        }
    }
}

void Action_GenerateShortestPaths()
{
    // Copy cost field from original terrain
    sim.costField.clearField();
    for (int i = 0; i < SF_RES; i++)
        for (int j = 0; j < SF_RES; j++)
            sim.costField.field[i][j] = sim.terrainOriginal.field[i][j];

    sim.costField.trimFieldWithPolygons(sim.nn.polygons);

    // Determine containment
    std::vector<std::vector<zVector>> polys;
    for (auto& parcel : sim.plots)
        for (auto& poly : sim.nn.polygons)
            if (pointInsidePolygon(parcel.centerOfBox, poly))
                polys.push_back(parcel.polyPoints);

    sim.costField.scale_scalar_within_polygons(polys);

    // Extract nodes from NN
    std::vector<Pose2D> poses;
    sim.nn.extractPoses(sim.nnOutput, poses, true);

    sim.calculatedPaths.clear();

    // Source / Sink vectors
    std::vector<zVector> sources, sinks;
    for (int n = 0; n < 1 /* && n < poses.size() */; n++)
        sources.push_back(poses[n].c);

    for (int m = 0; m < poses.size(); m++)
        sinks.push_back(poses[m].c);

    // Compute
    if (sim.nn.polygons.size() > 0)
    {
        shortest_paths_N_x_M(sources, sinks, sim.costField, sim.calculatedPaths, sim.nn.polygons[0]);
    }
}

void Action_PopulateParcels()
{
    std::vector<Pose2D> poses;
    sim.nn.extractPoses(sim.nnOutput, poses, true);

    int id = 0;
    sim.plots.clear();

    parcel tmpPlot;

    for (auto& pose : poses)
    {
        tmpPlot.centerOfBox = pose.c;

        zVector dir = pose.v;
        dir.normalize();
        dir.z = 0;

        tmpPlot.directionOfBox = dir;
        tmpPlot.setDefaultBox_OrientedRectangle(12 * 0.5, 5.5 * 0.5, dir);
        tmpPlot.invertBox(dir);
        tmpPlot.transformBox();
        tmpPlot.flipNormals();

        tmpPlot.id_u = id++;
        sim.plots.push_back(tmpPlot);
    }
}

void Action_ExpandParcels()
{
    for (auto& parcel : sim.plots)
        parcel.expand_withNormalCheck(sim.plots, true, &sim.grid);

    for (auto& parcel : sim.plots)
        parcel.smooth();

    // Update SpaceGrid
    sim.grid.clearBuckets();
    sim.grid.np = 0;

    for (auto& parcel : sim.plots)
        for (int i = 0; i < parcel.nPoints; i++)
            sim.grid.addPosition(parcel.polyPoints[i]);

    for (auto& poly : sim.nn.polygons)
        for (auto& p : poly)
            sim.grid.addPosition(p);

    for (auto& p : sim.terrainOriginal.lastShortestPath)
        sim.grid.addPosition(p);

    sim.grid.PartitionParticlesToBuckets();
}

void Action_ExtractNextContour()
{
    // Increment Z range
    sim.zRangeMin += 1.0 * sim.terrainField.scale;
    if (sim.zRangeMin >= sim.terrainField.zMax)
        sim.zRangeMin = sim.terrainField.zMin;

    float iso = ofMap(sim.zRangeMin, sim.terrainField.MLS_zMin, sim.terrainField.MLS_zMax, 0, 1);
    printf("Extracting Iso: %.2f at Z: %.2f\n", iso, sim.zRangeMin);

    sim.terrainField.computeIsocontours(iso);
    std::vector<std::vector<zVector>> contours = sim.terrainField.getOrderedContours();

    // Filter large islands
    std::vector<std::vector<zVector>> validPolys;
    for (int i = 0; i < contours.size(); i++)
    {
        if (sim.terrainField.area_of_contour_island(contours[i]) > 50)
            validPolys.push_back(contours[i]);
    }

    if (!validPolys.empty())
    {
        // Smooth
        for (auto& poly : validPolys)
            for (int i = 0; i < 15; i++)
                sim.terrainField.smoothPath(poly);

        // Update NN target
        sim.nn.setTargetPolygons(validPolys);
        sim.nn.generateSDFSamplePointsFromPolygons();
    }
}

void Action_TrainStep()
{
    trainSGD(sim.nn, sim.nnDummyInput, sim.nnDummyTarget, sim.nnOutput, sim.prevLoss, sim.learningRate);
}

void Action_TrainSinglePass()
{
    std::vector<float> y_pred = sim.nn.forward(sim.nnDummyInput);
    float loss = sim.nn.computeLoss(y_pred, sim.nnDummyTarget);

    std::vector<float> grad;
    sim.nn.computeGradient(sim.nnDummyInput, sim.nnDummyTarget, grad);

    if (fabs(loss - sim.prevLoss) < 1e-2)
        sim.learningRate *= 1.1;

    sim.learningRate = ofClamp(sim.learningRate, 1e-2, 0.25);
    sim.prevLoss = loss;
    sim.nn.backward(grad, sim.learningRate);

    sim.nnOutput = y_pred;
}

void Action_ExportRhino()
{
    write_3DM(sim.plots, sim.terrainField.scale, sim.terrainField.cDst, sim.terrainField.cSrc);
}

// - ----------- RENDERING -----------

void Render_Paths(const std::vector<std::vector<zVector>>& paths)
{
    for (auto& path : paths)
    {
        if (!path.empty())
        {
            for (size_t i = 0; i < path.size() - 1; i++)
                drawLine(zVecToAliceVec(const_cast<zVector&>(path[i])), zVecToAliceVec(const_cast<zVector&>(path[i + 1])));

            glPointSize(3);
            for (const auto& p : path)
                drawPoint(zVecToAliceVec(const_cast<zVector&>(p)));
        }
    }
}

void Render_SSAO_Pass()
{
    sim.ssao.clearQueue();

    // 1. Cabins / Parcels from NN Poses (COLOR = RED)
    vec3f cabinColor = { 0.8f, 0.3f, 0.3f };
    for (auto pose : sim.nn.poses)
    {
        float z = sim.terrainField.getFieldValue(pose.c);
        pose.c.z = sim.terrainField.mapIsoToActualHeight(z) + 1;

        // Calculate Transform (TRS)
        mat4f M = computeBoxTransform(vec3f{ pose.c.x, pose.c.y, pose.c.z }, vec3f{ pose.v.x, pose.v.y, pose.v.z });

        sim.ssao.addObject(&sim.orientedCubeMesh, M, cabinColor);
    }

    // 2. Terrain (COLOR = GREY)
    vec3f terrainColor = { 0.8f, 0.7f, 0.6f };
    sim.ssao.addObject(&sim.terrainMesh, identity4f(), terrainColor);

    sim.ssao.draw();
}

// - ----------- MVC CALLBACKS -----------

void setup()
{
    // UI Setup
    S.numSliders = 0;
    S.addSlider(&sim.isoThreshold, "tv");
    S.sliders[0].minVal = -1;
    S.sliders[0].maxVal = 1;

    S.addSlider(&sim.nn.o_weight, "o_w");
    S.addSlider(&sim.ssao.radius, "sao_r");
    S.sliders[2].maxVal = 100;

    B = *new ButtonGroup(Alice::vec(50, 125, 0));
    B.addButton(&sim.nn.o_flip_dir, "o_flip_dir");

    // Sim Init
    sim.initialize();

    // GL State
    glShadeModel(GL_SMOOTH);
}

void update(int value)
{
    if (sim.isTraining)
    {
        Action_TrainSinglePass();
    }
}

void draw()
{
    backGround(0.99);

    // 1. Render Scene with AO
    Render_SSAO_Pass();

    //// 2. Debug Visualization
    //drawGrid(50);

    // Terrain Points
    sim.terrainField.drawSamplePoints();

    // Parcels
    for (auto& parcel : sim.plots) parcel.display();

    // Debug Wireframes
    wireFrameOn();
    // sim.grid.drawBuckets(); // Optional debug
    wireFrameOff();

    // Paths
    glColor3f(0.25, 0, 0);
    Render_Paths(sim.calculatedPaths);
    Render_Paths(sim.existingPaths);

    // NN Viz
    sim.nn.visualize(zVector(50, 350, 0), 200, 250);
    sim.nn.drawCoveragePolygon();
    sim.nn.draw_output_and_loss();

    glColor3f(0, 0, 1);
}

void keyPress(unsigned char k, int xm, int ym)
{
    // SSAO Debug
    if (k == 'd') sim.ssao.mode = (sim.ssao.mode + 1) % 8;

    // Actions
    if (k == '2') Action_ImportRhinoData();
    if (k == 'g') Action_GenerateShortestPaths();
    if (k == '3') Action_ExportRhino();
    if (k == 'p') Action_PopulateParcels();
    if (k == 'e') Action_ExpandParcels();
    if (k == '=') Action_ExtractNextContour();

    // Training Control
    if (k == 'c') sim.isTraining = !sim.isTraining;
    if (k == 't') Action_TrainSinglePass();
    if (k == 'l') Action_TrainStep();
}

void mousePress(int b, int state, int x, int y)
{
    // Placeholder
}

void mouseMotion(int x, int y)
{
    // Placeholder
}

#endif // _MAIN_