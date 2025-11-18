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

bool pointInsidePolygon(zVector& pt, std::vector<zVector>& poly)
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

bool loadPolygonFromFile(std::string& filePath, std::vector<zVector>& polygon)
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




// - ----------- OTHER CUSTOM FUNCTIONS  -----------

#include "scalarField.h"
#include "HeightField.h"
#include "heightField_NN.h"
#include "parcel_vector.h"


void write_3DM(vector<parcel> plots, float scale, zVector cDst, zVector cSrc)
{
    // Init OpenNURBS
    ON::Begin();

    // Model + attributes
    ONX_Model model;
    model.m_properties.m_Notes.m_notes = L"Simple polyline via OpenNURBS";
    model.m_properties.m_Notes.m_bVisible = true;

    for (auto plot : plots)
    {

        ON_3dPointArray pts;
        for (int i = 0; i < plot.nPoints; i++)
        {
            zVector pt = plot.polyPoints[i];

            // reverse mapping: from destination back to source
            pt.x = cSrc.x + (pt.x - cDst.x) / scale * 1000;
            pt.y = cSrc.y + (pt.y - cDst.y) / scale * 1000;
            pt.z = 0;

            pts.Append(ON_3dPoint(pt.x, pt.y, pt.z));

        }

        ON_PolylineCurve* plc = new ON_PolylineCurve(pts);

        ON_3dmObjectAttributes attr;
        attr.m_name = L"SimplePolyline";

        model.AddModelGeometryComponent(plc, &attr);
    }

    // bounding box

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

        ON_PolylineCurve* plc = new ON_PolylineCurve(pts);

        ON_3dmObjectAttributes attr;
        attr.m_name = L"BBox";

        model.AddModelGeometryComponent(plc, &attr);
    }



    // --------------- write file via filename overload ---
    wchar_t* outfile = L"data/simplest_polyline.3dm";
    bool ok = model.Write(outfile);             // writes using default/latest version
    // If you want a specific version (e.g., Rhino 6 = 60, 7 = 70) and your SDK supports it:
    // bool ok = model.Write(outfile, 60);


    ok ? printf("Wrote 3dm: simplest_polyline.3dm\n") : printf("Failed to write 3dm file.\n");

    ON::End();

}

void drawText(string& str, float x = 50, float y = 100)
{
    unsigned int i;
    glRasterPos2f(x, y);


    for (i = 0; i < str.length(); i++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, str[i]);
}


// ------------------------------------------------------------
// Shuffle helper (as given)
// ------------------------------------------------------------
void shuffleSDFSamplePoints(std::vector<sdfSamples>& pts)
{
    static std::random_device rd;
    static std::mt19937 g(rd());

    std::shuffle(pts.begin(), pts.end(), g);
}

// ------------------------------------------------------------
// SGD training step: shuffle sdfSamplePoints + run your train logic
// ------------------------------------------------------------
void trainSGD(heightfieldNN& nn, vector<float>& dummyInput, vector<float>& dummyTarget, vector<float>& output, double& prevLoss, float& learningRate)
{
    if (nn.sdfSamplePoints.empty())
    {
        printf("ERROR: sdfSamplePoints is empty. Call generateSDFSamplePointsFromPolygon() first.\n");
        return;
    }

    // ----------------------------------
    // 1. Shuffle the SDF sample points
    // ----------------------------------
    shuffleSDFSamplePoints(nn.sdfSamplePoints);

    // ----------------------------------
    // 2. Forward pass using dummyInput
    // ----------------------------------
    if (dummyInput.size() != nn.inputDim)
    {
        dummyInput.clear();
        dummyInput.resize(nn.inputDim, 0.0f);   // zero input or fill with noise
    }

    std::vector<float> y_pred = nn.forward(dummyInput);

    // ----------------------------------
    // 3. Compute loss
    // ----------------------------------
    float loss = nn.computeLoss(y_pred, dummyTarget);

    // ----------------------------------
    // 4. Compute gradient
    // ----------------------------------
    std::vector<float> grad;
    nn.computeGradient(dummyInput, dummyTarget, grad);

    // ----------------------------------
    // 5. Learning-rate adaptation + backward update
    // ----------------------------------

    printf(" %.4f,%.4f \n", fabs(loss - prevLoss), learningRate);

    if (fabs(loss - prevLoss) < 1e-2) learningRate *= 1.1;

    learningRate = ofClamp(learningRate, 1e-2, 0.95);

    nn.backward(grad, learningRate);

    // ----------------------------------
    // 6. Update output + debug print
    // ----------------------------------
    output = y_pred;

    printf("Loss: %.8f | Output: [", loss);
    printf("]\n");

    prevLoss = loss;
}

// ------------------------------------------------------------
// shortest path helper 
// -
void drawShortestPaths(vector< vector<zVector> >& allPaths)
{
    for (auto& path : allPaths)
        if (!path.empty())
        {
            glColor3f(0, 0, 1);
            for (size_t i = 0; i < path.size() - 1; i++)
            {
                drawLine(zVecToAliceVec(path[i]), zVecToAliceVec(path[i + 1]));
            }
        }

}
// ------------------------- APP ----------------------------------
// ------------------------- - ----------------------------------
// ------------------------- - ----------------------------------

// -- height field
HeightField2D importedHeightField, siteHeightField;
double threshold;

// site definition

float zRangeMin;
std::vector<zVector> polygon;

// -- height field nn

heightfieldNN nn;
vector<float> output;
vector<float> dummyInput = { 0.0f };
std::vector<float> dummyTarget = { 1.0f };; // unused

float learningRate = 0.5f;


// -- parcels

vector<parcel> plots;
parcel plot;
spaceGrid SG;

//--- paths

vector< vector<zVector> > shortestPaths;

// -- app

bool compute = false;
double prevLoss = 0.0;

zVector grad;
zVector gradPt;

// -------------------------

void setup()
{


    S.numSliders = 0;
    S.addSlider(&threshold, "tv");// make a slider control for the variable called width;
    S.sliders[0].minVal = -1; // myHeightField.zScale * -1;
    S.sliders[0].maxVal = 1; //  myHeightField.zScale;

    // ----------- terrrain 

    importedHeightField = *new HeightField2D();
    siteHeightField = *new HeightField2D();

    importedHeightField.clearField();
    importedHeightField.readSamplesAndInterpolate(string("data/cabins_site.txt"));
    zRangeMin = importedHeightField.zMin;

    // ----------- terrain trim

    polygon.clear();
    loadPolygonFromFile(string("data/terrain_boundary_poly.txt"), polygon);
    importedHeightField.rescalePoints(polygon);// scale polygon by same amount as height feild points.
    importedHeightField.trimFieldWithPolygon(polygon);

    importedHeightField.computeGradient();

    // ----------- NN ----------------

    nn = heightfieldNN(30); // or however many poses you want

    //  ----- SDF loss polygon

    nn.setTargetPolygon(polygon);
    //nn.generateSDFSamplePointsFromPolygon();

    // ----- reserve input and output dimensions
    
    dummyInput.assign(nn.inputDim, 0.0f);
    output = nn.forward(dummyInput);


    // ----------- parcels

    plots.clear();
    int id = 0;

    //

    SG = *new spaceGrid();
}

void update(int value)
{
    if (compute) keyPress('l', 0, 0);
}

void draw()
{
    backGround(0.99);
    drawGrid(50);


    // ----------------------- imported heightfield and sample points 


     importedHeightField.drawSamplePoints();
   //  myHeightField.drawFieldPoints(true, false);
    // 

    // ----------------------- parcels

    for (auto& parcel : plots)parcel.display();

    wireFrameOn();
        //SG.drawBuckets();
        //SG.drawParticlesInBuckets();
    wireFrameOff();

    // ----------------------- paths
       // myHeightField1.drawPath();
       // myHeightField1.drawFieldPoints(false, false);
    glLineWidth(5);
    drawShortestPaths(shortestPaths);

    // ----------------------- nn

    nn.visualize(zVector(50, 350, 0), 200, 250);
    nn.drawPolygon();


    std::vector<Pose2D> poses;
    nn.extractPoses(output, poses, true);
    //
    /*vector<zVector> centers;
    for( auto &pose : poses)centers.push_back(pose.c);
    myHeightField1.drawStreamlinesFromSeeds(centers);*/
    //

    glPointSize(5);
    glColor3f(0, 0, 0);
    for (auto& pose : poses)
    {
        drawPoint(zVecToAliceVec(pose.c));
        // drawLine(zVecToAliceVec(pose.c), zVecToAliceVec(pose.c + pose.v * 5.0));
        zVector dir = importedHeightField.gradientAt(pose.c);
        dir.normalize();
        drawLine(zVecToAliceVec(pose.c), zVecToAliceVec(pose.c + dir * 2.0));

        drawCircle(zVecToAliceVec(pose.c), radius, 32);
    }
    glPointSize(1);

    setup2d();

        char s[200];
        sprintf(s, "%.4f", prevLoss);
        drawText(string(s), 50, 450);

    restore3d();


}

int n = 0;



void keyPress(unsigned char k, int xm, int ym)
{

    if (k == 'g') // ----------paths
    {
        siteHeightField.clearField();
        for (int i = 0; i < SF_RES; i++)
            for (int j = 0; j < SF_RES; j++)
                siteHeightField.field[i][j] = importedHeightField.field[i][j];

        siteHeightField.trimFieldWithPolygon(nn.polygon);

        std::vector<Pose2D> poses;
        nn.extractPoses(output, poses, true);


        //--

        vector<vector<zVector>> polys;
        for (auto& parcel : plots)
            if (pointInsidePolygon(parcel.centerOfBox, nn.polygon)) polys.push_back(parcel.polyPoints);
        siteHeightField.scale_scalar_within_polygons(polys);

        //
        shortestPaths.clear();
        for (int n = 0; n < poses.size(); n++)
        {
            for (int m = 0; m < poses.size(); m++)
            {
                if (m == n)continue;
                zVector str = poses[m].c;// poses[0].c;// nn.sdfSamplePoints[0].pt;
                zVector end = poses[n].c;//nn.sdfSamplePoints[n++].pt;

                if (!pointInsidePolygon(str, nn.polygon))continue;
                if (!pointInsidePolygon(end, nn.polygon))continue;

                siteHeightField.findShortestPath(str, end);


                for (int i = 0; i < 5; i++)
                {
                    siteHeightField.smoothPath();
                }

                shortestPaths.push_back(siteHeightField.lastShortestPath);
            }
        }
    }

    // --------------- WRITE 3DM ---------------


    if (k == '3')
    {
        write_3DM(plots, importedHeightField.scale, importedHeightField.cDst, importedHeightField.cSrc);
    }


    // --------------- PARCELS  ---------------

    if (k == 'p') // populate
    {
        polygon.clear();
        loadPolygonFromFile(string("data/cabin.txt"), polygon);

        for (auto& pt : polygon)
        {
            pt *= importedHeightField.scale;
        }

        ///
        std::vector<Pose2D> poses;
        nn.extractPoses(output, poses, true);

        int id = 0;

        plots.clear();
        for (auto& pose : poses)
        {
            plot.centerOfBox = pose.c;

            zVector dir = importedHeightField.gradientAt(pose.c);
            dir.normalize();
            dir.z = 0;
            dir = dir ^ zVector(0, 0, 1);
            dir.normalize();
            plot.directionOfBox = dir;


            plot.setDefaultBox();
            //plot.importPrimitive(polygon);
            plot.transformBox();
            plot.flipNormals();


            plot.id_u = id++;
            plots.push_back(plot);
        }
    }


    if (k == 'e') // expand
    {
        for (auto& parcel : plots)parcel.expand_withNormalCheck(plots, true, &SG);


        for (auto& parcel : plots)parcel.smooth();
        for (auto& plot : plots)
            printf("%.4f \n", plot.computeParcelArea());


        // ------------- update SG 

        // clear
        SG.clearBuckets();
        SG.np = 0;

        // fill
        for (auto& parcel : plots)
            for (int i = 0; i < parcel.nPoints; i++)
            {
                SG.addPosition(parcel.polyPoints[i]);
            }

        for (auto p : nn.polygon)SG.addPosition(p);

        // re-partition
        SG.PartitionParticlesToBuckets();

    }


    if (k == 'o')
    {
        plots[0].makeCentersEquiDistant(plots, nn.polygon);
    }

    // --------------- SITE DEFINITION AND NUERAL NET COVERAGE POLYGON  ---------------


    if (k == '=') // get next contour polygon
    {
        // nn.generateSamplesInRange(myHeightField, zRangeMin, zRangeMin+2);

        // -------- get contours and order them

        zRangeMin += 1.0;
        if (zRangeMin >= importedHeightField.zMax)zRangeMin = importedHeightField.zMin;


        float iso = ofMap(zRangeMin, importedHeightField.MLS_zMin, importedHeightField.MLS_zMax, 0, 1);
        printf(" %.2f iso, %.2f zRangeMin, \n", iso, zRangeMin);
        importedHeightField.computeIsocontours(iso);
        std::vector<std::vector<zVector>> contours = importedHeightField.getOrderedContours();


        // -------- find contour island with most points

        vector<zVector> poly;
        size_t maxPts = 0;

        for (int i = 0; i < contours.size(); i++)
        {

            if (contours[i].size() > maxPts)
            {
                maxPts = contours[i].size();
                poly = contours[i];
            }
        }

        cout << maxPts << " -- " << poly.size() << endl;

        // -------- if contour island is valid, set it as polygon for NN to cover with sites.

        if (poly.size() > 2)
        {
            // -------- smooth contour
            for (int i = 0; i < 15; i++) importedHeightField.smoothPath(poly);

            // -------- set contour as polygon of NN and generate sample points within
            nn.setTargetPolygon(poly);
            nn.generateSDFSamplePointsFromPolygon();

        }

        // cout << zRangeMin << " -- " << contours.size() << endl;
    }

    // ---------------  NUERAL NET TRAINING  ---------------

    if (k == 'c')compute = !compute; // iteratively train NN to minimise loss function

    if (k == 't')
    {
        // Forward pass

        std::vector<float> y_pred = nn.forward(dummyInput);

        // Loss
        float loss = nn.computeLoss(y_pred, dummyTarget);

        // Gradient (numerical)
        std::vector<float> grad;
        nn.computeGradient(dummyInput, dummyTarget, grad);

        // Backward update
        //printf(" %.4f,%.4f \n", fabs(loss - prevLoss), learningRate);

        if (fabs(loss - prevLoss) < 1e-2) learningRate *= 1.1;

        learningRate = ofClamp(learningRate, 1e-2, 0.15);
        prevLoss = loss;
        nn.backward(grad, learningRate);

        // Debug

        output = y_pred;

        // Print output vector
        //printf("Loss: %.8f | Output: [", loss);
        /*for (int i = 0; i < y_pred.size(); ++i)
        {
            printf("%.4f", y_pred[i]);
            if (i < y_pred.size() - 1) printf(", ");
        }*/
        //printf("]\n");


        //------------------------ 

    }

    // ---- Train SGD

    if (k == 'l')
    {
        trainSGD(nn, dummyInput, dummyTarget, output, prevLoss, learningRate);
    }

    // --------------- HEIGHT FIELD IMPORT  ---------------


    if (k == 's')
    {
        importedHeightField.smoothDiffuseIsotropic(0.15, 1, true);
        //myHeightField1.smoothDiffuseAnisotropic(0.2, 1, 0.1, ScalarField2D::PMVariant::Exp, ScalarField2D::DiffuseDir::AlongIsophote, 2, true);
    }


    

}

void mousePress(int b, int state, int x, int y)
{
}

void mouseMotion(int x, int y)
{
}

#endif // _MAIN_