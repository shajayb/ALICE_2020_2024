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
#include "RhinoIO.h"
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

void drawText(string& str, float x = 50, float y = 100)
{
    unsigned int i;
    glRasterPos2f(x, y);


    for (i = 0; i < str.length(); i++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, str[i]);
}



inline void drawCube_Lambert
(
    Alice::vec& minPt,
    Alice::vec& maxPt,
    Alice::vec& origin = Alice::vec(0, 0, 0),
    bool wire = false
)
{
    // -----------------------------------------------------------
    // Enable basic Lambert (diffuse) lighting
    // -----------------------------------------------------------
    //glEnable(GL_LIGHTING);
    //glEnable(GL_LIGHT0);
    //glEnable(GL_NORMALIZE);
    //glShadeModel(GL_FLAT);

    //GLfloat lightPos[] = { 0.3f, 0.6f, 1.0f, 0.0f };
    //GLfloat lightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    //GLfloat lightAmbient[] = { 0.15f, 0.15f, 0.15f, 1.0f };
    //GLfloat lightSpecular[] = { 0.0f, 0.0f, 0.0f, 1.0f };

    //glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    //glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);
    //glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);
    //glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpecular);

    //// -----------------------------------------------------------
    //// Material: flat grey Lambert
    //// -----------------------------------------------------------
    //GLfloat diffuseColor[] = { 0.5,0.5,0.5,1.0 };
    //GLfloat ambientColor[] = { 0.25f, 0.25f, 0.25f,1.0 };

    //glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuseColor);
    //glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambientColor);

    // Cube coordinates
    float xmin = minPt.x - origin.x;
    float ymin = minPt.y - origin.y;
    float zmin = minPt.z - origin.z;

    float xmax = maxPt.x - origin.x;
    float ymax = maxPt.y - origin.y;
    float zmax = maxPt.z - origin.z;

    if (wire)
    {
       // glDisable(GL_LIGHTING);
        glBegin(GL_LINES);
        glColor3f(1, 0, 0);

        // edges
        glVertex3f(xmin, ymin, zmin); glVertex3f(xmax, ymin, zmin);
        glVertex3f(xmin, ymax, zmin); glVertex3f(xmax, ymax, zmin);
        glVertex3f(xmin, ymin, zmax); glVertex3f(xmax, ymin, zmax);
        glVertex3f(xmin, ymax, zmax); glVertex3f(xmax, ymax, zmax);

        glVertex3f(xmin, ymin, zmin); glVertex3f(xmin, ymax, zmin);
        glVertex3f(xmax, ymin, zmin); glVertex3f(xmax, ymax, zmin);
        glVertex3f(xmin, ymin, zmax); glVertex3f(xmin, ymax, zmax);
        glVertex3f(xmax, ymin, zmax); glVertex3f(xmax, ymax, zmax);

        glVertex3f(xmin, ymin, zmin); glVertex3f(xmin, ymin, zmax);
        glVertex3f(xmax, ymin, zmin); glVertex3f(xmax, ymin, zmax);
        glVertex3f(xmin, ymax, zmin); glVertex3f(xmin, ymax, zmax);
        glVertex3f(xmax, ymax, zmin); glVertex3f(xmax, ymax, zmax);

        glEnd();
       // glEnable(GL_LIGHTING);
        return;
    }

    // -----------------------------------------------------------
    // Flat-shaded faces
    // -----------------------------------------------------------
    glColor3f(0.5, 0.5, 0.5);
    glBegin(GL_QUADS);

    glNormal3f(0, 0, 1);
    glVertex3f(xmin, ymin, zmax);
    glVertex3f(xmax, ymin, zmax);
    glVertex3f(xmax, ymax, zmax);
    glVertex3f(xmin, ymax, zmax);

    glNormal3f(0, 0, -1);
    glVertex3f(xmax, ymin, zmin);
    glVertex3f(xmin, ymin, zmin);
    glVertex3f(xmin, ymax, zmin);
    glVertex3f(xmax, ymax, zmin);

    glNormal3f(1, 0, 0);
    glVertex3f(xmax, ymin, zmax);
    glVertex3f(xmax, ymin, zmin);
    glVertex3f(xmax, ymax, zmin);
    glVertex3f(xmax, ymax, zmax);

    glNormal3f(-1, 0, 0);
    glVertex3f(xmin, ymin, zmin);
    glVertex3f(xmin, ymin, zmax);
    glVertex3f(xmin, ymax, zmax);
    glVertex3f(xmin, ymax, zmin);

    glNormal3f(0, 1, 0);
    glVertex3f(xmin, ymax, zmax);
    glVertex3f(xmax, ymax, zmax);
    glVertex3f(xmax, ymax, zmin);
    glVertex3f(xmin, ymax, zmin);

    glNormal3f(0, -1, 0);
    glVertex3f(xmin, ymin, zmin);
    glVertex3f(xmax, ymin, zmin);
    glVertex3f(xmax, ymin, zmax);
    glVertex3f(xmin, ymin, zmax);

    glEnd();

    /*glDisable(GL_LIGHTING);
    glDisable(GL_LIGHT0);
    glDisable(GL_NORMALIZE);*/
    //glShadeModel(GL_FLAT);
}


//---------------------------------------------------------------
// Draw an oriented cube aligned to `direction` and centered at `center`
//---------------------------------------------------------------
inline void drawOrientedCube
(
    const zVector& center,
    const zVector& direction,
    float len = 6.0f,
    float wid = 2.75f,
    float ht = 1.5f

)
{
    glPushMatrix();

    // --- 1. Translate to center ---
    glTranslatef(center.x, center.y, center.z);

    // --- 2. Compute rotation to align X-axis to 'direction' ---
    zVector dir = direction;
    dir.normalize();

    zVector xAxis(1, 0, 0);

    // Rotation axis = xAxis × dir
    zVector rotAxis = xAxis ^ dir;
    float axisLen = rotAxis.length();

    if (axisLen > 1e-6)
    {
        rotAxis /= axisLen;

        float dot = xAxis * dir;
        dot = std::max(-1.0f, std::min(1.0f, dot)); // clamp for safety
        float angleDeg = acos(dot) * 180.0f / PI;

        glRotatef(angleDeg, rotAxis.x, rotAxis.y, rotAxis.z);
    }
    else
    {
        // direction is parallel or opposite to xAxis
        if ((xAxis * dir) < 0.0f)
        {
            glRotatef(180.0f, 0, 1, 0);  // flip
        }
    }

    // --- 3. Scale unit cube to requested dimensions ---
    glScalef(len, wid, ht);

    // --- 4. Draw unit cube [-0.5, 0.5] using your utility ---
    Alice::vec mn(-0.5f, -0.5f, -0.5f);
    Alice::vec mx(0.5f, 0.5f, 0.5f);
    Alice::vec origin(0, 0, 0);

    

    drawCube_Lambert(mn, mx, origin, false);
    drawCube_Lambert(mn, mx, origin, true);


    glPopMatrix();
}


// - ----------- OTHER CUSTOM FUNCTIONS  -----------


#include "scalarField.h"
#include "HeightField.h"
#include "heightField_NN.h"
#include "parcel_vector.h"


void write_3DM(vector<parcel> plots, float scale, zVector cDst, zVector cSrc)
{
    RhinoIO rio;

    // --------- add parcel polygons to RIO
    vector< vector<zVector> > all_polygons;
    vector<zVector> poly;

    for (auto plot : plots)
    {
        poly.clear();
        for (int i = 0; i < plot.nPoints; i++)poly.push_back(plot.polyPoints[i] * 1000);

        all_polygons.push_back(poly);

        // -- add curve representing oriented 12 x 5 box ;

        poly.clear();
        //
            plot.setDefaultBox_OrientedRectangle(12 * 0.5, 5.5 * 0.5, plot.directionOfBox);
            plot.invertBox(plot.directionOfBox);
            plot.transformBox();
            plot.flipNormals();
        
        //
        for (int i = 0; i < plot.nPoints; i++)poly.push_back(plot.polyPoints[i] * 1000);
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

void shortest_paths_N_x_M
( 
    vector<zVector> &Source, vector<zVector>& Sinks, HeightField2D &sf_field,
    vector< vector<zVector> > &paths, vector<zVector>&clippingPolygon 
)
{
    paths.clear();
    for (int n = 0; n < Source.size(); n++)
    {
        for (int m = 0; m < Sinks.size(); m++)
        {
            //if (m == n)continue;
            zVector str = Source[n]; 
            zVector end = Sinks[m];

            if(clippingPolygon.size() > 3)
            {
                if (!pointInsidePolygon(str, clippingPolygon))continue;
                if (!pointInsidePolygon(end, clippingPolygon))continue;
            }

            sf_field.findShortestPath(str, end);


            for (int i = 0; i < 5; i++)
                sf_field.smoothPath();

            paths.push_back(sf_field.lastShortestPath);
        }
    }
}


void draw_path(vector<zVector> &path)
{
    if (path.size() < 1) return; 
    for (size_t i = 0; i < path.size() - 1; i++)
        drawLine(zVecToAliceVec(path[i]), zVecToAliceVec(path[i + 1]));

    // --- -- 
    glPointSize(3);

    for (size_t i = 0; i < path.size(); i++)
        drawPoint( zVecToAliceVec(path[i]) );

    glPointSize(3);
}
void draw_paths(vector< vector<zVector> >& allPaths)
{
    for (auto& path : allPaths)
        if (!path.empty())
            draw_path(path);
            

}




// ------------------------- APP ----------------------------------
// ------------------------- - ----------------------------------
// ------------------------- - ----------------------------------

// -- height field
HeightField2D importedHeightField, importedHeightField_original, siteHeightField, existingPathsField;
double threshold = -0.05;

// site definition

float zRangeMin;
vector<zVector> polygon;

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
vector< vector <zVector> > existing_paths;
vector <zVector> clippedContour;

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

    S.addSlider(&nn.o_weight, "o_w");

    B = *new ButtonGroup(Alice::vec(50, 75, 0));
    B.addButton(&nn.o_flip_dir, "o_flip_dir");

    // ----------- TERRAIN  -----------

    importedHeightField = *new HeightField2D();
    siteHeightField = *new HeightField2D();



    // ----------- NN ----------------

    nn = heightfieldNN(25); // or however many poses you want

    // ----- reserve input and output dimensions
    
    dummyInput.assign(nn.inputDim, 0.0f);
    output = nn.forward(dummyInput);


    // ----------- PARCELS -----------

    plots.clear();
    int id = 0;

    // ----- spatial bins

    SG = *new spaceGrid();

    // ----------- tmp 

    keyPress('2', 0, 0);
    for (int i = 0; i < 10; i++)keyPress('=', 0, 0);
    
}

void update(int value)
{
    if (compute) keyPress('t', 0, 0);
}

void draw()
{
    backGround(0.85);
    drawGrid(50);


    // ----------------------- imported heightfield and sample points 

    importedHeightField.drawSamplePoints();
    //importedHeightField.drawFieldPoints(true, false);
    //importedHeightField.drawIsocontours(threshold);

    // ----------------------- parcels

    for (auto& parcel : plots)parcel.display();

    wireFrameOn();
        //SG.drawBuckets();
        //SG.drawParticlesInBuckets();
    wireFrameOff();

    // ----------------------- paths

    glColor3f(0.25, 0, 0);
    draw_paths(shortestPaths);
    draw_paths(existing_paths);

    // ----------------------- nn

    nn.visualize(zVector(50, 350, 0), 200, 250);
    nn.drawCoveragePolygon();
    nn.draw_output_and_loss();
    //nn.drawCoverageSamples();

    for (auto pose : nn.poses)
    {
        float z = importedHeightField.getFieldValue(pose.c);
        pose.c.z = importedHeightField.mapIsoToActualHeight(z);

        
        drawOrientedCube(pose.c, pose.v);
        
        glColor3f(0, 0, 0);
        drawLine(zVecToAliceVec(pose.c), zVecToAliceVec(zVector(pose.c.x, pose.c.y, 0)));

        //drawLine(zVecToAliceVec(zVector(-35,50,0)), zVecToAliceVec(zVector(pose.c.x, pose.c.y, 0)));
    }
    
    //
    glColor3f(0, 0, 1);
    
    /*if(nn.polygons.size() > 0)
    {
        importedHeightField_original.clippedContour(threshold, clippedContour, nn.polygons[0]);
        draw_path(clippedContour);
    }*/
    
   // importedHeightField_original.computeGradient();
    //importedHeightField_original.drawStreamlinesFromSeeds(importedHeightField_original.lastShortestPath);

    

    //if( plots.size() > 0)
    {
       // siteHeightField.drawFieldPoints();
        //siteHeightField.drawIsocontours(threshold);

       // glTranslatef(100, 0, 0);
       // existingPathsField.drawFieldPoints();
        existingPathsField.clearField();
        float scale = importedHeightField.scale;
        float hlen = 12 * 0.5 * scale;
        float hw = 5.5 * 0.5 * scale;

        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                existingPathsField.field[i][j] = evalBlendedOrientedRectSDF(existingPathsField.gridPoints[i][j], nn.poses, hlen,hw );
            }
        }
        
        existingPathsField.rescaleFieldToRange(-1, 1);
        //existingPathsField.drawFieldPoints();
        existingPathsField.drawIsocontours(threshold);
    }



}

int n = 0;



void keyPress(unsigned char k, int xm, int ym)
{
    
    if (k == '2')
    {
        RhinoIO in_RIO;
        in_RIO.Read3dm(L"data/CF_beta_village_extract.3dm");//beta_village_paths.3dm
        auto names = in_RIO.GetObjectInfo();

        vector< RhinoObjectInfo> curves, meshes, pclouds;
        in_RIO.SeparateGeometryTypes(names,curves, meshes, pclouds);

        // ---------- import point cloud .. if more than one PC exists, all are processesd, but only last one is stored 

        for (const auto& obj : pclouds)
        {


            // Try POINT CLOUD
            const ON_PointCloud* PC = ON_PointCloud::Cast(obj.geometry);
            if ( PC /*&& obj.name == L"PC"*/ )
            {
                int count = PC->m_P.Count();
                importedHeightField.samples.clear();

                for (int i = 0; i < count; i++)
                {
                    const ON_3dPoint& p = PC->m_P[i];
                    importedHeightField.samples.emplace_back(zVector(p.x, p.y, p.z));
                 
                }

                
                importedHeightField.rescaleSamplesToBoundingBox(zVector(-50, -50, -50), zVector(50, 50, 50));
                // samples to scalar field
                importedHeightField.clearField();
                importedHeightField.interpolateToGrid_MLS();
                zRangeMin = importedHeightField.zMin;

                // make a copy 
                importedHeightField_original.clearField();
                for (int i = 0; i < SF_RES; i++)
                    for (int j = 0; j < SF_RES; j++)importedHeightField_original.field[i][j] = importedHeightField.field[i][j];
                importedHeightField_original.scale = importedHeightField.scale;
            }
            

        }

     

        // ---------- import existing paths, and BND polygon
        existing_paths.clear();
        vector <zVector> poly;

        for (const auto& obj : curves)
        {
            
            const ON_Curve* crv = ON_Curve::Cast(obj.geometry);

            if( crv && obj.name == L"EXIST_PATH") // all curves with name Attr == EXIST_PATH
            {
                wprintf(L"[name EXIST_PATH ] %ls\n", obj.name.c_str());
                poly.clear();
                in_RIO.sample_curve_unifrom(crv, poly);
                existing_paths.push_back(poly);
            }

            if (crv && obj.name == L"BND")
            {
                polygon.clear();
                in_RIO.sample_curve_unifrom(crv, polygon);
               
            }

        }

        for (auto& path : existing_paths)
            for (auto& p : path)p *= importedHeightField.scale;

        existingPathsField.clearField();
        existingPathsField.addSDFfromPolylines(existing_paths,3);
     
        // put the path back along terrain;
        for (auto& path : existing_paths)
            for (auto& p : path)
            {
                p.z = importedHeightField.getFieldValue(p);
                p.z = importedHeightField.mapIsoToActualHeight(p.z);
            }
        // ----------- terrain trim

        importedHeightField.rescalePoints(polygon);// scale polygon by same amount as height feild points.
        importedHeightField.trimFieldWithPolygon(polygon);
        importedHeightField.subtract(existingPathsField);

        nn.correspondingHeightField = &importedHeightField;
    }
    
    
    if (k == 'j')
    {
        siteHeightField.clearField();

        vector<zVector> plot_centers;
        for (auto& parcel : plots)
            for (auto& poly : nn.polygons)
                    if (pointInsidePolygon(parcel.centerOfBox, poly)) plot_centers.push_back(parcel.centerOfBox);

        siteHeightField.addVoronoi(plot_centers);
        siteHeightField.trimFieldWithPolygons(nn.polygons);

       // siteHeightField.rescaleFieldToRange();
    }

    if (k == 'h')
    {
        // ----------------- high line contour extraction
        
        // contour segments and ordered set
        /*zRangeMin += 1.0;
        if (zRangeMin >= importedHeightField.zMax)zRangeMin = importedHeightField.zMin;

        float iso = ofMap( zRangeMin, importedHeightField.MLS_zMin, importedHeightField.MLS_zMax, 0, 1);*/
        importedHeightField_original.computeIsocontours(threshold);
        std::vector<std::vector<zVector>> contours = importedHeightField_original.getOrderedContours();

        // extract largest contiguous set
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

        // resample contour, trim with polygon
        
        float scale = importedHeightField.scale;

        poly = importedHeightField.resamplePolyline(poly, 2.5 * (1.0f / scale));
        for (int n = 0; n < 5; n++)importedHeightField_original.smoothPath(poly);

        importedHeightField_original.lastShortestPath.clear();
        for (auto& p : poly)
            for (auto& poly : nn.polygons)
                if (pointInsidePolygon(p, poly)) importedHeightField_original.lastShortestPath.push_back(p);

        poly = importedHeightField_original.lastShortestPath;

       // ---------------------------- shortest paths to parcel from contour high line ;
       // cost field
        siteHeightField.clearField();
        for (int i = 0; i < SF_RES; i++)
            for (int j = 0; j < SF_RES; j++)
                siteHeightField.field[i][j] = importedHeightField_original.field[i][j];
        
        siteHeightField.trimFieldWithPolygons(nn.polygons);

        // cost field scaling

        vector<vector<zVector>> polys;
        for (auto& parcel : plots)
            for (auto& poly : nn.polygons)
                if (pointInsidePolygon(parcel.centerOfBox, poly)) polys.push_back(parcel.polyPoints);

        siteHeightField.scale_scalar_within_polygons(polys);


        // -- node sets N (poly) & M (poses);

        
        std::vector<Pose2D> poses;
        nn.extractPoses(output, poses, true);
        
        // -- paths from M nodes to N nodes
        // poly x poses[i].c

        // -- path storage
        shortestPaths.clear();
        vector<zVector> sources, sinks;
        //for (int n = 0; n < poly.size(); n++)sources.push_back(poly[n]);
        sources.push_back(poly[0]);
        sources.push_back( poly[poly.size() - 1] ); 
        shortestPaths.clear();

        for (int m = 0; m < poses.size(); m++)sinks.push_back(poses[m].c);

        if(nn.polygons.size() >0 )
        shortest_paths_N_x_M(sources, sinks, siteHeightField, shortestPaths, nn.polygons[0]);


    }

    if (k == 'g') // ----------paths : N nodes to N nodes (poses)
    {
        // cost field
        siteHeightField.clearField();
        for (int i = 0; i < SF_RES; i++)
            for (int j = 0; j < SF_RES; j++)
                siteHeightField.field[i][j] = importedHeightField_original.field[i][j];

        siteHeightField.trimFieldWithPolygons(nn.polygons);

        // cost field scaling

        vector< vector<zVector> > polys;
        for (auto& parcel : plots)
            for (auto& poly : nn.polygons)
                if (pointInsidePolygon(parcel.centerOfBox, poly)) polys.push_back(parcel.polyPoints);

        siteHeightField.scale_scalar_within_polygons(polys);

        // node set
        std::vector<Pose2D> poses;
        nn.extractPoses(output, poses, true);


        // paths N x N 
        shortestPaths.clear();

        vector<zVector> sources, sinks;
        for (int n = 0; n < 1/*poses.size()*/; n++)sources.push_back(poses[n].c);
        for (int m = 0; m < poses.size(); m++)sinks.push_back(poses[m].c);

        if(nn.polygons.size() > 0)
        shortest_paths_N_x_M(sources, sinks, siteHeightField, shortestPaths, nn.polygons[0]);

        
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

            /*float z = importedHeightField.getFieldValue(pose.c);
            z = importedHeightField.mapIsoToActualHeight(z);
            plot.centerOfBox.z = z;*/

            zVector dir = pose.v;
            dir.normalize();
            dir.z = 0;
           /* dir = dir ^ zVector(0, 0, 1);
            dir.normalize();*/
            plot.directionOfBox = dir;


            //plot.setDefaultBox();
            plot.setDefaultBox_OrientedRectangle(12 * 0.5, 5.5 * 0.5, dir);
            plot.invertBox(dir);
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

        for (auto poly : nn.polygons)
            for (auto p : poly)SG.addPosition(p);

        for (auto p : importedHeightField_original.lastShortestPath)SG.addPosition(p);

        // re-partition
        SG.PartitionParticlesToBuckets();

    }




    // --------------- SITE DEFINITION AND NUERAL NET COVERAGE POLYGON  ---------------


    if (k == '=') // get next contour polygon
    {
        // nn.generateSamplesInRange(myHeightField, zRangeMin, zRangeMin+2);

        // -------- get contours and order them

        zRangeMin += 1.0 * importedHeightField.scale; //real dims * scale factor ; zMAx,zMIn, MLS_zMIn,MLS_zMax all already scaled.
        if (zRangeMin >= importedHeightField.zMax)zRangeMin = importedHeightField.zMin;


        float iso = ofMap(zRangeMin, importedHeightField.MLS_zMin, importedHeightField.MLS_zMax, 0, 1);
        printf(" %.2f iso, %.2f zRangeMin, \n", iso, zRangeMin);
        importedHeightField.computeIsocontours(iso);
        std::vector<std::vector<zVector>> contours = importedHeightField.getOrderedContours();

         

        // -------- find contour island with most points

        vector< vector<zVector>>  polys;
        size_t maxPts = 0;

        for (int i = 0; i < contours.size(); i++)
            if ( importedHeightField.area_of_contour_island( contours[i]) > 50) polys.push_back(contours[i]);

        cout << maxPts << " -- " << polys.size() << endl;

        // -------- if contour island is valid, set it as polygon for NN to cover with sites.

        if (polys.size() > 0)
        {
            // -------- smooth contour
            for( auto &poly : polys)
                for (int i = 0; i < 15; i++) importedHeightField.smoothPath(poly);


            // -------- set contour as polygon of NN and generate sample points within
            nn.setTargetPolygons(polys);
            nn.generateSDFSamplePointsFromPolygons();

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

        learningRate = ofClamp(learningRate, 1e-2, 0.25);
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