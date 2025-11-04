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

// - ----------- OPEN NURBS  -----------


// defining OPENNURBS_PUBLIC_INSTALL_DIR enables automatic linking using pragmas
#define OPENNURBS_PUBLIC_INSTALL_DIR "C:/Users/shajay.b/source/repos/opennurbs"
// uncomment the next line if you want to use opennurbs as a DLL
#define OPENNURBS_IMPORTS
#include "C:/Users/shajay.b/source/repos/opennurbs/opennurbs_public.h"




// - ----------- OTHER CUSTOM CLASSES  -----------

#include "scalarField.h"
#include "HeightField.h"
#include "heightField_NN.h"
#include "parcel.h"


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
            zVector pt = plot.boxPoints[i];

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
    const wchar_t* outfile = L"data/simplest_polyline.3dm";
    bool ok = model.Write(outfile);             // writes using default/latest version
    // If you want a specific version (e.g., Rhino 6 = 60, 7 = 70) and your SDK supports it:
    // bool ok = model.Write(outfile, 60);


    ok ? printf("Wrote 3dm: simple_polyline.3dm\n") : printf("Failed to write 3dm file.\n");

    ON::End();

}



// ------------------------- APP ----------------------------------
// ------------------------- - ----------------------------------
// ------------------------- - ----------------------------------

// -- height field
HeightField2D myHeightField , myHeightField1 , myHeightField2;
double threshold;

// -- height field nn
heightfieldNN nn;
vector<float> output;
vector<float> dummyInput = { 1.0f };
std::vector<float> dummyTarget = { 1.0f };; // unused

float learningRate = 0.1f;
float zRangeMin;
std::vector<zVector> polygon;

// -- parcels

vector<parcel> plots;
parcel plot;
spaceGrid SG;

// -- app

bool compute = false;
double prevLoss = 0.0;

// -------------------------

void setup()
{
    
  
    
    S.numSliders = 0;
    S.addSlider(&threshold, "tv");// make a slider control for the variable called width;
    S.sliders[0].minVal = -1; // myHeightField.zScale * -1;
    S.sliders[0].maxVal = 1; //  myHeightField.zScale;

    // ----------- terrrain 
    myHeightField = *new HeightField2D();
    myHeightField1 = *new HeightField2D();
    myHeightField2 = *new HeightField2D();

    myHeightField.clearField();
    myHeightField.readSamplesAndInterpolate("data/cabins_site.txt");
    zRangeMin = myHeightField.zMin;

    myHeightField1.addCircleSDF(zVector(0, 0, 0), 4);
    myHeightField2.addCircleSDF(zVector(0, 0, 0), 4); 
    
    // ----------- terrain trim

    polygon.clear();
    loadPolygonFromFile("data/terrain_boundary_poly.txt", polygon);
    myHeightField.rescalePoints(polygon);
    myHeightField.trimFieldWithPolygon(polygon);
    
  

    // ----------- NN ----------------
   
    nn = heightfieldNN(25); // or however many poses you want

    //  ----- SDF loss polygon
    nn.setTargetPolygon(polygon);
    //nn.generateSDFSamplePointsFromPolygon();

    //
    dummyInput.clear();

    for (int i = 0; i <  nn.n; i++)
    {
        dummyInput.push_back( 0 ); // x 
        dummyInput.push_back( ofRandom(-1, 1) ); // y
    }

    nn.setInputSeeds(dummyInput);
    output = nn.forward(dummyInput);

   
    std::vector<float> y_pred = nn.forward(dummyInput);
    printf("Pred: [");
        for (float v : y_pred) printf("%.4f ", v);
    printf("]\n");

    // ----------- parcels

    plots.clear();
    int id = 0;

    /*for( int i = 0; i < 2; i+= 1)
    {
        for (int j = 0; j < 2; j++)
        {
            plot.centerOfBox = zVector(i * 20, j * 20, 0);
            plot.directionOfBox = zVector(1, 1, 0);;
            plot.setDefaultBox();
            plot.transformBox();
            plot.id_u = id++;
            plots.push_back(plot);
        }

    }*/

    //

    SG = *new spaceGrid();
}

void update(int value)
{
    if ( compute ) keyPress('t', 0, 0);
}

void draw()
{
    backGround(0.99);
    drawGrid(50);


    // ------ imported heightfield and sample points 

    myHeightField.drawSamplePoints();
    //myHeightField.drawFieldPoints(false, false);

    // ---------parcels

    for (auto& parcel : plots)parcel.display();

    wireFrameOn();
        SG.drawBuckets();
        //SG.drawParticlesInBuckets();
    wireFrameOff();

   // ----------------------- nn
  
   nn.visualize(zVector(50, 350, 0), 200, 250);
   nn.drawPolygon();


   std::vector<Pose2D> poses;
   nn.extractPoses(output, poses, true);

   glPointSize(5);
   glColor3f(0, 0, 0);
   for (auto& pose : poses)
   {
       drawPoint(zVecToAliceVec(pose.c ));
      // drawLine(zVecToAliceVec(pose.c), zVecToAliceVec(pose.c + pose.v * 5.0));
       drawCircle(zVecToAliceVec(pose.c), radius, 32);
   }
   glPointSize(1);





    //{
    //   // myHeightField.drawFieldPoints(false, false);
    //   float ht = myHeightField.zMin;

    //    glColor3f(0, 0, 0);
    //    /*for (double tv = 0; tv < threshold; tv += 0.1)
    //    {
    //        float h = ofMap(tv, 0, 1, myHeightField.MLS_zMin, myHeightField.MLS_zMax);
    //        
    //        glPushMatrix();
    //
    //        {
    //            glTranslatef(0.0f, 0.0f, h);
    //            myHeightField.drawIsocontours(tv);
    //        }
    //        glPopMatrix();

    //        
    //    }*/

    //    float iso = ofMap(zRangeMin, myHeightField.MLS_zMin, myHeightField.MLS_zMax, 0, 1);
    //    //myHeightField.drawIsocontours(iso);
    //   

    //    /// ------
    //    iso = ofMap(zRangeMin+1, myHeightField.MLS_zMin, myHeightField.MLS_zMax, 0, 1);
    //    //myHeightField.drawIsocontours(iso);

    //    glLineWidth(1);
    //}

    //glTranslatef(120, 0, 0);
    //{
    //    myHeightField1.drawFieldPoints(false, false);

    //    glColor3f(0, 0, 0);
    //    float iso = ofMap(0, myHeightField.MLS_zMin, myHeightField.MLS_zMax, 0, 1);
    //    myHeightField1.drawIsocontours(iso);
    //  
    //}

    //glTranslatef(120, 0, 0);
    //{
    //    myHeightField2.drawFieldPoints(false, false);

    //    glColor3f(0, 0, 0);
    //    float iso = ofMap(0, myHeightField.MLS_zMin, myHeightField.MLS_zMax, 0, 1);
    //    myHeightField2.drawIsocontours(threshold);

    //}

    //

}

void keyPress(unsigned char k, int xm, int ym)
{

    // --------------- WRITE 3DM ---------------


    if (k == '3')
    {
        write_3DM(plots, myHeightField.scale, myHeightField.cDst,myHeightField.cSrc);
    }
    
    
    // --------------- PARCELS  ---------------
    
    if (k == 'p') // populate
    {
        std::vector<Pose2D> poses;
        nn.extractPoses(output, poses, true);

        int id = 0;

        plots.clear();
        for( auto &pose : poses)
        {
            plot.centerOfBox = pose.c;
            plot.directionOfBox = zVector(1, 1, 0);;
            plot.setDefaultBox();
            plot.transformBox();
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
                    SG.addPosition(parcel.boxPoints[i]);
                }

            for( auto p : nn.polygon)SG.addPosition(p);

        // re-partition
        SG.PartitionParticlesToBuckets();
        
    }
    
    // --------------- NUERAL NET  ---------------


    if (k == '=') // get next cotnour polygon
    {
       // nn.generateSamplesInRange(myHeightField, zRangeMin, zRangeMin+2);

         // -------- get contours and order then
        zRangeMin += 1.0;
        if (zRangeMin >= myHeightField.zMax)zRangeMin = myHeightField.zMin;

       
        float iso = ofMap(zRangeMin, myHeightField.MLS_zMin, myHeightField.MLS_zMax, 0, 1);
        myHeightField.computeIsocontours(iso);
        std::vector<std::vector<zVector>> contours = myHeightField.getOrderedContours();


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
            
            nn.set_field_values_from_polygon(poly, myHeightField2);
            myHeightField2.subtract(myHeightField1);

            nn.set_field_values_from_polygon(poly, myHeightField1);
            
            //myHeightField1.smoothDiffuseIsotropic();
            //myHeightField1.smoothDiffuseIsotropic();
            //myHeightField1.rescaleFieldToRange(-1, 1);
           // myHeightField.rescaleFieldToRange(-1, 1);

            // -----------

            nn.setTargetPolygon(poly);
            nn.generateSDFSamplePointsFromPolygon();
            //nn.translate_SDFPolygon_and_samples_to_origin();
          
        }

       // cout << zRangeMin << " -- " << contours.size() << endl;
    }

    if (k == 'c')compute = !compute; // iteratively train NN to minimise loss function

    if (k == 'u') runUnitTest(); // NN unit test to check if a default MLP, from which heighField_NN is derived converges

    if (k == 'p')
    {
       
        double alpha_base = 0.125;
            double sigma_cells = 1.15;  // Gaussian splat radius in grid cells
            double pin_threshold = 0.9;// fraction of max weight to pin (Dirichlet)
            int max_iters = 1500;
            double omega = 1.88;
            double tol = 1e-4;

        myHeightField.reconstructScreenedPoisson(alpha_base,sigma_cells,pin_threshold);
        myHeightField.setGridPointHeights();
    }

    if (k == 's')
    {
        myHeightField.smoothDiffuseIsotropic(0.15, 1, true);
        myHeightField1.smoothDiffuseAnisotropic(0.2, 1, 0.1, ScalarField2D::PMVariant::Exp, ScalarField2D::DiffuseDir::AlongIsophote, 2, true);
    }


    if (k == 't')
    {
        // Forward pass
       // std::vector<float> noisyInput = dummyInput;
       // noisyInput[0] += ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        //for (auto &x : dummyInput)x = ofRandom(-1, 1);
        std::vector<float> y_pred = nn.forward(dummyInput);

        //std::vector<float> y_pred = nn.forward(dummyInput);

        // Loss
        float loss = nn.computeLoss(y_pred, dummyTarget);

        // Gradient (numerical)
        std::vector<float> grad;
        nn.computeGradient(dummyInput, dummyTarget, grad);

        // Backward update
        printf(" %.4f,%.4f \n", fabs(loss - prevLoss), learningRate);
        
        if (fabs(loss - prevLoss) < 1e-2) learningRate *= 1.1; 

        learningRate = ofClamp(learningRate, 1e-2, 0.15);
        prevLoss = loss;
        nn.backward(grad, learningRate);

        // Debug

        output = y_pred;

        // Print output vector
        printf("Loss: %.8f | Output: [", loss);
        /*for (int i = 0; i < y_pred.size(); ++i)
        {
            printf("%.4f", y_pred[i]);
            if (i < y_pred.size() - 1) printf(", ");
        }*/
        printf("]\n");


        //------------------------ 
        std::vector<Pose2D> poses;
        nn.extractPoses(output, poses, true); 
        
        zPointArray sites;
        for (int i = 0; i < poses.size(); i++)
                    sites.push_back(poses[i].c);

        for (int i = 0; i < myHeightField2.RES; i++)
            for (int j = 0; j < myHeightField2.RES; j++)
                myHeightField2.field[i][j] = evalBlendedCircleSDF(myHeightField2.gridPoints[i][j], poses, radius);

        myHeightField2.clearField();
        myHeightField2.addVoronoi(sites);
       ///myHeightField2.subtract(myHeightField1);
       // myHeightField2.normalise();
        myHeightField2.rescaleFieldToRange(-1, 1);
    }

    
}

void mousePress(int b, int state, int x, int y)
{
}

void mouseMotion(int x, int y)
{
}

#endif // _MAIN_
