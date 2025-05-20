#define _MAIN_
#ifdef _MAIN_

#include "main.h"


//// zSpace Core Headers
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;

Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

zVector AliceVecToZvec(Alice::vec& in)
{
    return zVector(in.x, in.y, in.z);
}

#include "scalarField.h"

// Derived class
class ScalarFieldWithCenters : public ScalarField2D
{
private:
    float animationPhase = 0.0f;
    float animationSpeed = 0.05f;

    std::vector<zVector> originalCenters;
    std::vector<zVector> targetCenters; // 🔸 NEW

    std::vector<std::vector<zVector>> stackedContours;
    float contourZOffset = 0.25f;

public:
    std::vector<zVector> centers;
    std::vector<zObjMesh> contourMeshes;
    zObjMesh CombinedMesh;
    bool CM_isValid;


    void initializeCentersOnCircle(int numPoints = 10, float radius = 20.0f)
    {
        centers.clear();
        originalCenters.clear();
        targetCenters.clear();

        float angleStep = TWO_PI / float(numPoints);
        for (int i = 0; i < numPoints; i++)
        {
            float x = radius * cos(i * angleStep);
            float y = radius * sin(i * angleStep);
            zVector c(x, y, 0);
            centers.push_back(c);
            originalCenters.push_back(c);
            targetCenters.push_back(zVector(0, 0, 0)); // default target at origin
        }
    }


    void addCirclesFromCenters(float sdfRadius = 5.0f)
    {
        clearField();

        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                zVector pt = gridPoints[i][j];
                float minDist = 1e6f;

                for (const auto& c : centers)
                {
                    float d = pt.distanceTo(zVector(c));  // copy to avoid const-ref issue
                    float sdf = d - sdfRadius;
                    minDist = std::min(minDist, sdf);
                }

                field[i][j] = minDist;
            }
        }

        rescaleFieldToRange(-1, 1);
    }

    void animateCentersToTargetsAndBack()
    {
        animationPhase += animationSpeed;
        float t = 0.5f * (1.0f + sin(animationPhase)); // [0,1]

        for (int i = 0; i < centers.size(); i++)
        {
            zVector orig = originalCenters[i];
            zVector target = targetCenters[i];

            orig *= (1.0f - t);
            target *= t;

            centers[i] = orig + target;
        }
    }


    void setAnimationSpeed(float s)
    {
        animationSpeed = s;
    }

    void setTargetForCenter(int index, const zVector& target)
    {
        if (index >= 0 && index < targetCenters.size())
        {
            targetCenters[index] = target;
        }
    }

    void setAllTargets(const std::vector<zVector>& targets)
    {
        int count = std::min(int(targetCenters.size()), int(targets.size()));
        for (int i = 0; i < count; i++)
        {
            targetCenters[i] = targets[i];
        }
    }

    void copyAndStackCurrentContours(float threshold)
    {
        computeIsocontours(threshold);
        auto contours = getOrderedContours();

        for (auto& contour : contours)
        {
            for (auto& pt : contour)
            {
                pt.z += stackedContours.size() * contourZOffset;
            }
            stackedContours.push_back(contour);
        }
    }

    void exportStackedContoursAsCSV(const std::string& filename = "stackedContours.csv")
    {
        std::ofstream out(filename);
        if (!out.is_open())
        {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }

        for (size_t layerIdx = 0; layerIdx < stackedContours.size(); ++layerIdx)
        {
            out << "Layer_" << layerIdx << "\n";
            for (const auto& pt : stackedContours[layerIdx])
            {
                out << std::fixed << std::setprecision(6)
                    << pt.x << "," << pt.y << "," << pt.z << "\n";
            }
            out << "\n";
        }

        out.close();
        std::cout << "Stacked contours exported to: " << filename << std::endl;
    }

    void convertContoursToPolygonDisks()
    {
        
        if (stackedContours.empty())return;

        contourMeshes.clear();
        contourMeshes.resize(stackedContours.size());
        for( auto &om : contourMeshes)om.setDisplayElements(true, false, true);

        int cnt = 0;

        for (const auto& contour : stackedContours)
        {
            if (contour.size() < 3) continue; // need at least a triangle

            zPointArray CM_positions;
            zIntArray CM_Connects, CM_Counts;

            int startIndex = 0;

            // Add contour points
            for (const auto& pt : contour)
            {
                CM_positions.push_back(zVector(pt)); // make non-const copy
            }

            // Generate fan from center
            zVector center(0, 0, 0);
            for (const auto& pt : contour) center += pt;
            center /= contour.size();
            CM_positions.push_back(center);
            int centerIndex = CM_positions.size() - 1;

            // Create triangle faces: (center, vi, vi+1)
            for (int i = 0; i < contour.size(); i++)
            {
                int next = (i + 1) % contour.size();

                CM_Connects.push_back(centerIndex);
                CM_Connects.push_back(i);
                CM_Connects.push_back(next);

                CM_Counts.push_back(3);
            }


            // Final sanity check
            int numVerts = CM_positions.size();
            int numFaces = CM_Counts.size();
            int numConnects = CM_Connects.size();

            printf(" nv %i nf %i ne %i  \n", numVerts, numFaces, numConnects);

            if (numFaces * 3 != numConnects)
            {
                std::cerr << "Invalid mesh data: mismatch in face counts and connects." << std::endl;
                continue;
            }

            // Create mesh object and function wrapper
            cout << contourMeshes.size() << " -- " << cnt << endl;
            if (cnt >= contourMeshes.size())continue;

           //zObjMesh omesh;
           zFnMesh fnMesh(contourMeshes[cnt++]);//
           fnMesh.create(CM_positions, CM_Counts, CM_Connects);

           //// contourMeshes.push_back(outMesh);
           //char s[200];
           //sprintf(s, "data/out_%i.obj", cnt);
           //fnMesh.to(s, zSpace::zOBJ);
           
          // cout << "success mesh" << cnt <<  endl;
        }

        cout << "success" << endl;
    }

    void combineMeshes()
    {
        int sizeOfArray = contourMeshes.size();

        {
            // arrays of Combined Mesh (CM): to collect and store points, face and edge data from each of the meshes in alignedMesh array
            zPointArray CM_positions;
            zIntArray CM_Counts, CM_Connects;

            //arrays of a Each Mesh (EM) in the array of alignedMeshes;
            zPointArray EM_positions;
            zIntArray EM_Counts, EM_Connects;

            // go through each object in the aligned mesh array. 
            for (int i = 0; i < sizeOfArray; i++)
            {
                //make an instance of a zFnMesh object, attach it to the oMesh stored in the alignedMesh object.
                zFnMesh fnMesh(contourMeshes[i]);
                if (!(fnMesh.numEdges() > 0 && fnMesh.numVertices() > 0))continue;

                int numV = CM_positions.size();

                // get and store the vertex positions of the mesh using fnMesh;
                EM_positions.clear();
                fnMesh.getVertexPositions(EM_positions);

                // add the vertex positions into the array of vertex positions of the combined mesh.
                // copy EM_positions array to the end of the CM_positions
                copy(EM_positions.begin(), EM_positions.end(), back_inserter(CM_positions));

                // similarly collect and add edge and face data from each alignedMesh object and,
                // add the data into the array of edge and faca data of the combined mesh
                EM_Counts.clear(); EM_Connects.clear();
                fnMesh.getPolygonData(EM_Connects, EM_Counts);

                // offset each face-vertex id by current number of vertices in the Combined mesh;
                for (auto& i : EM_Connects)i += numV;
                // copy EM_Connects array to the end of the CM_connects
                copy(EM_Connects.begin(), EM_Connects.end(), back_inserter(CM_Connects));

                // copy EM_Counts array to the end of the CM_Counts
                copy(EM_Counts.begin(), EM_Counts.end(), back_inserter(CM_Counts));
            }

            /// ------------- using the vertex, edge and face data collected to 
            // create a combined mesh and export it as OBJ.
            
            zFnMesh fnMesh(CombinedMesh);
            
            fnMesh.create(CM_positions, CM_Counts, CM_Connects);
            CombinedMesh.setDisplayElements(true, false, true);
            fnMesh.to("data/CM.obj", zOBJ);

            if (fnMesh.numEdges() > 0 && fnMesh.numVertices() > 0)CM_isValid = true;
        }
    }

    void drawCombinedMesh()
    {
        if (contourMeshes.empty())return;

        if (CM_isValid)CombinedMesh.draw();

        /*for (auto &mesh : contourMeshes)
        {
            zFnMesh fn(mesh);
            
            if( fn.numEdges() > 0 && fn.numVertices() > 0)mesh.draw();
            
        }*/
    }

    void draw(float threshold = 0.0f, bool showCenters = true)
    {
        drawFieldPoints();
        drawIsocontours(threshold);

        if (showCenters)
        {
            glPointSize(8);
            glColor3f(1, 0, 0);
            for (const auto& c : centers)
            {
                drawPoint(zVecToAliceVec(zVector(c)));
            }
            glPointSize(1);
        }
    }

    void drawStackedContours()
    {
        glColor3f(0.2f, 0.2f, 0.2f);
        glLineWidth(2);

        for ( auto contour : stackedContours)
        {
            smoothContour(contour, 15);
            for (size_t i = 0; i < contour.size() - 1; i++)
            {
                drawLine(zVecToAliceVec(zVector(contour[i])), zVecToAliceVec(zVector(contour[i + 1])));
            }
        }

        glLineWidth(1);
    }
};


//---------------------------------------------

ScalarFieldWithCenters myField;
double isoThreshold = 0.0f;
bool run = false;

void setup()
{
    S.addSlider(&isoThreshold, "thresh");
    S.sliders[0].minVal = -1.0;
    S.sliders[0].maxVal = 1.0;

    myField.initializeCentersOnCircle(16, 25.0f);
    myField.addCirclesFromCenters(6.0f);

    std::vector<zVector> targets;
    for (int i = 0; i < myField.centers.size(); i += 2)
    {
        // Target A (e.g. center inward)
        zVector targetA(0, 0, 0);

        // Target B (mirror of original, pushes outward)
        zVector c1 = myField.centers[i];
        zVector c2 = (i + 1 < myField.centers.size()) ? myField.centers[i + 1] : c1;

        zVector avg = (c1 + c2) * 0.5;
        zVector outward = avg * 1.25f; // push a bit further out

        myField.setTargetForCenter(i, targetA);
        if (i + 1 < myField.centers.size()) myField.setTargetForCenter(i + 1, outward);
    }
}

void update(int value)
{
    if (!run) return;

    myField.animateCentersToTargetsAndBack();
    myField.addCirclesFromCenters(6.0f);

    keyPress('s', 0, 0);
}


void draw()
{
    backGround(0.8);
    drawGrid(50);

    myField.draw(isoThreshold, true);
    myField.drawStackedContours(); // ← draw all stacked layers

    myField.drawCombinedMesh();
}


void keyPress(unsigned char k, int xm, int ym)
{
    if (k == '1')run = !run;

    if (k == 'e')
    {
        myField.exportStackedContoursAsCSV("data/stackedContours.csv");
    }

    if (k == 'm')
    {
        myField.convertContoursToPolygonDisks();
        myField.combineMeshes();
    }
    
    if (k == 'r') // regenerate centers randomly on a circle
    {
        myField.initializeCentersOnCircle(10, 25.0f);
        myField.addCirclesFromCenters(6.0f);
    }

    if (k == 's') // Stack current contours
    {
        myField.copyAndStackCurrentContours(isoThreshold);
    }
}

void mousePress(int b, int state, int x, int y)
{
}

void mouseMotion(int x, int y)
{
}

#endif // _MAIN_
