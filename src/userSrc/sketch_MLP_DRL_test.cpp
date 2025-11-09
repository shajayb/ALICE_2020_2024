#define _MAIN_
#ifdef _MAIN_

#include "main.h"

#include <vector>
#include <cmath>
#include <fstream>
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;

//------------------------------------------------------------------ Utility
Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

zVector AliceVecToZvec(Alice::vec& in)
{
    return zVector(in.x, in.y, in.z);
}

#include "scalarField.h" //// two functions must be turned on in scalarfIELD.H for sketch_circleSDF_fitter.cpp
#include "genericMLP.h" 

/// --------- sub class

//------------------------------------------------------------------ Utility

bool isInsidePolygon(zVector& p, std::vector<zVector>& poly)
{
    int windingNumber = 0;

    for (int i = 0; i < poly.size(); i++)
    {
        zVector& a = poly[i];
        zVector& b = poly[(i + 1) % poly.size()];

        if (a.y <= p.y)
        {
            if (b.y > p.y && ((b - a) ^ (p - a)).z > 0)
                ++windingNumber;
        }
        else
        {
            if (b.y <= p.y && ((b - a) ^ (p - a)).z < 0)
                --windingNumber;
        }
    }

    return (windingNumber != 0);
}
float polygonSDF(zVector& p, std::vector<zVector>& poly)
{
    float minDist = 1e6;
    int n = poly.size();

    for (int i = 0; i < n; i++)
    {
        zVector a = poly[i];
        zVector b = poly[(i + 1) % n];

        zVector ab = b - a;
        zVector ap = p - a;

        float t = std::max(0.0f, std::min(1.0f, (ab * ap) / (ab * ab)));
        zVector proj = a + ab * t;
        float d = p.distanceTo(proj);
        minDist = std::min(minDist, d);
    }

    return minDist * (isInsidePolygon(p, poly) ? -1.0f : 1.0f);
}

void loadPolygonFromCSV( std::string& filename, vector<zVector>& polygon)
{
    polygon.clear();
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string xStr, yStr;
        if (std::getline(ss, xStr, ',') && std::getline(ss, yStr))
        {
            float x = std::stof(xStr);
            float y = std::stof(yStr);
            polygon.emplace_back(x, y, 0);
        }
    }

    cout << polygon.size() << " polygon size" << endl;
}

void samplePoints(std::vector<zVector>& trainingSamples, std::vector<float>& sdfGT, vector<zVector>& polygon)
{
    // collect input-output pairs of information

    trainingSamples.clear();
    sdfGT.clear();

    for (float x = -50; x <= 50; x += 2.0f)
    {
        for (float y = -50; y <= 50; y += 2.0f)
        {
            zVector pt(x, y, 0);
            if (  isInsidePolygon(pt, polygon) )
            {
                trainingSamples.push_back(pt); // input exmaples
                sdfGT.push_back(polygonSDF(pt, polygon)); // known output expected for the input
            }
        }
    }

    std::cout << "Training samples: " << trainingSamples.size() << std::endl;
}

float blendCircleSDFs(zVector pt, std::vector<zVector>& centers, std::vector<float>& radii, float k)
{
    float d = 1e6;
    for (int i = 0; i < centers.size(); i++)
    {
        float dist = pt.distanceTo(centers[i]) - radii[i];
        d = smin(d, dist, k);
    }
    return d;
}

float orientedBoxSDF(zVector pt, zVector center, float width, float height, float angleRad)
{
    zVector d = pt - center;

    float cosA = cos(angleRad);
    float sinA = sin(angleRad);

    float localX = d.x * cosA + d.y * sinA;
    float localY = -d.x * sinA + d.y * cosA;

    float dx = fabs(localX) - width * 0.5f;
    float dy = fabs(localY) - height * 0.5f;

    float ax = std::max(dx, 0.0f);
    float ay = std::max(dy, 0.0f);

    float insideDist = std::min(std::max(dx, dy), 0.0f);
    return sqrtf(ax * ax + ay * ay) + insideDist;
}

float blendOrientedBoxSDFs(zVector pt, std::vector<zVector>& centers, std::vector<float>& angles, float width = 6.0f, float height = 3.0f, float k = 3.0f)
{
    float d = 1e6;
    for (int i = 0; i < centers.size(); i++)
    {
        float dist = orientedBoxSDF(pt, centers[i], width, height, angles[i]);
        d = min(d, dist);;// smin(d, dist, k);
    }
    return d;
}

zVector gradientAt(zVector pt, std::vector<zVector>& centers, std::vector<float>& angles, float h = 0.1f)
{
    float dx = blendOrientedBoxSDFs(pt + zVector(h, 0, 0), centers, angles) -
        blendOrientedBoxSDFs(pt - zVector(h, 0, 0), centers, angles);

    float dy = blendOrientedBoxSDFs(pt + zVector(0, h, 0), centers, angles) -
        blendOrientedBoxSDFs(pt - zVector(0, h, 0), centers, angles);

    zVector ret(dx, dy, 0);
    ret.normalize();
    return ret;
}

zVector gradientAt_polygonSDF(zVector pt, vector<zVector>&polygon, float h = 0.1f)
{
    float dx = polygonSDF(pt + zVector(h, 0, 0),polygon) -
        polygonSDF(pt - zVector(h, 0, 0), polygon);

    float dy = polygonSDF(pt + zVector(0, h, 0), polygon) -
        polygonSDF(pt - zVector(0, h, 0), polygon);

    zVector ret(dx, dy, 0);
    ret.normalize();
    return ret;
}


//------------------------------------------------------------------ MLP


class PolygonSDF_MLP : public MLP
{
public:
    using MLP::MLP;

    std::vector<zVector> polygon;
    std::vector<zVector> trainingSamples;
    std::vector<float> sdfGT;
    std::vector<float> losses;
    std::vector<float> losses_ang;
    std::vector<float> losses_sun;

    std::vector<zVector> fittedCenters;
    std::vector<float> fittedRadii;

    int number_sdf;
    double radius = 8.;
    float smoothK = 3.0f;
    zVector sunDir = zVector(1, -1, 0);

    ScalarField2D generatedField;
    ScalarField2D generatedField_1;
    int epoch = 0;

    std::vector<bool> lossOnBools = { true,false,false };


    void decodeOutput( std::vector<float>& out, std::vector<zVector>& centers, std::vector<float>& angles, std::vector<float>& heights)
    {
        centers.resize(number_sdf);
        angles.resize(number_sdf);
        heights.resize(number_sdf);


    

        float minHeight = 1.0f;
        float maxHeight = 20.0f;

        float inc = TWO_PI / float(number_sdf);
        float r = 20;

        for (int i = 0; i < number_sdf; i++)
        {
            int idx = i * 5;

            float x = r * cos(float(i) * inc);
            float y = r * sin(float(i) * inc);
            centers[i] = lossOnBools[0] ? zVector(out[idx + 0], out[idx + 1], 0) : zVector(x, y, 0);

            zVector dir(out[idx + 2], out[idx + 3], 0);
            dir.normalize();
            angles[i] = lossOnBools[1] ? atan2(dir.y, dir.x) : 0;

            heights[i] = lossOnBools[2] ? fabs(out[idx + 4]) : 5;// std::clamp(fabs(out[idx + 4]), minHeight, maxHeight);
           

            //centers[i] = (isInsidePolygon(centers[i], polygon)) ? centers[i] : zVector(x , y, 0);
        }
    }

    float angleBetween(zVector &a, zVector &b)
    {
        float dot = a.x * b.x + a.y * b.y;
        float det = a.x * b.y - a.y * b.x;
        return atan2(det, dot); // angle in radians
    }

    bool pointInsideFootprint
    (
        zVector& pt,
        zVector& center,
        float angle,
        float w,
        float d
    )
    {
        zVector u = zVector(cos(angle), sin(angle), 0);
        zVector v = zVector(-sin(angle), cos(angle), 0);
        zVector c = center;
        zVector p = pt;
        zVector rel = p - c;

        float uDot = rel * u;
        float vDot = rel * v;

        return (fabs(uDot) < w * 0.5f && fabs(vDot) < d * 0.5f);
    }

    inline float randBetween(float minVal, float maxVal)
{
    return minVal + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (maxVal - minVal);
}

    bool rayIntersectsFootprint
    (
        zVector& rayOrigin,
        zVector& rayDir,
        zVector& footprintCenter,
        float angle,
        float w,
        float d,
        float& out_t,
        zVector& out_hitPoint
    )
    {
        // Normalize direction
        zVector dir = rayDir;
        dir.normalize();

        // Build local footprint axes
        zVector u = zVector(cos(angle), sin(angle), 0);
        zVector v = zVector(-sin(angle), cos(angle), 0);

        // Transform ray to local box frame
        zVector localOrigin;
        localOrigin.x = (rayOrigin - footprintCenter) * u;
        localOrigin.y = (rayOrigin - footprintCenter) * v;

        zVector localDir;
        localDir.x = dir * u;
        localDir.y = dir * v;

        // Axis-aligned slab intersection (classic 2D)
        float tMin = -1e6;
        float tMax = 1e6;

        if (fabs(localDir.x) > 1e-6)
        {
            float t1 = (-w * 0.5f - localOrigin.x) / localDir.x;
            float t2 = (w * 0.5f - localOrigin.x) / localDir.x;

            tMin = std::max(tMin, std::min(t1, t2));
            tMax = std::min(tMax, std::max(t1, t2));
        }
        else if (fabs(localOrigin.x) > w * 0.5f)
        {
            return false; // Parallel and outside
        }

        if (fabs(localDir.y) > 1e-6)
        {
            float t1 = (-d * 0.5f - localOrigin.y) / localDir.y;
            float t2 = (d * 0.5f - localOrigin.y) / localDir.y;

            tMin = std::max(tMin, std::min(t1, t2));
            tMax = std::min(tMax, std::max(t1, t2));
        }
        else if (fabs(localOrigin.y) > d * 0.5f)
        {
            return false; // Parallel and outside
        }

        if (tMax < tMin || tMax < 0)
        {
            return false; // No valid intersection or behind ray
        }

        out_t = tMin > 0 ? tMin : tMax;

        zVector localHit = localOrigin + localDir * out_t;
        out_hitPoint = footprintCenter + u * localHit.x + v * localHit.y;

        return true;
    }


    // method / action
    float evaluateLoss
    (
        std::vector<zVector>& centers,
        std::vector<float>& angles,
        std::vector<float>& heights
    )
    {
         int N = trainingSamples.size();
         int numLossTypes = 3; // coverage, angular, shadow

        std::vector< std::vector<float> > lossesByType(numLossTypes, std::vector<float>(N, 0.0f));

        for (int i = 0; i < N; i++)
        {
            zVector pt = trainingSamples[i];

            // --- Loss 0: coverage ---
            float pred = blendCircleSDFs(pt, centers, angles, smoothK);
            float err = pred - sdfGT[i];
            lossesByType[0][i] = (std::isnan(err)) ? 0 : err * err; 
           

            // --- Loss 1: angular ---
            zVector grad = gradientAt(pt, centers, angles);
            zVector grad_polygon = gradientAt_polygonSDF(pt, polygon);
            grad.normalize();
            grad_polygon.normalize();
            //
            float angleErr = angleBetween(grad, sunDir);
            lossesByType[1][i] = (std::isnan(angleErr)) ? 0 : angleErr * angleErr;
           // printf("%0.2f,%0.2f \n", angleErr, angleErr);

            // 
            // --- Loss 2: robust local overshadow ---
            float tallestHere = 0.0f;
            std::vector<int> coveringTowers;
            bool t_exists = false;
            int tower_id = 0;

            if( lossOnBools[2])
            for (int t = 0; t < number_sdf; t++)
            {
                std::vector<zVector> c = { centers[t] };
                std::vector<float> a = { angles[t] };
                float d = blendOrientedBoxSDFs(pt, c, a);

                if (d < 1e-4) // inside tower footprint [t]
                {
                    tallestHere = std::max(tallestHere, heights[t]);
                    coveringTowers.push_back(t);
                    t_exists = true;
                    tower_id = t;
                  
                    glColor3f(0, 0, 1);
                    drawCircle(zVecToAliceVec(pt), 0.5, 32);
                   // drawLine(zVecToAliceVec(pt), zVecToAliceVec(pt + sunDir * -50));
                    break; // no need to check other footprints if one is found;
                }
            }


            //// Robust upstream search, if tower exists at training point
            float foundUpstreamHeight = 0.0f;
            bool foundHit = false;

            zVector probe = pt;

            float stepSize = 35.0f;
            float maxDist = 150.0f;

            
            if (t_exists && lossOnBools[2]) // tower exists at training point
            {
                zVector rayOrigin = pt;
                zVector rayDir = sunDir * -1; // assuming you step upstream

                for (int t = 0; t < number_sdf; t++)
                {
                    // Skip any tower that covered the base point
                    //if (std::find(coveringTowers.begin(), coveringTowers.end(), t) != coveringTowers.end())
                    if( t == tower_id)
                    {
                        continue;
                    }

                    float hit_t = 0.0f;
                    zVector hitPoint;

                    bool hit = rayIntersectsFootprint(rayOrigin, rayDir, centers[t], angles[t], 8, 4, hit_t, hitPoint);
                        

                    if (hit && hit_t >= 0 /*&& hit_t <= maxDist*/)
                    {
                        foundUpstreamHeight = std::max(foundUpstreamHeight, heights[t]);
                        foundHit = true;

                        probe = hitPoint;
                        
                        break; // Stop at first valid hit
                    }
                }
            }


            float shadowOverlap = foundHit ? 100.f - fabs(foundUpstreamHeight - tallestHere) : 1.f;

            shadowOverlap = std::clamp(shadowOverlap, 1.f, 100.f);
            lossesByType[2][i] = (std::isnan(shadowOverlap)) ? 1.f : shadowOverlap * shadowOverlap;
            

        //debug draw;//
           if ( t_exists && foundHit)
            {
               glColor3f(1, 0, 0);
               
               //drawLine(zVecToAliceVec(pt), zVecToAliceVec(pt) + Alice::vec(0, 0, tallestHere * 100));
              // drawLine(zVecToAliceVec(probe), zVecToAliceVec(probe) + Alice::vec(0, 0, foundUpstreamHeight ));
              // drawLine(zVecToAliceVec(pt), zVecToAliceVec(probe));
               drawCircle(zVecToAliceVec(pt), 0.5, 32);
               
            }
            
        }

        //
        //std::vector<bool> normalizeLoss = { false, false, true}; // match number of loss types

        //for (int t = 0; t < numLossTypes; t++)
        //{
        //    if (!normalizeLoss[t]) continue;

        //    float minVal = 1e6f, maxVal = -1e6f;
        //    for (float v : lossesByType[t])
        //    {
        //        minVal = std::min(minVal, v);
        //        maxVal = std::max(maxVal, v);
        //    }

        //    float range = std::max(maxVal - minVal, 1e-6f);
        //    for (float& v : lossesByType[t])
        //    {
        //        v = (v - minVal) / range;
        //    }
        //}
        

        // --- Combine ---
        std::vector<float> weights = { float(lossOnBools[0]),float(lossOnBools[1]),float(lossOnBools[2]) }; // adjust shadow weight
        float totalLoss = 0.0f;

        for (int i = 0; i < N; i++)
        {
            for (int t = 0; t < numLossTypes; t++)
            {
                if (lossesByType[t][i] < 1e4)
                totalLoss += weights[t] * lossesByType[t][i];
                
            }
        }

        // --- Height mean regulariser ---
        float meanHeight = 0.0f;

        for (int i = 0; i < number_sdf; i++)
        {
            meanHeight += heights[i];
        }

        meanHeight /= number_sdf;

        float targetHeight = 5.0f; // desired average
        float heightRegLoss = (meanHeight - targetHeight) * (meanHeight - targetHeight);

        float heightRegWeight = 1.0f; // adjust if needed

       // totalLoss += heightRegWeight * heightRegLoss;

        // Optional store
        losses = lossesByType[0];
        losses_ang = lossesByType[1];
        losses_sun = lossesByType[2];

        //printf("%0.2f,%0.2f totalloss, N \n", totalLoss, float(N));

        return totalLoss / float(N);
    }

    float computeLoss(std::vector<float>& x, std::vector<float>& dummy) override
    {
        auto out = forward(x);

        std::vector<zVector> centers;
        std::vector<float> angles;
        std::vector<float> heights;

        decodeOutput(out, centers, angles, heights);

        epoch++;
        return evaluateLoss(centers, angles, heights);
    }

    void computeGradient(std::vector<float>& x, std::vector<float>& dummy, std::vector<float>& gradOut) override
    {
        auto out = forward(x);
        float eps = 0.01f;

        std::vector<zVector> baseCenters;
        std::vector<float> baseAngles;
        std::vector<float> baseHeights;

        decodeOutput(out, baseCenters, baseAngles, baseHeights);

        float baseLoss = evaluateLoss(baseCenters, baseAngles, baseHeights);

        gradOut.assign(out.size(), 0.0f);

        for (int i = 0; i < out.size(); ++i)
        {
            std::vector<float> outPerturbed = out;
            outPerturbed[i] += eps;

            std::vector<zVector> centers;
            std::vector<float> angles;
            std::vector<float> heights;

            decodeOutput(outPerturbed, centers, angles, heights);

            float lossPerturbed = evaluateLoss(centers, angles, heights);
            gradOut[i] = (lossPerturbed - baseLoss) / eps;
        }
    }



    /// visualisation

    void GenerateField(std::vector<float>& x)
    {
        auto out = forward(x);

        std::vector<zVector> centers(number_sdf);
        std::vector<float> radii(number_sdf);
        std::vector<float> heights(number_sdf); // or radii
        
        decodeOutput(out,centers, radii, heights);

        GenerateField(centers, radii);

        
    }

    void GenerateField(std::vector<zVector>& centers, std::vector<float> &radii)
    {

       // for (auto& r : radii)r = radius;
        generatedField_1.clearField();
        generatedField.clearField();
        
        
        generatedField.addVoronoi(centers);
      //  generatedField_1.addVoronoi(centers);

        for (int i = 0; i < generatedField.RES; i++)
        {
            for (int j = 0; j < generatedField.RES; j++)
            {

                zVector pt = generatedField.gridPoints[i][j];
                
                float d_v = generatedField.field[i][j];
                float d_c =  blendOrientedBoxSDFs(pt, centers, radii);; // blendCircleSDFs(pt, centers, radii, smoothK);//
                float d_p = polygonSDF(pt, polygon) + 3;


                generatedField.field[i][j] = min(min(-d_v, d_c), -d_p); //  d_c;;/// min(d_c, -d_p); // min(d_v, -d_p);// min(min(-d_v, d_c), -d_p);
                //generatedField_1.field[i][j] = min(generatedField_1.field[i][j], -d_p);
            }
        }

        generatedField.rescaleFieldToRange(-1, 1);
       // generatedField_1.rescaleFieldToRange(-1, 1);


    }

    void drawLossText(float startY = 150)
    {
        setup2d();
        
        glColor3f(0, 0, 0);
        char s[100];

        float lossSum = 0;
        float loss_A_Sum = 0;
        float loss_B_Sum = 0;

        for (int i = 0; i < losses_ang.size(); i++)
        {
            lossSum += losses[i];
            loss_A_Sum += losses_ang[i];  
            loss_B_Sum += losses_sun[i];
        }


        sprintf(s, " loss %1.2f", lossSum / trainingSamples.size());
        drawText(string(s), 50, startY);

        sprintf(s, " loss_ang %1.2f", loss_A_Sum);
        drawText(string(s), 50, startY + 15);

        sprintf(s, " loss_ang %1.2f", loss_B_Sum);
        drawText(string(s), 50, startY + 30);

        restore3d();
    }
    
    void drawLossBarGraph( std::vector<float>& losses, float startPtX, float startPtY, float screenWidth = 800, float barHeight = 50)
    {
        if (losses.empty()) return;

        setup2d(); // Switch to orthographic 2D mode

            int N = losses.size();
            float barSpacing = screenWidth / (float)N;

            // --- Normalize losses to [0, 1]
            float minVal = 1e6f, maxVal = -1e6f;
            for (float v : losses)
            {
                minVal = std::min(minVal, v);
                maxVal = std::max(maxVal, v);
            }
            float range = std::max(maxVal - minVal, 1e-6f);  // avoid divide by zero

            float loss_A_Sum = 0;
            float lossSum = 0;

            for (int i = 0; i < N; i++)
            {
                float normalized = (losses[i] - minVal) / range;
                float x = startPtX + i * barSpacing;
                float h = barHeight * normalized;

                float r, g, b;
                getJetColor(normalized, r, g, b);

                glColor3f(r, g, b);
                drawLine(Alice::vec(x, startPtY, 0), Alice::vec(x, startPtY + h, 0));

                lossSum += losses[i];
                loss_A_Sum += losses_ang[i];  // unnormalized angular loss sum
            }
        
            


       

        restore3d(); // Restore to 3D mode
    }

    void drawText(string &str , float x = 50, float y=100)
    {
        unsigned int i;
        glRasterPos2f(x, y);

        
        for (i = 0; i < str.length(); i++)
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, str[i]);
    }

    void visualiseField( float threshold = 0.01 , bool drawField = true)
    {
        if(drawField)generatedField.drawFieldPoints();
        generatedField.drawIsocontours(threshold);

       // generatedField_1.drawIsocontours(threshold);
      
    }

    void visualiseGradients( vector<float> &x , bool showBldg = true , bool showLoss = true)
    {
       
       
        auto out = forward(x);

        std::vector<zVector> centers(number_sdf);
        std::vector<float> radii(number_sdf);
        std::vector<float> heights(number_sdf);

        decodeOutput(out, centers, radii, heights);

        for (int i = 0; i < number_sdf; i++)
        {
            (heights[i] < 1.f) ? glColor3f(0.0, 0.0, 0.0) : glColor3f(0.5, 0.0, 1.0); // purple towers
            drawLine(zVecToAliceVec(centers[i]), zVecToAliceVec(centers[i] + zVector(0, 0, heights[i] + 25)));
          
        }

      



        for (int i = 0; i < number_sdf; i++)
        {
            if(showBldg)
            {
                glPushMatrix();


                glTranslatef(centers[i].x, centers[i].y, centers[i].z);
                glRotatef(radii[i] * (180.0f / PI), 0, 0, 1);

                glColor3f(0, 0, 0);
                //wireFrameOn();
                drawCube(Alice::vec(-4, -2.0, 0), Alice::vec(4, 2.0, heights[i] + 10), Alice::vec(0, 0, 0), true);;
                drawCube(Alice::vec(-4, -2.0, 0), Alice::vec(4, 2.0, heights[i] + 10), Alice::vec(0, 0, 0), false);;
                //wireFrameOff();



                glPopMatrix();
            }


            //
            zVector grad_polygon = gradientAt_polygonSDF(centers[i], polygon);
            grad_polygon.normalize();

            Alice::vec a = zVecToAliceVec(centers[i]);

            glColor3f(0, 0, 0);
          //  drawLine(a, a + zVecToAliceVec(grad_polygon)*3);

            ///

            float cosA = cos(radii[i]);
            float sinA = sin(radii[i]);

             zVector axisX(cosA, -sinA, 0); // local X direction
             zVector axisY(sinA, cosA, 0); // local Y direction

            zVector grad = axisY;// gradientAt(centers[i], centers, radii);
            grad.normalize();

            glColor3f(1, 0, 0);
           // drawLine(a, a + zVecToAliceVec(grad) * 4);
          //  drawLine(a, a + zVecToAliceVec(sunDir) * 4 );

            
        }

        float minVal = 1e6f, maxVal = -1e6f;
        for (float v : losses)
        {
            minVal = std::min(minVal, v);
            maxVal = std::max(maxVal, v);
        }

        for (float &v : losses)
        {
            v = ofMap(v, minVal, maxVal, 0, 1);
        }

        float rr, gg, bb;
        for (int i = 0; i < trainingSamples.size(); i++)
        {
            glColor3f(0, 1, 0);
            Alice::vec a, b;
            if(showLoss)
            {
                a = zVecToAliceVec(trainingSamples[i]);
                b = zVecToAliceVec(zVector(0, 0, losses[i]));

                getJetColor(losses[i], rr, gg, bb);
                glColor3f(rr, gg, bb);
               // drawLine(a, a + b * 15);
            }

           /* zVector grad_polygon = gradientAt_polygonSDF(trainingSamples[i], polygon);
            grad_polygon.normalize();

            glColor3f(0, 0, 0);
            drawLine(a, a + zVecToAliceVec(grad_polygon));

            zVector grad = gradientAt(trainingSamples[i], centers,radii);
            grad.normalize();

            glColor3f(1, 0, 0);
            drawLine(a, a + zVecToAliceVec(grad));*/

        }
    }

};




//------------------------------------------------------------------ MVC test for subClassMLP
std::vector<zVector> polygon;
std::vector<zVector> trainingSamples;
std::vector<float> sdfGT;


#define NUM_CENTERS 9
PolygonSDF_MLP mlp;
std::vector<float> grads;
std::vector<float> mlp_input_data;

double lr = 0.1;
double tv = -0.005;
bool drawField = false;

bool stringSwitch = true;
string path = "data/polygonx.txt";
// function
void initializeMLP()
{
    int input_dim = NUM_CENTERS * 5;
    int output_dim = NUM_CENTERS * 5;
    std::vector<int> hidden_dims = { 16 };

    mlp = PolygonSDF_MLP(input_dim, hidden_dims, output_dim); // assumes MLP ructor initializes weights/biases
    mlp_input_data.assign(input_dim, 1.0f); // or use 0.0f for strict zero-input
    mlp.number_sdf = NUM_CENTERS;


}


bool showBldgs = false;
bool showLoss = true;



void setup()
{


    initializeMLP();  // create MLP

    // load boudnary polygon from a text file;
    loadPolygonFromCSV(path, polygon);

    //calculate training set
    samplePoints(trainingSamples, sdfGT, polygon);
    mlp.polygon = polygon;
    mlp.trainingSamples = trainingSamples;
    mlp.sdfGT = sdfGT;
    mlp.losses.resize(sdfGT.size());
    mlp.losses_ang.resize(sdfGT.size());

    //lr = 0.1;
    //S.numSliders = 0;
    //S.addSlider(&lr, "LR");
    //S.sliders[0].minVal = lr;
    //S.sliders[0].maxVal = lr*3;

    //
    S.numSliders = 0;
    S.addSlider(&tv, "TV");
    S.sliders[1].minVal = -1;
    S.sliders[1].maxVal = 1;

    //
    B.numButtons = 0;
    B.min = Alice::vec(50, 50, 0);
    B.numButtons = 0;
    B.addButton(&drawField, "field");
    B.addButton(&showBldgs, "bldgs");
    B.addButton(&showLoss, "loss");


    //

    for (int i = 0; i < mlp.generatedField.RES; i++)
    {
        for (int j = 0; j < mlp.generatedField.RES; j++)
        {

            zVector pt = mlp.generatedField.gridPoints[i][j];
            mlp.generatedField.field[i][j] = polygonSDF(pt, polygon);
        }
    }

    mlp.generatedField.rescaleFieldToRange(-1, 1);



}

bool run = false;
void update(int value)
{
    if (run)
    {
        keyPress('t', 0, 0);

        lr *= 0.99;
    }
}

void draw()
{
    backGround(0.95);
    //drawGrid(50);
    

    std::vector<float> dummy;
    float loss = mlp.computeLoss(mlp_input_data, dummy);

    //zVector pt(0,3,0);
    //float hit_t;
    //zVector hitPoint;

   
    //mlp.rayIntersectsFootprint(pt, zVector(1,0,0), zVector(13, 0, 0), 0, 8, 6, hit_t, hitPoint);
    //drawLine(zVecToAliceVec(pt), zVecToAliceVec(hitPoint));
    //cout << hit_t << endl;

    glColor3f(1, 0, 0);
    for (auto& c : mlp.fittedCenters)
    {
        drawCircle(zVecToAliceVec(c), 3, 32);
    }

    glColor3f(0, 0, 1);
    for (auto& p : trainingSamples)
    {
        drawPoint(zVecToAliceVec(p));
    }


    // --------------- polygon
    glColor3f(0, 0, 0);
    for (int i = 0; i < polygon.size(); i++)
    {
        int j = (i + 1) % polygon.size();
        drawLine(zVecToAliceVec(polygon[i]), zVecToAliceVec(polygon[j]));
    }


    /// MLP Viz
    mlp.visualize(zVector(50, 350, 0), 200, 250);
    mlp.drawLossBarGraph(mlp.losses, 50, 550, 200, 40);  // bottom-left start, 800px width, 40px bar height
    mlp.drawLossBarGraph(mlp.losses_ang, 50, 625, 200, 40);
    mlp.drawLossBarGraph(mlp.losses_sun, 50, 700, 200, 40);
    mlp.drawLossText(775);
  

    mlp.visualiseField(tv, drawField);
   /*if(run)*/ mlp.visualiseGradients(mlp_input_data, showBldgs, showLoss);

    

}

void keyPress(unsigned char k, int xm, int ym)
{
    if (k == 's') 
    {
        path = stringSwitch ? "data/polygonx.txt" : "data/polygon__.txt";
        cout << path << endl;
        stringSwitch = !stringSwitch;
    }
   
    if (k == '0') mlp.lossOnBools[0] = !mlp.lossOnBools[0];
    if (k == '1') mlp.lossOnBools[1] = !mlp.lossOnBools[1];
    if (k == '2') mlp.lossOnBools[2] = !mlp.lossOnBools[2];

    
    if (k == 't')
    {
        grads.clear();
        std::vector<float> dummy;
        float loss = mlp.computeLoss(mlp_input_data, dummy);
        mlp.computeGradient(mlp_input_data, dummy, grads);

        mlp.backward(grads, lr);

        cout << "loss :" << loss << endl;

        keyPress('u', 0, 0);
        
    }



    if (k == 'r')run = !run;

    if (k == 'u')
    {
        mlp.GenerateField(mlp_input_data);
    }

}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_
