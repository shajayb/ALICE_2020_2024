#define _MAIN_
#ifdef _MAIN_

#include "main.h"

// zSpace Core
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>
using namespace zSpace;

#include <fstream>
#include <sstream>

Alice::vec zVecToAliceVec(zVector& in)
{
    return Alice::vec(in.x, in.y, in.z);
}

zVector AliceVecToZvec(Alice::vec& in)
{
    return zVector(in.x, in.y, in.z);
}

inline zVector zMax(zVector& a, zVector& b)
{
    return zVector(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

inline zVector zMin(zVector& a, zVector& b)
{
    return zVector(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

#include "scalarField.h" //// two functiosn must be turned on in scalarfIELD.H for sketch_circleSDF_fitter.cpp


//inline float smin(float a, float b, float k)
//{
//    float h = std::max(k - fabs(a - b), 0.0f) / k;
//    return std::min(a, b) - h * h * k * 0.25f;
//}
std::vector<zVector> polygon;
std::vector<zVector> sdfCenters;
std::vector<float>predictedRadii;


ScalarField2D myField;

int numCircles = 16;
double thresholdValue = 0.0;
double radius = 8.0;
double smoothK = 3.0;

// ----------------- MLP 
std::vector<zVector> samplePts;
std::vector<float> sdfGT;
std::vector<zVector> fittedCenters;
std::vector<float> fittedRadii;
#define NUM_SDF 16
//std::vector<zVector> polygon;
double threshold;


//-------------------------------
// Utility
//-------------------------------
void loadPolygonFromCSV(const std::string& filename)
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

//-------------------------------
// Circle SDF Blending
//-------------------------------
inline float circleSDF( zVector& pt, zVector& center, float r)
{
    return pt.distanceTo(zVector(center)) - r; // signed: negative inside, 0 on boundary, positive outside
}

inline float blendCircleSDFs( zVector& pt,  std::vector<zVector>& centers, float r, float k)
{
    if (centers.empty()) return 1e6f;

    float d = circleSDF(pt, centers[0], r);
    for (int i = 1; i < centers.size(); i++)
    {
        float d_i = circleSDF(pt, centers[i], r);
        d = std::min(d,d_i);// smin(d, d_i, k);  // smooth union of signed distances
    }

    return d;
}

inline float blendCircleSDFs(zVector& pt, std::vector<zVector>& centers, vector<float> &radii, float k)
{
    if (centers.empty()) return 1e6f;

    float d = circleSDF(pt, centers[0], radii[0]);
    for (int i = 1; i < centers.size(); i++)
    {
        float d_i = circleSDF(pt, centers[i], radii[i]);
        d = std::min(d, d_i);// smin(d, d_i, k);  // smooth union of signed distances
    }

    return d;
}




bool isInsidePolygon( zVector& p,  std::vector<zVector>& poly)
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


float polygonSDF( zVector& pt,  std::vector<zVector>& poly)
{
    float minDist = 1e6f;
    for (int i = 0; i < poly.size(); i++)
    {
        zVector a = poly[i];
        zVector b = poly[(i + 1) % poly.size()];
        zVector ab = b - a;
        zVector ap = pt - a;

        float t = std::clamp((ap * ab) / (ab * ab), 0.0f, 1.0f);
        zVector closest = a + ab * t;
        float dist = pt.distanceTo(closest);
        minDist = std::min(minDist, dist);
    }

    bool inside = isInsidePolygon(pt, poly);
    return inside ? -minDist : minDist;
}


//-------------------------------
// Fit & Field
//-------------------------------

std::vector<zVector> candidatePts;
void fitSDFToPolygon()
{
    sdfCenters.clear();
    candidatePts.clear();
    

    // Sample a grid of points across bounding box
    for (float x = -50; x <= 50; x += 2.0f)
    {
        for (float y = -50; y <= 50; y += 2.0f)
        {
            zVector pt(x, y, 0);
            if (isInsidePolygon(pt, polygon))
            {
                candidatePts.push_back(pt);
            }
        }
    }

    cout << candidatePts.size() << " candidatePts size" << endl;

    for (int c = 0; c < numCircles; c++)
    {
        float maxResidual = -1e6;
        zVector bestPt;


     
        for (auto& pt : candidatePts)
        {
            float pred = blendCircleSDFs(pt, sdfCenters, radius, smoothK);
            float actual = polygonSDF(pt, polygon);

            float residual = fabs(pred - actual);

            if (residual > maxResidual)
            {
                maxResidual = residual;
                bestPt = pt;
            }
        }


       // sdfCenters.push_back(bestPt);
        sdfCenters.push_back( zVector(0,0,0) );
    }

    printf(" NSDFC %i, NC %i \n", sdfCenters.size(), numCircles);
}

void buildScalarField()
{
    for (int i = 0; i < ScalarField2D::RES; i++)
    {
        for (int j = 0; j < ScalarField2D::RES; j++)
        {
            zVector pt = myField.gridPoints[i][j];
            float d = polygonSDF(pt, polygon); ; // blendCircleSDFs(pt, sdfCenters, radius, smoothK);// polygonSDF(pt,polygon);
            myField.field[i][j] = d; // signed SDF directly
        }
    }

    myField.rescaleFieldToRange(-1, 1);
}

std::vector<zVector> trainingSamples;

void samplePoints()
{
    trainingSamples.clear();
    samplePts.clear(); // mlp
    for (float x = -50; x <= 50; x += 5.0f)
    {
        for (float y = -50; y <= 50; y += 5.0f)
        {
            zVector pt(x, y, 0);
            if (isInsidePolygon(pt, polygon))
            {
                trainingSamples.push_back(pt);
                samplePts.push_back(pt); //mlp
                sdfGT.push_back(polygonSDF(pt, polygon));//mlp
            }
        }
    }

    cout << " training samples " << trainingSamples.size() << endl;
}

float computeTotalError()
{
    float err = 0.0f;
    for (auto& pt : trainingSamples)
    {
        float pred = blendCircleSDFs(pt, sdfCenters, radius, smoothK);
        float actual = polygonSDF(pt, polygon);
        float diff = pred - actual;
        err += diff * diff;
    }
    return err / trainingSamples.size();
}

void optimiseCircleCenters(int iterations = 20, float step = 0.001f)
{
    const float eps = 1e-3f;

   //for (int it = 0; it < iterations; it++)
    {
       for (int c = 0; c < sdfCenters.size(); c++)
       {
           zVector center = sdfCenters[c];
           zVector grad(0, 0, 0);

           for (int d = 0; d < 2; d++)
           {
               zVector dir(0, 0, 0);
               if (d == 0) dir.x = eps;
               if (d == 1) dir.y = eps;

               std::vector<zVector> testCenters = sdfCenters;
               testCenters[c] = center + dir;
               float E_plus = 0;
               int i = 0;
               for (auto& pt : trainingSamples)
               {
                   float pred = blendCircleSDFs(pt, testCenters, radius, smoothK);
                   float actual = sdfGT[i++]; //polygonSDF(pt, polygon);
                   
                   float diff = pred - actual;
                   E_plus += diff * diff;
               }

               testCenters[c] = center - dir;
               float E_minus = 0;
               i = 0;
               for (auto& pt : trainingSamples)
               {
                   float pred = blendCircleSDFs(pt, testCenters, radius, smoothK);
                   float actual = sdfGT[i++];// polygonSDF(pt, polygon);
                   float diff = pred - actual;
                   E_minus += diff * diff;
               }

               float g = (E_plus - E_minus) / (2 * eps);
               if (d == 0) grad.x = g;
               if (d == 1) grad.y = g;
           }

           // Update
          
           sdfCenters[c] = center - grad * step;
           sdfCenters[c].x = std::clamp(sdfCenters[c].x, -50.0f, 50.0f);
          sdfCenters[c].y = std::clamp(sdfCenters[c].y, -50.0f, 50.0f);

       }


        std::cout << "Iteration " << " error: " << computeTotalError() << std::endl;
    }
}



class MLP
{
public:
    int inputDim, hiddenDim, outputDim;
    std::vector<std::vector<float>> W1, W2;
    std::vector<float> b1, b2;
    std::vector<float> input, hidden, output;
    float prevLoss = 0 ;

    MLP()
    {

    }
    MLP(int inDim, int hDim, int outDim)
        : inputDim(inDim), hiddenDim(hDim), outputDim(outDim)
    {
        W1.resize(hiddenDim, std::vector<float>(inputDim));
        W2.resize(outputDim, std::vector<float>(hiddenDim));
        b1.resize(hiddenDim);
        b2.resize(outputDim);
        input.resize(inputDim);
        hidden.resize(hiddenDim);
        output.resize(outputDim);

        std::default_random_engine eng;
        std::normal_distribution<float> dist(0.0, 0.1);

        for (auto& row : W1) for (auto& val : row) val = dist(eng);
        for (auto& row : W2) for (auto& val : row) val = dist(eng);
    }

    std::vector<float> forward(const std::vector<float>& x)
    {
        input = x;
        for (int i = 0; i < hiddenDim; ++i)
        {
            hidden[i] = b1[i];
            for (int j = 0; j < inputDim; ++j)
                hidden[i] += W1[i][j] * input[j];
            hidden[i] = std::tanh(hidden[i]);
        }
        for (int i = 0; i < outputDim; ++i)
        {
            output[i] = b2[i];
            for (int j = 0; j < hiddenDim; ++j)
                output[i] += W2[i][j] * hidden[j];
        }
        return output;
    }

    //float computeLossAndGradient( std::vector<float>& x, std::vector<zVector>& polygon, std::vector<float>& gradOut)
    //{
    //    std::vector<zVector> centers(outputDim / 2);
    //    std::vector<float> radii(outputDim / 2);

    //   
    //    auto out = forward(x);
    //    for (int i = 0; i < centers.size(); i++)
    //    {
    //        zVector pt(out[i * 2 + 0], out[i * 2 + 1], 0);
    //        centers[i] = (isInsidePolygon(pt, polygon)) ? pt : fittedCenters[i];
    //        radii[i] = radius; // std::clamp(std::abs(out[i * 3 + 2]), 8.f, 8.f);
    //    }

    //    // Save for visualization
    //    fittedCenters = centers;
    //    fittedRadii = radii;

    //    float loss = 0;
    //    gradOut.assign(outputDim, 0.0f);

    //    for (int j = 0; j < outputDim; ++j)
    //    {
    //        float eps = 0.1f;
    //        std::vector<float> perturbedInput = x;
    //        perturbedInput[j] += eps;

    //        auto perturbedOut = forward(perturbedInput);
    //        std::vector<zVector> cPert(centers.size());
    //        std::vector<float> rPert(centers.size());

    //        for (int i = 0; i < centers.size(); i++)
    //        {
    //            zVector pt(perturbedOut[i * 2 + 0], perturbedOut[i * 2 + 1], 0);

    //            cPert[i] = (isInsidePolygon(pt, polygon)) ? pt : fittedCenters[i];

    //            rPert[i] = radius;//  std::clamp(std::abs(perturbedOut[i * 3 + 2]), 8.0f, 8.f);
    //        }

    //        float gradLoss = 0;
    //        /*for (int s = 0; s < samplePts.size(); ++s)
    //        {
    //            float f = blendCircleSDFs(samplePts[s], centers, radii);
    //            float fPert = blendCircleSDFs(samplePts[s], cPert, rPert);
    //            float gt = sdfGT[s];
    //            float err = f - gt;
    //            loss += err * err;
    //            gradLoss += 2 * err * (fPert - f) / eps;
    //        }*/

    //        // Compute original loss
    //        float baseError = 0.0f;
    //        for (int s = 0; s < samplePts.size(); ++s)
    //        {
    //            float f = blendCircleSDFs(samplePts[s], centers, radii,smoothK);
    //            float gt = sdfGT[s];
    //            float err = f - gt;
    //            baseError += err * err;
    //        }
    //        loss += baseError;

    //        // Compute perturbed loss (x_j + eps)
    //        float errorPertPlus = 0.0f;
    //        for (int s = 0; s < samplePts.size(); ++s)
    //        {
    //            float fPert = blendCircleSDFs(samplePts[s], cPert, rPert, smoothK);
    //            float gt = sdfGT[s];
    //            float errPert = fPert - gt;
    //            errorPertPlus += errPert * errPert;
    //        }

    //        // Compute perturbed loss (x_j - eps)
    //        std::vector<float> perturbedInputMinus = x;
    //        perturbedInputMinus[j] -= eps;
    //        auto perturbedOutMinus = forward(perturbedInputMinus);
    //        std::vector<zVector> cPertMinus(centers.size());
    //        std::vector<float> rPertMinus(centers.size());
    //        for (int i = 0; i < centers.size(); i++)
    //        {
    //            cPertMinus[i] = zVector(perturbedOutMinus[i * 2 + 0], perturbedOutMinus[i * 2 + 1], 0);
    //            cPertMinus[i] = (isInsidePolygon(cPertMinus[i], polygon)) ? cPertMinus[i] : fittedCenters[i];
    //            rPertMinus[i] = radius;// std::abs(perturbedOutMinus[i * 3 + 2]);
    //        }
    //        float errorPertMinus = 0.0f;
    //        for (int s = 0; s < samplePts.size(); ++s)
    //        {
    //            float fPert = blendCircleSDFs(samplePts[s], cPertMinus, rPertMinus,smoothK);
    //            float gt = sdfGT[s];
    //            float errPert = fPert - gt;
    //            errorPertMinus += errPert * errPert;
    //        }

    //        // Central difference gradient
    //        gradLoss = (errorPertPlus - errorPertMinus) / (2.0f * eps);

    //        
    //        gradOut[j] = gradLoss;//  (loss > prevLoss) ? gradLoss * -1 : gradLoss;
    //        prevLoss = std::min(loss, prevLoss);
    //    }

    //    sdfCenters.clear();
    //    sdfCenters = fittedCenters;
    //    
    //    return computeTotalError(); //loss;
    //}

    // Inside MLP class
    float computeLossAndGradient(const std::vector<float>& x, std::vector<zVector>& polygon, std::vector<float>& gradOut)
    {
        // Make sure gradOut is correctly sized
        gradOut.assign(outputDim, 0.0f);

        // 1. Forward pass to get MLP's predicted parameters
        auto raw_mlp_output = forward(x); // This is the output before clamping or other transformations

        std::vector<zVector> centers(NUM_SDF);
        std::vector<float> radii(NUM_SDF);

        // Interpret MLP output as circle parameters.
        // Apply any necessary transformations (e.g., clamping for robustness, but ideally learned)
        // For now, let's keep the radius fixed as in your problem statement
        for (int i = 0; i < NUM_SDF; i++)
        {
            // For x, y, consider if they need to be clamped or scaled.
            // For now, using raw output for gradient calculation, but apply insidePolygon check for the actual usage.
            zVector pt(raw_mlp_output[i * 2 + 0], raw_mlp_output[i * 2 + 1], 0);

            // This check is for visualization/usage, the gradient should ideally flow through if possible.
            // For now, we'll keep it as is, but it can create discontinuous gradients.
            centers[i] = (isInsidePolygon(pt, polygon)) ? pt : fittedCenters[i]; // fittedCenters should ideally be initialized better
            radii[i] = radius; // Fixed radius as per your current setup. If MLP should learn, remove this.
        }

        // Save for visualization
        fittedCenters = centers;
        fittedRadii = radii;

        float totalLoss = 0.0f;

        // 2. Calculate the loss based on the predicted circle parameters
        // This part iterates through sample points and calculates the SDF loss.
        // We also need to compute the gradient of this loss with respect to each circle parameter (center.x, center.y, radius).

        // Numerical gradient of the *SDF blending function* with respect to circle parameters
        // This is the crucial part for the MLP's backprop.
        float eps_param = 0.01f; // Epsilon for numerical gradient of SDF w.r.t. circle params

        for (int s = 0; s < samplePts.size(); ++s)
        {
            zVector current_sample_pt = samplePts[s];
            float gt_sdf = sdfGT[s];

            // Calculate the predicted blended SDF with current parameters
            float predicted_sdf = blendCircleSDFs(current_sample_pt, centers, radii, smoothK);
            float error = predicted_sdf - gt_sdf;
            totalLoss += error * error; // Sum of squared errors

            // Now, compute the gradient of `error*error` (or `predicted_sdf`) with respect to each
            // of the *output parameters* (x, y for centers, and radius) that come from the MLP.

            for (int i = 0; i < NUM_SDF; ++i) // For each circle
            {
                // Gradient with respect to center.x of circle i
                std::vector<zVector> centers_plus_x = centers;
                centers_plus_x[i].x += eps_param;
                float predicted_sdf_plus_x = blendCircleSDFs(current_sample_pt, centers_plus_x, radii, smoothK);
                float grad_sdf_cx = (predicted_sdf_plus_x - predicted_sdf) / eps_param;
                // Chain rule: dLoss/d_param = dLoss/d_predicted_sdf * d_predicted_sdf/d_param
                // dLoss/d_predicted_sdf = 2 * (predicted_sdf - gt_sdf)
                gradOut[i * 2 + 0] += 2 * error * grad_sdf_cx; // Accumulate gradient for output x

                // Gradient with respect to center.y of circle i
                std::vector<zVector> centers_plus_y = centers;
                centers_plus_y[i].y += eps_param;
                float predicted_sdf_plus_y = blendCircleSDFs(current_sample_pt, centers_plus_y, radii, smoothK);
                float grad_sdf_cy = (predicted_sdf_plus_y - predicted_sdf) / eps_param;
                gradOut[i * 2 + 1] += 2 * error * grad_sdf_cy; // Accumulate gradient for output y

                // Gradient with respect to radius of circle i (if it were learned by MLP)
                // Currently, radius is fixed. If MLP were to learn radius, uncomment and adapt:
                /*
                std::vector<float> radii_plus_r = radii;
                radii_plus_r[i] += eps_param;
                float predicted_sdf_plus_r = blendCircleSDFs(current_sample_pt, centers, radii_plus_r, smoothK);
                float grad_sdf_r = (predicted_sdf_plus_r - predicted_sdf) / eps_param;
                gradOut[i * 3 + 2] += 2 * error * grad_sdf_r; // Accumulate gradient for output radius
                */
            }
        }

        // Normalize gradients by sample count for average gradient
        for (float& g : gradOut) 
        {
            g /= samplePts.size();

        }

        return totalLoss / samplePts.size(); // Return average loss
    }

    /*void backward(const std::vector<float>& gradOut, float lr)
    {
        std::vector<float> gradHidden(hiddenDim);

        for (int i = 0; i < hiddenDim; ++i)
        {
            gradHidden[i] = 0;
            for (int j = 0; j < outputDim; ++j)
            {
                gradHidden[i] += gradOut[j] * W2[j][i];
                W2[j][i] -= lr * gradOut[j] * hidden[i];
            }
        }

        for (int j = 0; j < outputDim; ++j)
        {
            b2[j] -= lr * gradOut[j];
        }

    }*/
    // Inside MLP class
    void backward(const std::vector<float>& gradOut, float lr)
    {
        // gradOut: dLoss/d_output_raw_mlp_output (e.g., dLoss/d(raw_x), dLoss/d(raw_y), dLoss/d(raw_radius))

        // Gradients for output layer weights (W2) and biases (b2)
        // dLoss/dW2_ji = dLoss/d_output_j * d_output_j/dW2_ji = dLoss/d_output_j * hidden_i
        // dLoss/db2_j = dLoss/d_output_j * d_output_j/db2_j = dLoss/d_output_j * 1

        std::vector<float> gradHidden_raw(hiddenDim); // Gradient before tanh activation

        for (int j = 0; j < outputDim; ++j) // Iterate through output neurons
        {
            // Update W2 (output_dim x hidden_dim)
            for (int i = 0; i < hiddenDim; ++i) // Iterate through hidden neurons
            {
                W2[j][i] -= lr * gradOut[j] * hidden[i];
            }

            // Update b2 (output_dim)
            b2[j] -= lr * gradOut[j];

            // Accumulate gradient for hidden layer (to backpropagate further)
            // This is dLoss/d_raw_hidden_i contribution from this output neuron j
            for (int i = 0; i < hiddenDim; ++i)
            {
                gradHidden_raw[i] += gradOut[j] * W2[j][i]; // Chain rule: dLoss/d_hidden_raw_i = sum(dLoss/d_output_j * d_output_j/d_hidden_raw_i)
                // d_output_j/d_hidden_raw_i = W2_ji * d_hidden_j/d_raw_hidden_j (tanh derivative)
                // NO, this is dLoss/d_output_j * W2_ji. The tanh derivative comes next.
            }
        }

        // Now, backpropagate through the tanh activation for the hidden layer
        // dLoss/d_hidden_i = dLoss/d_raw_hidden_i * d(tanh(raw_hidden_i))/d_raw_hidden_i
        // d(tanh(x))/dx = 1 - tanh(x)^2 = 1 - hidden[i]^2
        std::vector<float> gradHidden_activated(hiddenDim);
        for (int i = 0; i < hiddenDim; ++i)
        {
            gradHidden_activated[i] = gradHidden_raw[i] * (1.0f - hidden[i] * hidden[i]);
        }


        // Gradients for hidden layer weights (W1) and biases (b1)
        // dLoss/dW1_ij = dLoss/d_hidden_i * input_j
        // dLoss/db1_i = dLoss/d_hidden_i * 1
        for (int i = 0; i < hiddenDim; ++i) // Iterate through hidden neurons
        {
            // Update W1 (hidden_dim x input_dim)
            for (int j = 0; j < inputDim; ++j) // Iterate through input neurons
            {
                W1[i][j] -= lr * gradHidden_activated[i] * input[j];
            }

            // Update b1 (hidden_dim)
            b1[i] -= lr * gradHidden_activated[i];
        }
    }
};


void computeSampleData()
{
    /*samplePts.clear();
    sdfGT.clear();
    for (int i = -50; i <= 50; i += 5)
    {
        for (int j = -50; j <= 50; j += 5)
        {

            zVector pt(i, j, 0);

            if (!isInsidePolygon(pt, polygon))continue;

            samplePts.push_back(pt);
            sdfGT.push_back(polygonSDF(pt, polygon));
        }
    }*/

}

bool train = false;
std::vector<float> input(numCircles * 2, 0.0f);
std::vector<float> gradOut;
MLP mlp;
ScalarField2D sf;



//-------------------------------
// Visualisation
//-------------------------------
void drawPolygon()
{
    glColor3f(0, 0, 0);
    for (int i = 0; i < polygon.size(); i++)
    {
        int j = (i + 1) % polygon.size();
        drawLine(zVecToAliceVec(polygon[i]), zVecToAliceVec(polygon[j]));
    }
}

void drawCircles()
{
    glColor3f(0, 0, 1);
    for (auto& c : sdfCenters)
    {
        drawCircle(zVecToAliceVec(c), radius * 0.5, 32);
    }
}

//-------------------------------
// MVC
//-------------------------------
void setup()
{
    loadPolygonFromCSV("data/polygon.txt");
    fitSDFToPolygon();       // initial greedy placement
    samplePoints();          // prepare training set
    optimiseCircleCenters(); // run gradient descent
    buildScalarField();      // update field


    S.addSlider(&thresholdValue, "iso");
    S.sliders[0].maxVal = 1;
    S.sliders[0].minVal = -1.0;

    S.addSlider(&radius, "r");
    S.sliders[1].maxVal = 20;

    S.addSlider(&smoothK, "k");
    S.sliders[2].maxVal = 10;
}

bool opt = false;
void update(int value)
{
    if(opt)optimiseCircleCenters();
   
    if (train)keyPress('n', 0, 0);
    
    buildScalarField();
}

void draw()
{
    backGround(0.9);
    drawGrid(50);

    drawPolygon();
    drawCircles();

    /*glPointSize(5);
    for (auto& pt : candidatePts)drawPoint(zVecToAliceVec(pt));
    glPointSize(1);*/
    //mlp
    glColor3f(1, 0, 0);
    for (auto& pt : fittedCenters)drawCircle(zVecToAliceVec(pt), radius*0.5, 32);
  
    myField.drawFieldPoints();
    
    myField.drawIsocontours(thresholdValue, true);
   
}

void keyPress(unsigned char k, int xm, int ym)
{
    
  
    if (k == 'r')
    {
        /*loadPolygonFromCSV("data/polygon.csv");
        fitSDFToPolygon();
        buildScalarField();*/

       // optimiseCircleCenters();
        opt = !opt; 
    }
    ////

    if (k == 't')
    {
        mlp = MLP(numCircles * 2, NUM_SDF * 16, 2 * numCircles);
        train = true;
    }

    if (k == 'n')
    {
        /*for (int i = 0; i < input.size() - 1; i += 2)
        {
            int n = (ofRandom(0, samplePts.size()));
            zVector pt(ofRandom(-1, 1), ofRandom(-1, 1), 0);
            input[i] = pt.x;
            input[i + 1] = pt.y;
        }*/

        //for (int epoch = 0; epoch < 250; ++epoch)
        {

            float loss = mlp.computeLossAndGradient(input, polygon, gradOut);
            mlp.backward(gradOut, 0.01);
           // input = mlp.forward(input);
            std::cout << "Epoch " << ", Loss: " << loss << std::endl;
        }
    }
}

void mousePress(int b, int state, int x, int y) {}
void mouseMotion(int x, int y) {}

#endif // _MAIN_
