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



#include "scalarField.h"
#include "parcel.h"

#pragma once

#include "scalarField.h"
#include <fstream>
#include <sstream>
#include <map>
#include "genericMLP.h"


float radius = 5.0f; // or expose as a parameter

class HeightField2D : public ScalarField2D
{
public:

    std::vector<zVector> samples;
    double zMin = 0.0f, zMax = 1.0f;
    double MLS_zMin = 0.0f, MLS_zMax = 1.0f;
    double zScale = 5;
    //
    float scale;
    zVector cSrc;
    zVector cDst;

    void readSamplesAndInterpolate(const std::string& filename)
    {
        samples.clear();
        zMin = 0.0f, zMax = 1.0f;
        zScale = 5;

        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Failed to open " << filename << std::endl;
            return;
        }

        std::string line;
        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string xStr, yStr, zStr;

            if (std::getline(ss, xStr, ',') &&
                std::getline(ss, yStr, ',') &&
                std::getline(ss, zStr))
            {
                float x = std::stof(xStr);
                float y = std::stof(yStr);
                float z = std::stof(zStr);
                samples.emplace_back(x, y, z);
            }
        }

        file.close();

        rescaleSamplesToBoundingBox(zVector(-50, -50, -50), zVector(50, 50, 50));

        //interpolateToGrid();
        interpolateToGrid_MLS();
    }
  
    void rescalePoints( vector<zVector> &pts)
    {
        for (auto& s : pts)
        {
            s.x = cDst.x + (s.x - cSrc.x) * scale;
            s.y = cDst.y + (s.y - cSrc.y) * scale;
            // z remains untouched
        }
    }
    
    void rescaleSamplesToBoundingBox(zVector& targetMin, zVector& targetMax)
    {
        if (samples.empty())
        {
            return;
        }

        // --- 1) Source 2D bbox (x,y only)
        zVector bmin( 1e6, 1e6,0); //  samples[0];
        zVector bmax = bmin * -1;; // samples[0];

        for (const auto& s : samples)
        {
            bmin.x = std::min(bmin.x, s.x);
            bmin.y = std::min(bmin.y, s.y);
            bmax.x = std::max(bmax.x, s.x);
            bmax.y = std::max(bmax.y, s.y);
        }

        zVector src = bmax - bmin;                 // source width/height
        zVector dst = targetMax - targetMin;       // target width/height

        if (src.x < 1e-9f || src.y < 1e-9f)
        {
            printf("rescaleSamplesToBoundingBox: degenerate source bbox.\n");
            return;
        }

        // --- 2) Uniform "contain" scale (never exceeds target on any axis)
        scale = std::min(dst.x / src.x, dst.y / src.y);

        // --- 3) Centered placement: scale about source center, move to target center
        cSrc = (bmin + bmax) * 0.5f;
        cDst = (targetMin + targetMax) * 0.5f;

        for (auto& s : samples)
        {
            s.x = cDst.x + (s.x - cSrc.x) * scale;
            s.y = cDst.y + (s.y - cSrc.y) * scale;
            // z remains untouched
        }

        // --- 4) z-range (diagnostic)
        zMin = 1e6;;// samples[0].z;
        zMax = -zMin;// samples[0].z;
        for (const auto& s : samples)
        {
            zMin = std::min(float(zMin), s.z);
            zMax = std::max(float(zMax), s.z);
        }

        printf("rescaleSamplesToBoundingBox: contain scale=%.6f  src(%.3f,%.3f) -> dst(%.3f,%.3f)  z[%.3f,%.3f]\n",
            scale, src.x, src.y, dst.x, dst.y, zMin, zMax);
    }

    // Add this method inside HeightField2D
// Assumes members: RES, gridMin, gridMax, field[RES][RES], samples {x,y,z}
// Improves fit to dotted iso-samples while remaining smooth between contours.

    void reconstructScreenedPoisson
    (
        double alpha_base = 0.05,
        double sigma_cells = 1.25,   // Gaussian splat radius in grid cells
        double pin_threshold = 0.65,  // fraction of max weight to pin (Dirichlet)
        int max_iters = 1500,
        double omega = 1.88,
        double tol = 1e-4
    )
    {
        if (samples.empty()) return;

        const int nx = RES;
        const int ny = RES;
        zVector gridMin(-50, -50, 0);
        zVector gridMax( 50, 50, 0);

        const double dx = (gridMax.x - gridMin.x) / (nx - 1);
        const double dy = (gridMax.y - gridMin.y) / (ny - 1);

        const double invDx = (nx > 1) ? 1.0 / (gridMax.x - gridMin.x) * (nx - 1) : 0.0;
        const double invDy = (ny > 1) ? 1.0 / (gridMax.y - gridMin.y) * (ny - 1) : 0.0;

        // ----------------------------
        // 1) Gaussian splat of samples -> data (b) and weights (w)
        // ----------------------------
        std::vector<double> b(nx * ny, 0.0);
        std::vector<double> w(nx * ny, 0.0);

        auto I = [&](int i, int j) -> int
            {
                return i + nx * j;
            };

        auto clampi = [&](int v, int lo, int hi) -> int
            {
                return (v < lo) ? lo : (v > hi ? hi : v);
            };

        const double twoSigma2 = 2.0 * sigma_cells * sigma_cells;
        const int r = std::max(1, (int)std::ceil(3.0 * sigma_cells)); // 3σ footprint

        for (auto& s : samples)
        {
            // grid coords (float index space)
            double gx = (s.x - gridMin.x) * invDx;
            double gy = (s.y - gridMin.y) * invDy;

            int ix0 = (int)std::floor(gx);
            int iy0 = (int)std::floor(gy);

            for (int dj = -r; dj <= r; ++dj)
            {
                int jy = iy0 + dj;
                if (jy < 0 || jy >= ny) continue;

                for (int di = -r; di <= r; ++di)
                {
                    int ix = ix0 + di;
                    if (ix < 0 || ix >= nx) continue;

                    double dxg = (gx - (double)ix);
                    double dyg = (gy - (double)jy);

                    // isotropic Gaussian in cell units
                    double ww = std::exp(-(dxg * dxg + dyg * dyg) / twoSigma2);

                    int k = I(ix, jy);
                    b[k] += ww * s.z;
                    w[k] += ww;
                }
            }
        }

        // Normalize b by weights where present, and track max weight
        double wmax = 0.0;
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                int k = I(i, j);
                if (w[k] > 0.0)
                {
                    b[k] /= w[k];
                    if (w[k] > wmax) wmax = w[k];
                }
                else
                {
                    // reasonable initial guess far from data
                    b[k] = field[i][j];
                }
            }
        }
        if (wmax <= 0.0) wmax = 1.0; // safety

        // ----------------------------
        // 2) Build adaptive alpha per cell, and hard-pin mask
        // ----------------------------
        std::vector<double> alpha(nx * ny, alpha_base);
        std::vector<unsigned char> pin(nx * ny, 0);

        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                int k = I(i, j);

                // normalize weight to [0,1]
                double wn = (w[k] > 0.0) ? (w[k] / wmax) : 0.0;

                // data adherence grows near samples, but stays >= alpha_base
                // factor 1 + 4*wn biases to data without overfitting
                alpha[k] = alpha_base * (1.0 + 4.0 * wn);

                // hard-pin if this cell is very strongly supported by nearby samples
                if (wn >= pin_threshold)
                {
                    pin[k] = 1;
                }
            }
        }

        // ----------------------------
        // 3) Initialize solution with current field
        // ----------------------------
        std::vector<double> f(nx * ny, 0.0);
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                f[I(i, j)] = field[i][j];
            }
        }

        // ----------------------------
        // 4) Red–Black SOR with anisotropic Laplacian and Neumann (clamped) edges
        //     Discrete equation:
        //     (fE+fW)/dx^2 + (fN+fS)/dy^2 - (2/dx^2 + 2/dy^2 + α) f = -α b
        //     =>
        //     f = [ (fE+fW)/dx^2 + (fN+fS)/dy^2 + α b ] / [ 2/dx^2 + 2/dy^2 + α ]
        // ----------------------------
        const double invdx2 = 1.0 / (dx * dx);
        const double invdy2 = 1.0 / (dy * dy);

        auto neighborSample = [&](int i, int j) -> double
            {
                i = clampi(i, 0, nx - 1);
                j = clampi(j, 0, ny - 1);
                return f[I(i, j)];
            };

        auto sweep_color = [&](int color)
            {
                for (int j = 0; j < ny; ++j)
                {
                    for (int i = (j & 1) ^ color; i < nx; i += 2)
                    {
                        int k = I(i, j);

                        if (pin[k])
                        {
                            f[k] = b[k];
                            continue;
                        }

                        double fE = neighborSample(i + 1, j);
                        double fW = neighborSample(i - 1, j);
                        double fN = neighborSample(i, j + 1);
                        double fS = neighborSample(i, j - 1);

                        double A = (2.0 * invdx2 + 2.0 * invdy2) + alpha[k];
                        double rhs = (fE + fW) * invdx2 + (fN + fS) * invdy2 + alpha[k] * b[k];

                        double f_new = rhs / A;

                        // SOR relaxation
                        f[k] = (1.0 - omega) * f[k] + omega * f_new;
                    }
                }
            };

        auto residual_norm_inf = [&]() -> double
            {
                double rmax = 0.0;
                for (int j = 0; j < ny; ++j)
                {
                    for (int i = 0; i < nx; ++i)
                    {
                        int k = I(i, j);
                        if (pin[k]) continue;

                        double fC = f[k];
                        double fE = neighborSample(i + 1, j);
                        double fW = neighborSample(i - 1, j);
                        double fN = neighborSample(i, j + 1);
                        double fS = neighborSample(i, j - 1);

                        double L = (fE - 2.0 * fC + fW) * invdx2 + (fN - 2.0 * fC + fS) * invdy2;
                        double r = (L - alpha[k] * (fC - b[k]));

                        rmax = (std::fabs(r) > rmax) ? std::fabs(r) : rmax;
                    }
                }
                return rmax;
            };

        // Iterate
        for (int it = 0; it < max_iters; ++it)
        {
            sweep_color(0); // red
            sweep_color(1); // black

            if ((it & 31) == 31) // every 32 iters, check residual
            {
                double rinf = residual_norm_inf();
                if (rinf < tol) break;
            }
        }

        // ----------------------------
        // 5) Write back to field and (optionally) normalize
        // ----------------------------
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                field[i][j] = f[I(i, j)];
            }
        }

        // keep your existing normalization utilities if you wish
        // normalise();
         rescaleFieldToRange(-1, 1);
    }


    void interpolateToGrid()
    {
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                zVector gp = gridPoints[i][j];
                float num = 0.0f;
                float den = 0.0f;

                for (const auto& s : samples)
                {
                    float d = gp.distanceTo(zVector(s.x, s.y, 0));
                    if (d < 1e-3f) d = 1e-3f;

                    float w = 1.0f / (d * d);
                    num += w * s.z;
                    den += w;
                }

                field[i][j] = (den > 0.0f) ? num / den : 0.0f;
            }
        }

        // normalise doesnt affect field values.. normalised values are stored in a separate array for visualisation purposes
        normalise(); 
        rescaleFieldToRange(-1, 1);
    }

    // Drop-in: smoother, artefact-free interpolation for contours sampled as dots.
// Add inside HeightField2D and call instead of interpolateToGrid().
//
// Params:
//   supportRadiusCells : kernel radius in *grid cells* (use 2.0–4.0)
//   minNeighbors       : require this many points inside support
//   eps                : tiny to avoid singularities
//
    void interpolateToGrid_MLS(double supportRadiusCells = 3.0,
        int    minNeighbors = 6,
        double eps = 1e-12)
    {
        const int nx = RES;
        const int ny = RES;

        // world grid spacing
        zVector gridMax (50, 50, 0);
        zVector gridMin (-50, -50, 0);
        const double dx = (gridMax.x - gridMin.x) / (nx - 1);
        const double dy = (gridMax.y - gridMin.y) / (ny - 1);

        // helper lambdas
        auto wendlandC2 = [](double q) -> double
            {
                // q in [0,1]; C2 smooth, compact support
                if (q >= 1.0) return 0.0;
                double t = 1.0 - q;
                double t2 = t * t;
                return t2 * t2 * (1.0 + 4.0 * q); // (1-q)^4 * (1+4q)
            };

        auto solve3x3 = [&](double A00, double A01, double A02,
            double A10, double A11, double A12,
            double A20, double A21, double A22,
            double b0, double b1, double b2,
            double& x0, double& x1, double& x2) -> bool
            {
                // Gaussian elimination with partial pivoting (tiny 3x3)
                double M[3][4] =
                {
                    { A00, A01, A02, b0 },
                    { A10, A11, A12, b1 },
                    { A20, A21, A22, b2 }
                };

                for (int col = 0; col < 3; ++col)
                {
                    // pivot
                    int piv = col;
                    double amax = std::fabs(M[col][col]);
                    for (int r = col + 1; r < 3; ++r)
                    {
                        double v = std::fabs(M[r][col]);
                        if (v > amax)
                        {
                            amax = v;
                            piv = r;
                        }
                    }
                    if (amax < eps) return false; // singular

                    if (piv != col)
                    {
                        for (int c = col; c < 4; ++c) std::swap(M[piv][c], M[col][c]);
                    }

                    // normalize row
                    double invp = 1.0 / M[col][col];
                    for (int c = col; c < 4; ++c) M[col][c] *= invp;

                    // eliminate in other rows
                    for (int r = 0; r < 3; ++r)
                    {
                        if (r == col) continue;
                        double f = M[r][col];
                        if (std::fabs(f) < eps) continue;
                        for (int c = col; c < 4; ++c)
                        {
                            M[r][c] -= f * M[col][c];
                        }
                    }
                }

                x0 = M[0][3];
                x1 = M[1][3];
                x2 = M[2][3];
                return true;
            };

        // main loop: fit local affine z(x,y) = a0 + a1*(x-xg) + a2*(y-yg)
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                const zVector gp = gridPoints[i][j];
                const double xg = gp.x;
                const double yg = gp.y;

                // build weighted normal equations
                double S_w = 0.0;
                double S_x = 0.0, S_y = 0.0;
                double S_xx = 0.0, S_xy = 0.0, S_yy = 0.0;
                double S_z = 0.0, S_xz = 0.0, S_yz = 0.0;

                int nbh = 0;

                // choose support radius in *world* so kernel is isotropic when dx != dy
                // convert cells -> world by averaging
                const double R = supportRadiusCells * 0.5 * (dx + dy);
                const double invR = (R > eps) ? 1.0 / R : 1.0;

                double idw_num = 0.0;
                double idw_den = 0.0;

                for (const auto& s : samples)
                {
                    const double sx = s.x;
                    const double sy = s.y;
                    const double sz = s.z;

                    const double rx = sx - xg;
                    const double ry = sy - yg;

                    const double r = std::sqrt(rx * rx + ry * ry);
                    const double q = r * invR;

                    // IDW (fallback) accumulators
                    double d = r;
                    if (d < 1e-6) d = 1e-6;
                    double widw = 1.0 / (d * d);
                    idw_num += widw * sz;
                    idw_den += widw;

                    // kernel weight
                    double w = wendlandC2(q);
                    if (w <= 0.0) continue;

                    ++nbh;

                    S_w += w;
                    S_x += w * rx;
                    S_y += w * ry;
                    S_xx += w * rx * rx;
                    S_xy += w * rx * ry;
                    S_yy += w * ry * ry;

                    S_z += w * sz;
                    S_xz += w * rx * sz;
                    S_yz += w * ry * sz;
                }

                // not enough neighbors -> safe IDW fallback
                if (nbh < minNeighbors || S_w < eps)
                {
                    field[i][j] = (idw_den > 0.0) ? (idw_num / idw_den) : 0.0;
                    continue;
                }

                // solve 3x3 normal equations for (a0, a1, a2)
                // [ S_w  S_x  S_y ] [a0] = [ S_z  ]
                // [ S_x  S_xx S_xy ] [a1]   [ S_xz ]
                // [ S_y  S_xy S_yy ] [a2]   [ S_yz ]
                double a0, a1, a2;
                bool ok = solve3x3(
                    S_w, S_x, S_y,
                    S_x, S_xx, S_xy,
                    S_y, S_xy, S_yy,
                    S_z, S_xz, S_yz,
                    a0, a1, a2
                );

                if (!ok)
                {
                    // Robust fallback if local system is ill-conditioned
                    field[i][j] = (idw_den > 0.0) ? (idw_num / idw_den) : 0.0;
                }
                else
                {
                    // evaluate at the grid point (rx=0, ry=0) -> z = a0
                    field[i][j] = a0;
                }
            }
        }

        /// store interpolated min-max
        float mn = 1e6f, mx = -1e6f;
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                if (fabs(field[i][j] - OUT) < 1e-6)continue; //exclude outside

                mn = std::min(mn, field[i][j]);
                mx = std::max(mx, field[i][j]);
            }
        }
        MLS_zMin = mn;
        MLS_zMax = mx;
        printf(" MLS min %.4f, MLS max %.4f, zMin %.4f, zMax %.4f \n", MLS_zMin, MLS_zMax, zMin, zMax);

        // ------------------------- 

        normalise(); // normalise doesnt change field values ;
        rescaleFieldToRange(0, 1);
    }

    void trimFieldWithPolygon( vector<zVector>&poly)
    {
        for (int i = 0; i < RES; i++)
            for (int j = 0; j < RES; j++)
                if ( !pointInsidePolygon(gridPoints[i][j], poly)) field[i][j] = 1;
               
    }

    // --------------------------

    void setGridPointHeights()
    {
        if (samples.empty()) return;

        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                gridPoints[i][j].z = ofMap(field[i][j], -1, 1, -zScale, zScale); ;// ofMap(field_normalized[i][j], 0, 1, -zScale, zScale);
            }
        }
    }


    void drawSamplePoints()
    {
        if (samples.empty()) return;

        glPointSize(1);
        glBegin(GL_POINTS);
        for (const auto& ptRaw : samples)
        {
            zVector pt = ptRaw;
            //pt.z = ofMap(ptRaw.z, zMin, zMax, -1.0f, 1.0f) * zScale; // normalized and scaled

            float color = ofMap(pt.z, zMin, zMax, 0.0f, 1.0f);
            glColor3f(color, 0.0f, 1.0f - color);

            Alice::vec av = zVecToAliceVec(pt);
            glVertex3f(av.x, av.y, av.z);
        }
        glEnd();
        glPointSize(1);
    }

};

//--------------------------------------------------------------------
// Custom HeightFieldNN derived from MLP
//--------------------------------------------------------------------


struct Pose2D
{
    zVector c;   // center (x, y, 0)
    zVector v;   // 2D vector (vx, vy, 0)
};


float signedDistanceToPolygon( zVector& p, const std::vector<zVector>& poly)
{
    float minDist = 1e6f;
    int n = poly.size();

    for (int i = 0; i < n; ++i)
    {
         zVector a = poly[i];
         zVector b = poly[(i + 1) % n];

        // project p onto edge ab
        zVector ab = b - a;
        zVector ap = p - a;

        float t = std::max(0.0f, std::min(1.0f, ap * ab / (ab * ab)));
        zVector proj = a + ab * t;

        float d = p.distanceTo(proj);
        if (d < minDist) minDist = d;
    }

    // Signed: negative if outside
    return pointInsidePolygon(p, poly) ? -minDist : minDist;
}

// Evaluates the SDF of the given polygon
float evalPolygonSDF( zVector& p,  std::vector<zVector>& poly)
{
    return signedDistanceToPolygon(const_cast<zVector&>(p), poly);
}

// Computes blended SDF from all circles defined by pose centers
float evalBlendedCircleSDF( zVector& p, const std::vector<Pose2D>& poses)
{
    float sdf = 1e6f;


    for ( auto& pose : poses)
    {
        zVector temp = pose.c;

        float d = p.distanceTo(temp) - radius;
        sdf = smin(sdf, d, 3); // union of circles
    }



    return sdf;
}

struct sdfSamples
{
    zVector pt;
    float val;
};

class heightfieldNN : public MLP
{
public:
    int n = 4;
    std::vector<zVector> polygon;
    std::vector<float> poseSeeds; // 2 floats per pose: x, y
    std::vector<sdfSamples> sdfSamplePoints;
    zVector sdfSample_centroid;
    
    heightfieldNN() {}

    heightfieldNN(int _n)
    {
        n = _n;
        initialize(2 * n, { 16, 16 }, 4 * n); // dummy input = 1; output = n × (center + dir)
    }

    // ------------------
    void generateSamplesInRange(HeightField2D& htField, float minZ = 18.0f, float maxZ = 20.0f )
    {
       
        // check if htField.rescaleInRange had -ve range prior t rescaling, if so, use {-1,1} below
        minZ = ofMap(minZ, htField.MLS_zMin, htField.MLS_zMax, 0, 1);
        maxZ = ofMap(maxZ, htField.MLS_zMin, htField.MLS_zMax, 0, 1);
        
        sdfSamplePoints.clear();

        sdfSample_centroid = zVector(0, 0, 0);
        for (int i = 0; i < htField.RES; i+=2)
        {
            for (int j = 0; j < htField.RES; j+=2)
            {
                float val = htField.field[i][j];

                // Optional: Only if the Z component stores elevation
                //if (val >= minZ && val <= maxZ)
                {
                    sdfSamples sample;
                    sample.pt = htField.gridPoints[i][j];
                    sample.val = (val >= minZ && val <= maxZ) ? -val : +val;
                    sdfSamplePoints.push_back(sample);
                    //
                    sdfSample_centroid += sample.pt;
                }
            }
        }
        sdfSample_centroid /= sdfSamplePoints.size();
    }
    
    void set_field_values_from_polygon( vector<zVector> &polygon, HeightField2D& htField)
    {

        for (int i = 0; i < htField.RES; ++i)
        {
            for (int j = 0; j < htField.RES; ++j)
            {
                zVector pt = htField.gridPoints[i][j];

                if (pointInsidePolygon(pt, polygon))
                {
                    htField.field[i][j] = evalPolygonSDF(pt, polygon);
                    cout << "yes" << endl;
                }
                
            }
        }

        htField.normalise();
        htField.rescaleFieldToRange(-1, 1);
    }

    void generateSDFSamplePointsFromPolygon()
    {
        sdfSamplePoints.clear();

        if (polygon.empty()) return;

        // Compute bounding box of polygon
        zVector bmin = polygon[0];
        zVector bmax = polygon[0];

        for (auto& p : polygon)
        {
            bmin = zMin(bmin, p);
            bmax = zMax(bmax, p);
        }

        int gridResX = 50;
        int gridResY = 50;

        for (int i = 0; i < gridResX; ++i)
        {
            for (int j = 0; j < gridResY; ++j)
            {
                float u = (float)i / (gridResX - 1);
                float v = (float)j / (gridResY - 1);

                zVector pt;
                pt.x = zLerp(bmin.x, bmax.x, u);
                pt.y = zLerp(bmin.y, bmax.y, v);
                pt.z = 0;

                if (pointInsidePolygon(pt, polygon))
                {
                    sdfSamples sample;
                    sample.pt = pt;
                    sample.val = evalPolygonSDF(pt, polygon);
                    sdfSamplePoints.push_back(sample);
                    sdfSample_centroid += sample.pt;
                }
            }
        }

        sdfSample_centroid /= sdfSamplePoints.size();

        if (!pointInsidePolygon(sdfSample_centroid, polygon))
        {
            float mn = 1e6;
            for (auto& s : sdfSamplePoints)
            {
                if (s.val < mn)
                {
                    mn = s.val;
                    sdfSample_centroid = s.pt;
                }

            }
        }

        
        printf("Sample points generated: %zu\n", sdfSamplePoints.size());
    }

    void translate_SDFPolygon_and_samples_to_origin()
    {
        if (!pointInsidePolygon(sdfSample_centroid, polygon))
        {
            float mn = 1e6;
            for (auto& s : sdfSamplePoints)
            {
                if( s.val < mn)
                {
                    mn = s.val;
                    sdfSample_centroid = s.pt;
                }

            }
        }


        for (auto& s : sdfSamplePoints)
        {
            s.pt -= sdfSample_centroid;
        }

        for (auto& s : polygon)
        {
            s -= sdfSample_centroid;
        }
    }

    void translate_SDFPolygon_and_samples_to_original()
    {
        for (auto& s : sdfSamplePoints)
        {
            s.pt += sdfSample_centroid;
        }

        for (auto& s : polygon)
        {
            s += sdfSample_centroid;
        }
    }
    
    void setInputSeeds(const std::vector<float>& seeds)
    {
        poseSeeds = seeds;
    }

    void setTargetPolygon(const std::vector<zVector>& poly)
    {
        polygon = poly;
    }
    // ------------------

    void computePolygonBBox(const std::vector<zVector>& polygon, zVector& bmin, zVector& bmax)
    {
        if (polygon.empty()) return;

        bmin = zVector(1e6, 1e6, 0);
        bmax = zVector(-1e6, -1e6, 0);

        for (const auto& p : polygon)
        {
            bmin.x = std::min(bmin.x, p.x);
            bmin.y = std::min(bmin.y, p.y);
            bmax.x = std::max(bmax.x, p.x);
            bmax.y = std::max(bmax.y, p.y);
        }
    }

    void extractPoses(std::vector<float>& output, std::vector<Pose2D>& poses , bool rawCenter = false)
    {
        poses.resize(n);
        if (polygon.empty()) return;

        // --- Compute polygon bounding box (target range)
        zVector bmin(1e6, 1e6, 0);
        zVector bmax(-1e6, -1e6, 0);
        for (const auto& p : polygon)
        {
            bmin.x = std::min(bmin.x, p.x);
            bmin.y = std::min(bmin.y, p.y);
            bmax.x = std::max(bmax.x, p.x);
            bmax.y = std::max(bmax.y, p.y);
        }

        /* bmin *= 0.5;
         bmax *= 0.5;*/
         // --- Collect raw centers from network outputs
        std::vector<zVector> rawCenters;
        rawCenters.reserve(n);
        for (int i = 0; i < n; ++i)
        {
            rawCenters.push_back(zVector(output[i * 4 + 0], output[i * 4 + 1], 0));
        }

        // --- Compute input range of raw centers
        zVector inMin(1e6, 1e6, 0);
        zVector inMax(-1e6, -1e6, 0);
        for (const auto& c : rawCenters)
        {
            inMin.x = std::min(inMin.x, c.x);
            inMin.y = std::min(inMin.y, c.y);
            inMax.x = std::max(inMax.x, c.x);
            inMax.y = std::max(inMax.y, c.y);
        }

        // --- Avoid degenerate ranges
        float rangeX = std::max(1e-6f, inMax.x - inMin.x);
        float rangeY = std::max(1e-6f, inMax.y - inMin.y);

        // --- Remap raw centers to polygon bounding box
        for (int i = 0; i < n; ++i)
        {
            const zVector& raw = rawCenters[i];

            float u = (raw.x - inMin.x) / rangeX;
            float v = (raw.y - inMin.y) / rangeY;

            zVector mapped;
            mapped.x = bmin.x + u * (bmax.x - bmin.x);
            mapped.y = bmin.y + v * (bmax.y - bmin.y);
            mapped.z = 0;

            zVector rawDir(output[i * 4 + 2], output[i * 4 + 3], 0);

            poses[i].c = rawCenter ? raw : mapped;
            poses[i].c += sdfSample_centroid;
            poses[i].v = rawDir;
            poses[i].v.normalize();
        }
    }

    /// ----------------
    float computeLoss(std::vector<float>& y_pred, std::vector<float>& y_dummy) override
    {
        return coverageLoss(y_pred);
    }

    float coverageLoss(std::vector<float>& output)
    {
        if (polygon.empty()) return 1e6f;

        std::vector<Pose2D> poses;
        extractPoses(output, poses , true);// for Term 0, raw centers work better.

        //visualiseBlendedSDFs(poses);
        // -- Term 0: SDF field mismatch at fixed sample points : a coverage objective function
        float sdfLoss = 0.0f;
        if (!sdfSamplePoints.empty())
        {
            for (auto& sample : sdfSamplePoints)
            {
                float sdfTarget = sample.val;
                float sdfPred = evalBlendedCircleSDF(sample.pt, poses);

              float diff = sdfTarget - sdfPred;
              sdfLoss += isnan(diff) ? 0 : (diff * diff);

            }

            sdfLoss /= sdfSamplePoints.size(); // average

            
        }


        
        // -- Term 1: Repulsion (even spread); repulsion doesnt work as well for non-convex polygons
        // https://chatgpt.com/s/t_690658c9da2c8191a2570cf2cebc09bf

        /*float repulsion = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                float d2 = poses[i].c.distanceTo(poses[j].c);
                if (d2 > 1e-4f)
                {
                    repulsion += 1.0f / (1e-4f + d2 * d2);
                }
            }
        }*/

        // -- Term 2: Penalty for deviation from input seeds
        float displacement = 0.0f;

        if ( poseSeeds.size() == 2 * n )
        {
            for (int i = 0; i < n; ++i)
            {
                zVector inputPos( poseSeeds[i * 2 + 0], poseSeeds[i * 2 + 1], 0 );
                float d = poses[i].c.distanceTo(inputPos * 100); // poses scaled in extractPoses
                displacement += 1.0 / (d * d);
            }
        }

        //-- Term 3: PIP penalty
        float PIP_pen = 0;
        for (int i = 0; i < n; ++i)
        {
            if (!pointInsidePolygon(poses[i].c, polygon))PIP_pen -= 0.1;
        }

        //-- Term 4 : pull to sdfSamples centroid

        float minD = 1e6;
        for (int i = 0; i < n; ++i)
        {
            minD += poses[i].c.distanceTo(sdfSample_centroid);// std::min(, minD);
        }
        //minD *= 0.1;
        /// rescale ;
        //sdfLoss /= 100;
        /*displacement *= 100; 
        if (sdfLoss < 150)sdfLoss = 0;*/


        printf(" %.4f,%.4f,%.4f \n", sdfLoss, minD, PIP_pen);

        //if (minD > 1) sdfLoss = minD;

        return sdfLoss /*+ displacement + PIP_pen */;
    }

    void computeGradient(std::vector<float>& x, std::vector<float>& y_dummy, std::vector<float>& gradOut) override
    {
        // 1) Single forward to get the baseline outputs
        std::vector<float> y0 = forward(x);

        // 2) Central-difference step. 
        //    Because you scale centers by 100 in extractPoses(), use a slightly larger eps.
        const float eps = 1e-2f;

        gradOut.assign(outputDim, 0.0f);

        // 3) Compute baseline loss once
        float baseLoss = computeLoss(y0, y_dummy);

        // 4) For each output dimension, perturb that output value and re-evaluate loss
        for (int i = 0; i < outputDim; ++i)
        {
            // Positive perturbation
            std::vector<float> y_plus = y0;
            y_plus[i] += eps;
            float L_plus = computeLoss(y_plus, y_dummy);

            // Negative perturbation
            std::vector<float> y_minus = y0;
            y_minus[i] -= eps;
            float L_minus = computeLoss(y_minus, y_dummy);

            // Central difference derivative dL/dy_i
            gradOut[i] = (L_plus - L_minus) / (2.0f * eps);

           // printf("%.4f,%.4f \n", L_plus, baseLoss);
        }

        // Note:
        // - Only the center coords (indices 4*k + 0 and 4*k + 1) affect your current loss,
        //   so gradients for the orientation slots (4*k + 2, 4*k + 3) will be ~0 — this is expected.
        // - Call backward(gradOut, learningRate) after this to update weights.

      /*  printf("GRAD: [");
            for (float v : gradOut) printf("  %.4f \n ", v);
        printf("]\n");*/

    }


    ///

    void visualiseBlendedSDFs(vector<Pose2D>& poses)
    {
        for (auto& sample : sdfSamplePoints)
            sample.val = evalBlendedCircleSDF(sample.pt, poses);

    }

    void drawPolygon()
    {
        if (polygon.empty()) return;

        glColor3f(0.1f, 0.1f, 0.1f);
        glLineWidth(2.0f);

        glBegin(GL_LINE_LOOP);
        for (const zVector& pt : polygon)
        {
            glVertex3f(pt.x, pt.y, pt.z);
        }
        glEnd();

        glLineWidth(1.0f);

        if (sdfSamplePoints.empty())return;

        /*for (auto& sample : sdfSamplePoints)
            drawPoint(zVecToAliceVec(sample.pt));*/

        // --- SDF Sample Points: Jet Color Visualization
        {
            // Compute min-max val
            float vmin = 1e6;
            float vmax = -vmin;

            for (auto& s : sdfSamplePoints)
            {
                vmin = std::min(vmin, s.val);
                vmax = std::max(vmax, s.val);
            }


            //glPointSize(3);
            //for (auto& s : sdfSamplePoints)
            //{

            //    float r, g, b;
            //    getJetColor(ofMap(s.val,vmin,vmax,0,1), r, g, b); // map to [-1,1] before jetColor
            //    
            //    glColor3f(r, g, b);
            //    drawPoint(zVecToAliceVec(s.pt));
            //}
            //glPointSize(1);
            //glColor3f(0, 0, 0);

        }

    }

};

// ------------------------- APP ----------------------------------
// ------------------------- - ----------------------------------
// ------------------------- - ----------------------------------


HeightField2D myHeightField , myHeightField1 , myHeightField2;
double threshold;

heightfieldNN nn;
vector<float> output;
vector<float> dummyInput = { 1.0f };
std::vector<float> dummyTarget = { 1.0f };; // unused

float learningRate = 0.1f;
float zRangeMin;
std::vector<zVector> polygon;

vector<parcel> plots;
parcel plot;
spaceGrid SG;

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

void runUnitTest()
{
    MLP net(2, { 8, 8 }, 1);

    std::vector<std::vector<float>> X, Y;
    for (int i = 0; i < 100; ++i)
    {
        float x0 = ((float)rand() / RAND_MAX) * 6.28f - 3.14f;
        float x1 = ((float)rand() / RAND_MAX) * 6.28f - 3.14f;
        float y = std::sin(x0) + std::cos(x1);
        X.push_back({ x0, x1 });
        Y.push_back({ y });
    }

    float lr = 0.01f;
    for (int epoch = 0; epoch < 1200; ++epoch)
    {
        float totalLoss = 0.0f;
        for (int i = 0; i < X.size(); ++i)
        {
            std::vector<float> grads;
            net.computeGradient(X[i], Y[i], grads);
            net.backward(grads, lr);
            auto out = net.forward(X[i]);
            totalLoss += net.computeLoss(out, Y[i]);
        }
        totalLoss /= X.size();
        if (epoch % 50 == 0)
            std::cout << "Epoch " << epoch << " Avg Loss: " << totalLoss << std::endl;
    }

    std::cout << "Test prediction:\n";
    for (int i = 0; i < 5; ++i)
    {
        auto out = net.forward(X[i]);
        std::cout << "Input: (" << X[i][0] << ", " << X[i][1] << ") Target: " << Y[i][0] << " Pred: " << out[0] << std::endl;
    }
}

bool compute = false;
void update(int value)
{
    if ( compute ) keyPress('t', 0, 0);
}

void draw()
{
    backGround(0.9);
    drawGrid(50);


    // ---------parcels

    for (auto& parcel : plots)parcel.drawBox();

    wireFrameOn();
        SG.drawBuckets();
        SG.drawParticlesInBuckets();
    wireFrameOff();

   // ----------------------- nn
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

   nn.visualize(zVector(50, 350, 0), 200, 250);
   /////////
  myHeightField.drawSamplePoints();
 // myHeightField.drawFieldPoints(false, false);


    {
       // myHeightField.drawFieldPoints(false, false);
       float ht = myHeightField.zMin;

        glColor3f(0, 0, 0);
        /*for (double tv = 0; tv < threshold; tv += 0.1)
        {
            float h = ofMap(tv, 0, 1, myHeightField.MLS_zMin, myHeightField.MLS_zMax);
            
            glPushMatrix();
    
            {
                glTranslatef(0.0f, 0.0f, h);
                myHeightField.drawIsocontours(tv);
            }
            glPopMatrix();

            
        }*/

        float iso = ofMap(zRangeMin, myHeightField.MLS_zMin, myHeightField.MLS_zMax, 0, 1);
        //myHeightField.drawIsocontours(iso);
       

        /// ------
        iso = ofMap(zRangeMin+1, myHeightField.MLS_zMin, myHeightField.MLS_zMax, 0, 1);
        //myHeightField.drawIsocontours(iso);

        glLineWidth(1);
    }

    glTranslatef(120, 0, 0);
    {
        myHeightField1.drawFieldPoints(false, false);

        glColor3f(0, 0, 0);
        float iso = ofMap(0, myHeightField.MLS_zMin, myHeightField.MLS_zMax, 0, 1);
        myHeightField1.drawIsocontours(iso);
      
    }

    glTranslatef(120, 0, 0);
    {
        myHeightField2.drawFieldPoints(false, false);

        glColor3f(0, 0, 0);
        float iso = ofMap(0, myHeightField.MLS_zMin, myHeightField.MLS_zMax, 0, 1);
        myHeightField2.drawIsocontours(threshold);

    }

    




    /*for (int i = 0; i < parcels.size(); i++)
        for (int j = i + 1; j < parcels.size(); j++)
            parcels[i].parcel_parcel_intersect(parcels[j]);*/





}

std::vector<std::vector<zVector>> contour_bands;

void keyPress(unsigned char k, int xm, int ym)
{
    
    if (k == 'p')
    {
        std::vector<Pose2D> poses;
        nn.extractPoses(output, poses, true);

        int id = 0;

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

    if (k == 'o')
    {
        plots[0].makeCentersEquiDistant(plots);
    }
    
    if (k == 'e')
    {
        for (auto& parcel : plots)parcel.expand_withNormalCheck(plots, true, &SG);

        /*for (int i = 0; i < parcels.size(); i++)
            for (int j = i + 1; j < parcels.size(); j++)
                parcels[i].parcel_parcel_intersect(parcels[j]);*/

        for (auto& parcel : plots)parcel.smooth();

        // ------------- update SG 


        SG.clearBuckets();
        SG.np = 0;

        for (auto& parcel : plots)
            for (int i = 0; i < parcel.nPoints; i++)
            {
                SG.addPosition(parcel.boxPoints[i]);
            }

        for( auto p : nn.polygon)SG.addPosition(p);

        SG.PartitionParticlesToBuckets();

    }
    
    //------------------------------ N

    if (k == '-')
    {
        nn.translate_SDFPolygon_and_samples_to_original();
        compute = false;
    }
    if (k == '=')
    {
       // nn.generateSamplesInRange(myHeightField, zRangeMin, zRangeMin+2);
        zRangeMin += 1.0;

        if (zRangeMin >= myHeightField.zMax)zRangeMin = myHeightField.zMin;

       
        float iso = ofMap(zRangeMin, myHeightField.MLS_zMin, myHeightField.MLS_zMax, 0, 1);
        myHeightField.computeIsocontours(iso);
        std::vector<std::vector<zVector>> contours = myHeightField.getOrderedContours();

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
    if (k == 'c')compute = !compute;

    if (k == 'u') runUnitTest();

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
        {
            sites.push_back(poses[i].c);
        }

        for (int i = 0; i < myHeightField2.RES; i++)
        {
            for (int j = 0; j < myHeightField2.RES; j++)
            {
                myHeightField2.field[i][j] = evalBlendedCircleSDF(myHeightField2.gridPoints[i][j], poses);
            }
        }

        myHeightField2.clearField();
        myHeightField2.addVoronoi(sites);
        myHeightField2.subtract(myHeightField1);
       // myHeightField2.normalise();
        //myHeightField2.rescaleFieldToRange(-1, 1);
    }

    
}

void mousePress(int b, int state, int x, int y)
{
}

void mouseMotion(int x, int y)
{
}

#endif // _MAIN_
