#ifndef _HEIGHT_FIELD_
#define _HEIGHT_FIELD_





#include <vector>
#include <algorithm>
#include <cmath>
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

using namespace zSpace;

#include "scalarField.h"



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

    void rescalePoints(vector<zVector>& pts)
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
        zVector bmin(1e6, 1e6, 0); //  samples[0];
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
        zVector gridMax(50, 50, 0);

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
        zVector gridMax(50, 50, 0);
        zVector gridMin(-50, -50, 0);
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

    void trimFieldWithPolygon(vector<zVector>& poly)
    {
        for (int i = 0; i < RES; i++)
            for (int j = 0; j < RES; j++)
                if (!pointInsidePolygon(gridPoints[i][j], poly)) field[i][j] = 1;

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

#endif