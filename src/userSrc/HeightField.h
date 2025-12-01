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
#include "parcel_vector.h"



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

    void readSamplesAndInterpolate( std::string& filename)
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

        for ( auto& s : samples)
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

        for (auto& s : samples)  s = cDst + (s - cSrc) * scale;



        // --- 3) z-range (diagnostic)
        zMin = 1e6;;// samples[0].z;
        zMax = -zMin;// samples[0].z;
        for (auto& s : samples)
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

    void reconstruct_screened_poisson
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

         int nx = SF_RES;
         int ny = SF_RES;
        zVector gridMin(-50, -50, 0);
        zVector gridMax(50, 50, 0);

         double dx = (gridMax.x - gridMin.x) / (nx - 1);
         double dy = (gridMax.y - gridMin.y) / (ny - 1);

         double invDx = (nx > 1) ? 1.0 / (gridMax.x - gridMin.x) * (nx - 1) : 0.0;
         double invDy = (ny > 1) ? 1.0 / (gridMax.y - gridMin.y) * (ny - 1) : 0.0;

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

         double twoSigma2 = 2.0 * sigma_cells * sigma_cells;
         int r = std::max(1, (int)std::ceil(3.0 * sigma_cells)); // 3σ footprint

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
         double invdx2 = 1.0 / (dx * dx);
         double invdy2 = 1.0 / (dy * dy);

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
        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
            {
                zVector gp = gridPoints[i][j];
                float num = 0.0f;
                float den = 0.0f;

                for ( auto& s : samples)
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
    void interpolateToGrid_MLS(double supportRadiusCells = 1.0,
        int    minNeighbors = 10,
        double eps = 1e-12)
    {
         int nx = SF_RES;
         int ny = SF_RES;

        // world grid spacing
        zVector gridMax(50, 50, 0);
        zVector gridMin(-50, -50, 0);
         double dx = (gridMax.x - gridMin.x) / (nx - 1);
         double dy = (gridMax.y - gridMin.y) / (ny - 1);

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
                 zVector gp = gridPoints[i][j];
                 double xg = gp.x;
                 double yg = gp.y;

                // build weighted normal equations
                double S_w = 0.0;
                double S_x = 0.0, S_y = 0.0;
                double S_xx = 0.0, S_xy = 0.0, S_yy = 0.0;
                double S_z = 0.0, S_xz = 0.0, S_yz = 0.0;

                int nbh = 0;

                // choose support radius in *world* so kernel is isotropic when dx != dy
                // convert cells -> world by averaging
                 double R = supportRadiusCells * 0.5 * (dx + dy);
                 double invR = (R > eps) ? 1.0 / R : 1.0;

                double idw_num = 0.0;
                double idw_den = 0.0;

                for ( auto& s : samples)
                {
                     double sx = s.x;
                     double sy = s.y;
                     double sz = s.z;

                     double rx = sx - xg;
                     double ry = sy - yg;

                     double r = std::sqrt(rx * rx + ry * ry);
                     double q = r * invR;

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
        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
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



    inline int Mod(int a, int n)
    {
        a = a % n;
        return (a < 0) ? a + n : a;
    }

    void trimFieldWithPolygon(vector<zVector>& poly)
    {
        if (poly.size() < 3)return;

        zVector centroid;
        for (auto& p : poly)centroid += p; 
        centroid /= poly.size();


        float area = 0.0f;
        for (int i = 0; i < poly.size(); i++)
        {
            int nxt = Mod(i + 1, poly.size());

            area +=
                ((poly[nxt] - poly[i]) ^
                    (centroid - poly[i])).length() * 0.5f;
        }

        if (area < 1e-2 || std::isnan(area)) return;


        for (int i = 0; i < SF_RES; i++)
            for (int j = 0; j < SF_RES; j++)
                if (!pointInsidePolygon(gridPoints[i][j], poly)) field[i][j] = 1e4;

    }

    void trimFieldWithPolygons(vector< vector<zVector> >& polys)
    {
        for (auto& poly : polys)trimFieldWithPolygon(poly);
    }

    //void scale_feild_values_within(zVector *pts, int N)
    //{
    //    for (int i = 0; i < SF_RES; i++)
    //        for (int j = 0; j < SF_RES; j++)
    //            if (!insidePOly) field[i][j] = 1e4;

    //}

    // -------------------------- PATH FINDING

    void scale_scalar_within_polygons( vector<vector<zVector>> polygons)
    {
        for (int i = 0; i < SF_RES; i++)
            for (int j = 0; j < SF_RES; j++)
            {
                for( auto &poly : polygons)
                 if ( pointInsidePolygon(gridPoints[i][j], poly)) field[i][j] *= 1e4;
            }
                
    }

    //------------------------------------------------------------
    // PATHFINDING SUPPORT STRUCTURES
    //------------------------------------------------------------
    struct Node
    {
        int x, y;
        float cost, heuristic;
        Node* parent;

        Node(int _x, int _y, float _c, float _h, Node* _p)
            : x(_x), y(_y), cost(_c), heuristic(_h), parent(_p)
        {}
    };

    struct CompareNode
    {
        bool operator()(Node* a, Node* b)
        {
            return (a->cost + a->heuristic) > (b->cost + b->heuristic);
        }
    };

    //------------------------------------------------------------
    // MAPPING FUNCTIONS
    //------------------------------------------------------------
    inline int worldToGrid(float x)
    {
        // world: [-50,50] → grid index: [0, SF_RES-1]
        float t = (x + 50.0f) / 100.0f;
        t = std::clamp(t, 0.0f, 1.0f);
        return int(t * (SF_RES - 1));
    }

    inline float gridToWorld(int i)
    {
        // grid index: [0, SF_RES-1] → world [-50,50]
        float t = float(i) / float(SF_RES - 1);
        return -50.0f + t * 100.0f;
    }


    //------------------------------------------------------------
    // A* SHORTEST PATH ON field[][] USING 8-NEIGHBORS
    //------------------------------------------------------------

    std::vector<zVector> lastShortestPath;

    void findShortestPath(zVector start, zVector end)
    {
        lastShortestPath.clear();

        auto heuristic = [](int x1, int y1, int x2, int y2)
            {
                return sqrt((x2 - x1) * (x2 - x1)
                    + (y2 - y1) * (y2 - y1));
            };

        // --------------------------------------------------------
        // MAP WORLD TO GRID INDICES
        // --------------------------------------------------------
        int sx = worldToGrid(start.x);
        int sy = worldToGrid(start.y);
        int ex = worldToGrid(end.x);
        int ey = worldToGrid(end.y);

        // clamp in case of rounding edge cases
        sx = std::clamp(sx, 0, SF_RES - 1);
        sy = std::clamp(sy, 0, SF_RES - 1);
        ex = std::clamp(ex, 0, SF_RES - 1);
        ey = std::clamp(ey, 0, SF_RES - 1);

        std::priority_queue<Node*, std::vector<Node*>, CompareNode> openSet;
        std::unordered_map<int, Node*> visited;

        Node* startNode = new Node(sx, sy, 0.0f, heuristic(sx, sy, ex, ey), nullptr);
        openSet.push(startNode);

        // 8-connected grid
        int directions[8][2] =
        {
            { 1,  0}, {-1,  0}, { 0,  1}, { 0, -1},
            { 1,  1}, { 1, -1}, {-1,  1}, {-1, -1}
        };

        // --------------------------------------------------------
        // A* LOOP
        // --------------------------------------------------------
        while (!openSet.empty())
        {
            Node* curr = openSet.top();
            openSet.pop();

            // goal reached
            if (curr->x == ex && curr->y == ey)
            {
                Node* v = curr;
                while (v)
                {
                    float wx = gridToWorld(v->x);
                    float wy = gridToWorld(v->y);
                    float wz = gridPoints[v->x][v->y].z;

                    lastShortestPath.push_back(zVector(wx, wy, wz));
                    v = v->parent;
                }

                std::reverse(lastShortestPath.begin(), lastShortestPath.end());
                return;
            }

            int key = curr->x * SF_RES + curr->y;
            if (visited.count(key)) continue;
            visited[key] = curr;

            // explore neighbors
            for (auto& d : directions)
            {
                int nx = curr->x + d[0];
                int ny = curr->y + d[1];

                if (nx < 0 || nx >= SF_RES || ny < 0 || ny >= SF_RES)
                    continue;

                // diagonal costs sqrt(1^2 + 1^2)
                bool diag = (d[0] != 0 && d[1] != 0);
                float stepCost = diag ? 1.41421356f : 1.0f;

                float costHere = field[nx][ny];
                float newCost = curr->cost + stepCost + costHere;
                
                //float costHere = field[nx][ny];
                //if (costHere < 1e-6f) costHere = 1e-6f;   // avoid divide-by-zero

                //float invCost = 1.0f / costHere;
                //float newCost = curr->cost + stepCost + invCost;


                openSet.push(new Node(
                    nx, ny,
                    newCost,
                    heuristic(nx, ny, ex, ey),
                    curr
                ));
            }
        }

        printf("No path found. Path length = %i\n", lastShortestPath.size());
    }

    void clippedContour( float threshold, vector<zVector> &poly, vector<zVector>& clippingPoly)
    {
        computeIsocontours(threshold);
        std::vector<std::vector<zVector>> contours = getOrderedContours();

        // extract largest contiguous set
        size_t maxPts = 0;

        for (int i = 0; i < contours.size(); i++)
        {

            if (contours[i].size() > maxPts)
            {
                maxPts = contours[i].size();
                poly = contours[i];
            }
        }


        // resample contour, trim with polygon

        poly = resamplePolyline(poly, 2.5 * (1.0f / scale));
        for (int n = 0; n < 5; n++)smoothPath(poly);

        lastShortestPath.clear();
        for (auto& p : poly)if (pointInsidePolygon(p, clippingPoly)) lastShortestPath.push_back(p);

        poly = lastShortestPath;
    }

    

    //------------------------------------------------------------
    // SMOOTHING
    //------------------------------------------------------------
    void smoothPath()
    {
        if (lastShortestPath.size() < 3) return;

        std::vector<zVector> out = lastShortestPath;

        for (size_t i = 1; i < lastShortestPath.size() - 1; i++)
        {
            out[i] =
                lastShortestPath[i - 1] * 0.3f +
                lastShortestPath[i] * 0.4f +
                lastShortestPath[i + 1] * 0.3f;
        }

        lastShortestPath = out;
    }

    void smoothPath(std::vector<zVector>& path)
    {
        if (path.size() < 3) return;

        std::vector<zVector> out = path;

        for (size_t i = 1; i < path.size() - 1; i++)
        {
            out[i] =
                path[i - 1] * 0.3f +
                path[i] * 0.4f +
                path[i + 1] * 0.3f;
        }

        path = out;
    }

    //------------------------------------------------------------
    // CURVE RESAMPLING (Equal Spacing)
    //------------------------------------------------------------

    /**
     * Resamples a polyline to have points equally spaced by 'spacing'.
     * @param inputPoly The original dense curve (e.g., from Marching Squares).
     * @param spacing   The desired distance between points. Higher value = fewer points.
     * @return          A new vector containing the resampled points.
     */
    std::vector<zVector> resamplePolyline( std::vector<zVector>& inputPoly, float spacing)
    {
        if (inputPoly.size() < 2 || spacing <= 1e-5f)
        {
            return inputPoly;
        }

        std::vector<zVector> outPoly;

        // Always keep the starting point
        outPoly.push_back(inputPoly[0]);

        float currentDist = 0.0f;       // Distance traversed on the current walk
        float nextTargetDist = spacing; // The next distance threshold to place a point

        // Iterate over original segments
        for (size_t i = 0; i < inputPoly.size() - 1; i++)
        {
            zVector p0 = inputPoly[i];
            zVector p1 = inputPoly[i + 1];

            float segmentLength = p0.distanceTo(p1);

            // If duplicate points exist in input, skip them to avoid div/0
            if (segmentLength < 1e-6f) continue;

            // Check if the target distance falls within this segment (potentially multiple times)
            while (currentDist + segmentLength >= nextTargetDist)
            {
                float distNeeded = nextTargetDist - currentDist;
                float t = distNeeded / segmentLength;

                // Linear Interpolation: P = p0 + (p1 - p0) * t
                zVector newPt = p0 + (p1 - p0) * t;

                outPoly.push_back(newPt);

                // Advance the target
                nextTargetDist += spacing;
            }

            // Add this segment's full length to the accumulator before moving to next segment
            currentDist += segmentLength;
        }

        // Ensure the exact end point of the original curve is included
        // (unless the last resampled point is practically identical to it)
        if (!outPoly.empty())
        {
            if (outPoly.back().distanceTo(inputPoly.back()) > 1e-4f)
            {
                outPoly.push_back(inputPoly.back());
            }
        }
        else
        {
            // Fallback for degenerate cases
            outPoly.push_back(inputPoly.back());
        }

        return outPoly;
    }

    //------------------------------------------------------------
    // DRAW PATH
    //------------------------------------------------------------
 
    void drawPth(std::vector<zVector>& path)
    {
        if (path.empty()) return;

        glColor3f(0, 0, 1);
        for (size_t i = 0; i < path.size() - 1; i++)
        {
            drawLine(zVecToAliceVec(path[i]), zVecToAliceVec(path[i + 1]));
        }
    }


    //------------------------------------------------------------
    // VECTOR FIELD FROM HEIGHT FIELD
    //------------------------------------------------------------
    //------------------------------------------------------------
    // BILINEAR GRADIENT LOOKUP
    //------------------------------------------------------------
    zVector getGradientBilinear(float worldX, float worldY)
    {
        // 1. Map World Space -> Grid Space (Float)
        // Matches your [-50, 50] world bounds
        float u = (worldX + 50.0f) / 100.0f;
        float v = (worldY + 50.0f) / 100.0f;

        // Clamp to safe range [0, 1]
        u = std::clamp(u, 0.0f, 1.0f);
        v = std::clamp(v, 0.0f, 1.0f);

        float gx = u * (SF_RES - 1);
        float gy = v * (SF_RES - 1);

        // 2. Determine Integer Indices and Fractional Weights
        int x0 = (int)floor(gx);
        int y0 = (int)floor(gy);

        // Ensure we don't access out of bounds (x0+1)
        int x1 = std::min(x0 + 1, SF_RES - 1);
        int y1 = std::min(y0 + 1, SF_RES - 1);

        float tx = gx - float(x0); // Fractional part X
        float ty = gy - float(y0); // Fractional part Y

        // 3. Fetch Neighbors
        zVector v00 = gradient[x0][y0];
        zVector v10 = gradient[x1][y0];
        zVector v01 = gradient[x0][y1];
        zVector v11 = gradient[x1][y1];

        // 4. Bilinear Interpolation
        // Lerp X (Bottom Row)
        zVector bot = v00 * (1.0f - tx) + v10 * tx;
        // Lerp X (Top Row)
        zVector top = v01 * (1.0f - tx) + v11 * tx;

        // Lerp Y (Combine Bottom and Top)
        return bot * (1.0f - ty) + top * ty;
    }

    //------------------------------------------------------------
    // BILINEAR SCALAR LOOKUP (world-space input)
    float mapIsoToActualHeight(float iso)
    {

        return iso * (MLS_zMax - MLS_zMin) + MLS_zMin;
    }


    float getFieldValue(zVector& p)
    {
        // Map world space → [0,1]
        float u = (p.x + 50.0f) / 100.0f;
        float v = (p.y + 50.0f) / 100.0f;

        // Clamp
        u = std::clamp(u, 0.0f, 1.0f);
        v = std::clamp(v, 0.0f, 1.0f);

        // Convert to grid index
        float gx = u * (SF_RES - 1);
        float gy = v * (SF_RES - 1);

        int x0 = (int)floor(gx);
        int y0 = (int)floor(gy);
        int x1 = std::min(x0 + 1, SF_RES - 1);
        int y1 = std::min(y0 + 1, SF_RES - 1);

        // Fractional coords
        float tx = gx - float(x0);
        float ty = gy - float(y0);

        // Fetch values
        float f00 = field[x0][y0];
        float f10 = field[x1][y0];
        float f01 = field[x0][y1];
        float f11 = field[x1][y1];

        // Bilinear interpolation
        float bot = f00 * (1.0f - tx) + f10 * tx;
        float top = f01 * (1.0f - tx) + f11 * tx;

        return bot * (1.0f - ty) + top * ty;
    }


    //------------------------------------------------------------
    // STREAMLINES (OPTIONAL, MATCHING LVM STRUCTURE)
    //------------------------------------------------------------
    void drawStreamlinesFromSeeds
    (
        const std::vector<zVector>& seeds,
        int numDirections = 1,
        float stepSize = 1.0f,
        int maxSteps = 600
    )
    {
        std::vector<std::vector<zVector>> streamlines;

        // Thresholds for stopping
        float sinkThreshold = 1e-5f; // Gradient length considered "zero"
        float minMoveDist = stepSize * 0.05f; // Stop if we move less than 5% of stepSize

        for (const auto& seedWS : seeds)
        {
            for (int d = 0; d < numDirections; d++)
            {
                // Calculate Initial Direction (only used if starting exactly on a flat spot)
                float angle = float(d) / float(numDirections) * TWO_PI;
                zVector initialDir(cos(angle), sin(angle), 0);

                zVector pWS = seedWS;
                std::vector<zVector> path;
                path.push_back(pWS);

                // Track previous velocity to detect 180 reversals (ping-ponging)
                zVector prevVelocity = zVector(0, 0, 0);

                for (int s = 0; s < maxSteps; s++)
                {
                    // --------------------------------------------------
                    // Step 1: Sample vector at current position (Predictor)
                    // --------------------------------------------------
                    int ix = worldToGrid(pWS.x);
                    int iy = worldToGrid(pWS.y);

                    // Safety: Stop if we left the grid
                    if (ix < 0 || ix >= SF_RES || iy < 0 || iy >= SF_RES) break;

                    zVector g1 = getGradientBilinear(pWS.x, pWS.y);
                    //zVector g1 = gradient[ix][iy];

                    // STOP CONDITION 1: Gradient is zero (Sink/Peak reached)
                    if (g1.length() < sinkThreshold)
                    {
                        // If this is the very first step, use radial dir, otherwise stop
                        if (s == 0) g1 = initialDir;
                        else break;
                    }

                    zVector v1 = g1;
                    v1.normalize();
                    v1 *= stepSize;

                    // --------------------------------------------------
                    // Step 2: Midpoint Sample (Corrector)
                    // --------------------------------------------------
                    zVector midWS = pWS + v1 * 0.5f;
                    int mx = worldToGrid(midWS.x);
                    int my = worldToGrid(midWS.y);

                    // Safety: Midpoint out of bounds
                    if (mx < 0 || mx >= SF_RES || my < 0 || my >= SF_RES) break;

                    //zVector g2 = gradient[mx][my];
                    zVector g2 = getGradientBilinear(midWS.x, midWS.y);

                    // STOP CONDITION 1 (Repeated for midpoint)
                    if (g2.length() < sinkThreshold) break;

                    zVector v2 = g2;
                    v2.normalize();
                    v2 *= stepSize;

                    // --------------------------------------------------
                    // STOP CONDITION 2: Direction Reversal (Oscillation)
                    // --------------------------------------------------
                    // If the new velocity opposes the previous velocity, we overshot the sink.
                    if (s > 0 && v2*(prevVelocity) < 0.0f)
                    {
                        break;
                    }

                    // --------------------------------------------------
                    // Update Position
                    // --------------------------------------------------
                    zVector nextWS = pWS + v2;

                    // STOP CONDITION 3: Stagnation
                    // If we barely moved (e.g. hitting a wall or very slow convergence)
                    if (nextWS.distanceTo(pWS) < minMoveDist) break;

                    // STOP CONDITION 4: World Bounds
                    if (nextWS.x < -50 || nextWS.x > 50 || nextWS.y < -50 || nextWS.y > 50)
                        break;

                    path.push_back(nextWS);

                    pWS = nextWS;
                    prevVelocity = v2;
                }

                if (path.size() > 1)
                {
                    streamlines.push_back(path);
                }
            }
        }

        // ------------------------------------------------------------
        // Draw
        // ------------------------------------------------------------
        glLineWidth(3);
        glColor3f(1, 0, 0); // Red color for streamlines

        for (auto& stream : streamlines)
        {
            // Resample to smooth out the RK2 segments
            // Using your existing resamplePolyline method
            if (stream.size() > 2)
            {
                stream = resamplePolyline(stream, stepSize * 0.5f);

                // Optional smoothing pass
                smoothPath(stream);
                smoothPath(stream);
            }

            for (size_t i = 0; i < stream.size() - 1; i++)
            {
                drawLine(zVecToAliceVec(stream[i]), zVecToAliceVec(stream[i + 1]));
            }
        }
        glLineWidth(1);
    }



    /// ----------------------------
    

    void setGridPointHeights()
    {
        if (samples.empty()) return;

        for (int i = 0; i < SF_RES; i++)
        {
            for (int j = 0; j < SF_RES; j++)
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
        for ( auto& ptRaw : samples)
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

    void drawPath()
    {
        if (!lastShortestPath.empty())
        {
            
            for (size_t i = 0; i < lastShortestPath.size() - 1; i++)
            {
                drawLine(zVecToAliceVec(lastShortestPath[i]), zVecToAliceVec(lastShortestPath[i + 1]));
                
                glPointSize(5);
                drawPoint( zVecToAliceVec(lastShortestPath[i]));

            }
        }

        glPointSize(1);

    }

};

#endif