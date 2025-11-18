#pragma once

#define _PARCEL_VECTOR_
#ifdef _PARCEL_VECTOR_

#include "main.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

#include "spatialBin.h"
#include "Matrix3x3.h"

// ==========================================================
// POLYGON WINDING TEST
// ==========================================================

int is_point_inside_polygon(zVector* polygon, int N, zVector& p, int bound)
{
    int count = 0;
    zVector p1 = polygon[0];
    zVector p2;

    for (int i = 1; i <= N; i++)
    {
        if (p == p1) return bound;

        p2 = polygon[i % N];

        if (p.y < MIN(p1.y, p2.y) || p.y > MAX(p1.y, p2.y))
        {
            p1 = p2;
            continue;
        }

        if (p.y > MIN(p1.y, p2.y) && p.y < MAX(p1.y, p2.y))
        {
            if (p.x <= MAX(p1.x, p2.x))
            {
                if (p1.y == p2.y && p.x >= MIN(p1.x, p2.x)) return bound;

                if (p1.x == p2.x)
                {
                    if (p.x == p1.x) return bound;
                    else count++;
                }
                else
                {
                    float xinters =
                        (p.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x;

                    if (fabs(p.x - xinters) < EPS) return bound;
                    if (p.x < xinters) count++;
                }
            }
        }
        else
        {
            if (p.y == p2.y && p.x <= p2.x)
            {
                zVector& p3 = polygon[(i + 1) % N];
                if (p.y >= MIN(p1.y, p3.y) && p.y <= MAX(p1.y, p3.y))
                    count++;
                else
                    count += 2;
            }
        }

        p1 = p2;
    }

    return (count % 2 == 0) ? 0 : 1;
}

// ==========================================================
// SEGMENT INTERSECTION
// ==========================================================

enum IntersectResult { PARALLEL, COINCIDENT, NOT_INTERESECTING, INTERESECTING };

IntersectResult Intersect_segment2d(double* x, double* y, double& u, double& v)
{
    double denom =
        (y[3] - y[2]) * (x[1] - x[0]) -
        (x[3] - x[2]) * (y[1] - y[0]);

    double nume_a =
        (x[3] - x[2]) * (y[0] - y[2]) -
        (y[3] - y[2]) * (x[0] - x[2]);

    double nume_b =
        (x[1] - x[0]) * (y[0] - y[2]) -
        (y[1] - y[0]) * (x[0] - x[2]);

    if (fabs(denom) < 1e-06)
    {
        if (fabs(nume_a) < 1e-06 && fabs(nume_b) < 1e-06)
            return COINCIDENT;

        u = v = 0.0;
        return PARALLEL;
    }

    u = nume_a / denom;
    v = nume_b / denom;

    if (u >= 0.0f && u <= 1.0f &&
        v >= 0.0f && v <= 1.0f)
        return INTERESECTING;

    return NOT_INTERESECTING;
}

double X_[4], Y_[4], U_, V_;

zVector IntersectEdges(zVector& p1, zVector& p2,
    zVector& p3, zVector& p4,
    IntersectResult& IR)
{
    X_[0] = p1.x;  Y_[0] = p1.y;
    X_[1] = p2.x;  Y_[1] = p2.y;
    X_[2] = p3.x;  Y_[2] = p3.y;
    X_[3] = p4.x;  Y_[3] = p4.y;

    IR = Intersect_segment2d(X_, Y_, U_, V_);
    return p1 + (p2 - p1) * U_;
}


// ==========================================================
// PARCEL CLASS (DYNAMIC VECTOR VERSION)
// ==========================================================

class parcel
{
public:

    zVector directionOfBox;
    zVector centerOfBox;
    float areaOfBox = 0;

    std::vector<zVector> polyPoints;
    std::vector<zVector> boxPointsNormals;
    std::vector<bool>    boolMove;

    std::vector<zVector> centerPoints;
    std::vector<zVector> forces;

    int id_u = 0, id_v = 0;
    int nPoints = 0;
    int n_cen = 0;

    float normScale = 0.25f;
    float collisionRad = 0.75f;
    bool  flipNormals_always = true;

    float restLength = 0.2f;
    int* nborIds = NULL;

public:

    parcel()
    {
        polyPoints.reserve(200);
        boxPointsNormals.reserve(200);
        boolMove.reserve(200);
        centerPoints.reserve(200);
        forces.reserve(200);
    }

    inline int Mod(int a, int n)
    {
        a = a % n;
        return (a < 0) ? a + n : a;
    }

    // ======================================================
    // SHAPE CONSTRUCTION
    // ======================================================

    void setDefaultBox(float r = 1.0f, int N = 75)
    {
        nPoints = N;

        polyPoints.resize(nPoints);
        boxPointsNormals.resize(nPoints);
        boolMove.assign(nPoints, true);

        float inc = TWO_PI / float(nPoints);

        for (int i = 0; i < nPoints; i++)
        {
            float x = r * sin(i * inc);
            float y = r * cos(i * inc);
            polyPoints[i] = zVector(x, y, 0);
        }

        invertBox();
        computeNormals();
    }

    void importPrimitive(std::vector<zVector>& polygon)
    {
        nPoints = polygon.size();
        polyPoints = polygon;

        boxPointsNormals.assign(nPoints, zVector());
        boolMove.assign(nPoints, true);
    }

    // ======================================================
    // ORIENTATION
    // ======================================================

    void invertBox(zVector u = zVector(1, 1, 0))
    {
        zTransform TM;
        TM.setIdentity();

        zVector v = u;
        std::swap(v.x, v.y);
        v.y *= -1;

        zVector w(0, 0, 1);

        u.normalize();
        v.normalize();
        w.normalize();

        zVector c(0, 0, 0);

        TM.col(0) << u.x, u.y, u.z, 1;
        TM.col(1) << v.x, v.y, v.z, 1;
        TM.col(2) << w.x, w.y, w.z, 1;
        TM.col(3) << c.x, c.y, c.z, 1;

        TM.inverse();

        for (auto& p : polyPoints)
            p = p * TM;
    }

    void computeNormals()
    {
        for (int i = 0; i < nPoints; i++)
        {
            int next = Mod(i + 1, nPoints);
            int prev = Mod(i - 1, nPoints);

            zVector e1 =
                (polyPoints[i] - polyPoints[prev]) ^ zVector(0, 0, 1);

            zVector e2 =
                (polyPoints[next] - polyPoints[i]) ^ zVector(0, 0, 1);

            zVector nrm = (e1 + e2) * 0.5;

            zVector toC = polyPoints[i] - centerOfBox;
            if (nrm * toC > 0) nrm *= -1;

            nrm.normalize();
            boxPointsNormals[i] =
                nrm * normScale * (flipNormals_always ? -1 : 1);
        }
    }

    void flipNormals()
    {
        for (int i = 0; i < boxPointsNormals.size(); i++) boxPointsNormals[i] *= -1;

    }

    void computePCA()
    {
        Matrix3x3 mat;
        zVector mean, eigenVals, eigenVecs[3];

        mat.PCA(polyPoints.data(), nPoints, mean, eigenVals, eigenVecs);

        eigenVecs[2].normalize();
        directionOfBox = eigenVecs[2] * 5.0f;
    }

    // ======================================================
    // GEOMETRY
    // ======================================================

    float computeParcelArea()
    {
        areaOfBox = 0.0f;

        for (int i = 0; i < nPoints; i++)
        {
            int nxt = Mod(i + 1, nPoints);

            areaOfBox +=
                ((polyPoints[nxt] - polyPoints[i]) ^
                    (centerOfBox - polyPoints[i])).length() * 0.5f;
        }

        return areaOfBox;
    }

    void transformBox()
    {
        zTransform TM;
        TM.setIdentity();

        zVector u = directionOfBox;
        zVector v = u;

        std::swap(v.x, v.y);
        v.y *= -1;

        zVector w(0, 0, 1);

        u.normalize();
        v.normalize();
        w.normalize();

        TM.col(0) << u.x, u.y, u.z, 1;
        TM.col(1) << v.x, v.y, v.z, 1;
        TM.col(2) << w.x, w.y, w.z, 1;
        TM.col(3) << centerOfBox.x, centerOfBox.y, centerOfBox.z, 1;

        for (auto& p : polyPoints)
            p = p * TM;

        computeNormals();
    }

    // ======================================================
    // EXPANSION (NO GRID)
    // ======================================================

    void expand(std::vector<parcel>& AB)
    {
        for (int i = 0; i < nPoints; i++)
        {
            zVector np = boxPointsNormals[i];

            if (boolMove[i])
                polyPoints[i] += np;

            for (auto& pr : AB)
            {
                if (pr.id_u == id_u) continue;

                for (int j = 0; j < pr.nPoints; j++)
                {
                    float dist =
                        polyPoints[i].distanceTo(pr.polyPoints[j]);

                    if (dist < collisionRad)
                        boolMove[i] = false;
                }
            }
        }
    }

    // ======================================================
    // EXPANSION (SPACE GRID)
    // ======================================================

    void expand(spaceGrid* SG)
    {
        for (int i = 0; i < nPoints; i++)
        {
            zVector np = boxPointsNormals[i];

            if (boolMove[i])
                polyPoints[i] += np;

            boolMove[i] = true;

            int num_nbors =
                SG->getNeighBors(nborIds, polyPoints[i], collisionRad * 2);

            for (int j = 0; j < num_nbors; j++)
            {
                if (nborIds[j] >= id_u * nPoints &&
                    nborIds[j] < (id_u + 1) * nPoints)
                    continue;

                float dist =
                    polyPoints[i].distanceTo(SG->positions[nborIds[j]]);

                if (dist < collisionRad)
                    boolMove[i] = false;
            }

            if (!(num_nbors > 0)) continue;

            for (int j = 0; j < num_nbors; j++)
            {
                if (nborIds[j] >= id_u * nPoints &&
                    nborIds[j] < (id_u + 1) * nPoints)
                    continue;

                int next_j =
                    (j == num_nbors - 1) ? nborIds[j] + 1 : nborIds[j + 1];

                if (next_j >= SG->np) continue;

                np.normalize();

                IntersectResult IR;
                IntersectEdges(polyPoints[i],
                    polyPoints[i] + np * collisionRad * 2,
                    SG->positions[nborIds[j]],
                    SG->positions[next_j],
                    IR);

                if (IR == INTERESECTING)
                    boolMove[i] = false;
            }
        }
    }

    // ======================================================
    // EXPANSION WITH NORMAL CHECK
    // ======================================================

    void expand_withNormalCheck(
        std::vector<parcel>& AB,
        bool expandBeforeNormalCheck = false,
        spaceGrid* SG = NULL)
    {
        if (expandBeforeNormalCheck && SG)
            expand(SG);

        computeNormals();
    }

    // ======================================================
    // PARCEL-PARCEL INTERSECTION
    // ======================================================

    void parcel_parcel_intersect(parcel& other)
    {
        for (int i = 0; i < nPoints; i++)
        {
            int nxt = (i + 1) % nPoints;

            if (other.id_u == id_u) continue;

            for (int j = 0; j < other.nPoints; j++)
            {
                int next_j = (j + 1) % other.nPoints;

                IntersectResult IR;

                zVector ip =
                    IntersectEdges(polyPoints[i], polyPoints[nxt],
                        other.polyPoints[j], other.polyPoints[next_j],
                        IR);

                if (IR == INTERESECTING)
                {
                    boolMove[i] = false;
                    boolMove[nxt] = false;
                    other.boolMove[j] = false;
                    other.boolMove[next_j] = false;
                }

                bool i_in =
                    is_point_inside_polygon(other.polyPoints.data(),
                        other.nPoints,
                        polyPoints[i], 0);

                bool nxt_in =
                    is_point_inside_polygon(other.polyPoints.data(),
                        other.nPoints,
                        polyPoints[nxt], 0);

                if (i_in)
                {
                    polyPoints[i] -= boxPointsNormals[i];
                    boolMove[i] = false;
                }

                if (nxt_in)
                {
                    polyPoints[nxt] -= boxPointsNormals[nxt];
                    boolMove[nxt] = false;
                }
            }
        }
    }

    // ======================================================
    // EDGE-LENGTH SMOOTHING
    // ======================================================

    void equaliseEdgeLengths()
    {
        for (int i = 0; i < nPoints; i++)
        {
            int nxt = Mod(i + 1, nPoints);
            zVector edge = polyPoints[nxt] - polyPoints[i];

            float disp = edge.length() - restLength;

            edge.normalize();

            if (boolMove[i])
                polyPoints[i] += edge * disp * 0.4f;

            if (boolMove[nxt])
                polyPoints[nxt] -= edge * disp * 0.4f;
        }
    }

    // ======================================================
    // SMOOTHING
    // ======================================================

    void smooth()
    {
        std::vector<zVector> newPts(nPoints);

        for (int i = 0; i < nPoints; i++)
        {
            int prev = Mod(i - 1, nPoints);
            int next = Mod(i + 1, nPoints);

            if (!boolMove[i])
                newPts[i] =
                polyPoints[prev] * 0.15 +
                polyPoints[i] * 0.70 +
                polyPoints[next] * 0.15;
            else
                newPts[i] =
                polyPoints[prev] * 0.30 +
                polyPoints[i] * 0.40 +
                polyPoints[next] * 0.30;
        }

        polyPoints = newPts;
    }

    // ======================================================
    // CENTER MANAGEMENT
    // ======================================================

    void addCenter()
    {
        centerPoints.push_back(
            centerOfBox + zVector(ofRandom(-1, 1), ofRandom(-1, 1), 0));

        forces.push_back(zVector());
        n_cen = centerPoints.size();
    }

    void normaliseForces()
    {
        if (forces.empty()) return;

        float mn = 1e9, mx = -1e9;

        for (auto& f : forces)
        {
            float d = f.length();
            mn = MIN(mn, d);
            mx = MAX(mx, d);
        }

        for (auto& f : forces)
        {
            float d = f.length();
            if (d > 1e-6)
            {
                f.normalize();
                f *= ofMap(d, mn, mx, 0, 1);
            }
        }
    }

    void makeCentersEquiDistant(std::vector<parcel>& plots,
        std::vector<zVector>& polygon)
    {
        int N = plots.size();
        forces.assign(N, zVector(0, 0, 0));

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (plots[i].id_u == plots[j].id_u) continue;

                zVector e = plots[j].centerOfBox - plots[i].centerOfBox;
                float d = e.length();

                if (d > 1e-2)
                {
                    e.normalize();
                    e /= (d * d);
                    forces[i] -= e;
                }
            }
        }

        normaliseForces();

        for (int i = 0; i < N; i++)
        {
            if (forces[i].length() < 1)
            {
                if (is_point_inside_polygon(polygon.data(),
                    polygon.size(),
                    plots[i].centerOfBox, 0))
                {
                    plots[i].centerOfBox += forces[i];
                }
            }
        }
    }

    // ======================================================
    // DISPLAY
    // ======================================================

    void display()
    {
        glPointSize(5);
        drawPoint(zVecToAliceVec(centerOfBox));
        glPointSize(1);

        drawLine(zVecToAliceVec(centerOfBox),
            zVecToAliceVec(centerOfBox + directionOfBox * 3));

        for (int i = 0; i < nPoints; i++)
        {
            glColor3f(1, 0, 0);
            drawLine(zVecToAliceVec(polyPoints[i]),
                zVecToAliceVec(polyPoints[(i + 1) % nPoints]));

            glColor3f(boolMove[i] ? 1 : 0,
                0,
                boolMove[i] ? 0 : 1);

            drawPoint(zVecToAliceVec(polyPoints[i]));
            drawLine(zVecToAliceVec(polyPoints[i]),
                zVecToAliceVec(polyPoints[i] + boxPointsNormals[i]));
        }

        for (int i = 0; i < centerPoints.size(); i++)
        {
            drawPoint(zVecToAliceVec(centerPoints[i]));
            drawLine(zVecToAliceVec(centerPoints[i]),
                zVecToAliceVec(centerPoints[i] + forces[i]));
        }
    }
};

#endif // _PARCEL_VECTOR_
