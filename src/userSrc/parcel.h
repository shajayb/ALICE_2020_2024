#define _PARCEL_
#ifdef _PARCEL_

#include "main.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>

#include "spatialBin.h"
#include "Matrix3x3.h"


// USAGE 

// SETUP : 
// array of parcels, each with center and direction. default box is actually a circle.
// Space Grid needs to setup

// vector<parcel> plots
//plots.clear();
//int id = 0;
//
///*for( int i = 0; i < 2; i+= 1)
//{
//	for (int j = 0; j < 2; j++)
//	{
//		plot.centerOfBox = zVector(i * 20, j * 20, 0);
//		plot.directionOfBox = zVector(1, 1, 0);;
//		plot.setDefaultBox();
//		plot.transformBox();
//		plot.id_u = id++;
//		plots.push_back(plot);
//	}
//
//}*/

//

//SG = *new spaceGrid();

//EXPAND COLLIDE STOP
// expand_withNormalCheck(plots, true, &SG); is the main method.
// SG needs to be cleared, and updated with new locations for ALL collision points
// SG.PartitionParticlesToBuckets(); needs to be called at the end.


//if (k == 'e')
//{
//	for (auto& parcel : plots)parcel.expand_withNormalCheck(plots, true, &SG);
//	for (auto& parcel : plots)parcel.smooth();
//
//	// ------------- update SG 
//
//
//	SG.clearBuckets();
//	SG.np = 0;
//
//	for (auto& parcel : plots)
//		for (int i = 0; i < parcel.nPoints; i++)
//		{
//			SG.addPosition(parcel.boxPoints[i]);
//		}
//
//	for (auto p : nn.polygon)SG.addPosition(p);
//
//	SG.PartitionParticlesToBuckets();
//
//}

//

#define nPoly 75
#define num_centers 60
double width = 1.0;

int insidePolygon(zVector* polygon, int N, zVector& p, int bound)
{
	//cross points count of x
	int __count = 0;

	//neighbour bound vertices
	zVector p1, p2;

	//left vertex
	p1 = polygon[0];

	//check all rays
	for (int i = 1; i <= N; ++i)
	{
		//point is an vertex
		if (p == p1) return bound;

		//right vertex
		p2 = polygon[i % N];

		//ray is outside of our interests
		if (p.y < MIN(p1.y, p2.y) || p.y > MAX(p1.y, p2.y))
		{
			//next ray left point
			p1 = p2; continue;
		}

		//ray is crossing over by the algorithm (common part of)
		if (p.y > MIN(p1.y, p2.y) && p.y < MAX(p1.y, p2.y))
		{
			//x is before of ray
			if (p.x <= MAX(p1.x, p2.x))
			{
				//overlies on a horizontal ray
				if (p1.y == p2.y && p.x >= MIN(p1.x, p2.x)) return bound;

				//ray is vertical
				if (p1.x == p2.x)
				{
					//overlies on a ray
					if (p1.x == p.x) return bound;
					//before ray
					else ++__count;
				}

				//cross point on the left side
				else
				{
					//cross point of x
					double xinters = (p.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x;

					//overlies on a ray
					if (fabs(p.x - xinters) < EPS) return bound;

					//before ray
					if (p.x < xinters) ++__count;
				}
			}
		}
		//special case when ray is crossing through the vertex
		else
		{
			//p crossing over p2
			if (p.y == p2.y && p.x <= p2.x)
			{
				//next vertex
				const zVector& p3 = polygon[(i + 1) % N];

				//p.y lies between p1.y & p3.y
				if (p.y >= MIN(p1.y, p3.y) && p.y <= MAX(p1.y, p3.y))
				{
					++__count;
				}
				else
				{
					__count += 2;
				}
			}
		}

		//next ray left point
		p1 = p2;
	}

	//EVEN
	if (__count % 2 == 0) return(0);
	//ODD
	else return(1);
}

enum IntersectResult { PARALLEL, COINCIDENT, NOT_INTERESECTING, INTERESECTING };
IntersectResult Intersect_segment2d(double* x, double* y, double& u, double& v)
{
	double denom = ((y[3] - y[2]) * (x[1] - x[0])) - ((x[3] - x[2]) * (y[1] - y[0]));
	double nume_a = ((x[3] - x[2]) * (y[0] - y[2])) - ((y[3] - y[2]) * (x[0] - x[2]));
	double nume_b = ((x[1] - x[0]) * (y[0] - y[2])) - ((y[1] - y[0]) * (x[0] - x[2]));

	if (fabs(denom) < 1e-06)
	{
		if (fabs(nume_a) < 1e-06 && fabs(nume_b) < 1e-06)return COINCIDENT;

		u = v = -0.0;
		return PARALLEL; // lines are parallel
	}
	u = nume_a;// ((x[3] - x[2]) * (y[0] - y[2])) - ((y[3] - y[2]) * (x[0] - x[2]));
	u /= denom;// ((y[3] - y[2]) * (x[1] - x[0])) - ((x[3] - x[2]) * (y[1] - y[0]));

	v = nume_b;// ((x[1] - x[0]) * (y[0] - y[2])) - ((y[1] - y[0]) * (x[0] - x[2]));
	v /= denom;// ((y[3] - y[2]) * (x[1] - x[0])) - ((x[3] - x[2]) * (y[1] - y[0]));

	if (u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f)return INTERESECTING;

	return NOT_INTERESECTING;
}

zVector Intersect_linesegments(zVector* pts, double& u, double& v)
{
	double x[4], y[4];
	for (int i = 0; i < 4; i++)
	{
		x[i] = pts[i].x;
		y[i] = pts[i].y;

	}

	Intersect_segment2d(x, y, u, v);
	return pts[0] + (pts[1] - pts[0]) * u;
}



double x[4], y[4], u, v;
zVector IntersectEdges(zVector& p1, zVector& p2, zVector& p3, zVector& p4, IntersectResult& IR)
{

	x[0] = p1.x;
	y[0] = p1.y;

	x[1] = p2.x;
	y[1] = p2.y;

	x[2] = p3.x;
	y[2] = p3.y;

	x[3] = p4.x;
	y[3] = p4.y;

	IR = Intersect_segment2d(x, y, u, v);
	return  (p1 + (p2 - p1) * u);
}


class parcel
{

public:

	//declare class variables.
	zVector directionOfBox;
	zVector centerOfBox;
	float areaOfBox;

	zVector boxPoints[nPoly];
	zVector boxPointsNormals[nPoly];

	zVector centerPoints[num_centers];
	int n_cen = 0;
	zVector forces[num_centers];

	int nPoints;
	bool boolMove[nPoly];
	int id_u, id_v;

	float normScale = 0.01;
	float collisionRad = 0.5;
	bool flipNormals_always = false;

	float restLength = 0.2;

	void setDefaultBox(float r = 1)
	{


		nPoints = nPoly;

		float inc = TWO_PI / float(nPoly);
		//float r =  3; // ofRandom(1, 3);// *sqrt(2);

		for (int i = 0; i < nPoints; i++)
		{
			float x = r * sin(float(i) * inc);
			float y = r * cos(float(i) * inc);

			boxPoints[i] = zVector(x, y, 0);
			boolMove[i] = true;
		}

		invertBox();
		computeNormals();

		normScale = 0.25;
		//collisionRad = normScale * 1;

		//importPrimitive();
		//computeNormals();
	}

	void importPrimitive()
	{
		zObjMesh o_fieldMesh;
		zFnMesh fnMesh(o_fieldMesh);

		fnMesh.from("data/parcelPrim.obj", zOBJ, false);

		if (fnMesh.numVertices() < nPoly)
		{
			nPoints = fnMesh.numVertices();

			zPointArray vertexPositions;
			fnMesh.getVertexPositions(vertexPositions);

			for (int i = 0; i < nPoints; i++)
			{
				boxPoints[i] = vertexPositions[i];
				boolMove[i] = true;
			}
		}
	}

	void importPrimitive(vector<zVector>& polygon)
	{
		nPoints = polygon.size();
		for (int i = 0; i < nPoints; i++)
		{
			boxPoints[i] = polygon[i];
			boolMove[i] = true;
		}
	}

	void invertBox(zVector u = zVector(1, 1, 0))
	{
		zTransform TM;
		TM.setIdentity();
		zVector v, w;
		v = u; swap(v.x, v.y); v.y *= -1;
		w = zVector(0, 0, 1);
		u.normalize(); v.normalize(); w.normalize();
		zVector c(0, 0, 0);


		//assign the values to the matrix
		TM.col(0) << u.x, u.y, u.z, 1;
		TM.col(1) << v.x, v.y, v.z, 1;
		TM.col(2) << w.x, w.y, w.z, 1;
		TM.col(3) << c.x, c.y, c.z, 1;

		TM.inverse();

		for (int i = 0; i < nPoints; i++)
			boxPoints[i] = boxPoints[i] * TM;
	}

	int Mod(int a, int n)
	{
		a = a % n;
		return (a < 0) ? a + n : a;
	}

	void shortenNormal(bool lengthen = false)
	{
		normScale *= lengthen ? 1.1 : 0.9;
		for (int i = 0; i < nPoints; i++)
		{
			boxPointsNormals[i].normalize();
			boxPointsNormals[i] *= normScale;
		}
	}

	void flipNormals()
	{
		for (int i = 0; i < nPoints; i++) boxPointsNormals[i] *= -1;

	}

	void computeNormals()
	{
		for (int i = 0; i < nPoints; i++)
		{
			int next = Mod(i + 1, nPoints);// (i + 1) % nPoints;
			int prev = Mod(i - 1, nPoints); //(nPoly - 1 + i) % nPoints;
			zVector e1 = (boxPoints[i] - boxPoints[prev]) ^ zVector(0, 0, 1);
			zVector e2 = (boxPoints[next] - boxPoints[i]) ^ zVector(0, 0, 1);;

			zVector norm = ((e1 + e2) * 0.5);
			zVector compareVec = boxPoints[i] - centerOfBox;
			if (norm * compareVec) norm *= -1;

			norm.normalize();
			boxPointsNormals[i] = norm * normScale * (flipNormals_always ? -1 : 1);
		}
	}

	float computeParcelArea()
	{
		areaOfBox = 0;
		for (int i = 0; i < nPoints; i++)
		{
			int nxt = Mod(i + i, nPoints);
			areaOfBox += ((boxPoints[nxt] - boxPoints[i]) ^ (centerOfBox - boxPoints[i])).length() * 0.5;

		}

		return areaOfBox;
	}

	void computePCA()
	{
		Matrix3x3 mat;
		zVector mean, eigenVals, eigenVecs[3];
		//double wts[3];

		mat.PCA(boxPoints, nPoints, mean, eigenVals, eigenVecs);

		for (int i = 0; i < 3; i++)
		{
			printf("%1.2f,%1.2f,%1.f  -- %i \n", eigenVecs[i].x, eigenVecs[i].y, eigenVecs[i].z, i);
		}

		cout << " ----------------- " << endl;

		eigenVecs[2].normalize();
		directionOfBox = eigenVecs[2] * 5;
	}
	// actions / functions / methods
	void transformBox()
	{
		zTransform TM;
		TM.setIdentity();

		zVector u, v, w;
		u = directionOfBox;
		v = u; swap(v.x, v.y); v.y *= -1;
		w = zVector(0, 0, 1);

		u.normalize(); v.normalize(); w.normalize();
		zVector c = centerOfBox;
		v *= width;

		//assign the values to the matrix
		TM.col(0) << u.x, u.y, u.z, 1;
		TM.col(1) << v.x, v.y, v.z, 1;
		TM.col(2) << w.x, w.y, w.z, 1;
		TM.col(3) << c.x, c.y, c.z, 1;

		for (int i = 0; i < nPoints; i++)
			boxPoints[i] = boxPoints[i] * TM;

		computeNormals();
	}

	void expand(vector<parcel>& AB)
	{
		zVector np;
		for (int i = 0; i < nPoints; i++)
		{
			np = boxPointsNormals[i];

			if (boolMove[i])boxPoints[i] += np;

			for (auto& parcel : AB)
			{
				if (parcel.id_u != id_u)
					for (int j = 0; j < nPoints; j++)
					{
						float dist = boxPoints[i].distanceTo(parcel.boxPoints[j]);

						if (dist < collisionRad)
						{
							boolMove[i] = false;

						}
					}
			}
		}


		//computeNormals();
	}

	int* nborIds;
	void expand(spaceGrid* SG)
	{

		zVector np;
		for (int i = 0; i < nPoints; i++)
		{
			np = boxPointsNormals[i];

			if (boolMove[i])boxPoints[i] += np;

			boolMove[i] = true; // reset;

			//for (auto& parcel : AB)
			{
				//if (parcel.id_u != id_u)
				int num_nbors = SG->getNeighBors(nborIds, boxPoints[i], collisionRad * 2);

				/// collision test with vertices.
				for (int j = 0; j < num_nbors; j++)
				{
					if (nborIds[j] >= (id_u * nPoints) && nborIds[j] < (id_u + 1) * nPoints)continue;

					float dist = boxPoints[i].distanceTo(SG->positions[nborIds[j]]);

					if (dist < collisionRad)boolMove[i] = false;

				}

				if (!(num_nbors > 0))continue;

				/// look-up intersection test with edges.
				for (int j = 0; j < num_nbors; j++)
				{
					if (nborIds[j] >= (id_u * nPoints) && nborIds[j] < (id_u + 1) * nPoints)continue;

					int next_j = (j == num_nbors - 1) ? nborIds[j] + 1 : nborIds[j + 1];// !!! this assumes the nbors[j] ids are in consequtive order;
					if (next_j >= SG->np)continue;

					np.normalize();
					IntersectResult IR;
					IntersectEdges(boxPoints[i], boxPoints[i] + np * collisionRad * 2, SG->positions[nborIds[j]], SG->positions[next_j], IR);

					if (IR == INTERESECTING)
					{
						boolMove[i] = false;
						//boxPoints[i] -= np * collisionRad * 0.5;;

					}
				}
			}
		}


		//computeNormals();
	}

	void expand_withNormalCheck(vector<parcel>& AB, bool expandBeforeNormalCheck = false, spaceGrid* SG = NULL)
	{

		//if( expandBeforeNormalCheck)expand(AB);
		if (expandBeforeNormalCheck)expand(SG);

		zVector np;


		//double x[4], y[4], u, v;
		//for (int i = 0; i < nPoints; i++)
		//{
		//	np = boxPointsNormals[i];
		//	np.normalize();
		//	
		//	for (auto& parcel : AB)
		//	{
		//		

		//		if (parcel.id_u != id_u)
		//			for (int j = 0; j < nPoints; j++)
		//			{
		//				
		//				int next_j = Mod(j + i, nPoints);// (j + 1) % nPoly;
		//				IntersectResult IR;
		//				IntersectEdges( boxPoints[i], boxPoints[i] + np * collisionRad, parcel.boxPoints[j], parcel.boxPoints[next_j], IR);

		//				if (IR == INTERESECTING)
		//				{
		//					boolMove[i] = false;
		//					boxPoints[i] -= np * collisionRad * 0.5;;

		//				}
		//			}
		//	}
		//}

		computeNormals();
		//flipNormals();
	}

	void parcel_All_parcel_intersect(vector<parcel>& AB)
	{
		for (int i = 0; i < nPoints; i++)
		{
			int nxt = Mod(i + 1, nPoints);// (i + 1) % nPoly;

			for (auto& parcel : AB)
			{
				parcel_parcel_intersect(parcel);
			}

		}

	}

	void parcel_parcel_intersect(parcel& otherBox)
	{
		for (int i = 0; i < nPoints; i++)
		{
			int nxt = (i + 1) % nPoints;

			//for (auto& parcel : AB)
			//auto parcel = otherBox;
			{
				if (otherBox.id_u != id_u)
					for (int j = 0; j < nPoints; j++)
					{
						int next_j = (j + 1) % nPoints;

						IntersectResult IR;
						zVector int_pt = IntersectEdges(boxPoints[i], boxPoints[nxt], otherBox.boxPoints[j], otherBox.boxPoints[next_j], IR);

						if (IR == INTERESECTING)
						{
							//boxPoints[i] -= np;
							boolMove[i] = false;
							boolMove[nxt] = false;
							otherBox.boolMove[j] = false;
							otherBox.boolMove[next_j] = false;
						}

						bool i_in, nxt_in;
						i_in = insidePolygon(otherBox.boxPoints, nPoints, boxPoints[i], 0);
						nxt_in = insidePolygon(otherBox.boxPoints, nPoly, boxPoints[nxt], 0);

						zVector np = boxPointsNormals[i];

						if (i_in)
						{
							boxPoints[i] -= boxPointsNormals[i];
							boolMove[i] = false;

						}
						if (nxt_in)
						{
							boxPoints[nxt] -= boxPointsNormals[nxt];
							boolMove[nxt] = false;
						}


						glPointSize(8);
						if (IR == INTERESECTING) drawPoint(zVecToAliceVec(int_pt));
						glLineWidth(1);
					}
			}

		}

	}

	void equaliseEdgeLengths()
	{
		for (int i = 0; i < nPoints; i++)
		{
			int nxt = Mod(i + 1, nPoints);
			zVector edge = boxPoints[nxt] - boxPoints[i];
			double displacement = edge.length() - restLength;

			edge.normalize();
			if (boolMove[i])
				boxPoints[i] += edge * displacement * 0.4;

			if (boolMove[nxt])
				boxPoints[nxt] -= edge * displacement * 0.4;
		}

	}

	void smooth()
	{

		for (int i = 0; i < nPoints; i++)
		{
			int prev = Mod(i - 1, nPoints);// (i + nPoints - 1) % nPoints;
			int next = Mod(i + 1, nPoints);// (i + 1) % nPoints;
			if (!boolMove[i])
				boxPoints[i] = boxPoints[prev] * 0.15 + boxPoints[i] * 0.7 + boxPoints[next] * 0.15;
			else
				boxPoints[i] = boxPoints[prev] * 0.3 + boxPoints[i] * 0.4 + boxPoints[next] * 0.3;
		}

	}

	void addCenter()
	{

		centerPoints[n_cen++] = centerOfBox + zVector(ofRandom(-1, 1), ofRandom(-1, 1), 0);
		if (n_cen >= num_centers)n_cen = 0;
	}

	void makeCentersEquiDistant(vector<parcel>& plots , vector<zVector>&polygon) // not working
	{
		// reset forces
		
		for (int i = 0; i < plots.size(); i++)forces[i] = zVector(0, 0, 0);

		//calculate & store repulsive force per point
		for (int i = 0; i < plots.size(); i++)
		{
			for (int j = 0; j < plots.size(); j++)
			{
				if (plots[i].id_u == plots[j].id_u) continue;

				zVector e = plots[j].centerOfBox - plots[i].centerOfBox;
				float d = e.length();

				if (d > 1e-2)
				{
					e.normalize();
					e /= d * d;
					forces[i] -= e;
				}

			}
		}

		/*for (int i = 0; i < plots.size(); i++)
		{
			zVector grad = gradientAt(plots[i].centerOfBox, polygon);
			grad = grad ^ zVector(0, 0, -1);
			grad.normalize();
			forces[i] += grad * 1e-2;
		}*/
		// calculate the maximum and minimum magnitude of reuplisve force
		normaliseForces();

		// move each of the points, by applying their respective forces, if the magnitude of force is less than 1 and the point is with a radius of 10 from the origin;
		for (int i = 0; i < plots.size(); i++)
			if (forces[i].length() < 1)
			{
				if (insidePolygon(polygon.data(), polygon.size(), plots[i].centerOfBox, 0))
					plots[i].centerOfBox += forces[i];
				
				//centerPoints[i] -= forces[i] * 2;

			}


	}

	void normaliseForces()
	{
		double force_max, force_min;
		force_min = 1e6; force_max = -force_min;

		for (int i = 0; i < n_cen; i++)
		{
			float d = forces[i].length();
			force_max = MAX(force_max, d);
			force_min = MIN(force_min, d);
		}

		// re-scale all forces to be within 0 & 1
		for (int i = 0; i < n_cen; i++)
		{
			float d = forces[i].length();
			forces[i].normalize();
			forces[i] *= ofMap(d, force_min, force_max, 0, 1);

		}
	}

	zVector norm;
	void display()
	{
		glPointSize(1);

		//drawLine and drawPoint accept data of type : Alice::vec
		// so we need to convert zVector to Alice::vec;

		glPointSize(5);

			drawPoint(zVecToAliceVec(centerOfBox));
		
		glPointSize(1);
		
		drawLine(zVecToAliceVec(centerOfBox), zVecToAliceVec(centerOfBox + directionOfBox * 3));

		for (int i = 0; i < nPoints; i++)
		{
			//drawLine and drawPoint accept data of type : Alice::vec
			// so we need to convert zVector to Alice::vec;
			glColor3f(1, 0, 0);
			drawLine(zVecToAliceVec(boxPoints[i]), zVecToAliceVec(boxPoints[(i + 1) % nPoints]));
			norm = boxPointsNormals[i];


			//drawLine and drawPoint accept data of type : Alice::vec
			// so we need to convert zVector to Alice::vec;
			(boolMove[i]) ? glColor3f(1, 0, 0) : glColor3f(0, 0, 1);

			drawPoint(zVecToAliceVec(boxPoints[i]));
			drawLine(zVecToAliceVec(boxPoints[i]), zVecToAliceVec(boxPoints[i] + norm));


			/*for (int i = 0; i < nPoints; i++)
				if (boolMove[i])
				{
					(boolMove[i]) ? glColor3f(1, 0, 0) : glColor3f(0, 0, 1);
					drawCircle(zVecToAliceVec(boxPoints[i]), collisionRad, 32);
				}*/

		}

		for (int i = 0; i < n_cen; i++)
		{
			drawPoint(zVecToAliceVec(centerPoints[i]));
			drawLine(zVecToAliceVec(centerPoints[i]), zVecToAliceVec(centerPoints[i] + forces[i]));
		}

		glPointSize(1);


	}
};
#endif // _PARCEL_