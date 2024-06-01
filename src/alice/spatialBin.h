#ifndef SPATIAL_BIN
#define SPATIAL_BIN

#include "ALICE_DLL.h"


#define bucketSize 150
struct bucket
{
	int n;
	int ids[bucketSize];
	bucket()
	{
		n = 0;
	}
	int addId(int &i)
	{

		if (n >= bucketSize)return 0;
		ids[n] = i;
		n++;

		return 1;
	}
	void clearBucket()
	{
		n = 0;
	}
	bool containsParticles()
	{
		return (n > 0);
	}
};


class spaceGrid
{

#define MAX_SPATIAL_BIN_POINTS 100000
public:

	zVector* positions;

	int np;

	#define RESX 22 	
	#define RESY 22 	
	#define RESZ 5 	

	bucket buckets[RESX*RESY*RESZ]; /// pre-allocoation is susceptible to crashing --> change RES
	double dx, dy, dz;
	zVector min, max;

	spaceGrid()
	{
		np = 0;
		positions = new zVector[MAX_SPATIAL_BIN_POINTS];
	}

	spaceGrid( zVector *pos, int _np)
	{
		positions = pos;
		np = _np;
	}
	////////////////////////////////////////////////////////////////////////// utility methods 
	
	void addPosition(zVector & pt)
	{
		positions[np] = pt;
		np++;
	}
	
	int getNearestNeighbor(zVector inPt, double searchRadius, bool &suc)
	{
		int* nbors;
		int num_nbors = getNeighBors(nbors, inPt, searchRadius);
		

		if (num_nbors == 0)
		{
			suc = false;
			return 0;
		}

		float minD = 1e06;
		int nearestNbor = nbors[0];
		for (int i = 0; i < num_nbors; i++)
		{
			
			if (nbors[i] == (np - 1))continue;// last point is dummy point added for bounding box calcs

			float d = inPt.distanceTo(positions[nbors[i]]);
			if ( d < minD)
			{
				minD = d;
				nearestNbor = nbors[i];
			}
		}

		suc = true;
		return nearestNbor;
	}
	
	int getNeighBors(int *&nBorIds, zVector inPt, double searchRadius)
	{
		int sRad_x, sRad_y, sRad_z;


		sRad_x = (searchRadius / dx) + 2;
		sRad_y = (searchRadius / dy) + 2;
		sRad_z =( dz > EPS) ? (searchRadius / dz) + 2 : 1;

		//printf(" %i,%i,%i sRads\n", sRad_x, sRad_y, sRad_z);
		//printf(" %1.2f,%1.2f,%1.2f divs\n", dx,dy,dz);
		////////////////////////////////////////////////////////////////////////// calculate voxel nBor search-index based on link-radius

		int u, v, w, index, nborId; // to store 1D and 3D voxel indices per particles, and neighbor id.
		zVector pt, p_n, vel, v_n, diff; // to temp.store positions/velocities of particle and its neighbor.
		double dSq; // to store square of the distance

		int nBorHdSize = bucketSize * sRad_x * 2 + bucketSize * sRad_y * 2 + bucketSize * sRad_z * 2;
		nBorIds = new int[nBorHdSize];
		//cout << nBorHdSize << "--nBorHdSize-- " << endl;
		int n_cnt = 0;

		{
			pt = inPt;
			threeDIndexFromPosition(pt, u, v, w);
			if (u == -1 || v == -1 || w == -1)return 0;
			//printf(" %i,%i,%i uvw\n", u, v, w);

			//search in nborhood

			for (int i = u - sRad_x; i <= u + sRad_x; i++)
				for (int j = v - sRad_y; j <= v + sRad_y; j++)
					for (int k = w - sRad_z; k <= w + sRad_z; k++)
					{
						index = threeDIndexToIndex(i, j, k);

						//if(index !=-1)printf(" %i,%i,%i s_ijk \n",i,j,k);

						if (index < RESX*RESY*RESZ && index >= 0)
							if (buckets[index].containsParticles())
								for (int nb = 0; nb < buckets[index].n; nb++)
								{
										if(nb >= bucketSize) continue;

									nborId = buckets[index].ids[nb];
									
										if (nborId >= np || nborId < 0)continue;

									p_n = positions[nborId];
									diff = p_n - pt;
									if (diff * diff < searchRadius * searchRadius && n_cnt < nBorHdSize)
									{
										nBorIds[n_cnt] = nborId;
										n_cnt++;
									}
								}
					}
		}

		return (n_cnt);
	}
	void indexToThreeDIndex(int &index, int &u, int &v, int &w)
	{
		u = index % RESX;
		v = (index / RESX) % RESY;
		w = index / (RESX * RESY);
		
		/*
		x = i % max_x
		y = (i / max_x) % max_y
		z = i / (max_x * max_y)
		https://coderwall.com/p/fzni3g/bidirectional-translation-between-1d-and-3d-arrays	
		*/
	}
	int threeDIndexToIndex(int &i, int &j, int &k)
	{


		int index = (i + j*RESX + k*RESX*RESY);
		
		if (index < 0)return -1;
		if (index >= RESX*RESY*RESZ)return -1;

		return index;
	}
	void threeDIndexFromPosition(zVector p, int &u, int &v, int &w)
	{
		p.x -= min.x;
		p.y -= min.y;
		p.z -= min.z;
		u = floor(p.x / dx); v = floor(p.y / dy); w = floor(p.z / dz);

		/*if (u >= RES || u < 0)u = -1;
		if (v >= RES || v < 0)v = -1;
		if (w >= RES || w < 0)w = -1;*/
	}
	zVector centerOfBucket(int &u, int &v, int &w)
	{
		return (min + (zVector(u*dx, v*dy, w*dz) + zVector(dx, dy, dz)) * 0.5);
	}
	void clearBuckets()
	{
		int index;
		for (int i = 0; i < RESX * RESY * RESZ; i++)buckets[i].clearBucket();

	}
	void computeBucketDimensions()
	{
		if ((max.z - min.z) < 0) max.z = min.z = 0;

		zVector diff = max - min;
		dx = diff.x / (float)RESX;
		dy = diff.y / (float)RESY;
		dz = diff.z / (float)RESZ;

	}
	void getBoundingBox(zVector &min, zVector &max)
	{
		double x = pow(10, 10);
		min = zVector(x, x, x);
		max = min * -1;

		for (int i = 0; i < np; i++)
		{
			min.x = MIN(positions[i].x, min.x);
			min.y = MIN(positions[i].y, min.y);
			min.z = MIN(positions[i].z, min.z);

			max.x = MAX(positions[i].x, max.x);
			max.y = MAX(positions[i].y, max.y);
			max.z = MAX(positions[i].z, max.z);
		}
	}
	int getNumberofPointsinBuckets()
	{
		int sum = 0;
		for (int i = 0; i < RESX; i++)
			for (int j = 0; j < RESY; j++)
				for (int k = 0; k < RESZ ; k++)
					sum += buckets[threeDIndexToIndex(i, j, k)].n;

		return sum;
	}
	//////////////////////////////////////////////////////////////////////////

	void PartitionParticlesToBuckets()
	{
		clearBuckets();

		getBoundingBox(min, max);

		if (fabs(max.z - min.z) < EPS)
		{
			min.z = -0.1;
			max.z = 0.1;
		}
		zVector diff = (max - min);
		diff.normalize();
		min -= diff*0.1;
		max += diff * 0.1;

		computeBucketDimensions();
		

		zVector pt; int index;
		int u, v, w;
		for (int i = 0; i < np; i++)
		{
			pt = positions[i];
			threeDIndexFromPosition(pt, u, v, w);
			index = threeDIndexToIndex(u, v, w);

			/*if (u == -1 || v == -1 || w == -1)
			{
				cout << " point not added" << i << " -- index " << index << "-- " << RES * RES* RES << endl;
				zVector a = pt - min;
				a.print();
				printf(" %1.2f %1.2f %1.2f \n", floor(a.x / dx), floor(a.y / dy), floor(a.z / dz));
			}*/

			if (index != -1)
			{
				int suc =	buckets[index].addId(i);
				//if (!suc)cout << " point not added" << i << " -- bucketsize" << buckets[index].n << endl;
				
			}
			else
				{
					/*cout << " point not added" << i << endl;
					printf(" %i %i %i \n", u, v, w);*/
					
				}

		}

		//cout << getNumberofPointsinBuckets() << " --- SG --  " << np << endl;
	}

	//////////////////////////////////////////////////////////////////////////
	void  drawBucket(zVector& min, zVector& max, zVector& origin, bool lines, zVector clr)
	{
		glEnableClientState(GL_NORMAL_ARRAY);
		//glEnableClientState(GL_COLOR_ARRAY);
		glEnableClientState(GL_VERTEX_ARRAY);
		glNormalPointer(GL_FLOAT, 0, normals);
		//glColorPointer(3, GL_FLOAT, 0, colors);
		glVertexPointer(3, GL_FLOAT, 0, vertices);

		//glEnable( GL_BLEND );
		//glBlendFunc( GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

		glPushMatrix();

		//glScalef( max.x-min.x, max.y-min.y, max.z-min.z);	
		glTranslatef((min.x + max.x) * 0.5, (min.y + max.y) * 0.5, (min.z + max.z) * 0.5);
		glScalef((max.x - min.x) * 0.5, (max.y - min.y) * 0.5, (max.z - min.z) * 0.5);

		// move to upper-right

		//glEnable(GL_BLEND);
		//glBlendFunc(GL_ONE_MINUS_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

		if (lines)
		{
			glColor4f(0.1, 0.1, 0.1, 1);
			glDrawElements(GL_LINES, 24, GL_UNSIGNED_BYTE, indices);

		}
		else
		{

			glColor4f(clr.x, clr.y, clr.z, 1);
			glDrawElements(GL_QUADS, 24, GL_UNSIGNED_BYTE, indices);
		}

		//glDisable(GL_BLEND);
		glPopMatrix();




		glDisableClientState(GL_VERTEX_ARRAY);  // disable vertex arrays
		//glDisableClientState(GL_COLOR_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);
	}
	
	void drawBuckets()
	{
		glLineWidth(1);
		int index;
		for (int i = 0; i < RESX; i++)
			for (int j = 0; j < RESY; j++)
				for (int k = 0; k < RESZ; k++)
					if (buckets[threeDIndexToIndex(i, j, k)].containsParticles())
					{
						drawBucket( (min + zVector(i * dx, j * dy, k * dz)), (min + zVector(i * dx, j * dy, k * dz) + zVector(dx, dy, dz)), (min + zVector(i * dx, j * dy, k * dz) + zVector(dx, dy, dz)), false, zVector(0.35, 0.35, 0.35));

						/*for (int nb = 0; nb < buckets[threeDIndexToIndex(i, j, k)].n; nb++)
						{
							glColor3f(ofMap(i, 0, RES, 0, 1), ofMap(j, 0, RES, 0, 1), ofMap(k, 0, RES, 0, 1));
							if (nb < bucketSize)
								drawPoint(positions[buckets[threeDIndexToIndex(i, j, k)].ids[nb]]);
						}*/
					}
	}

	void drawParticlesInBuckets()
	{
		glPointSize(5);

		int index;
		for (int i = 0; i < RESX; i++)
			for (int j = 0; j < RESY; j++)
				for (int k = 0; k < RESZ; k++)
				{
					int ind = threeDIndexToIndex(i, j, k);
					if (index == -1)continue;

					if (buckets[ind].containsParticles())
					{
						for (int nb = 0; nb < buckets[ind].n; nb++)
						{
							glColor3f(ofMap(i, 0, RESX, 0, 1), ofMap(j, 0, RESY, 0, 1), ofMap(k, 0, RESZ, 0, 1));
							if (nb < bucketSize)
								drawPoint( zVecToAliceVec(positions[buckets[ind].ids[nb]]) );
						}
					}
				}

		glPointSize(1);

	}

};
#endif // !SPATIAL_BIN
