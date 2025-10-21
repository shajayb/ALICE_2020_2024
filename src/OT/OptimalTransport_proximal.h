
#ifndef _OPTIMAL_TRANSPORT_PROXIMAL
#define  _OPTIMAL_TRANSPORT_PROXIMAL

////////////////////////////////////////////////////////////////
/// Faster and specialized C++ implementation of
// "Optimal Transport with Proximal Splitting" [Papadakis et al.]
// based on Gabriel Peyré's original Primal-Dual Matlab implementation.
//
// author: Nicolas Bonneel (nbonneel@seas.harvard.edu)
// June 2013
// Revised August 2013 : GCC compatibility
//
// Specialized as: 
// - alpha=1 (pure optimal transport)
// - dimension=3 (2 spatial, 1 temporal) ; i.e., for images
// - boundary condition = Neumann
// - uniform grid : Lx=Ly=Lz
//
// Modified as:
// - uses the FFTW library for solving Poisson's equation
// - always solve for the cubic polynomial with 2 Newton iterations (unless 
//   EXACT_ROOT is #defined). This seems precise enough.
// - Row major storage contrary to Matlab's column major.
// - can use SSE if SSE is #defined. If so, W*H*Time should be a multiple of 4 
//   for single precision data or 2 for double precision.
// - can use AVX if AVX_SUPPORT is #defined. If so, W*H*Time should be multiple
//   of 8 for single precision data or 4 for double precision.
// 
// Note:
// - AVX uses 256bits registers but it is available only on recent machines 
//   and with Win 7 SP1 and above, in 64bits (I don't know for GNU/Linux). 
//   Otherwise, switch to regular SSE (do not define AVX_SUPPORT)
// - Cannot compute exact roots of the cubic equation if SSE is #defined
// - Known issues: on my laptop there are some artifacts that are not present on 
//   my desktop machine + the obstacle gives incorrect results
// - to load png/jpg images, you need to install ImageMagick (binary)
// - Compile time options set below ; runtime options set in the main() 
//
// Benchmark
// - AVX is not faster than SSE on my machine
// - SSE is about 25% faster than nothing for floats, and 7% for doubles
// - single precision about 30% to 70% faster than double precision
// - much faster for power of two images
// - Overall, up to 37x speedup in double and 56x in single precision
// - Typical example: Gaussians in 256x256:  
//        Matlab: 7604 s
//        double precision, exact cubic polynomial with exact cubic root: 313s
//        double precision, exact cubic polynomial 2 Newton steps: 264s
//        double precision, 2 Newton steps for the entire cubic polynomial: 222s
//        double precision, SSE : 208s
//        double precision, AVX : 203s
//        single precision, exact cubic polynomial with exact cubic root: 220s
//        single precision, exact cubic polynomial 2 Newtons steps: 180s
//        single precision, 2 Newton steps for the entire cubic polynomial: 164s
//        single precision, SSE : 131s
//        single precision, AVX : 136s
// - Typical example: Cloud image in 300x158:  
//        Matlab: 1715s
//        double precision, exact cubic polynomial with exact cubic root: 256s
//        double precision, exact cubic polynomial 2 Newton steps: 226s
//        double precision, 2 Newton steps for the entire cubic polynomial: 203s
//        double precision, SSE : 193s
//        double precision, AVX : 197s
//        single precision, exact cubic polynomial with exact cubic root: 196s
//        single precision, exact cubic polynomial 2 Newton steps: 174s
//        single precision, 2 Newton steps for the entire cubic polynomial: 159s
//        single precision, SSE : 130s
//        single precision, AVX : 138s
//        
//////////////////////////////////////////////////////////////


#define OTFP_SSE            // incompatible with EXACT_ROOT
#define AVX_SUPPORT   // incompatible with EXACT_ROOT ; only on recent INTEL processors ; untested on gcc

typedef int size_type;  // {int | __int64}  __int64 for large problems (W*H*time*8 > 1024*1024*256*8-1 on VStudio)
typedef float data_type; // {double | float} Single precision introduces artifacts for large problems. It is ~30% faster.

#define EXACT_ROOT  // uses the exact roots for the cubic polynomial rather than Newton's method. "exact" is misleading though, since the cubic root in Cardan's method can also be approximated (see below for different options)

#ifdef EXACT_ROOT
#include "fast_cube_root.h"

// Choice of approximation for the cube root
//#define cube_root_double(x) halley_cbrt1d(x) // gives 16 bits of precision but somewhat slower that 2 newton iterations on my machine although it's supposed to be faster (http://metamerist.com/cbrt/cbrt.htm)
#define cube_root_double(x) newton_cbrt2d(x)   // Two Newton steps. gives 20 bit of precision ; faster on my machine
//#define cube_root_double(x) pow(x, 1./3.) // the real deal (slower - 52 bits of precision)

//#define cube_root_float(x) halley_cbrt1f(x) // gives 15 bits of precision but somewhat slower that 2 newton iterations on my machine
#define cube_root_float(x) newton_cbrt2f(x)   // Two Newton steps. Gives 20 bit of precision ; faster on my machine. Beware, a single Newton iteration results in some ringing for 256x256 gaussians (although not with 128x128)
//#define cube_root_float(x) powf(x, 1.f/3.f) // the real deal (slower - 23 bits of precision)
#endif


// -------------------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------- end of compile time options --------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>


#include <fftw3.h>
#include "CImg.h"
#include "chrono.h"
#include <omp.h>



using namespace std;
using namespace cimg_library;


#if defined(OTFP_SSE)||defined(AVX_SUPPORT)
#include "sse_helpers.h"
#endif



#define M_PI 3.1415926535897932385626


/// Image IO

template<typename T>
T load_grayscale_image(const char* filename, std::vector<T> &img, size_type &w, size_type &h) { // load red channel and return sum of pixel values
	CImg<unsigned char> img_in(filename);
	w = img_in.width();
	h = img_in.height();
	img.resize(w*h);

	T s(0);
	for (size_type j = 0; j < h; j++) {
		for (size_type i = 0; i < w; i++) {
			img[j*w + i] = *img_in.data(i, j, 2);
			s += *img_in.data(i, j, 2);
		}
	}
	return s;
}

template<typename T>
void save_grayscale_image(const char* filename, T* img, int w, int h, T sumpix) { // save grayscale image and adjusts the sum of pixel values to match sumpix
	T s(0);
	for (int i = 0; i < w*h; i++) {
		s += img[i];
	}
	s = sumpix / s;
	std::vector<unsigned char> result(w*h * 3);
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			for (int k = 0; k < 3; k++) {
				result[k*w*h + j * w + i] = (unsigned char)std::max(T(0), std::min(T(255), img[j*w + i] * s));
			}
		}
	}

	CImg<unsigned char> img_result(&result[0], w, h, 1, 3);
	img_result.save(filename);
}

// product of elements in array
template<typename T>
T prod(T* vals, int n) {
	T p = vals[0];
	for (int i = 1; i < n; i++) {
		p *= vals[i];
	}
	return p;
}


/////////////////////////////////////////////////////////////////////////////////////////////////
// Staggered grid. Dimension is templated but currently only implemented for DIM==3
// Data is stored in row major order, in each of the three functions M[0][:], M[1][:] and M[2][:]
/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int DIM>
class Staggered {};

template<typename T>
class Staggered<T, 3> {
public:
	static const int DIM = 3;

	Staggered(const size_type ddim[DIM]) {

		memcpy(dim, ddim, DIM * sizeof(dim[0]));
		for (int i = 0; i < DIM; i++) {
			memcpy(dims[i], ddim, DIM * sizeof(dims[i][0]));
			dims[i][i]++;
			size_type nelems = prod(dims[i], DIM);
			M[i] = new T[nelems];
			memset(&M[i][0], 0, nelems * sizeof(T));
		}
	}
	~Staggered() {
		for (int i = 0; i < DIM; i++) {
			delete[] M[i];
		}
	}

	Staggered<T, DIM>& operator=(const Staggered<T, DIM> &b) { // beware : does not allocate any memory! Not actually used, but don't want any mistake.

		memcpy(dim, b.dim, DIM * sizeof(dim[0]));

		int i;
#pragma omp parallel private(i)
		{
#pragma omp for schedule(static, 1) // seems somewhat better with this parallelization
			for (i = 0; i < DIM; i++) {
				memcpy(dims[i], b.dims[i], DIM * sizeof(dims[i][0]));
				memcpy(&M[i][0], &b.M[i][0], prod(dims[i], DIM) * sizeof(T));
			}
		}
		return *this;
	}


	size_type dim[DIM];
	size_type dims[DIM][DIM]; //first index = which function, second = which dimension
	T* M[DIM];
	char boundary;
};


// DCT class, that also stores the coefficients to solve for the Poisson equation
// Also only for DIM==3. Should specialize for dimensions 2 and 4 eventually...
template<typename T, int DIM>
class DCT {};

template<typename T>
class DCT<T, 3> {
public:
	static const int DIM = 3;

	DCT(size_type sizes[DIM], T* data) {
		memcpy(siz, sizes, DIM * sizeof(siz[0]));

		const ptrdiff_t L = siz[0], M = siz[1], N = siz[2];

		/* create plan for in-place REDFT10 x REDFT10 x REDFT10 */
		if (sizeof(T) == sizeof(double)) {
			fftw_plan_with_nthreads(omp_get_max_threads());
			plan = fftw_plan_r2r_3d(L, M, N, (double*)data, (double*)data, FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE); // if using FFTW_MEASURE, make sure the data can be overwritten safely ; otherwise FFTW_ESTIMATE 		 
			plan_inverse = fftw_plan_r2r_3d(L, M, N, (double*)data, (double*)data, FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);
		}
		else {
			fftwf_plan_with_nthreads(omp_get_max_threads());
			planf = fftwf_plan_r2r_3d(L, M, N, (float*)data, (float*)data, FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE); // if using FFTW_MEASURE, make sure the data can be overwritten safely ; otherwise FFTW_ESTIMATE 		 
			planf_inverse = fftwf_plan_r2r_3d(L, M, N, (float*)data, (float*)data, FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);
		}
		data_ = data;

		depn = new T[L];
		depm = new T[M];
		depr = new T[N];

		for (size_type i = 0; i < L; i++) {
			depn[i] = T((2.*cos(M_PI*i / (double)L) - 2.)*L*L);
		}
		for (size_type i = 0; i < M; i++) {
			depm[i] = T((2.*cos(M_PI*i / (double)M) - 2.)*M*M);
		}
		for (size_type i = 0; i < N; i++) {
			depr[i] = T((2.*cos(M_PI*i / (double)N) - 2.)*N*N);
		}
	}

	~DCT() {
		if (sizeof(T) == sizeof(double)) {
			fftw_destroy_plan(plan);
			fftw_destroy_plan(plan_inverse);
			fftw_cleanup_threads();
		}
		else {
			fftwf_destroy_plan(planf);
			fftwf_destroy_plan(planf_inverse);
			fftwf_cleanup_threads();
		}
		delete[] depn;
		delete[] depm;
		delete[] depr;
	}

	void dct() {
		if (sizeof(T) == sizeof(double)) {
			fftw_execute(plan);
		}
		else {
			fftwf_execute(planf);
		}
	}
	void idct() {
		if (sizeof(T) == sizeof(double)) {
			fftw_execute(plan_inverse);
		}
		else {
			fftwf_execute(planf_inverse);
		}
		size_type count = prod(siz, DIM);
		T factor = 1. / T(count*(1 << DIM));
		T* dataPtr = data_;
		for (size_type i = 0; i < count; i++) {
			*dataPtr *= factor;
			dataPtr++;
		}
	}

	size_type siz[DIM];
	fftw_plan plan, plan_inverse;
	fftwf_plan planf, planf_inverse;
	T* data_;

	T* depn, *depm, *depr;  // we store the coeffs for the Poisson solver here : it will avoid reallocating and recomputing them at each iteration
};

template<typename T, int DIM>
void interp(const Staggered<T, DIM>* U, T* V[DIM], bool has_zero_boundary) { // V of size [DIM][prod(U.dim)]

	switch (DIM) {
	case 2:

		if (has_zero_boundary) {
			for (size_type i = 0; i < U->dim[0]; i++) {
				if (i == 0) {
					for (size_type j = 0; j < U->dim[1]; j++) {
						V[0][i*U->dim[1] + j] = U->M[0][(i + 1)*U->dims[0][1] + j] * T(0.5);
					}
				}
				else {
					if (i == U->dim[0] - 1) {
						for (size_type j = 0; j < U->dim[1]; j++) {
							V[0][i*U->dim[1] + j] = U->M[0][(i)*U->dims[0][1] + j] * T(0.5);
						}
					}
					else {
						for (size_type j = 0; j < U->dim[1]; j++) {
							V[0][i*U->dim[1] + j] = (U->M[0][i*U->dims[0][1] + j] + U->M[0][(i + 1)*U->dims[0][1] + j])* T(0.5);
						}
					}
				}
				V[1][i*U->dim[1] + 0] = U->M[1][i*U->dims[1][1] + 1] * T(0.5);
				for (size_type j = 1; j < U->dim[1] - 1; j++) {
					V[1][i*U->dim[1] + j] = (U->M[1][i*U->dims[1][1] + j] + U->M[1][i*U->dims[1][1] + j + 1])* T(0.5);
				}
				V[1][i*U->dim[1] + U->dims[1][1] - 1] = U->M[1][i*U->dims[1][1] + U->dim[1] - 1] * T(0.5);
			}
		}
		else {
			for (size_type i = 0; i < U->dim[0]; i++) {
				for (size_type j = 0; j < U->dim[1]; j++) {
					V[0][i*U->dim[1] + j] = (U->M[0][i*U->dims[0][1] + j] + U->M[0][(i + 1)*U->dims[0][1] + j]) * T(0.5);
					V[1][i*U->dim[1] + j] = (U->M[1][i*U->dims[1][1] + j] + U->M[1][i*U->dims[1][1] + j + 1])* T(0.5);
				}
			}
		}

		break;

	case 3:
		size_type i;


		const size_type dim0 = U->dim[0];
		const size_type dim1 = U->dim[1];
		const size_type dim2 = U->dim[2];
		const size_type dim01 = U->dims[0][1];
		const size_type dim02 = U->dims[0][2];
		const size_type dim11 = U->dims[1][1];
		const size_type dim12 = U->dims[1][2];
		const size_type dim21 = U->dims[2][1];
		const size_type dim22 = U->dims[2][2];

		if (has_zero_boundary) {
#pragma omp parallel private(i)
			{
#pragma omp for schedule(static, dim0/omp_get_max_threads()) // in parallel: it seems to be quite slow in serie for large images ; computes the three functions separately: fewer cache misses, and easier to code
				for (i = 0; i < dim0; i++) {

					const size_type idim01 = i * dim01;
					const size_type idim11 = i * dim11;
					const size_type idim1dim2 = i * dim1*dim2;

					// first function
					T* Vptr = &V[0][idim1dim2];
					T* Uptr1 = &U->M[0][idim01*dim02];         // == [i][0][0]
					T* Uptr2 = &U->M[0][(idim01 + dim01)*dim02]; // == [i+1][0][0]  => to average in the first index
					if (i == 0) { // boundary = 0  (i==0)
						for (size_type j = 0; j < dim1*dim2; j++) {
							*Vptr = *Uptr2 * T(0.5);
							Vptr++;	Uptr2++;
						}
					}
					else {
						if (i == dim0 - 1) { // boundary = 0  (i==end)
							for (size_type j = 0; j < dim1*dim2; j++) {
								*Vptr = *Uptr1 * T(0.5);
								Vptr++;	Uptr1++;
							}
						}
						else {
							for (size_type j = 0; j < dim1*dim2; j++) {
								*Vptr = (*Uptr1 + *Uptr2)* T(0.5);
								Vptr++;	Uptr1++; Uptr2++;
							}
						}
					}

					// second function
					Vptr = &V[1][idim1dim2];
					Uptr2 = &U->M[1][(idim11 + 1)*dim12]; // == [i][1][0]  => to average in the second index
					for (size_type k = 0; k < dim2; k++) { // boundary = 0 (j==0)
						*Vptr = *Uptr2* T(0.5);
						Vptr++;
						Uptr2++;
					}
					Uptr1 = &U->M[1][idim11*dim12 + dim2];         // == [i][0][0] but dim2 iterations later, done just above
					for (size_type j = 0; j < (dim1 - 2)*dim2; j++) {
						*Vptr = (*Uptr1 + *Uptr2)* T(0.5);
						Vptr++;
						Uptr1++;
						Uptr2++;
					}
					for (size_type k = 0; k < dim2; k++) { // boundary = 0 (j==end)
						*Vptr = *Uptr1* T(0.5);
						Vptr++;
						Uptr1++;
					}


					// third function
					Vptr = &V[2][idim1dim2];
					Uptr1 = &U->M[2][i*dim21*dim22];         // == [i][0][0]
					Uptr2 = Uptr1 + 1; // == [i][0][1]  => to average in the third index
					for (size_type j = 0; j < dim1; j++) {
						*Vptr = *Uptr2* T(0.5);  // boundary = 0
						Vptr++;
						Uptr2++;
						Uptr1++;
						for (size_type k = 1; k < dim2 - 1; k++) {
							*Vptr = (*Uptr1 + *Uptr2)* T(0.5);
							Vptr++;
							Uptr2++;
							Uptr1++;
						}
						*Vptr = (*Uptr1)* T(0.5);  // boundary = 0
						Vptr++;
						Uptr2 += 2; // because U is actually bigger in the dimension dim2
						Uptr1 += 2;
					}
				}
			}

		}
		else {
#pragma omp parallel private(i)
			{
#pragma omp for schedule(static, dim0/omp_get_max_threads()) // the naive way ; only used once at the end, so no need to sacrifice readability here.
				for (i = 0; i < dim0; i++) {
					for (size_type j = 0; j < dim1; j++) {
						for (size_type k = 0; k < dim2; k++) {
							V[0][(i*dim1 + j)*dim2 + k] = (U->M[0][(i*dim01 + j)*dim02 + k] + U->M[0][((i + 1)*dim01 + j)*dim02 + k])* T(0.5);
							V[1][(i*dim1 + j)*dim2 + k] = (U->M[1][(i*dim11 + j)*dim12 + k] + U->M[1][(i*dim11 + (j + 1))*dim12 + k])* T(0.5);
							V[2][(i*dim1 + j)*dim2 + k] = (U->M[2][(i*dim21 + j)*dim22 + k] + U->M[2][(i*dim21 + j)*dim22 + k + 1])* T(0.5);
						}
					}
				}
			}
		}
		break;

	}
}


template<typename T, int DIM>
void interp_adj(T* V[DIM], Staggered<T, DIM>* U) { // V of size [DIM][dims], U already constructed

	switch (DIM) {
	case 2:
		for (size_type i = 0; i < U->dim[0] + 1; i++) {
			if (i == 0) {
				for (size_type j = 0; j < U->dim[1]; j++) {
					U->M[0][i*U->dims[0][1] + j] = (V[0][i*U->dim[1] + j])* T(0.5);
				}
			}
			else {
				if (i == U->dim[0]) {
					for (size_type j = 0; j < U->dim[1]; j++) {
						U->M[0][i*U->dims[0][1] + j] = (V[0][(i - 1)*U->dim[1] + j])* T(0.5);
					}
				}
				else {
					for (size_type j = 0; j < U->dim[1]; j++) {
						U->M[0][i*U->dims[0][1] + j] = (V[0][(i - 1)*U->dim[1] + j] + V[0][(i)*U->dim[1] + j])* T(0.5);
					}
				}
			}
		}
		for (size_type i = 0; i < U->dim[0]; i++) {
			U->M[1][i*U->dims[1][1] + 0] = (V[1][i*U->dim[1] + 0])* T(0.5);
			for (size_type j = 0; j < U->dim[1] - 1; j++) {
				U->M[1][i*U->dims[1][1] + j + 1] = (V[1][i*U->dim[1] + j] + V[1][i*U->dim[1] + j + 1])* T(0.5);
			}
			U->M[1][i*U->dims[1][1] + U->dim[1]] = (V[1][i*U->dim[1] + U->dim[1] - 1])* T(0.5);
		}
		break;

	case 3:

		size_type i;

		const size_type dim0 = U->dim[0];
		const size_type dim1 = U->dim[1];
		const size_type dim2 = U->dim[2];
		const size_type dim01 = U->dims[0][1];
		const size_type dim02 = U->dims[0][2];
		const size_type dim11 = U->dims[1][1];
		const size_type dim12 = U->dims[1][2];
		const size_type dim21 = U->dims[2][1];
		const size_type dim22 = U->dims[2][2];

#pragma omp parallel private(i)
		{
#pragma omp for schedule(static, (size_type)floor(1+(dim0+1.)/omp_get_max_threads())) // in parallel: it seems to be quite slow in serie for large images
			for (i = 0; i < dim0 + 1; i++) {

				//first function
				T* Uptr = &U->M[0][i*dim01*dim02]; //[i][0][0]
				T* Vptr1 = &V[0][i*dim1*dim2];		// [i][0][0]				
				T* Vptr2;
				if (i == 0) {
					for (size_type j = 0; j < dim1*dim2; j++) {
						*Uptr = (*Vptr1)* T(0.5);
						Uptr++; Vptr1++;
					}
				}
				else {
					Vptr2 = &V[0][(i - 1)*dim1*dim2];  // [i-1][0][0]
					if (i == dim0) {
						for (size_type j = 0; j < dim1*dim2; j++) {
							*Uptr = (*Vptr2)* T(0.5);
							Uptr++; Vptr2++;
						}
					}
					else {
						for (size_type j = 0; j < dim1*dim2; j++) {
							*Uptr = (*Vptr1 + *Vptr2)* T(0.5);
							Uptr++; Vptr1++; Vptr2++;
						}
					}
				}

				if (i != dim0) {
					// second function
					Uptr = &U->M[1][i*dim11*dim12];  // [i][0][0]
					Vptr1 = &V[1][i*dim1*dim2];      // [i][0][0]
					for (size_type k = 0; k < dim2; k++) {  // boundary
						*Uptr = *Vptr1* T(0.5);
						Uptr++;	Vptr1++;
					}
					Vptr2 = &V[1][i*dim1*dim2];  // [i][-1][0] but dim2 iterations later
					for (size_type j = 0; j < (dim1 - 1)*dim2; j++) {
						*Uptr = (*Vptr1 + *Vptr2)* T(0.5);
						Uptr++;	Vptr1++; Vptr2++;
					}
					for (size_type k = 0; k < dim2; k++) {	// boundary		
						*Uptr = *Vptr2* T(0.5);
						Uptr++;	Vptr2++;
					}


					// third function
					Uptr = &U->M[2][i*dim21*dim22];  // [i][0][0]
					Vptr1 = &V[2][i*dim1*dim2];	     // [i][0][0]  		
					for (size_type j = 0; j < dim1; j++) {
						*Uptr = (*Vptr1)* T(0.5);
						Uptr++;
						Vptr2 = Vptr1;  // Vptr2 = [i][j-1][0]
						Vptr1++;
						for (size_type k = 0; k < dim2 - 1; k++) {
							*Uptr = (*Vptr1 + *Vptr2)* T(0.5);
							Uptr++;	Vptr1++; Vptr2++;
						}
						*Uptr = (*Vptr2)* T(0.5);
						Uptr++;
					}
				}
			}
		}
		break;
	}

}


template<typename T, int DIM>
Staggered<T, DIM>* zero_boundary(Staggered<T, DIM>* U) {

	switch (DIM) {
	case 2:
		for (size_type i = 0; i < U->dims[0][1]; i++) {
			U->M[0][i] = 0;
			U->M[0][(U->dims[0][0] - 1)*U->dims[0][1] + i] = 0;
		}
		for (size_type i = 0; i < U->dims[1][0]; i++) {
			U->M[1][i*U->dims[1][1] + 0] = 0;
			U->M[1][i*U->dims[1][1] + U->dims[1][1] - 1] = 0;
		}

		break;
	case 3:
		const size_type dim0 = U->dim[0];
		const size_type dim1 = U->dim[1];
		const size_type dim2 = U->dim[2];
		const size_type dim00 = U->dims[0][0];
		const size_type dim01 = U->dims[0][1];
		const size_type dim02 = U->dims[0][2];
		const size_type dim10 = U->dims[1][0];
		const size_type dim11 = U->dims[1][1];
		const size_type dim12 = U->dims[1][2];
		const size_type dim20 = U->dims[2][0];
		const size_type dim21 = U->dims[2][1];
		const size_type dim22 = U->dims[2][2];

		// first function
		T* Uptr1 = &U->M[0][0];   // [0][0][0]
		T* Uptr2 = &U->M[0][(dim00 - 1)*dim01*dim02]; // [end][0][0]
		for (size_type i = 0; i < dim01*dim02; i++) {
			*Uptr1 = 0;  Uptr1++;
			*Uptr2 = 0;  Uptr2++;
		}

		// second function
		Uptr1 = &U->M[1][0];   // [0][0][0]
		Uptr2 = &U->M[1][(dim11 - 1)*dim12];   // [0][end][0]
		for (size_type i = 0; i < dim10; i++) {
			for (size_type j = 0; j < dim12; j++) {
				*Uptr1 = 0; Uptr1++;
				*Uptr2 = 0; Uptr2++;
			}
			Uptr1 += dim12 * (dim11 - 1); // jump to the next slice
			Uptr2 += dim12 * (dim11 - 1);
		}

		// third function
		Uptr1 = &U->M[2][0];   // [0][0][0]
		Uptr2 = &U->M[2][dim22 - 1];   // [0][0][end]
		for (size_type i = 0; i < dim20*dim21; i++) {
			*Uptr1 = 0; Uptr1 += dim22; // beware of the cache misses :s
			*Uptr2 = 0; Uptr2 += dim22;
		}
	}
	return U;
}

template<typename T, int DIM>
void pd_operator_direction_p1(const Staggered<T, DIM>* X, T* Y[DIM]) { // Y already allocated
	interp<T, DIM>(X, Y, true);
}
template<typename T, int DIM>
void pd_operator_direction_m1(T* X[DIM], Staggered<T, DIM>* Y) { // Y already created
	interp_adj<T, DIM>(X, Y);
	zero_boundary(Y);
}


#ifndef OTFP_SSE

#ifdef EXACT_ROOT
double poly_root_new4_first_root_normalized(double p1, double p2, double p3) { // degree 3 only ; DIM==4 without imaginary part, gives the first root,  p0= 1 
	double p12 = p1 * p1 / 3.;
	double p = -p12 + p2;
	double q = p1 / 3.*(2. / 3.*p12 - p2) + p3;
	double delt = q * q + 4. / 27.*p*p*p;

	if (delt > 0.) {
		double sd = sqrt(delt);
		double u = cube_root_double((-q + sd)*0.5);
		double v = cube_root_double((-q - sd)*0.5);
		return  u + v - p1 / 3.;
	}
	if (delt < 0.) {
		double u = cube_root_double(-q / 2.);
		return u + u - p1 / 3.;
	}
	// if delta == 0    	
	return 3.*q / p - p1 / 3.;
}


float poly_root_new4_first_root_normalized(float p1, float p2, float p3) { // degree 3 only ; DIM==4 without imaginary part, gives the first root,  p0= 1 
	float p12 = p1 * p1 / 3.f;
	float p = -p12 + p2;
	float q = p1 / 3.f*(2.f / 3.f*p12 - p2) + p3;
	float delt = q * q + 4.f / 27.f*p*p*p;

	if (delt > 0.f) {
		float sd = sqrtf(delt);
		float u = cube_root_float((-q + sd)*0.5f);
		float v = cube_root_float((-q - sd)*0.5f);
		return  u + v - p1 / 3.f;
	}
	if (delt < 0.f) {
		float u = cube_root_float(-q / 2.f);
		return 2.f*u - p1 / 3.f;
	}
	// if delta == 0    	
	return 3.f*q / p - p1 / 3.f;
}
#else

template<typename T>
T poly_root_new4_first_root_normalized(T v2, T p1, T p2, T p3) { // degree 3 only ; DIM==4 without imaginary part, gives the first root,  p0= 1 ;   Newton's method. If initialized with the previous value, no need for more than 2 iterations (manually unrolled here). 
	T x = v2;
	T y = x * (x*(x + p1) + p2) + p3;
	T yp = x * (x*3. + 2.*p1) + p2;
	x -= y / yp;
	y = x * (x*(x + p1) + p2) + p3;
	yp = x * (x*3. + 2.*p1) + p2;
	x -= y / yp;
	y = x * (x*(x + p1) + p2) + p3;
	yp = x * (x*3. + 2.*p1) + p2;
	x -= y / yp;
	return x;
}

#endif


template<typename T>
void proxJ(T &v0, T &v1, T &v2, T gamm, T epsilon, T alpha, T obstacle) { // assumes alpha==1 ; v2=f0 ; also assumes DIM=3

#ifdef EXACT_ROOT
	T f = poly_root_new4_first_root_normalized(T(4.*gamm - v2), T(4.*gamm*(gamm - v2)), T(-gamm * (v0*v0 + v1 * v1 + 4 * gamm*v2)));
#else
	T f = poly_root_new4_first_root_normalized(v2, T(4.*gamm - v2), T(4.*gamm*(gamm - v2)), T(-gamm * (v0*v0 + v1 * v1 + 4 * gamm*v2)));
#endif

	f = std::max(f, epsilon);
	if (obstacle > 0) f = epsilon;
	//T coeff = 1./(1.+2.*gamm/pow(f, alpha));
	T coeff = 1. / (1. + 2.*gamm / f); // beware: in our case, alpha==1
	v0 *= coeff;
	v1 *= coeff;
	v2 = f;
}
template<typename T, int DIM> // == ProxFS
void compute_dual_prox(T &v0, T &v1, T &v2, T sigma, T epsilon, T alpha, T obstacle, T b) { // b=0 almost everywhere ; assumes DIM=3
	sigma = 1. / sigma; // two divisions instead of four
	T v0b = v0 * sigma;
	T v1b = v1 * sigma;
	T v2b = v2 * sigma + b;
	proxJ<T>(v0b, v1b, v2b, sigma, epsilon, alpha, obstacle);
	sigma = 1. / sigma;
	v0 -= v0b * sigma;
	v1 -= v1b * sigma;
	v2 -= (v2b - b)*sigma;
}
#endif

template<typename T, int DIM>
void poisson3d_Neumann(T* divu, size_type* dims, DCT<T, DIM> &dct) { // l = 1  ,  inplace

	size_type N = dims[0];
	size_type M = dims[1];
	size_type R = dims[2];

	dct.dct();

	size_type i;
#pragma omp parallel private(i)
	{
#pragma omp for schedule(static, N/omp_get_max_threads()) 
		for (i = 0; i < N; i++) {
			T* divuPtr = &divu[i*M*R];
			T coeff1 = dct.depn[i];
			for (size_type j = 0; j < M; j++) {
				T coeff2 = dct.depm[j];
				for (size_type k = 0; k < R; k++) {
					T denom2 = coeff1 + coeff2 + dct.depr[k];
					if (denom2 == 0.) denom2 = 1.;
					*divuPtr /= -denom2;
					divuPtr++;
				}
			}
		}
	}

	dct.idct();
}


template<typename T, int DIM>
void debugSums(Staggered<T, DIM> *U) {
	double eps = 1E-5; // with our truncated Newton method, tests2 and testt2 can be around 1E-6
	double tests0 = 0, tests1 = 0, tests2 = 0;
	double testt0 = 0, testt1 = 0, testt2 = 0;
	for (size_type j = 0; j < U->dims[0][1]; j++) {
		for (size_type k = 0; k < U->dims[0][2]; k++) {
			tests0 += U->M[0][(0 * U->dims[0][1] + j)*U->dims[0][2] + k] * U->M[0][(0 * U->dims[0][1] + j)*U->dims[0][2] + k];
			testt0 += U->M[0][((U->dims[0][0] - 1)*U->dims[0][1] + j)*U->dims[0][2] + k] * U->M[0][((U->dims[0][0] - 1)*U->dims[0][1] + j)*U->dims[0][2] + k];
		}
	}
	tests0 = sqrt(tests0);
	testt0 = sqrt(testt0);
	for (size_type i = 0; i < U->dims[1][0]; i++) {
		for (size_type k = 0; k < U->dims[1][2]; k++) {
			tests1 += U->M[1][(i*U->dims[1][1] + 0)*U->dims[1][2] + k] * U->M[1][(i*U->dims[1][1] + 0)*U->dims[1][2] + k];
			testt1 += U->M[1][(i*U->dims[1][1] + U->dims[1][1] - 1)*U->dims[1][2] + k] * U->M[1][(i*U->dims[1][1] + U->dims[1][2] - 1)*U->dims[1][2] + k];
		}
	}
	tests1 = sqrt(tests1);
	testt1 = sqrt(testt1);
	for (size_type i = 0; i < U->dims[2][0]; i++) {
		for (size_type j = 0; j < U->dims[2][1]; j++) {
			tests2 += U->M[2][(i*U->dims[2][1] + j)*U->dims[2][2] + 0];
			testt2 += U->M[2][(i*U->dims[2][1] + j)*U->dims[2][2] + U->dims[2][2] - 1];
		}
	}
	tests2 = abs(tests2 - 1.);
	testt2 = abs(testt2 - 1.);
	if (tests0 > eps || testt0 > eps || tests1 > eps || testt1 > eps || tests2 > eps || testt2 > eps) {
		std::cout << "bug in projection" << std::endl;
	}
}

template<typename T, int DIM> // DIM == 3 only
void div_proj(Staggered<T, DIM> *U, T* div, DCT<T, DIM> &dct) { // supposes lx==1  div should be allocated of size nelems = prod(U->dim, DIM);

#ifdef _DEBUG
	debugSums(U);
#endif

	size_type i;
	const size_type dim0 = U->dim[0];
	const size_type dim1 = U->dim[1];
	const size_type dim2 = U->dim[2];
	const size_type dim00 = U->dims[0][0];
	const size_type dim01 = U->dims[0][1];
	const size_type dim02 = U->dims[0][2];
	const size_type dim10 = U->dims[1][0];
	const size_type dim11 = U->dims[1][1];
	const size_type dim12 = U->dims[1][2];
	const size_type dim20 = U->dims[2][0];
	const size_type dim21 = U->dims[2][1];
	const size_type dim22 = U->dims[2][2];

#pragma omp parallel private(i)
	{
#pragma omp for schedule(static, dim0/omp_get_max_threads()) 
		for (i = 0; i < dim0; i++) {

			T* divPtr = &div[i*dim1*dim2];
			T* Uptr01 = &U->M[0][i*dim01*dim02]; // U0: [i][0][0]
			T* Uptr02 = Uptr01 + dim01 * dim02; // U0: [i+1][0][0]
			T* Uptr11 = &U->M[1][i*dim11*dim12]; // U1: [i][0][0]
			T* Uptr12 = Uptr11 + dim12; // U1: [i][1][0]
			T* Uptr21 = &U->M[2][i*dim21*dim22]; // U2: [i][0][0]
			T* Uptr22 = Uptr21 + 1; // U2: [i][0][1]

			for (size_type j = 0; j < dim1; j++) {
				for (int k = 0; k < dim2; k++) {
					*divPtr = (*Uptr01 - *Uptr02) * dim0 + (*Uptr11 - *Uptr12) * dim1 + (*Uptr21 - *Uptr22) * dim2;  // directly the opposite of the divergence
					divPtr++;
					Uptr01++; Uptr02++;
					Uptr11++; Uptr12++;
					Uptr21++; Uptr22++;
				}
				Uptr21++; Uptr22++;	// the third dimension of U2 is larger by 1
			}
		}
	}

	poisson3d_Neumann<T, DIM>(div, U->dim, dct);


	// update of U
#pragma omp parallel private(i)
	{
#pragma omp for schedule(static, dim0/omp_get_max_threads()) 
		for (i = 0; i < dim0; i++) {

			// first function
			if (i != dim0 - 1) {
				T* Uptr = &U->M[0][(i + 1)*dim01*dim02];  // [i+1][0][0]
				T* divPtr1 = &div[i*dim1*dim2];  // [i][0][0]
				T* divPtr2 = divPtr1 + dim1 * dim2;  // [i+1][0][0]
				for (size_type j = 0; j < dim1*dim2; j++) {
					*Uptr += (*divPtr1 - *divPtr2)*dim0;
					Uptr++;
					divPtr1++; divPtr2++;
				}
			}

			// second function
			T* Uptr = &U->M[1][i*dim11*dim12 + dim12];  // [i][1][0]
			T* divPtr1 = &div[i*dim1*dim2];  // [i][0][0]
			T* divPtr2 = divPtr1 + dim2;  // [i][1][0]
			for (size_type j = 0; j < (dim1 - 1)*dim2; j++) {
				*Uptr += (*divPtr1 - *divPtr2)*dim1;
				Uptr++;
				divPtr1++; divPtr2++;
			}

			//third function
			Uptr = &U->M[2][i*dim21*dim22 + 1];  // [i][0][1]
			divPtr1 = &div[i*dim1*dim2];  // [i][0][0]
			divPtr2 = divPtr1 + 1;  // [i][0][1]
			for (size_type j = 0; j < dim1; j++) {
				for (size_type k = 0; k < dim2 - 1; k++) {
					*Uptr += (*divPtr1 - *divPtr2)*dim2;
					Uptr++;
					divPtr1++; divPtr2++;
				}
				Uptr += 2; // because the third dimension of U2 is bigger and we only looped dim2-1 times
				divPtr1++; // because we only looped dim2-1 times
				divPtr2++;
			}
		}
	}
}



#ifdef OTFP_SSE
template<int DIM>
void update_y(float* y[DIM], float sigma, float epsilon, float alpha, const float* obstacle, float* b, float* Kx1[DIM], size_type nelems) {

	size_type j;

#pragma omp parallel private(j)
	{
		simd_float sigma2 = simd_set1_f(sigma);
		simd_float alpha2 = simd_set1_f(alpha);
		simd_float epsilon2 = simd_set1_f(epsilon);
		simd_float zero = simd_set1_f(0.);
		simd_float one = simd_set1_f(1.);
		simd_float two = simd_set1_f(2.);
		simd_float three = simd_set1_f(3.);
		simd_float mone = simd_set1_f(-1.);
		simd_float four = simd_set1_f(4.);

#pragma omp for schedule(static, nelems/VECSIZEFLOAT/omp_get_max_threads()) 
		for (j = 0; j < nelems; j += VECSIZEFLOAT) {

			simd_float v0 = simd_load_f(&y[0][j]);
			simd_float v1 = simd_load_f(&y[1][j]);
			simd_float v2 = simd_load_f(&y[2][j]);
			simd_float Kx10 = simd_load_f(&Kx1[0][j]);
			simd_float Kx11 = simd_load_f(&Kx1[1][j]);
			simd_float Kx12 = simd_load_f(&Kx1[2][j]);
			simd_float b2 = simd_load_f(&b[j]);
			simd_float obstacle2 = simd_load_f(&obstacle[j]);

			Kx10 = simd_mul_f(sigma2, Kx10);
			Kx11 = simd_mul_f(sigma2, Kx11);
			Kx12 = simd_mul_f(sigma2, Kx12);
			v0 = simd_add_f(v0, Kx10);
			v1 = simd_add_f(v1, Kx11);
			v2 = simd_add_f(v2, Kx12);

			// compute_dual_prox
			sigma2 = simd_div_f(one, sigma2);
			simd_float v0b = simd_mul_f(v0, sigma2);
			simd_float v1b = simd_mul_f(v1, sigma2);
			simd_float v2b = simd_add_f(simd_mul_f(v2, sigma2), b2);

			// proxJ
			// polynomial coefficients
			simd_float p1 = simd_sub_f(simd_mul_f(four, sigma2), v2b);
			simd_float p2 = simd_mul_f(four, simd_mul_f(sigma2, simd_sub_f(sigma2, v2b)));
			simd_float p3 = simd_mul_f(mone, simd_mul_f(sigma2, simd_add_f(simd_add_f(simd_mul_f(v0b, v0b), simd_mul_f(v1b, v1b)), simd_mul_f(four, simd_mul_f(sigma2, v2b)))));

			// Newton (3 steps) ; directly performed on x = v2b					
			simd_float yval = simd_add_f(simd_mul_f(v2b, simd_add_f(simd_mul_f(v2b, simd_add_f(v2b, p1)), p2)), p3);
			simd_float yp = simd_add_f(simd_mul_f(v2b, simd_add_f(simd_mul_f(three, v2b), simd_mul_f(two, p1))), p2);
			v2b = simd_sub_f(v2b, simd_div_f(yval, yp));
			yval = simd_add_f(simd_mul_f(v2b, simd_add_f(simd_mul_f(v2b, simd_add_f(v2b, p1)), p2)), p3);
			yp = simd_add_f(simd_mul_f(v2b, simd_add_f(simd_mul_f(three, v2b), simd_mul_f(two, p1))), p2);
			v2b = simd_sub_f(v2b, simd_div_f(yval, yp));
			yval = simd_add_f(simd_mul_f(v2b, simd_add_f(simd_mul_f(v2b, simd_add_f(v2b, p1)), p2)), p3);    // last step optional (feel free to remove), but otherwise can produce large artifacts with number of iterations is large
			yp = simd_add_f(simd_mul_f(v2b, simd_add_f(simd_mul_f(three, v2b), simd_mul_f(two, p1))), p2);
			v2b = simd_sub_f(v2b, simd_div_f(yval, yp));

			// clamp
			v2b = simd_max_f(v2b, epsilon2);
			simd_float mask = simd_gt_f(obstacle2, zero); // if (obstacle>0) f=epsilon;
			v2b = simd_or_f(simd_and_f(mask, epsilon2), simd_andnot_f(mask, v2b));
			//result
			simd_float coeff = simd_div_f(one, (simd_add_f(one, simd_mul_f(two, simd_div_f(sigma2, v2b)))));
			v0b = simd_mul_f(v0b, coeff);
			v1b = simd_mul_f(v1b, coeff);
			// end proxJ

			sigma2 = simd_div_f(one, sigma2);
			v0 = simd_sub_f(v0, simd_mul_f(v0b, sigma2));
			v1 = simd_sub_f(v1, simd_mul_f(v1b, sigma2));
			v2 = simd_sub_f(v2, simd_mul_f(simd_sub_f(v2b, b2), sigma2));

			simd_store_f(&y[0][j], v0);
			simd_store_f(&y[1][j], v1);
			simd_store_f(&y[2][j], v2);
		}
	}
}


template<int DIM>
void update_y(double* y[DIM], double sigma, double epsilon, double alpha, const double* obstacle, double* b, double* Kx1[DIM], size_type nelems) {

	size_type j;

#pragma omp parallel private(j)
	{
		simd_double sigma2 = simd_set1(sigma);
		simd_double alpha2 = simd_set1(alpha);
		simd_double epsilon2 = simd_set1(epsilon);
		simd_double zero = simd_set1(0.);
		simd_double one = simd_set1(1.);
		simd_double two = simd_set1(2.);
		simd_double three = simd_set1(3.);
		simd_double mone = simd_set1(-1.);
		simd_double four = simd_set1(4.);

#pragma omp for schedule(static, nelems/VECSIZEDOUBLE/omp_get_max_threads()) 
		for (j = 0; j < nelems; j += VECSIZEDOUBLE) {

			simd_double v0 = simd_load(&y[0][j]);
			simd_double v1 = simd_load(&y[1][j]);
			simd_double v2 = simd_load(&y[2][j]);
			simd_double Kx10 = simd_load(&Kx1[0][j]);
			simd_double Kx11 = simd_load(&Kx1[1][j]);
			simd_double Kx12 = simd_load(&Kx1[2][j]);
			simd_double b2 = simd_load(&b[j]);
			simd_double obstacle2 = simd_load(&obstacle[j]);

			Kx10 = simd_mul(sigma2, Kx10);
			Kx11 = simd_mul(sigma2, Kx11);
			Kx12 = simd_mul(sigma2, Kx12);
			v0 = simd_add(v0, Kx10);
			v1 = simd_add(v1, Kx11);
			v2 = simd_add(v2, Kx12);

			// compute_dual_prox
			sigma2 = simd_div(one, sigma2);
			simd_double v0b = simd_mul(v0, sigma2);
			simd_double v1b = simd_mul(v1, sigma2);
			simd_double v2b = simd_add(simd_mul(v2, sigma2), b2);

			// proxJ
			// polynomial coefficients
			simd_double p1 = simd_sub(simd_mul(four, sigma2), v2b);
			simd_double p2 = simd_mul(four, simd_mul(sigma2, simd_sub(sigma2, v2b)));
			simd_double p3 = simd_mul(mone, simd_mul(sigma2, simd_add(simd_add(simd_mul(v0b, v0b), simd_mul(v1b, v1b)), simd_mul(four, simd_mul(sigma2, v2b)))));

			// Newton (3 steps) ; directly performed on x = v2b					
			simd_double yval = simd_add(simd_mul(v2b, simd_add(simd_mul(v2b, simd_add(v2b, p1)), p2)), p3);
			simd_double yp = simd_add(simd_mul(v2b, simd_add(simd_mul(three, v2b), simd_mul(two, p1))), p2);
			v2b = simd_sub(v2b, simd_div(yval, yp));
			yval = simd_add(simd_mul(v2b, simd_add(simd_mul(v2b, simd_add(v2b, p1)), p2)), p3);
			yp = simd_add(simd_mul(v2b, simd_add(simd_mul(three, v2b), simd_mul(two, p1))), p2);
			v2b = simd_sub(v2b, simd_div(yval, yp));
			yval = simd_add(simd_mul(v2b, simd_add(simd_mul(v2b, simd_add(v2b, p1)), p2)), p3);   // last step optional (feel free to remove), but otherwise can produce large artifacts with number of iterations is large
			yp = simd_add(simd_mul(v2b, simd_add(simd_mul(three, v2b), simd_mul(two, p1))), p2);
			v2b = simd_sub(v2b, simd_div(yval, yp));

			// clamp
			v2b = simd_max(v2b, epsilon2);
			simd_double mask = simd_gt(obstacle2, zero); // if (obstacle>0) f=epsilon;
			v2b = simd_or(simd_and(mask, epsilon2), simd_andnot(mask, v2b));
			//result
			simd_double coeff = simd_div(one, (simd_add(one, simd_mul(two, simd_div(sigma2, v2b)))));
			v0b = simd_mul(v0b, coeff);
			v1b = simd_mul(v1b, coeff);
			// end proxJ

			sigma2 = simd_div(one, sigma2);
			v0 = simd_sub(v0, simd_mul(v0b, sigma2));
			v1 = simd_sub(v1, simd_mul(v1b, sigma2));
			v2 = simd_sub(v2, simd_mul(simd_sub(v2b, b2), sigma2));

			simd_store(&y[0][j], v0);
			simd_store(&y[1][j], v1);
			simd_store(&y[2][j], v2);
		}
	}
}
#else
template<typename T, int DIM>
void update_y(T* y[DIM], T sigma, T epsilon, T alpha, const T* obstacle, T* b, T* Kx1[DIM], size_type nelems) {

	size_type j;

#pragma omp parallel private(j)
	{
#pragma omp for schedule(static, nelems/omp_get_max_threads()) 
		for (j = 0; j < nelems; j++) {
			y[0][j] += sigma * Kx1[0][j];
			y[1][j] += sigma * Kx1[1][j];
			y[2][j] += sigma * Kx1[2][j];
			compute_dual_prox<T, DIM>(y[0][j], y[1][j], y[2][j], sigma, epsilon, alpha, obstacle[j], b[j]);   // compute_dual_prox = proxFS
		}
	}
}
#endif

template<typename T, int DIM>
void perform_primal_dual(Staggered<T, DIM>* x, const T* f0, const T* f1, const T* obstacle, int niter = 2000, T epsilon = 1E-8, T alpha = 1., T theta = 1., T sigma = 85., int L = 1) { //only handles alpha = 1.

	T tau = T(0.99 / (sigma*L));
	size_type nelems = prod(x->dim, DIM), j;

	Staggered<T, DIM> xold(x->dim); // because our assignment operator will not allocate memory!
	Staggered<T, DIM> x1(x->dim);
	Staggered<T, DIM> KSy(x->dim);

#ifdef _DEBUG
	debugSums(x);
#endif

	x1 = *x;
	xold = *x;
	T* y[DIM];
	T* Kx1[DIM];

#ifdef OTFP_SSE
	T* b = (T*)malloc_simd(nelems * sizeof(T), ALIGN);
	for (int i = 0; i < DIM; i++) {
		y[i] = (T*)malloc_simd(nelems * sizeof(T), ALIGN);
		Kx1[i] = (T*)malloc_simd(nelems * sizeof(T), ALIGN);
	}
#else
	T* b = new T[nelems];
	for (int i = 0; i < DIM; i++) {
		y[i] = new T[nelems];
		Kx1[i] = new T[nelems];
	}
#endif
	memset(b, 0, nelems * sizeof(T));

	pd_operator_direction_p1<T, DIM>(x, y);

	for (size_type i = 0; i < x->dim[0] * x->dim[1]; i++) {
		b[i*x->dim[2] + 0] = f0[i] * T(0.5);
		b[i*x->dim[2] + x->dim[2] - 1] = f1[i] * T(0.5);
	}

	T* div;
	if (sizeof(T) == sizeof(double)) {
		div = (T*)fftw_malloc(nelems * sizeof(T));  // for faster access (aligned memory) ; storage space for the divergence ; also referenced in the FFTW class below
	}
	else {
		div = (T*)fftwf_malloc(nelems * sizeof(T)); // float version
	}


	int ret = fftw_init_threads();
	DCT<T, DIM> dct(x->dim, div);

	for (int i = 0; i < niter; i++) {
		if (i % (std::max(1, niter / 100)) == 0) {
			std::cout << i * 100. / niter << std::endl;
		}

		// ProxFS => compute_dual_prox
		pd_operator_direction_p1<T, DIM>(&x1, Kx1);

#ifdef OTFP_SSE
		update_y<DIM>(y, sigma, epsilon, alpha, obstacle, b, Kx1, nelems);
#else
		update_y<T, DIM>(y, sigma, epsilon, alpha, obstacle, b, Kx1, nelems);
#endif

		// ProxG
		pd_operator_direction_m1<T, DIM>(y, &KSy);

#pragma omp parallel private(j)
		{
#pragma omp for schedule(static, 1) 
			for (j = 0; j < DIM; j++) {
				T* KsyPtr = &KSy.M[j][0];
				T* xPtr = &x->M[j][0];
				T* xoldPtr = &xold.M[j][0];
				size_type n = prod(x->dims[j], DIM);
				for (size_type k = 0; k < n; k++) {
					*xoldPtr = *xPtr;
					*xPtr -= *KsyPtr*tau;
					xPtr++; xoldPtr++; KsyPtr++;
				}
			}
		}


		div_proj(x, div, dct);

#pragma omp parallel private(j)
		{
#pragma omp for schedule(static, 1) 
			for (j = 0; j < DIM; j++) {
				size_type n = prod(x->dims[j], DIM);
				T* x1Ptr = &x1.M[j][0];
				T* xPtr = &x->M[j][0];
				T* xoldPtr = &xold.M[j][0];
				for (size_type k = 0; k < n; k++) {
					*x1Ptr = *xPtr + theta * (*xPtr - *xoldPtr);
					x1Ptr++; xPtr++; xoldPtr++;
				}
			}
		}
	}

#ifdef OTFP_SSE
	for (int i = 0; i < DIM; i++) {
		free_simd(y[i]);
		free_simd(Kx1[i]);
	}
	free_simd(b);
#else
	for (int i = 0; i < DIM; i++) {
		delete[] y[i];
		delete[] Kx1[i];
	}
	delete[] b;
#endif


	if (sizeof(T) == sizeof(double)) {
		fftw_free(div);
	}
	else {
		fftwf_free(div);
	}

}

#ifdef AVX_SUPPORT
#define OSXSAVEFlag (1UL<<27)
#define AVXFlag     ((1UL<<28)|OSXSAVEFlag)

bool DetectFeature(unsigned int feature)
{
	int CPUInfo[4], InfoType = 1;
	__cpuidex(CPUInfo, 1, 1);       // read the desired CPUID format
	unsigned int ECX = CPUInfo[2];  // the output of CPUID in the ECX register. 
	if ((ECX & feature) != feature) // Missing feature 
		return false;
	__int64 val = _xgetbv(0);       // read XFEATURE_ENABLED_MASK register
	if ((val & 6) != 6)               // check OS has enabled both XMM and YMM support.
		return false;
	return true;
}
#endif 


int OT_proximal( string OTF1_filename = "data/cloud1.bmp" , string OTF2_filename = "data/cloud2.bmp" , string OTF_outPrefix = "data/out")
{
	int niter = 2000;
	data_type alpha = 1.; // only supports alpha=1
	data_type theta = 1.; // for the update
	data_type sigma = 85.;
	int L = 1; // step size in the grid

	size_type M = 128; // height (Gaussian case ; otherwise, overwritten by image height)
	size_type N = 128; // width (Gaussian case ; otherwise, overwritten by image width)	
	size_type K = 32;  // number of time steps

	const char test_case = 'i';  // 'g' for Gaussian ; 'i' for loading an image.
	const char *image_nameA;
	const char *image_nameB;
	image_nameA = OTF1_filename.c_str();
	image_nameB = OTF2_filename.c_str();
	

	/////////////////////////  end of runtime options ////////////////////////////



#ifdef AVX_SUPPORT
	bool avx_supported = DetectFeature(AVXFlag);
	if (!avx_supported) {
		std::cout << "AVX not supported" << std::endl;
		return -1;
	}
#endif




	std::vector<data_type> f0;
	std::vector<data_type> f1;
	data_type variance;
	data_type sum_pixelsA(0), sum_pixelsB(0);

	switch (test_case) {
	case 'g':
		variance = M * N / ((data_type)(6.*6.));
		f0.resize(M*N);
		f1.resize(M*N);
		for (size_type i = 0; i < M; i++) {
			for (size_type j = 0; j < N; j++) {
				f0[i*N + j] = data_type(1E-12 + 255.*exp(-((i - M / 3.)*(i - M / 3.) + (j - N / 3.)*(j - N / 3.)) / (2.*variance)));
				f1[i*N + j] = data_type(1E-12 + 255.*exp(-((i - 2.*M / 3.)*(i - 2.*M / 3.) + (j - 2.*N / 3.)*(j - 2.*N / 3.)) / (2.*variance)));
				sum_pixelsA += f0[i*N + j];
				sum_pixelsB += f1[i*N + j];
			}
		}
		break;
	case 'i':
		sum_pixelsA = load_grayscale_image(image_nameA, f0, N, M);
		sum_pixelsB = load_grayscale_image(image_nameB, f1, N, M);
		break;
	}

#ifdef OTFP_SSE
	data_type* obstacle = (data_type*)malloc_simd(M*N*K * sizeof(data_type), ALIGN);
#else
	data_type* obstacle = new data_type[M*N*K];
#endif
	memset(obstacle, 0, M*N*K * sizeof(obstacle[0]));

	//for ( int u = 20 ; u < 32;u++)
	//	for (int v = 20; v < 120; v++)
	//		for (int n = 0; n < K; n++)
	//		{
	//			int index = (n * 128 * 128) + (v * 128) + u;
	//			obstacle[index ] = 1;
	//		}

	size_type d[3] = { M, N, K };
	data_type epsilon = 1E9;
	for (size_type i = 0; i < M*N; i++) {
		f0[i] /= sum_pixelsA;
		f1[i] /= sum_pixelsB;
		epsilon = std::min(epsilon, f0[i]);//std::min(epsilon, f0[i]);
	}
	save_grayscale_image("data/fieldA.bmp", &f0[0], N, M, (data_type)(sum_pixelsA*(1. - 0) + 0 * sum_pixelsB)); //0.8E5*255./64.);
	save_grayscale_image("data/fieldB.bmp", &f1[0], N, M, (data_type)(sum_pixelsA*(1. - 1) + 1 * sum_pixelsB)); //0.8E5*255./64.);

	Staggered<data_type, 3> U0(d);
	for (size_type i = 0; i < d[0]; i++) {
		for (size_type j = 0; j < d[1]; j++) {
			for (size_type k = 0; k < d[2] + 1; k++) {
				data_type t = k / ((data_type)d[2]);
				U0.M[2][(i*U0.dims[2][1] + j)*U0.dims[2][2] + k] = (1 - t)*f0[i*d[1] + j] + t * f1[i*d[1] + j];
			}
		}
	}

	// Primal Dual
	PerfChrono chrono;
	chrono.Start();
	perform_primal_dual(&U0, &f0[0], &f1[0], obstacle, niter, epsilon, alpha, theta, sigma, L);
	std::cout << chrono.GetDiffMs() / 1000. << std::endl;


	// interpolate and saves result
	std::vector<data_type> img(M*N);
	data_type* V[3];
	for (int i = 0; i < 3; i++) {
		V[i] = new data_type[M*N*K];
	}
	interp<data_type, 3>(&U0, V, false);
	for (size_type k = 0; k < K; k++) {

		for (size_type i = 0; i < M; i++) {
			for (size_type j = 0; j < N; j++) {
				img[i*N + j] = V[2][(i*N + j)*K + k];
			}
		}
		data_type t = k / (data_type)(K - 1);
		std::ostringstream ss;
		ss << OTF_outPrefix << k << ".bmp";
		save_grayscale_image(ss.str().c_str(), &img[0], N, M, (data_type)(sum_pixelsA*(1. - t) + t * sum_pixelsB)); // 0.8E5*255./64.); //(data_type)(sum_pixelsA*(1.-t)+t*sum_pixelsB));
	}

#ifdef OTFP_SSE
	free_simd(obstacle);
#else
	delete[] obstacle;
#endif
}


#endif // !_OPTIMAL_TRANSPORT_PROXIMAL

