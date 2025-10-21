/// author: Nicolas Bonneel (nbonneel@seas.harvard.edu)
// small helper for SSE and AVX

#pragma once

#include <malloc.h>
#include <xmmintrin.h> 
#include <emmintrin.h>

#include <intrin.h>

#ifndef _MSC_VER
#include <intrin.h>
#endif 

#ifdef AVX_SUPPORT
#define OTFP_SSE
#endif

void * malloc_simd(const size_t size, const size_t alignment)
{
#if defined WIN32           // WIN32
    return _aligned_malloc(size, alignment);
#elif defined __linux__     // Linux
    return memalign(alignment, size);
#elif defined __MACH__      // Mac OS X
    return malloc(size);
#else                       // other (use valloc for page-aligned memory)
    return valloc(size);
#endif
}

void free_simd(void* mem)
{
#if defined WIN32           // WIN32
    return _aligned_free(mem);
#elif defined __linux__     // Linux
    free(mem);
#elif defined __MACH__      // Mac OS X
    free(mem);
#else                       // other (use valloc for page-aligned memory)
    free(mem);
#endif
}


#ifdef OTFP_SSE

#ifdef AVX_SUPPORT
#define ALIGN 32
#define VECSIZEDOUBLE 4
#define VECSIZEFLOAT 8
#define simd_add(x,y) _mm256_add_pd(x,y)
#define simd_sub(x,y) _mm256_sub_pd(x,y)
#define simd_mul(x,y) _mm256_mul_pd(x,y)
#define simd_div(x,y) _mm256_div_pd(x,y)
#define simd_max(x,y) _mm256_max_pd(x,y)
#define simd_load(x) _mm256_load_pd(x)
#define simd_store(x,y) _mm256_store_pd(x,y)
#define simd_set1(x) _mm256_set1_pd(x)
#define simd_or(x,y) _mm256_or_pd(x,y)
#define simd_gt(x,y) _mm256_cmp_pd(x,y,_CMP_GT_OS)
#define simd_and(x,y) _mm256_and_pd(x,y)
#define simd_andnot(x,y) _mm256_andnot_pd(x,y)

#define simd_add_f(x,y) _mm256_add_ps(x,y)
#define simd_sub_f(x,y) _mm256_sub_ps(x,y)
#define simd_mul_f(x,y) _mm256_mul_ps(x,y)
#define simd_div_f(x,y) _mm256_div_ps(x,y)
#define simd_max_f(x,y) _mm256_max_ps(x,y)
#define simd_load_f(x) _mm256_load_ps(x)
#define simd_store_f(x,y) _mm256_store_ps(x,y)
#define simd_set1_f(x) _mm256_set1_ps(x)
#define simd_or_f(x,y) _mm256_or_ps(x,y)
#define simd_gt_f(x,y) _mm256_cmp_ps(x,y,_CMP_GT_OS)
#define simd_and_f(x,y) _mm256_and_ps(x,y)
#define simd_andnot_f(x,y) _mm256_andnot_ps(x,y)

typedef __m256d simd_double;
typedef __m256 simd_float;

#else

#define ALIGN 16
#define VECSIZEDOUBLE 2
#define VECSIZEFLOAT 4
#define simd_add(x,y) _mm_add_pd(x,y)
#define simd_sub(x,y) _mm_sub_pd(x,y)
#define simd_mul(x,y) _mm_mul_pd(x,y)
#define simd_div(x,y) _mm_div_pd(x,y)
#define simd_max(x,y) _mm_max_pd(x,y)
#define simd_load(x) _mm_load_pd(x)
#define simd_store(x,y) _mm_store_pd(x,y)
#define simd_set1(x) _mm_set1_pd(x)
#define simd_or(x,y) _mm_or_pd(x,y)
#define simd_gt(x,y) _mm_cmpgt_pd(x,y)
#define simd_and(x,y) _mm_and_pd(x,y)
#define simd_andnot(x,y) _mm_andnot_pd(x,y)

#define simd_add_f(x,y) _mm_add_ps(x,y)
#define simd_sub_f(x,y) _mm_sub_ps(x,y)
#define simd_mul_f(x,y) _mm_mul_ps(x,y)
#define simd_div_f(x,y) _mm_div_ps(x,y)
#define simd_max_f(x,y) _mm_max_ps(x,y)
#define simd_load_f(x) _mm_load_ps(x)
#define simd_store_f(x,y) _mm_store_ps(x,y)
#define simd_set1_f(x) _mm_set1_ps(x)
#define simd_or_f(x,y) _mm_or_ps(x,y)
#define simd_gt_f(x,y) _mm_cmpgt_ps(x,y)
#define simd_and_f(x,y) _mm_and_ps(x,y)
#define simd_andnot_f(x,y) _mm_andnot_ps(x,y)

typedef __m128d simd_double;
typedef __m128 simd_float;
#endif

#endif