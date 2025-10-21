#pragma once
#include <math.h>


#ifndef _MSC_VER
#define __forceinline
#endif

typedef float  (*cuberootfnf) (float);
typedef double (*cuberootfnd) (double);

// estimate bits of precision (32-bit float case)
inline int bits_of_precision(float a, float b)
{
	const double kd = 1.0 / log(2.0);

	if (a==b)
		return 23;

	const double kdmin = pow(2.0, -23.0);

	double d = fabs(a-b);
	if (d < kdmin)
		return 23;

	return int(-log(d)*kd);
}

// estiamte bits of precision (64-bit double case)
inline int bits_of_precision(double a, double b)
{
	const double kd = 1.0 / log(2.0);

	if (a==b)
		return 52;

	const double kdmin = pow(2.0, -52.0);

	double d = fabs(a-b);
	if (d < kdmin)
		return 52;

	return int(-log(d)*kd);
}

// cube root via x^(1/3)
inline float pow_cbrtf(float x)
{
	return pow(x, 1.0f/3.0f);
}

// cube root via x^(1/3)
inline double pow_cbrtd(double x)
{
	return pow(x, 1.0/3.0);
}

// cube root approximation using bit hack for 32-bit float
__forceinline float cbrt_5f(float f)
{
	unsigned int* p = (unsigned int *) &f;
	*p = *p/3 + 709921077;
	return f;
}

// cube root approximation using bit hack for 64-bit float 
// adapted from Kahan's cbrt
__forceinline double cbrt_5d(double d)
{
	const unsigned int B1 = 715094163;
	double t = 0.0;
	unsigned int* pt = (unsigned int*) &t;
	unsigned int* px = (unsigned int*) &d;
	pt[1]=px[1]/3+B1;
	return t;
}

// cube root approximation using bit hack for 64-bit float 
// adapted from Kahan's cbrt
__forceinline double quint_5d(double d)
{
	const unsigned int B1 = 71509416*5/3;
	double t = 0.0;
	unsigned int* pt = (unsigned int*) &t;
	unsigned int* px = (unsigned int*) &d;
	pt[1]=px[1]/5+B1;
	return t;
}

// iterative cube root approximation using Halley's method (float)
__forceinline float cbrta_halleyf(const float a, const float R)
{
	const float a3 = a*a*a;
    const float b= a * (a3 + R + R) / (a3 + a3 + R);
	return b;
}

// iterative cube root approximation using Halley's method (double)
__forceinline double cbrta_halleyd(const double a, const double R)
{
	const double a3 = a*a*a;
    const double b= a * (a3 + R + R) / (a3 + a3 + R);
	return b;
}

// iterative cube root approximation using Newton's method (float)
__forceinline float cbrta_newtonf(const float a, const float x)
{
//    return (1.0 / 3.0) * ((a + a) + x / (a * a));
	return a - (1.0f / 3.0f) * (a - x / (a*a));
}

// iterative cube root approximation using Newton's method (double)
__forceinline double cbrta_newtond(const double a, const double x)
{
	return (1.0/3.0) * (x / (a*a) + 2*a);
}

// cube root approximation using 1 iteration of Halley's method (double)
inline double halley_cbrt1d(double d)
{
	double a = cbrt_5d(d);
	return cbrta_halleyd(a, d);
}

// cube root approximation using 1 iteration of Halley's method (float)
inline float halley_cbrt1f(float d)
{	
	float a = cbrt_5f(d);
	return cbrta_halleyf(a, d);
}

// cube root approximation using 2 iterations of Halley's method (double)
inline double halley_cbrt2d(double d)
{
	double a = cbrt_5d(d);
	a = cbrta_halleyd(a, d);
	return cbrta_halleyd(a, d);
}

// cube root approximation using 3 iterations of Halley's method (double)
inline double halley_cbrt3d(double d)
{
	double a = cbrt_5d(d);
	a = cbrta_halleyd(a, d);
	a = cbrta_halleyd(a, d);
	return cbrta_halleyd(a, d);
}


// cube root approximation using 2 iterations of Halley's method (float)
inline float halley_cbrt2f(float d)
{
	float a = cbrt_5f(d);
	a = cbrta_halleyf(a, d);
	return cbrta_halleyf(a, d);
}

// cube root approximation using 1 iteration of Newton's method (double)
inline double newton_cbrt1d(double d)
{
	double a = cbrt_5d(d);
	return cbrta_newtond(a, d);
}

// cube root approximation using 2 iterations of Newton's method (double)
inline double newton_cbrt2d(double d)
{
	double a = cbrt_5d(d);
	a = cbrta_newtond(a, d);
	return cbrta_newtond(a, d);
}

// cube root approximation using 3 iterations of Newton's method (double)
inline double newton_cbrt3d(double d)
{
	double a = cbrt_5d(d);
	a = cbrta_newtond(a, d);
	a = cbrta_newtond(a, d);
	return cbrta_newtond(a, d);
}

// cube root approximation using 4 iterations of Newton's method (double)
inline double newton_cbrt4d(double d)
{
	double a = cbrt_5d(d);
	a = cbrta_newtond(a, d);
	a = cbrta_newtond(a, d);
	a = cbrta_newtond(a, d);
	return cbrta_newtond(a, d);
}

// cube root approximation using 1 iterations of Newton's method (float)
inline float newton_cbrt1f(float d)
{
	float a = cbrt_5f(d);
	return cbrta_newtonf(a, d);
}

// cube root approximation using 2 iterations of Newton's method (float)
inline float newton_cbrt2f(float d)
{
	float a = cbrt_5f(d);
	a = cbrta_newtonf(a, d);
	return cbrta_newtonf(a, d);
}

// cube root approximation using 3 iterations of Newton's method (float)
inline float newton_cbrt3f(float d)
{
	float a = cbrt_5f(d);
	a = cbrta_newtonf(a, d);
	a = cbrta_newtonf(a, d);
	return cbrta_newtonf(a, d);
}

// cube root approximation using 4 iterations of Newton's method (float)
inline float newton_cbrt4f(float d)
{
	float a = cbrt_5f(d);
	a = cbrta_newtonf(a, d);
	a = cbrta_newtonf(a, d);
	a = cbrta_newtonf(a, d);
	return cbrta_newtonf(a, d);
}
