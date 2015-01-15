/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#define RANDOM
#ifdef RANDOM
#include <curand_kernel.h>
#endif
#include "optix_common.h"

//#define FILL_GAPS

/* OptiX method declaration in the style of RT_PROGRAM */
#define RT_METHOD	static __inline__ __device__

#ifndef FTINY
#define  FTINY		(1e-6f)
#endif

#ifndef AVGREFL
#define  AVGREFL	0.5f	/* assumed average reflectance */
#endif

#define RAY_START	(1e-4f)	/* FTINY does not seem to be large enough to miss self */
#define RAY_END		RT_DEFAULT_MAX	/* RT_DEFAULT_MAX squared is greater than Float.Inf */
#define AMBIENT_RAY_LENGTH	(1e-2f)

/* view types from view.h */
#define  VT_PER		'v'		/* perspective */
#define  VT_PAR		'l'		/* parallel */
#define  VT_ANG		'a'		/* angular fisheye */
#define  VT_HEM		'h'		/* hemispherical fisheye */
#define  VT_PLS		's'		/* planispheric fisheye */
#define  VT_CYL		'c'		/* cylindrical panorama */

#ifdef RANDOM
typedef curandState_t rand_state;
#else
typedef float rand_state;
#endif

#ifdef __CUDACC__
/* Ray payloads */
struct PerRayData_radiance
{
	float3 result;
	float weight;
	float distance;
	int depth;
	int ambient_depth;
	rand_state* state;
#ifdef FILL_GAPS
	int primary;
#endif
#ifdef RAY_COUNT
	int ray_count;
#endif
#ifdef HIT_COUNT
	int hit_count;
#endif
#ifdef HIT_TYPE
	int hit_type;
#endif
};

struct PerRayData_shadow
{
	float3 result;
	int target;
};

struct PerRayData_ambient
{
	float3 result;
	float3 surface_normal;
	float weight;
	float wsum;
	int ambient_depth;
	rand_state* state;
#ifdef HIT_COUNT
	int hit_count;
#endif
};

struct PerRayData_ambient_record
{
	AmbientRecord result;
	AmbientRecord* parent;
	rand_state* state;
};

struct PerRayData_point_cloud
{
	PointDirection result;
	PointDirection backup;
	//rand_state* state;
};

/* Ambient data structures */
#ifdef OLDAMB
typedef struct {
	float3 v;		/* division sum (partial) */
	float  r;		/* 1/distance sum */
	float  k;		/* variance for this division */
	int    n;		/* number of subsamples */
	unsigned short  t, p;	/* theta, phi indices */
} AMBSAMP;		/* ambient sample division */

typedef struct {
	float3 ux, uy, uz;	/* x, y and z axis directions */
	float3 acoef;		/* division contribution coefficient */
	int    ns;		/* number of super-samples */
	int    nt, np;		/* number of theta and phi directions */
} AMBHEMI;		/* ambient sample hemisphere */
#endif /* OLDAMB */

/* The following is from the random.h associated with OptiX. */

//// Generate random unsigned int in [0, 2^24)
//RT_METHOD unsigned int lcg(unsigned int &prev)
//{
//	const unsigned int LCG_A = 1664525u;
//	const unsigned int LCG_C = 1013904223u;
//	prev = (LCG_A * prev + LCG_C);
//	return prev & 0x00FFFFFF;
//}
//
//// Generate random float in [0, 1)
//RT_METHOD float rnd(unsigned int &prev)
//{
//	return ((float) lcg(prev) / (float) 0x01000000);
//}

RT_METHOD int isnan( const float3& v );
RT_METHOD int isinf( const float3& v );
RT_METHOD int isfinite( const float3& v );
RT_METHOD float2 sqrtf( const float2& v );
RT_METHOD float3 sqrtf( const float3& v );
RT_METHOD float3 cross_direction( const float3& v );
RT_METHOD float ray_start( const float3& hit, const float& t );
RT_METHOD float ray_start( const float3& hit, const float3& dir, const float3& normal, const float& t );
RT_METHOD float3 exceptionToFloat3( const unsigned int& code );

/* Test if any vector elements are NaN. */
RT_METHOD int isnan( const float3& v )
{
	return isnan(v.x) || isnan(v.y) || isnan(v.z);
}

/* Test if any vector elements are infinite. */
RT_METHOD int isinf( const float3& v )
{
	return isinf(v.x) || isinf(v.y) || isinf(v.z);
}

/* Test if all vector elements are finite. */
RT_METHOD int isfinite( const float3& v )
{
	return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

/* Element-wise square root. */
RT_METHOD float2 sqrtf( const float2& v )
{
	return make_float2(sqrtf(v.x), sqrtf(v.y));
}

/* Element-wise square root. */
RT_METHOD float3 sqrtf( const float3& v )
{
	return make_float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
}

/* Create a normal vector near orthogonal to the given vector. */
RT_METHOD float3 cross_direction( const float3& v )
{
	if ( v.x < 0.6f && v.x > -0.6f )
		return make_float3( 1.0f, 0.0f, 0.0f );
	if ( v.y < 0.6f && v.y > -0.6f )
		return make_float3( 0.0f, 1.0f, 0.0f );
	return make_float3( 0.0f, 0.0f, 1.0f );
}

/* Determine a safe starting value for ray t normal to surface. */
RT_METHOD float ray_start( const float3& hit, const float& t )
{
	return t * fmaxf( 1.0f, fabsf( optix::length( hit ) ) );
}

/* Determine a safe starting value for ray t at any angle to surface. */
RT_METHOD float ray_start( const float3& hit, const float3& dir, const float3& normal, const float& t )
{
	return t * fmaxf( 1.0f, fabsf( optix::length( hit ) / optix::dot( dir, normal ) ) );
}

/* Convert exception to float3 code. */
RT_METHOD float3 exceptionToFloat3( const unsigned int& code )
{
	if ( code < RT_EXCEPTION_USER )
		return make_float3( 0.0f, RT_EXCEPTION_USER - code, 0.0f );
	return make_float3( 1.0f, code - RT_EXCEPTION_USER, 0.0f );
}

#ifndef RANDOM
RT_METHOD void curand_init( const int& x, const int& y, const int& z, rand_state* state );
RT_METHOD float curand_uniform( rand_state* state );

/* Initialize the non-random number generator with zero. */
RT_METHOD void curand_init( const int& x, const int& y, const int& z, rand_state* state )
{
	*state = 0.0f;
}

/* Return the value of the non-random number. */
RT_METHOD float curand_uniform( rand_state* state )
{
	return *state;
}
#endif /* RANDOM */

#ifndef OLDAMB
#define	DCSCALE		11585.2f		/* (1<<13)*sqrt(2) */
#define FXNEG		01
#define FYNEG		02
#define FZNEG		04
#define F1X		010
#define F2Z		020
#define F1SFT		5
#define F2SFT		18
#define FMASK		0x1fff

RT_METHOD int encodedir( const float3& dv );
RT_METHOD float3 decodedir( const int& dc );
RT_METHOD void SDsquare2disk( float2& ds, const float& seedx, const float& seedy );
RT_METHOD unsigned int quadratic( float2* r, const float& a, const float& b, const float& c );

/* Encode a normalized direction vector. */
RT_METHOD int encodedir( const float3& dv )
{
	int dc = 0;
	int	cd[3], cm;

	if ( dv.x < 0.0f ) {
		cd[0] = (int)(dv.x * -DCSCALE);
		dc |= FXNEG;
	} else
		cd[0] = (int)(dv.x * DCSCALE);
	if ( dv.y < 0.0f ) {
		cd[1] = (int)(dv.y * -DCSCALE);
		dc |= FYNEG;
	} else
		cd[1] = (int)(dv.y * DCSCALE);
	if ( dv.z < 0.0f ) {
		cd[2] = (int)(dv.z * -DCSCALE);
		dc |= FZNEG;
	} else
		cd[2] = (int)(dv.z * DCSCALE);
	if (!(cd[0] | cd[1] | cd[2]))
		return(0);		/* zero normal */
	if (cd[0] <= cd[1]) {
		dc |= F1X | cd[0] << F1SFT;
		cm = cd[1];
	} else {
		dc |= cd[1] << F1SFT;
		cm = cd[0];
	}
	if (cd[2] <= cm)
		dc |= F2Z | cd[2] << F2SFT;
	else
		dc |= cm << F2SFT;
	if (!dc)	/* don't generate 0 code normally */
		dc = F1X;
	return(dc);
}

/* Decode a normalized direction vector. */
RT_METHOD float3 decodedir( const int& dc )
{
	if (!dc)		/* special code for zero normal */
		return make_float3( 0.0f );

	float3 dv;
	const float2 d = ( make_float2( (dc>>F1SFT & FMASK), (dc>>F2SFT & FMASK) ) + 0.5f ) * ( 1.0f / DCSCALE );
	const float der = sqrtf( 1.0f - optix::dot( d, d ) );
	if (dc & F1X) {
		if (dc & F2Z) dv = make_float3( d.x, der, d.y );
		else dv = make_float3( d, der );
	} else {
		if (dc & F2Z) dv = make_float3( der, d );
		else dv = make_float3( d.y, d.x, der );
	}
	if (dc & FXNEG) dv.x = -dv.x;
	if (dc & FYNEG) dv.y = -dv.y;
	if (dc & FZNEG) dv.z = -dv.z;
	return dv;
}

/* Map a [0,1]^2 square to a unit radius disk */
RT_METHOD void SDsquare2disk( float2& ds, const float& seedx, const float& seedy )
{
	float phi, r;
	const float a = 2.0f * seedx - 1.0f;   /* (a,b) is now on [-1,1]^2 */
	const float b = 2.0f * seedy - 1.0f;

	if (a > -b) {		/* region 1 or 2 */
		if (a > b) {	/* region 1, also |a| > |b| */
			r = a;
			phi = M_PI_4f * (b/a);
		} else {		/* region 2, also |b| > |a| */
			r = b;
			phi = M_PI_4f * (2.0f - (a/b));
		}
	} else {			/* region 3 or 4 */
		if (a < b) {	/* region 3, also |a| >= |b|, a != 0 */
			r = -a;
			phi = M_PI_4f * (4.0f + (b/a));
		} else {		/* region 4, |b| >= |a|, but a==0 and b==0 could occur. */
			r = -b;
			if (b != 0.0f)
				phi = M_PI_4f * (6.0f - (a/b));
			else
				phi = 0.0f;
		}
	}
	r *= 0.9999999999999;	/* prophylactic against MS sin()/cos() impl. (probably unnecessary on GPU) */
	ds.x = r * cosf(phi);
	ds.y = r * sinf(phi);
}

/* find real roots of quadratic equation (from zeros.c) */
RT_METHOD unsigned int quadratic( float2* r, const float& a, const float& b, const float& c )
{
	int  first;

	if (a < -FTINY)
		first = 1;
	else if (a > FTINY)
		first = 0;
	else if (fabsf(b) > FTINY) {	/* solve linearly */
		*r = make_float2( -c / b );
		return(1);
	} else {
		*r = make_float2( 0.0f );
		return(0);		/* equation is c == 0 ! */
	}
	
	float b2 = b * 0.5f;		/* simplifies formula */
	
	float disc = b2*b2 - a*c;	/* discriminant */

	if (disc < -FTINY*FTINY) {	/* no real roots */
		*r = make_float2( 0.0f );
		return(0);
	}

	if (disc <= FTINY*FTINY) {	/* double root */
		*r = make_float2( -b2 / a );
		return(1);
	}
	
	disc = sqrtf(disc);

	if (first)
		*r = make_float2( (-b2 + disc)/a, (-b2 - disc)/a );
	else
		*r = make_float2( (-b2 - disc)/a, (-b2 + disc)/a );
	return(2);
}
#endif /* OLDAMB */
#endif /* __CUDACC__ */
