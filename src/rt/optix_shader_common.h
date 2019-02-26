/*
 *  optix_shader_common.h - shader routines for ray tracing on GPUs.
 */

#pragma once

#include "accelerad_copyright.h"

#include "optix_common.h"
#ifdef CONTRIB_DOUBLE
#include "optix_double.h"
#endif

/* OptiX method declaration in the style of RT_PROGRAM */
#define RT_METHOD	static __forceinline__ __device__

#ifndef FTINY
#define  FTINY		(1e-6f)
#endif

#ifndef AVGREFL
#define  AVGREFL	0.5f	/* assumed average reflectance */
#endif

#define RAY_START	(1e-4f)	/* FTINY does not seem to be large enough to miss self */
#define RAY_END		RT_DEFAULT_MAX	/* RT_DEFAULT_MAX squared is greater than Float.Inf */
#define AMBIENT_RAY_LENGTH	(1e-2f)

/* estimate of Fresnel function */
#define FRESNE(ci)	(expf(-5.85f*(ci)) - 0.00287989916f)
#define FRESTHRESH	0.017999f	/* minimum specularity for approx. */

/* view types from view.h */
#define  VT_PER		'v'		/* perspective */
#define  VT_PAR		'l'		/* parallel */
#define  VT_ANG		'a'		/* angular fisheye */
#define  VT_HEM		'h'		/* hemispherical fisheye */
#define  VT_PLS		's'		/* planispheric fisheye */
#define  VT_CYL		'c'		/* cylindrical panorama */

#ifdef CONTRIB_DOUBLE
typedef double3 contrib3;
typedef double4 contrib4;
#define make_contrib3(...) make_double3(__VA_ARGS__)
#define make_contrib4(...) make_double4(__VA_ARGS__)
#else
typedef float3 contrib3;
typedef float4 contrib4;
#define make_contrib3(...) make_float3(__VA_ARGS__)
#define make_contrib4(...) make_float4(__VA_ARGS__)
#endif

#ifdef __CUDACC__
RT_METHOD void checkFinite( const float2& v );
RT_METHOD void checkFinite( const float3& v );
RT_METHOD int isnan( const float2& v );
RT_METHOD int isnan( const float3& v );
RT_METHOD int isinf( const float2& v );
RT_METHOD int isinf( const float3& v );
RT_METHOD int isfinite( const float2& v );
RT_METHOD int isfinite( const float3& v );
RT_METHOD float2 sqrtf( const float2& v );
RT_METHOD float3 sqrtf( const float3& v );
RT_METHOD float3 cross_direction( const float3& v );
RT_METHOD float3 getperpendicular( const float3& v );
RT_METHOD float ray_start( const float3& hit, const float& t );
RT_METHOD float ray_start( const float3& hit, const float3& dir, const float3& normal, const float& t );
RT_METHOD float bright(const float3& rgb);
RT_METHOD void SDsquare2disk( float2& ds, const float& seedx, const float& seedy );
RT_METHOD float3 exceptionToFloat3( const unsigned int& code );
RT_METHOD float4 exceptionToFloat4( const unsigned int& code );

/* Throw exception if any vector elements are NaN or infinte. */
RT_METHOD void checkFinite( const float2& v )
{
	if (isfinite(v)) return;
	if (isnan(v)) rtThrow(RT_EXCEPTION_NAN);
	rtThrow(RT_EXCEPTION_INF);
}

/* Throw exception if any vector elements are NaN or infinte. */
RT_METHOD void checkFinite( const float3& v )
{
	if (isfinite(v)) return;
	if (isnan(v)) rtThrow(RT_EXCEPTION_NAN);
	rtThrow(RT_EXCEPTION_INF);
}

/* Test if any vector elements are NaN. */
RT_METHOD int isnan( const float2& v )
{
	return isnan(v.x) || isnan(v.y);
}

/* Test if any vector elements are NaN. */
RT_METHOD int isnan( const float3& v )
{
	return isnan(v.x) || isnan(v.y) || isnan(v.z);
}

/* Test if any vector elements are infinite. */
RT_METHOD int isinf( const float2& v )
{
	return isinf(v.x) || isinf(v.y);
}

/* Test if any vector elements are infinite. */
RT_METHOD int isinf( const float3& v )
{
	return isinf(v.x) || isinf(v.y) || isinf(v.z);
}

/* Test if all vector elements are finite. */
RT_METHOD int isfinite( const float2& v )
{
	return isfinite(v.x) && isfinite(v.y);
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

/* Choose deterministic perpedicular direction */
RT_METHOD float3 getperpendicular( const float3& v )
{
	return optix::normalize(optix::cross(cross_direction(v), v));
}

/* Determine a safe starting value for ray t normal to surface. */
RT_METHOD float ray_start( const float3& hit, const float& t )
{
	return t * fmaxf( 1.0f, optix::length( hit ) );
}

/* Determine a safe starting value for ray t at any angle to surface. */
RT_METHOD float ray_start( const float3& hit, const float3& dir, const float3& normal, const float& t )
{
	return t * fmaxf( 1.0f, optix::length( hit ) / fabsf( optix::dot( dir, normal ) ) );
}

rtDeclareVariable(float3, CIE_rgbf, , ); /* This is the value [ CIE_rf, CIE_gf, CIE_bf ] from color.h */

/* Determine the CIE brightness of a color. */
RT_METHOD float bright(const float3& rgb)
{
	return optix::dot(rgb, CIE_rgbf);
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
	r *= 0.999999f;	/* prophylactic against MS sin()/cos() impl. */
	ds.x = r * cosf(phi);
	ds.y = r * sinf(phi);
}

/* Convert exception to float3 code. */
RT_METHOD float3 exceptionToFloat3( const unsigned int& code )
{
	return make_float3(code, 0.0f, 0.0f);
}

/* Convert exception to float4 code. */
RT_METHOD float4 exceptionToFloat4( const unsigned int& code )
{
	return make_float4(code, 0.0f, 0.0f, -1.0f);
}

#endif /* __CUDACC__ */
