/*
 *  optix_double.h - declaration for operations on double vectors on GPUs.
 */

#ifndef OPTIX_DOUBLE_HEADER
#define OPTIX_DOUBLE_HEADER

#include "accelerad_copyright.h"

__forceinline__ RT_HOSTDEVICE double3 make_double3(const double d)
{
	double3 t; t.x = d; t.y = d; t.z = d; return t;
}

__forceinline__ RT_HOSTDEVICE double3 make_double3(const float3& f)
{
	double3 t; t.x = f.x; t.y = f.y; t.z = f.z; return t;
}

__forceinline__ RT_HOSTDEVICE double4 make_double4(const double d)
{
	double4 t; t.x = d; t.y = d; t.z = d; t.w = d; return t;
}

__forceinline__ RT_HOSTDEVICE double4 make_double4(const double3& d)
{
	double4 t; t.x = d.x; t.y = d.y; t.z = d.z; t.w = 0.0f; return t;
}

__forceinline__ RT_HOSTDEVICE double3 operator*(const double3& d, const float3& f)
{
	double3 t; t.x = d.x * f.x; t.y = d.y * f.y; t.z = d.z * f.z; return t;
}

__forceinline__ RT_HOSTDEVICE void operator*=(double3& d, const float& f)
{
	d.x *= f; d.y *= f; d.z *= f;
}

__forceinline__ RT_HOSTDEVICE void operator*=(double3& d, const float3& f)
{
	d.x *= f.x; d.y *= f.y; d.z *= f.z;
}

__forceinline__ RT_HOSTDEVICE void operator+=(double4& a, const double4& b)
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

#endif /* OPTIX_DOUBLE_HEADER */
