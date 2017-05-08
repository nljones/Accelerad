/*
 *  fisheye.cu - program for removal of fisheye distortion to acheive equiangular projection on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>

using namespace optix;

struct Transform
{
	optix::Matrix<3, 3> m;
};

/* Program variables */
rtDeclareVariable(int, data, , ); /* texture ID */
rtDeclareVariable(int, type, , ); /* type of data (true for float) */
rtDeclareVariable(Transform, transform, , ); /* transformation matrix */

// Correct lens distortion for Sigma 4.5mm fisheye lens.
RT_CALLABLE_PROGRAM float3 fisheye(const float3 direction, const float3 ignore)
{
	const float3 dir = transform.m * direction;

	float phi = acosf(dir.z); // phi = 0 +z, phi = pi/2 xy equator, phi = pi -z
	float theta = atan2f(dir.x, -dir.y); // theta = 0 along -y axis

	float x = phi * M_2_PIf; // normalize
	float x2 = x * x;
	float x3 = x2 * x;
	float x4 = x3 * x;
	float y = 0.7617f * (x4 * x) - 1.9134f * x4 + 1.5577f * x3 - 0.6087f * x2 + 1.2056f * x; // fisheye correction

	if (y > 1.0f) // outside of image
		return make_float3(0.0f);

	y /= 2.0f; // radius
	float u = 0.5f + y * sin(theta);
	float v = 0.5f + y * cos(theta);

	/* Renormalize to remove edges */
	uint3 ne = rtTexSize(data);
	u = (u * (ne.x - 1) + 0.5f) / ne.x;
	v = (v * (ne.y - 1) + 0.5f) / ne.y;

	if (type)
		return make_float3(rtTex2D<float>(data, u, v)); // this is corr from source.cal
	float4 tex = rtTex2D<float4>(data, u, v);
	return make_float3(tex.y, tex.z, tex.w);
}
