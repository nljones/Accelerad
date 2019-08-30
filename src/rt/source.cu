/*
 *  source.cu - program for source distribution on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>
#include "optix_shader_common.h"

using namespace optix;

struct Transform
{
	optix::Matrix<3,3> m;
};

/* Program variables */
rtDeclareVariable(int, data, , ); /* texture ID */
rtDeclareVariable(int, type, , ); /* type of data (true for float) */
rtDeclareVariable(float3, org, , ); /* texture minimum coordinates */
rtDeclareVariable(float3, siz, , ); /* texture coordinates extent */
rtDeclareVariable(Transform, transform, , ); /* transformation matrix */
rtDeclareVariable(int, transpose, , ) = 0; /* flag to transpose texture to swap phi and theta */
rtDeclareVariable(float, symmetry, , ) = 0.0f; /* radial symmetry angle in radians */
rtDeclareVariable(float, multiplier, , ) = 1.0f; /* multiplier for light source intensity */
rtDeclareVariable(float3, bounds, , ); /* dimensions of axis-aligned box or Z-aligned cylinder in meters */

// Calculate source distribution.
RT_METHOD float3 source(const float3& dir)
{
	float theta = acosf(dir.z);
	float phi = atan2f(-dir.y, -dir.x);
	phi += 2.0f * M_PIf * (phi < 0.0f);

	if (symmetry > 0.0f) {
		phi = fabsf(symmetry - fmodf(phi + symmetry, 2 * symmetry));
	}

	if (transpose) {
		float temp = phi;
		phi = theta;
		theta = temp;
	}

	/* Normalize to [0, 1] within range */
	theta = (180.0f * M_1_PIf * theta - org.x) / siz.x;
	phi = (180.0f * M_1_PIf * phi - org.y) / siz.y;

	/* Renormalize to remove edges */
	uint3 ne = rtTexSize(data);
	theta = (theta * (ne.x - 1) + 0.5f) / ne.x;
	phi = (phi * (ne.y - 1) + 0.5f) / ne.y;

	if (type)
		return make_float3(multiplier * rtTex2D<float>(data, theta, phi)); // this is corr from source.cal
	float4 tex = rtTex2D<float4>(data, theta, phi);
	return multiplier * make_float3(tex.y, tex.z, tex.w);
}

// Calculate source distribution.
RT_CALLABLE_PROGRAM float3 corr(const float3 direction, const float3 ignore)
{
	const float3 dir = transform.m * direction;
	return source(dir); // this is corr from source.cal
}

// Calculate source distribution with correction for flat sources.
RT_CALLABLE_PROGRAM float3 flatcorr(const float3 direction, const float3 normal)
{
	const float3 dir = transform.m * direction;
	const float rdot = dot(direction, normal);
	return source(dir) / fabsf(rdot); // this is flatcorr from source.cal
}

// Calculate source distribution with correction for emitting boxes.
RT_CALLABLE_PROGRAM float3 boxcorr(const float3 direction, const float3 ignore)
{
	const float3 dir = transform.m * direction;
	const float boxprojection = fabsf(dir.x) * bounds.y * bounds.z + fabsf(dir.y) * bounds.x * bounds.z + fabsf(dir.z) * bounds.x * bounds.y;
	return source(dir) / boxprojection; // this is boxcorr from source.cal
}

// Calculate source distribution with correction for emitting cylinders.
RT_CALLABLE_PROGRAM float3 cylcorr(const float3 direction, const float3 ignore)
{
	const float3 dir = transform.m * direction;
	const float cylprojection = bounds.x * bounds.y * sqrtf(fmaxf(1.0f - dir.z * dir.z, 0.0f)) + M_PIf / 4.0f * bounds.x * bounds.x * fabsf(dir.z);
	return source(dir) / cylprojection; // this is cylcorr from source.cal
}