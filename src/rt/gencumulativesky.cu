/*
 *  gencumulativesky.cu - program for gencumulativesky distribution on GPUs.
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
rtDeclareVariable(Transform, transform, , ); /* transformation matrix */

// This function replicates the output .cal file from GenCumulativeSky.
RT_CALLABLE_PROGRAM float3 gencumsky(const float3 direction, const float3 ignore)
{
	const float3 dir = transform.m * direction;

	float alt = asinf(dir.z) * 180 * M_1_PIf;
	if (alt < 0.0f) return make_float3(0.0f);
	float az = atan2f(dir.x, dir.y) * 180 * M_1_PIf;
	if (az < 0.0f) az += 360.0f;
	
	int x = 0;
	if (alt < 12.0f)
		x = (int)(0.5f + az / 12.0f) % 30;
	else if (alt < 24.0f)
		x = (int)(0.5f + az / 12.0f) % 30 + 30;
	else if (alt < 36.0f)
		x = (int)(0.5f + az / 15.0f) % 24 + 60;
	else if (alt < 48.0f)
		x = (int)(0.5f + az / 15.0f) % 24 + 84;
	else if (alt < 60.0f)
		x = (int)(0.5f + az / 20.0f) % 18 + 108;
	else if (alt < 72.0f)
		x = (int)(0.5f + az / 30.0f) % 12 + 126;
	else if (alt < 84.0f)
		x = (int)(0.5f + az / 60.0f) % 6 + 138;
	else
		x = 144;

	return make_float3(rtTex1D<float>(data, x));
}
