/*
 *  disk2square.cu - program for Shirley-Chiu mapping on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float3, normal, , );	/* Normal direction */
rtDeclareVariable(float3, up, , );		/* Up direction */
rtDeclareVariable(int, RHS, , ) = 1;	/* Coordinate system handedness: set to -1 for left-handed system */
rtDeclareVariable(int, SCdim, , ) = 1;	/* Side length of square */

// Calculate the Shirley-Chiu mapping based on disk2square.cal.
RT_CALLABLE_PROGRAM int scbin(const float3 direction)
{
	// Compute oriented axis angles
	const float inc_dz = -dot(direction, normal);
	const float inc_rx = -RHS * dot(direction, cross(up, normal));
	const float inc_ry = -dot(direction, up) - inc_dz * dot(normal, up);

	/* -1 if behind surface */
	if (inc_dz <= 0.0f) return -1;

	const float inc_den2 = inc_rx * inc_rx + inc_ry * inc_ry;
	const float inc_radf = inc_den2 > 1e-7f ? sqrtf((1 - inc_dz*inc_dz) / inc_den2) : 0.0f;

	/* Compute square position from disk coordinates */
	const float2 in_disk = make_float2(inc_rx, inc_ry) * inc_radf;
	const float in_disk_r = length(in_disk);
	float in_disk_phi = atan2f(in_disk.y, in_disk.x);
	if (in_disk_phi < -M_PI_4f)
		in_disk_phi += 2.0f * M_PIf;

	float2 out_square;
	switch ((int)floor((in_disk_phi + M_PI_4f) / M_PI_2f)) {
	case 0:
		out_square = make_float2(in_disk_r, in_disk_phi * in_disk_r / M_PI_4f);
		break;
	case 1:
		out_square = make_float2((M_PI_2f - in_disk_phi) * in_disk_r / M_PI_4f, in_disk_r);
		break;
	case 2:
		out_square = make_float2(-in_disk_r, (M_PIf - in_disk_phi) * in_disk_r / M_PI_4f);
		break;
	case 3:
		out_square = make_float2((in_disk_phi - 3 * M_PI_2f) * in_disk_r / M_PI_4f, -in_disk_r);
		break;
	case 4:
		out_square = make_float2(in_disk_r, -in_disk_r); /* Edge case */
		break;
	default:
		return -1; /* Something's wrong */
	}
	out_square = (out_square + 1.0f) / 2.0f;

	/* Compute final bin */
	return (int)floor(out_square.x * SCdim) * SCdim + (int)floor(out_square.y * SCdim);
}