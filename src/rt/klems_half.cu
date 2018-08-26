/*
 *  klems_half.cu - program for Klems bin identification on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>
#include "optix_shader_common.h"

using namespace optix;

rtDeclareVariable(float3, normal, , );	/* Normal direction */
rtDeclareVariable(float3, up, , );		/* Up direction */
rtDeclareVariable(int, RHS, , ) = 1;	/* Coordinate system handedness: set to -1 for left-handed system */

// Calculate the Klems bin based on klems_full.cal.
RT_CALLABLE_PROGRAM int kbin(const float3 direction)
{
	const float DdotN = dot(direction, normal);
	const float DdotU = dot(direction, up);
	const float NdotU = dot(normal, up);

	if (DdotN > 0) return -1; // Wrong-side ray, probably bad
	if (DdotN <= -1) return 0;
	float pol = acosf(-DdotN) * 180 * M_1_PIf;
	float azi = atan2f(-DdotU + DdotN * NdotU, -RHS * dot(direction, cross(up, normal))) * 180 * M_1_PIf;
	if (azi < 0.0f) azi += 360.0f;

	const float kpola[] = { 6.5f, 19.5f, 32.5f, 46.5f, 61.5f, 76.5f, 90.0f };
	const int knaz[] = { 1, 8, 12, 16, 20, 12, 4 };	// Number of patches per row

	int row = 0;
	while (pol > kpola[row]) // This is kfindrow
		row++;

	float inc = 360.0f / knaz[row];
	int kaccum = ((360.0f - 0.5f * inc) > azi) ? (int)floor((azi + 0.5f * inc) / inc) : 0; // This is kazn

	for (int r = 0; r < row; r++)
		kaccum += knaz[r];

	return kaccum;
}
