/*
 *  reinhartb.cu - program for Reinhart sky patch identification on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float3, normal, , );	/* Normal direction */
rtDeclareVariable(float3, up, , );		/* Up direction */
rtDeclareVariable(int, RHS, , ) = 1;	/* Coordinate system handedness: set to -1 for left-handed system */
rtDeclareVariable(int, mf, , ) = 1;		/* Number of divisions per Tregenza patch */

// Calculate the Reinhart patch based on reinhartb.cal.
RT_CALLABLE_PROGRAM int rbin(const float3 direction)
{
	// Compute oriented axis angles
	const float inc_dz = -dot(direction, normal);
	const float inc_rx = -RHS * dot(direction, cross(up, normal));
	const float inc_ry = dot(direction, up) + inc_dz * dot(normal, up);

	if (inc_dz <= 0.0f) return -1;
	float alt = (inc_dz >= 1.0f) ? 90.0f : asinf(inc_dz) * 180 * M_1_PIf;
	float azi = atan2f(inc_rx, inc_ry) * 180 * M_1_PIf;
	if (azi < 0.0f) azi += 360.0f;

	const int tnaz[] = { 30, 30, 24, 24, 18, 12, 6 };	// Number of patches per row

	float alpha = 90.0f / (mf * 7 + 0.5f);		// Separation between rows in degrees
	int r_row = (int)floor(alt / alpha);
	int rnaz = (r_row > (7 * mf - 0.5f)) ? 1 : mf * tnaz[(int)floor((r_row + 0.5f) / mf)];
	float r_inc = 360.0f / rnaz;
	int raccum = (359.9999f > 0.5f * r_inc + azi) ? (int)floor((azi + 0.5f * r_inc) / r_inc) : 0; // This is r_azn

	for (int r = 0; r < r_row; r++) {
		raccum += (r > (7 * mf - 0.5f)) ? 1 : mf * tnaz[(int)floor((r + 0.5f) / mf)];
	}

	return raccum;
}