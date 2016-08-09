/*
 * Copyright (c) 2013-2016 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix_world.h>
//#include "optix_shader_common.h"

rtDeclareVariable(int, mf, , );	/* Number of divisions per Tregenza patch */

// Calculate the Reinhart patch based on reinhart.cal.
RT_CALLABLE_PROGRAM int rbin(const float3 direction)
{
	if (direction.z < 0.0f) return 0;
	float alt = (direction.z >= 1.0f) ? 90.0f : asinf(direction.z) * 180 * M_1_PIf;
	float azi = atan2f(direction.x, direction.y) * 180 * M_1_PIf;
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

	return raccum + 1;
}