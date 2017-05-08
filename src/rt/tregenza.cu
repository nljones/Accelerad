/*
 *  isotropsky.cu - program for Tregenza sky patch identification on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>

// Calculate the Tregenza patch based on tregenza.cal.
RT_CALLABLE_PROGRAM int tbin(const float3 direction)
{
	if (direction.z >= 1.0f) return 145;
	if (direction.z < 0.0f) return 0;
	float alt = asinf(direction.z) * 180 * M_1_PIf;
	float azi = atan2f(direction.x, direction.y) * 180 * M_1_PIf;
	if (azi < 0.0f) azi += 360.0f;

	int x = 0, inc = 0;
	if (alt < 12.0f) {
		x = 1; inc = 12;
	}
	else if (alt < 24.0f) {
		x = 31; inc = 12;
	}
	else if (alt < 36.0f) {
		x = 61; inc = 15;
	}
	else if (alt < 48.0f) {
		x = 85; inc = 15;
	}
	else if (alt < 60.0f) {
		x = 109; inc = 20;
	}
	else if (alt < 72.0f) {
		x = 127; inc = 30;
	}
	else if (alt < 84.0f) {
		x = 139; inc = 60;
	}
	else
		return 145;

	int y = azi + inc / 2;
	if (y < 360)
		x += y / inc;

	return x;
}