/*
 *  utah.cu - program for Utah sky distribution on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>
#include "optix_shader_common.h"

struct Transform
{
	optix::Matrix<3,3> m;
};

/* Program variables */
rtDeclareVariable(unsigned int, monochrome, , ) = 0u; /* output brightness only */
rtDeclareVariable(float, turbidity, , ); /* turbidity */
rtDeclareVariable(float3, sun, , ); /* sun direction */
rtDeclareVariable(Transform, transform, , ); /* transformation matrix */


RT_METHOD float perez(const float& theta, const float& gamma, const float& a, const float& b, const float& c, const float& d, const float& e)
{
	const float cosGamma = cosf(gamma);
	//return (1.0f + a * expf(b / cosf(theta))) * (1.0f + c * expf(d * gamma) + e * cosGamma * cosGamma);
	return (1.0f + a * expf(b / (0.004f + abs(cosf(theta))))) * (1.0f + c * expf(d * gamma) + e * cosGamma * cosGamma);
}

// From Preetham, Shirley, Smits, "A Practical, Analytical Model for Daylight"
// Converted to Radiance by Mark J.Stock, mstock@umich.edu
// This function replicates the algorithm in utah.cal.
RT_CALLABLE_PROGRAM float3 skybr(const float3 direction, const float3 ignore)
{
	const float3 dir = transform.m * direction;

	const float cosgamma = 0.999999f * optix::dot(dir, sun); // cosgamma = Dx*A8 + Dy*A9 + Dz*A10; // Adjusted to keep in range [-1,1]
	const float gamma = acosf(cosgamma); // gamma = Acos(cosgamma);		{ angle from sun to this point in sky }
	const float theta = acosf(dir.z); // theta = Acos(Dz);		{ angle from zenith to this point in sky }
	const float thetas = acosf(sun.z); // thetas = Acos(A4);		{ angle from zenith to sun }
	const float turb = optix::clamp(turbidity, 1.2f, 6.0f); // turb = bound(1.2, A1, 6.);	{ clamp the bounds of turbidity }

	/* zenith brightness, chromaticity */
	float yyz = (4.0453f * turb - 4.971f) * tanf((0.4444f - turb / 120.0f) * (3.1415927f - 2.0f * thetas)) - 0.2155f * turb + 2.4192f;
	if (yyz <= 0.0f)
		yyz = 0.0f;
	const float xz = 0.25886f + 0.00394f * turb + thetas * (0.06052f - 0.03202f * turb * (1.0f - 0.065272f * turb) + thetas * (-0.21196f + 0.06377f * turb * (1.0f - 0.058805f * turb) + thetas * (0.11693f - 0.02903f * turb * (1.0f - 0.057182f * turb))));
	const float yz = 0.26688f + 0.00516f * turb + thetas * (0.0667f - 0.04153f * turb * (1.0f - 0.07633f * turb) + thetas * (-0.26756f + 0.0897f * turb * (1.0f - 0.068004f * turb) + thetas * (0.15346f - 0.04214f * turb * (1.0f - 0.065259f * turb))));

	/* distribution coefficients for luminance, chromaticity; functions of turbidity */
	const float ayy = 0.1787f * turb - 1.463f;
	const float byy = -0.3554f * turb + 0.4275f;
	const float cyy = -0.0227f * turb + 5.3251f;
	const float dyy = 0.1206f * turb - 2.5771f;
	const float eyy = -0.067f * turb + 0.3703f;

	const float ax = -0.0193f * turb - 0.2593f;
	const float bx = -0.0665f * turb + 0.0008f;
	const float cx = -0.0004f * turb + 0.2125f;
	const float dx = -0.0641f * turb - 0.8989f;
	const float ex = -0.0033f * turb + 0.0452f;

	const float ay = -0.0167f * turb - 0.2608f;
	const float by = -0.095f * turb + 0.0092f;
	const float cy = -0.0079f * turb + 0.2102f;
	const float dy = -0.0441f * turb - 1.6537f;
	const float ey = -0.0109f * turb + 0.0529f;

	/* point values for luminance, chromaticity */
	float yyp = yyz * perez(theta, gamma, ayy, byy, cyy, dyy, eyy) / perez(0.0f, thetas, ayy, byy, cyy, dyy, eyy);
	const float xp = xz * perez(theta, gamma, ax, bx, cx, dx, ex) / perez(0.0f, thetas, ax, bx, cx, dx, ex);
	const float yp = yz * perez(theta, gamma, ay, by, cy, dy, ey) / perez(0.0f, thetas, ay, by, cy, dy, ey);

	/* hack to allow stars to shine through haze at dusk and dawn */
	if (sun.z <= 0.05f)
		yyp *= expf(20.0f * (sun.z - 0.05f));

	/* output brightness */
	float3 skybr = make_float3(yyp);
	if (monochrome)
		return skybr;

	/* output radiance */

	/* first, tristimulus values(are these CIE XYZ ? ) */
	skybr.x *= xp / yp;
	if (xp + yp < 1.0f)
		skybr.z *= (1.0f - xp - yp) / yp;
	else
		skybr.z = 0.0f;

	/* convert using CIE M^-1 matrix from http://www.brucelindbloom.com/Eqn_RGB_XYZ_Matrix.html */
	const float xyz2rgb[9] = {
		2.3706743f, -0.9000405f, -0.4706338f,
		-0.513885f, 1.4253036f, 0.0885814f,
		0.0052982f, -0.0146949f, 1.0093968f };
	return (optix::Matrix<3, 3>)xyz2rgb * skybr;
}
