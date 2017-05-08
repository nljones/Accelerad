/*
 *  perezlum.cu - program for Perez All-Weather Sky distribution on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>

struct Perez_coef
{
	float a, b, c, d, e;
};

struct Transform
{
	optix::Matrix<3,3> m;
};

/* Program variables */
rtDeclareVariable(float,      diffuse, , ); /* diffuse normalization */
rtDeclareVariable(float,      ground, , ); /* ground plane brightness */
rtDeclareVariable(Perez_coef, coef, , ); /* coefficients for the Perez model */
rtDeclareVariable(float3,     sun, , ); /* sun direction */
rtDeclareVariable(Transform,  transform, , ); /* transformation matrix */

// Calculate the All-weather Angular Sky Luminance Distribution value for the current ray direction.
// This function replicates the algorithm in perezlum.cal.
RT_CALLABLE_PROGRAM float3 skybright(const float3 direction, const float3 ignore)
{
	//rtPrintf("PerezLum Recieved (%f, %f, %f)\n", direction.x, direction.y, direction.z);

	const float3 dir = transform.m * direction;

	const float cosgamma = 0.999999f * optix::dot(dir, sun); // cosgamma = Dx*A8 + Dy*A9 + Dz*A10; // Adjusted to keep in range [-1,1]
	const float gamma = acosf(cosgamma); // gamma = Acos(cosgamma);		{ angle from sun to this point in sky }
	//float zt = acos(sun.z); // zt = Acos(A10);			{ angle from zenith to sun }
	//float eta = acos(dir.z); // eta = Acos(Dz);			{ angle from zenith to this point in sky }

	float dz = dir.z;
	if (dz < 0.01f) {
		dz = 0.01f;
	}

	const float intersky = diffuse * (1.0f + coef.a * expf( coef.b / dz ) ) * ( 1.0f + coef.c * expf(coef.d * gamma) + coef.e * cosgamma * cosgamma );
	//intersky = if( (Dz-0.01),  
	//		A1 * (1 + A3*Exp(A4/Dz) ) * ( 1 + A5*Exp(A6*gamma) + A7*cos(gamma)*cos(gamma) ),
	//		A1 * (1 + A3*Exp(A4/0.01) ) * ( 1 + A5*Exp(A6*gamma) + A7*cos(gamma)*cos(gamma) ) );

	const float a = powf(dir.z + 1.01f, 10.0f);
	const float b = powf(dir.z + 1.01f, -10.0f);

	const float skybright = (a * intersky + b * ground) / (a + b); // wmean(a, x, b, y) = (a*x+b*y)/(a+b);
	return make_float3(skybright);
}