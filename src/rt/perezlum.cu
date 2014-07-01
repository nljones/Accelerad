/*
 * Copyright (c) 2013-2014 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix_world.h>
//#include "optix_shader_common.h"

/* Program variables */
rtDeclareVariable(float,  diffuse, , ); /* diffuse normalization */
rtDeclareVariable(float,  ground, , ); /* ground plane brightness */
rtDeclareVariable(float3, coef0, , ); /* coefficients for the Perez model */
rtDeclareVariable(float2, coef1, , ); /* coefficients for the Perez model */
rtDeclareVariable(float3, sun, , ); /* sun direction */

// Calculate the All-weather Angular Sky Luminance Distribution value for the current ray direction.
// This function replicates the algorithm in perezlum.cal.
RT_CALLABLE_PROGRAM float perez_lum( const float3 direction )
{
	rtPrintf("PerezLum Recieved (%f, %f, %f)\n", direction.x, direction.y, direction.z);
	float cosgamma = optix::dot( direction, sun ); // cosgamma = Dx*A8 + Dy*A9 + Dz*A10;
	float gamma = acosf(cosgamma); // gamma = Acos(cosgamma);		{ angle from sun to this point in sky }
	//float zt = acos(sun.z); // zt = Acos(A10);			{ angle from zenith to sun }
	//float eta = acos(direction.z); // eta = Acos(Dz);			{ angle from zenith to this point in sky }

	float dz = direction.z;
	if (dz < 0.01f) {
		dz = 0.01f;
	}

	float intersky = diffuse * (1.0f + coef0.x * expf( coef0.y / dz ) ) * ( 1.0f + coef0.z * expf(coef1.x * gamma) + coef1.y * cosgamma * cosgamma );
	//intersky = if( (Dz-0.01),  
	//		A1 * (1 + A3*Exp(A4/Dz) ) * ( 1 + A5*Exp(A6*gamma) + A7*cos(gamma)*cos(gamma) ),
	//		A1 * (1 + A3*Exp(A4/0.01) ) * ( 1 + A5*Exp(A6*gamma) + A7*cos(gamma)*cos(gamma) ) );

	float a = powf(direction.z + 1.01f, 10.0f);
	float b = powf(direction.z + 1.01f, -10.0f);

	float skybright = (a * intersky + b * ground) / (a + b); // wmean(a, x, b, y) = (a*x+b*y)/(a+b);
	return skybright;
}