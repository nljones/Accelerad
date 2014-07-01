/*
 * Copyright (c) 2013-2014 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix_world.h>
//#include "optix_shader_common.h"

/* Program variables */
rtDeclareVariable(unsigned int, type, , ); /* 1 for CIE clear, 2 for CIE overcast, 3 for uniform, 4 for CIE intermediate */
rtDeclareVariable(float,        zenith, , ); /* zenith brightness */
rtDeclareVariable(float,        ground, , ); /* ground plane brightness */
rtDeclareVariable(float,        factor, , ); /* normalization factor based on sun direction */
rtDeclareVariable(float3,       sun, , ); /* sun direction */

// Calculate the sky brightness function for sunny and cloudy skies.
// This function replicates the algorithm in skybright.cal.
RT_CALLABLE_PROGRAM float sky_bright( const float3 direction )
{
	rtPrintf("SkyBright Recieved (%f, %f, %f)\n", direction.x, direction.y, direction.z);
	float cosgamma = optix::dot( direction, sun ); // cosgamma = Dx*A8 + Dy*A9 + Dz*A10;
	float gamma = acosf(cosgamma); // gamma = Acos(cosgamma);		{ angle from sun to this point in sky }
	//float dz = direction.z;
	float sky = zenith; // unifsky = A2; select(A1, sunnysky, cloudysky, unifsky, intersky)

	if (type == 1u) { // CIE clear
		sky *= ( 0.91f + 10.0f * expf( -3.0f * gamma ) + 0.45f * cosgamma * cosgamma ) / factor;
		if ( direction.z > 0.01f ) {
			sky *= 1.0f - expf(-0.32f / direction.z );
		}
		//sunnysky = A2 * (.91 + 10*exp(-3*gamma) + .45*cosgamma*cosgamma)
	    //  *  if( Dz - .01, 1.0 - exp(-.32/Dz), 1.0) / A4;
	} else if (type == 2u) { // CIE overcast
		sky *= ( 1.0f + 2.0f * direction.z ) / 3.0f; // cloudysky = A2 * (1 + 2*Dz)/3;
	} else if (type == 4u) { // CIE intermediate
		float zt = acosf(sun.z); // zt = Acos(A10);			{ angle from zenith to sun }
		float eta = acosf(direction.z); // eta = Acos(Dz);			{ angle from zenith to this point in sky }
		sky *= ( ( 1.35f * sinf( 5.631f - 3.59f * eta ) + 3.12f ) * sinf( 4.396f - 2.6f * zt) + 6.37f - eta ) / 2.326f *
			expf( gamma * -0.563f * ( ( 2.629f - eta ) * ( 1.562f - zt ) + 0.812f ) ) / factor;
		//intersky = A2 * ( (1.35*sin(5.631-3.59*eta)+3.12)*sin(4.396-2.6*zt)
		//	+ 6.37 - eta ) / 2.326 *
		// exp(gamma*-.563*((2.629-eta)*(1.562-zt)+.812)) / A4;
	}

	float a = powf(direction.z + 1.01f, 10.0f);
	float b = powf(direction.z + 1.01f, -10.0f);

	float skybr = (a * sky + b * ground) / (a + b); // wmean(a, x, b, y) = (a*x+b*y)/(a+b);
	return skybr;
}