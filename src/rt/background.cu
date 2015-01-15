/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

//#define RT_USE_TEMPLATED_RTCALLABLEPROGRAM
#include <optix_world.h>
#include "optix_shader_common.h"

using namespace optix;

/* Program variables */
#ifdef HIT_TYPE
rtDeclareVariable(unsigned int, type, , ); /* The material type representing "source" */
#endif

/* Context variables */
rtBuffer<DistantLight> lights;
#ifdef CALLABLE
rtBuffer<rtCallableProgramId<float(const float3)> > functions;
//rtDeclareVariable(rtCallableProgramId<float(float3)>, func, , );
//rtDeclareVariable(rtCallableProgramX<float(float3)>, func, , );
#else
rtBuffer<SkyBright> sky_brights;
rtBuffer<PerezLum> perez_lums;
#endif

/* OptiX variables */
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

#ifndef CALLABLE
// Calculate the sky brightness function for sunny and cloudy skies.
// This function replicates the algorithm in skybright.cal.
RT_METHOD float sky_bright( SkyBright cie )
{
	float cosgamma = optix::dot( ray.direction, cie.sun ); // cosgamma = Dx*A8 + Dy*A9 + Dz*A10;
	float gamma = acosf(cosgamma); // gamma = Acos(cosgamma);		{ angle from sun to this point in sky }
	float dz = ray.direction.z;
	float select = cie.zenith; // unifsky = A2; select(A1, sunnysky, cloudysky, unifsky, intersky)

	if (cie.type == 1) { // CIE clear
		select *= ( 0.91f + 10.0f * expf( -3.0f * gamma ) + 0.45f * cosgamma * cosgamma ) / cie.factor;
		if ( dz > 0.01f ) {
			select *= 1.0f - expf(-0.32f / dz );
		}
		//sunnysky = A2 * (.91 + 10*exp(-3*gamma) + .45*cosgamma*cosgamma)
	    //  *  if( Dz - .01, 1.0 - exp(-.32/Dz), 1.0) / A4;
	} else if (cie.type == 2) { // CIE overcast
		select *= ( 1.0f + 2.0f * dz ) / 3.0f; // cloudysky = A2 * (1 + 2*Dz)/3;
	} else if (cie.type == 4) { // CIE intermediate
		float zt = acosf(cie.sun.z); // zt = Acos(A10);			{ angle from zenith to sun }
		float eta = acosf(dz); // eta = Acos(Dz);			{ angle from zenith to this point in sky }
		select *= ( ( 1.35f * sinf( 5.631f - 3.59f * eta ) + 3.12f ) * sinf( 4.396f - 2.6f * zt) + 6.37f - eta ) / 2.326f *
			expf( gamma * -0.563f * ( ( 2.629f - eta ) * ( 1.562f - zt ) + 0.812f ) ) / cie.factor;
		//intersky = A2 * ( (1.35*sin(5.631-3.59*eta)+3.12)*sin(4.396-2.6*zt)
		//	+ 6.37 - eta ) / 2.326 *
		// exp(gamma*-.563*((2.629-eta)*(1.562-zt)+.812)) / A4;
	}

	float a = powf(ray.direction.z + 1.01f, 10.0f);
	float b = powf(ray.direction.z + 1.01f, -10.0f);

	float skybr = (a * select + b * cie.ground) / (a + b); // wmean(a, x, b, y) = (a*x+b*y)/(a+b);
	return skybr;
}

// Calculate the All-weather Angular Sky Luminance Distribution value for the current ray direction.
// This function replicates the algorithm in perezlum.cal.
RT_METHOD float perez_lum( PerezLum perez )
{
	float cosgamma = optix::dot( ray.direction, perez.sun ); // cosgamma = Dx*A8 + Dy*A9 + Dz*A10;
	float gamma = acosf(cosgamma); // gamma = Acos(cosgamma);		{ angle from sun to this point in sky }
	//float zt = acos(perez.sun.z); // zt = Acos(A10);			{ angle from zenith to sun }
	//float eta = acos(ray.direction.z); // eta = Acos(Dz);			{ angle from zenith to this point in sky }

	float dz = ray.direction.z;
	if (dz < 0.01f) {
		dz = 0.01f;
	}

	float intersky = perez.diffuse * (1.0f + perez.coef[0] * expf( perez.coef[1] / dz ) ) * ( 1.0f + perez.coef[2] * expf(perez.coef[3] * gamma) + perez.coef[4] * cosgamma * cosgamma );
	//intersky = if( (Dz-0.01),  
	//		A1 * (1 + A3*Exp(A4/Dz) ) * ( 1 + A5*Exp(A6*gamma) + A7*cos(gamma)*cos(gamma) ),
	//		A1 * (1 + A3*Exp(A4/0.01) ) * ( 1 + A5*Exp(A6*gamma) + A7*cos(gamma)*cos(gamma) ) );

	float a = powf(ray.direction.z + 1.01f, 10.0f);
	float b = powf(ray.direction.z + 1.01f, -10.0f);

	float skybright = (a * intersky + b * perez.ground) / (a + b); // wmean(a, x, b, y) = (a*x+b*y)/(a+b);
	return skybright;
}
#endif

RT_PROGRAM void miss()
{
	prd_radiance.result = make_float3( 0.0f );
	prd_radiance.distance = ray.tmax;
	if ( ray.tmax < RAY_END ) // ray length was truncated
		return;

	const float3 H = optix::normalize(ray.direction);

	// compute direct lighting
	unsigned int num_lights = lights.size();
	for (int i = 0; i < num_lights; ++i) {
		DistantLight light = lights[i];

		// get the angle bwetween the light direction and the view
		float3 L = optix::normalize(light.pos);
		float lDh = optix::dot( L, H );
		float solid_angle = 2.0f * M_PIf * (1.0f - lDh);

		if (solid_angle <= light.solid_angle) {
			float3 color = light.color;
			if (light.function > -1) {
#ifdef CALLABLE
				//rtPrintf( "Sending (%f, %f, %f)\n", H.x, H.y, H.z);
				color *= functions[light.function]( H );
#else
				if (light.type == SKY_CIE) {
					color *= sky_bright( sky_brights[light.function] );
				} else if (light.type == SKY_PEREZ) {
					color *= perez_lum( perez_lums[light.function] );
				}
#endif
			}
			if ( light.function > -1 || prd_radiance.ambient_depth == 0 ) //TODO need a better test, see badcomponent() in source.c
				// no contribution to ambient calculation
				prd_radiance.result += color;
		}
	}

#ifdef HIT_TYPE
	prd_radiance.hit_type = type;
#endif
}

RT_PROGRAM void miss_shadow()
{
	float3 result = make_float3( 0.0f );

	const float3 H = optix::normalize(ray.direction);

	// compute direct lighting
	if ( prd_shadow.target >= 0 && prd_shadow.target < lights.size() ) {
		DistantLight light = lights[prd_shadow.target];
		if (light.casts_shadow) {

			// get the angle bwetween the light direction and the view
			float3 L = optix::normalize(light.pos);
			float lDh = optix::dot( L, H );
			float solid_angle = 2.0f * M_PIf * (1.0f - lDh);

			if (solid_angle <= light.solid_angle) {
				float3 color = light.color;
				if (light.function > -1) {
#ifdef CALLABLE
					color *= functions[light.function]( H );
#else
					if (light.type == SKY_CIE) {
						color *= sky_bright( sky_brights[light.function] );
					} else if (light.type == SKY_PEREZ) {
						color *= perez_lum( perez_lums[light.function] );
					}
#endif
				}
				result += color;
			}
		}
	}
	prd_shadow.result = result;
}
