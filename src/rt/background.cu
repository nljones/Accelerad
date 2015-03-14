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
rtBuffer<rtCallableProgramId<float(const float3)> > functions;
//rtDeclareVariable(rtCallableProgramId<float(float3)>, func, , );
//rtDeclareVariable(rtCallableProgramX<float(float3)>, func, , );
#ifdef DAYSIM
rtDeclareVariable(int, daysimSortMode, , ); /* how the daylight coefficients are sorted */
#endif

/* OptiX variables */
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );


#ifdef DAYSIM
RT_METHOD int daysimComputePatch(const float3 dir);
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
				//rtPrintf( "Sending (%f, %f, %f)\n", H.x, H.y, H.z);
				color *= functions[light.function]( H );
			}
			if (light.function > -1 || prd_radiance.ambient_depth == 0) { //TODO need a better test, see badcomponent() in source.c
				// no contribution to ambient calculation
				prd_radiance.result += color;
#ifdef DAYSIM
				if (daylightCoefficients >= 2) {
					int patch = DAYSIM_MAX_COEFS; // Ignore by default
					if (daysimSortMode == 1)
						patch = i; // TODO This assumes that all sources are sun positions in numerical order
					else if (daysimSortMode == 2)
						patch = daysimComputePatch(ray.direction);
					daysimAddCoef(prd_radiance.dc, patch, color.x);
				}
#endif /* DAYSIM */
			}
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
					color *= functions[light.function]( H );
				}
				result += color;
#ifdef DAYSIM
				if (daylightCoefficients >= 2) {
					int patch = DAYSIM_MAX_COEFS; // Ignore by default
					if (daysimSortMode == 1)
						patch = prd_shadow.target; // TODO This assumes that all sources are sun positions in numerical order
					else if (daysimSortMode == 2)
						patch = daysimComputePatch(ray.direction);
					daysimAddCoef(prd_shadow.dc, patch, color.x);
				}
#endif /* DAYSIM */
			}
		}
	}
	prd_shadow.result = result;
}

#ifdef DAYSIM
/*
* Computes the sky/ground patch hit by a ray in direction (dx,dy,dz)
* according to the Tregenza sky division.
*/
RT_METHOD int daysimComputePatch(const float3 dir)
{
	int patch;

	if (dir.z > 0.0f) { // sky
		const int number[8] = { 0, 30, 60, 84, 108, 126, 138, 144 };
		const float ring_division[8] = { 30.0f, 30.0f, 24.0f, 24.0f, 18.0f, 12.0f, 6.0f, 0.0f };
		int ringnumber = (int)(asinf(dir.z) * 15.0f * M_1_PIf);
		// origin of the number "15":
		// according to Tregenza, the celestial hemisphere is divided into 7 bands and
		// the zenith patch. The bands range from:
		//												altitude center
		// Band 1		0 to 12 Deg			30 patches	6
		// Band 2		12 to 24 Deg		30 patches	18
		// Band 3		24 to 36 Deg		24 patches	30
		// Band 4		36 to 48 Deg		24 patches	42
		// Band 5		48 to 60 Deg		18 patches	54
		// Band 6		60 to 72 Deg		12 patches	66
		// Band 7		72 to 84 Deg		 6 patches	78
		// Band 8		84 to 90 Deg		 1 patch 	90
		// since the zenith patch is only takes 6Deg instead of 12, the arc length
		// between 0 and 90 Deg (equlas o and Pi/2) is divided into 7.5 units:
		// Therefore, 7.5 units = (int) asin(z=1)/(Pi/2)
		//				1 unit = asin(z)*(2*7.5)/Pi)
		//				1 unit = asin(z)*(15)/Pi)
		// Note that (int) always rounds to the next lowest integer
		patch = number[ringnumber] + ring_division[ringnumber] * (atan2f(dir.y, dir.x) * 0.5f * M_1_PIf + (dir.y >= 0.0f ? 0.0f : 1.0f));
	} else { // ground
		if (dir.z >= -0.17365f)
			patch = 145;
		else if (dir.z >= -0.5f)
			patch = 146;
		else
			patch = 147;
	}

	return patch;
}
#endif /* DAYSIM */
