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

/* OptiX variables */
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );


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
					color *= functions[light.function]( H );
				}
				result += color;
			}
		}
	}
	prd_shadow.result = result;
}
