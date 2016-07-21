/*
 * Copyright (c) 2013-2016 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix_world.h>
#include "optix_shader_common.h"

using namespace optix;

/* Program variables */
rtDeclareVariable(unsigned int, do_irrad, , ) = 0u; /* Calculate irradiance (-i) */

/* Contex variables */
rtBuffer<float3, 2>              origin_buffer;
rtBuffer<float3, 2>              direction_buffer;
rtBuffer<float3, 3>              contrib_buffer;
#ifdef RAY_COUNT
rtBuffer<unsigned int, 2>        ray_count_buffer;
#endif
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(rtObject, top_irrad, , );
rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, radiance_primary_ray_type, , );
rtDeclareVariable(unsigned int, imm_irrad, , ) = 0u; /* Immediate irradiance (-I) */
rtDeclareVariable(unsigned int, lim_dist, , ) = 0u; /* Limit ray distance (-ld) */

/* OptiX variables */
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );


RT_PROGRAM void ray_generator()
{
	PerRayData_radiance prd;
	init_rand(&prd.state, launch_index.x + launch_dim.x * launch_index.y);
	prd.weight = 1.0f;
	prd.depth = 0;
	prd.ambient_depth = 0;
	//prd.seed = rnd_seeds[launch_index];
#ifdef ANTIMATTER
	prd.mask = 0u;
	prd.inside = 0;
#endif
	setupPayload(prd);

	/* Zero the output */
	for (int i = 0; i < contrib_buffer.size().z; i++)
		contrib_buffer[make_uint3(launch_index.x, launch_index.y, i)] = make_float3(0.0f);

	float3 org = origin_buffer[launch_index];
	float3 dir = direction_buffer[launch_index];

	const float tmin = ray_start(org, RAY_START);
	if (imm_irrad) {
		Ray ray = make_Ray(org, -normalize(dir), radiance_ray_type, -tmin, tmin);
		rtTrace(top_irrad, ray, prd);
	}
	else {
		Ray ray = make_Ray(org, normalize(dir), do_irrad ? radiance_primary_ray_type : radiance_ray_type, tmin, lim_dist ? length(dir) : RAY_END);
		rtTrace(top_object, ray, prd);
	}

	checkFinite(prd.result);

#ifdef RAY_COUNT
	ray_count_buffer[launch_index] = prd.ray_count;
#endif
}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf("Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y);
	contrib_buffer[make_uint3(launch_index.x, launch_index.y, 0)] = make_float3(code, 0.0f, -1.0f);;
}
