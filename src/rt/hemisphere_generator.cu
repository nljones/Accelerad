/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix_world.h>
#include "optix_shader_common.h"

using namespace optix;

/* Contex variables */
rtBuffer<PointDirection, 1>      cluster_buffer; /* input */
rtBuffer<PointDirection, 3>      seed_buffer; /* output */
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  point_cloud_ray_type, , );

/* OptiX variables */
rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim,   rtLaunchDim, );

RT_PROGRAM void hemisphere_camera()
{
	PerRayData_point_cloud prd;
	clear(prd.backup);

	PointDirection eye = cluster_buffer[launch_index.z];

	// Check for valid input
	if ( isfinite( eye.pos ) && isfinite( eye.dir ) && dot( eye.dir, eye.dir ) > FTINY ) { // NaN values will be false
		// Init random state
		rand_state* state;
		init_rand(&state, launch_index.x + launch_dim.x * (launch_index.y + launch_dim.y * launch_index.z));

		// Make axes
		float3 uz = normalize(eye.dir);
		float3 ux = getperpendicular(uz, state);
		float3 uy = cross(uz, ux);
						/* avoid coincident samples */
		float2 spt = 0.1f + 0.8f * make_float2(curand_uniform(state), curand_uniform(state));
		SDsquare2disk(spt, (launch_index.y + spt.y) / launch_dim.y, (launch_index.x + spt.x) / launch_dim.x);
		float zd = sqrtf(1.0f - dot(spt, spt));
		float3 rdir = normalize(spt.x * ux + spt.y * uy + zd * uz);

		// Trace the current ray
		Ray ray = make_Ray(eye.pos, rdir, point_cloud_ray_type, ray_start( eye.pos, rdir, uz, RAY_START ), RAY_END);
		rtTrace(top_object, ray, prd);

		// Check for a valid result
		if ( isfinite( prd.result.pos ) && dot( prd.result.dir, prd.result.dir ) > FTINY ) { // NaN values will be false
			seed_buffer[launch_index] = prd.result; // This could contain points on glass materials
			return;
		}
	}
	seed_buffer[launch_index] = prd.backup;
}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf( "Caught exception 0x%X at launch index (%d,%d,%d)\n", code, launch_index.x, launch_index.y, launch_index.z );
	seed_buffer[launch_index].pos = exceptionToFloat3( code );
	seed_buffer[launch_index].dir = make_float3( 0.0f );
#ifdef AMBIENT_CELL
	seed_buffer[launch_index].cell = make_uint2(0);
#endif
}
