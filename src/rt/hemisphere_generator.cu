/*
 * Copyright (c) 2013-2014 Nathaniel Jones
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
	prd.backup.pos = make_float3( 0.0f );
	prd.backup.dir = make_float3( 0.0f );

	PointDirection eye = cluster_buffer[launch_index.x];

	// Check for valid input
	if ( isfinite( eye.pos ) && isfinite( eye.dir ) && dot( eye.dir, eye.dir ) > FTINY ) { // NaN values will be false
		// Init random state
		rand_state state;
		curand_init( launch_index.x + launch_dim.x * ( launch_index.y + launch_dim.y * launch_index.z ), 0, 0, &state );

		// Make axes
		float3 uz = normalize( eye.dir );
		float3 uy = cross_direction( uz );
		float3 ux = normalize( cross( uy, uz ) );
		uy = normalize( cross( uz, ux ) );

		// Set ray direction
		float zd = sqrtf( ( launch_index.y + curand_uniform( &state ) ) / launch_dim.y );
		float phi = 2.0f*M_PIf * ( launch_index.z + curand_uniform( &state ) ) / launch_dim.z;
		float xd = cosf(phi) * zd;
		float yd = sinf(phi) * zd;
		zd = sqrtf(1.0f - zd*zd);
		float3 rdir = normalize( xd * ux + yd * uy + zd * uz );

		// Trace the current ray
		Ray ray = make_Ray(eye.pos, rdir, point_cloud_ray_type, ray_start( eye.pos, rdir, eye.dir, RAY_START ), RAY_END);
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
}
