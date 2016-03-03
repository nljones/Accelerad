/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix_world.h>
#include "optix_shader_common.h"

using namespace optix;

/* Program variables */
rtDeclareVariable(unsigned int,  do_irrad, , ); /* Calculate irradiance (-i) */

/* Contex variables */
rtBuffer<RayData, 2>             ray_buffer;
rtBuffer<PointDirection, 3>      seed_buffer;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(rtObject,      top_irrad, , );
rtDeclareVariable(unsigned int,  point_cloud_ray_type, , );
rtDeclareVariable(unsigned int,  imm_irrad, , ) = 0u; /* Immediate irradiance (-I) */

/* OptiX variables */
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

RT_PROGRAM void cloud_generator()
{
	PerRayData_point_cloud prd;

	// Init random state
	rand_state* state;
	init_rand(&state, launch_index.x + launch_dim.x * launch_index.y);

	uint3 index = make_uint3( launch_index, 0u );

	float tmin = ray_start( ray_buffer[launch_index].origin, RAY_START );
	float tmax;
	if ( imm_irrad ) {
		tmax = tmin;
		tmin = -tmin;
	} else {
		// Zero or negative aft clipping distance indicates infinity
		tmax = ray_buffer[launch_index].max;
		if (tmax <= FTINY) {
			tmax = RAY_END;
		}
	}

	Ray ray = make_Ray(ray_buffer[launch_index].origin, ray_buffer[launch_index].dir, point_cloud_ray_type, tmin, tmax);

	const unsigned int seeds = seed_buffer.size().z;
	unsigned int loop = 2u * seeds; // Prevent infinite looping
	while ( index.z < seeds && loop-- ) {
		clear(prd.backup);
#ifdef ANTIMATTER
		prd.mask = 0u;
		prd.inside = 0;
#endif

		// Trace the current ray
		if ( imm_irrad )
			rtTrace(top_irrad, ray, prd);
		else
			rtTrace(top_object, ray, prd);

		// Check for a valid result
		if ( isfinite( prd.result.pos ) && dot( prd.result.dir, prd.result.dir ) > FTINY ) { // NaN values will be false
			seed_buffer[index] = prd.result; // This could contain points on glass materials
			index.z++;
#ifndef AMBIENT_CELL
		} else {
			prd.result.pos = ray_buffer[launch_index].origin;
			prd.result.dir = ray_buffer[launch_index].dir;
#endif /* AMBIENT_CELL */
		}

#ifdef AMBIENT_CELL
		if (!(isfinite(prd.backup.pos) && dot(prd.backup.dir, prd.backup.dir) > FTINY)) // NaN values will be false
			break;

		// Prepare for next ray
		ray.origin = prd.backup.pos;
		ray.direction = reflect(ray.direction, prd.backup.dir);
		ray.tmin = RAY_START;// ray_start(ray.origin, ray.direction, prd.backup.dir, RAY_START);
#else /* AMBIENT_CELL */
		// Prepare for next ray
		ray.origin = prd.result.pos;

		float3 uz = normalize( prd.result.dir );
		float3 ux = getperpendicular(uz);
		float3 uy = normalize(cross(uz, ux));

		float zd = sqrtf( curand_uniform( state ) );
		float phi = 2.0f*M_PIf * curand_uniform( state );
		float xd = cosf(phi) * zd;
		float yd = sinf(phi) * zd;
		zd = sqrtf(1.0f - zd*zd);
		ray.direction = normalize( xd*ux + yd*uy + zd*uz );

		ray.tmin = ray_start(ray.origin, RAY_START);
#endif /* AMBIENT_CELL */
		ray.tmax = RAY_END;
	}

	// If outdoors, there are no bounces, but we need to prevent junk data
	while ( index.z < seeds ) {
		clear(seed_buffer[index]);
		index.z++;
	}
}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
	uint3 index = make_uint3(launch_index, seed_buffer.size().z - 1u); // record error to last segment
	seed_buffer[index].pos = exceptionToFloat3( code );
	seed_buffer[index].dir = make_float3( 0.0f );
#ifdef AMBIENT_CELL
	seed_buffer[index].cell = make_uint2(0);
#endif
}
