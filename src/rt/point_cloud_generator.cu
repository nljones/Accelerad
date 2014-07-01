/*
 * Copyright (c) 2013-2014 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix_world.h>
#include "optix_shader_common.h"

using namespace optix;

/* Program variables */
rtDeclareVariable(unsigned int,  camera, , ) = 0u;
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , ); /* view.hvec */
rtDeclareVariable(float3,        V, , ); /* view.vvec */
rtDeclareVariable(float3,        W, , ); /* view.vdir */
rtDeclareVariable(float2,        fov, , );
rtDeclareVariable(float2,        shift, , );
rtDeclareVariable(float2,        clip, , );
rtDeclareVariable(float,         dstrpix, , ); /* Pixel sample jitter (-pj) */

/* Contex variables */
rtBuffer<PointDirection, 3>      seed_buffer;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  point_cloud_ray_type, , );
rtDeclareVariable(unsigned int,  seeds, , ) = 1u; /* number of seed points to discover per thread */

/* OptiX variables */
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

static __device__ float3 uniform_solid_angle( float2 in )
{
	float2 d = 2.0f * in - 1.0f; // map input from [0, 1] to [-1, 1]
	float z = d.y > 0.0f ? 1.0f : -1.0f;
	if ( ( abs( d.x ) < FTINY ) && ( abs( d.y ) < FTINY ) )
		return make_float3( 0.0f, 0.0f, z );

	d.y = 2.0f * d.y - z;
	float s, x, y;
	if ( abs( d.x ) > abs( d.y ) ) {
		float angle = M_PI_4f * d.y / d.x;
		s = d.x;
		x = cosf( angle );
		y = sinf( angle );
	} else {
		float angle = M_PI_4f * d.x / d.y;
		s = d.y;
		x = sinf( angle );
		y = cosf( angle );
	}
	float r = s * sqrtf( 2.0f - s * s );
	return make_float3( r * x, r * y, z - z * s * s );
}

RT_PROGRAM void point_cloud_camera()
{
	PerRayData_point_cloud prd;
	prd.backup.pos = make_float3( 0.0f );
	prd.backup.dir = make_float3( 0.0f );

	// Init random state
	rand_state state;
	curand_init( launch_index.x + launch_dim.x * launch_index.y, 0, 0, &state );

	uint3 index = make_uint3( launch_index, 0u );

	float3 direction;
	float2 d = make_float2( curand_uniform( &state ), curand_uniform( &state ) );
	d = 0.5f + dstrpix * ( 0.5f - d ); // this is pixjitter() from rpict.c

	// Set initial ray direction
	if ( camera ) { // using the camera viewport
		d = shift + ( make_float2( launch_index ) + d ) / make_float2( launch_dim ) - 0.5f;
		float3 ray_origin = eye;
		float z = 1.0f;

		// This is adapted from viewray() in image.c.
  		if( camera == VT_PAR ) { /* parallel view */
			ray_origin += d.x*U + d.y*V;
			d = make_float2( 0.0f );
		} else if ( camera == VT_HEM ) { /* hemispherical fisheye */
			z = 1.0f - d.x*d.x * dot( U, U ) - d.y*d.y * dot( V, V );
			if (z < 0.0f) {
				seed_buffer[index] = prd.backup;//TODO throw an exception?
				return;
			}
			z = sqrtf(z);
		} else if ( camera == VT_CYL ) { /* cylindrical panorama */
			float dd = d.x * fov.x * ( M_PIf / 180.0f );
			z = cosf( dd );
			d.x = sinf( dd );
		} else if ( camera == VT_ANG ) { /* angular fisheye */
			d *= fov / 180.0f;
			float dd = sqrtf( dot( d, d ) );
			if (dd > 1.0f) {
				seed_buffer[index] = prd.backup;//TODO throw an exception?
				return;
			}
			z = cosf( M_PIf * dd );
			d *= sqrtf( 1.0f - z*z ) / dd;
		} else if ( camera == VT_PLS ) { /* planispheric fisheye */
			d *= make_float2( sqrtf( dot( U, U ) ), sqrtf( dot( V, V ) ) );
			float dd = dot( d, d );
			z = ( 1.0f - dd ) / ( 1.0f + dd );
			d *= 1.0f + z;
		}

		direction = normalize(d.x*U + d.y*V + z*W);
	} else { // using a sphere with equal solid angle divisions
		d = ( make_float2( launch_index ) + d ) / make_float2( launch_dim );// - 0.5f;

		// Get the position and normal of the first ray
		direction = uniform_solid_angle( d );
	}

	Ray ray = make_Ray(eye, direction, point_cloud_ray_type, RAY_START, RAY_END);

	unsigned int loop = 2u * seeds; // Prevent infinite looping
	while ( index.z < seeds && loop-- ) {
		// Trace the current ray
		rtTrace(top_object, ray, prd);

		// Check for a valid result
		if ( isfinite( prd.result.pos ) && dot( prd.result.dir, prd.result.dir ) > FTINY ) { // NaN values will be false
			seed_buffer[index] = prd.result; // This could contain points on glass materials
			index.z++;
		} else {
			prd.result.pos = eye;
			prd.result.dir = direction;
		}

		// Prepare for next ray
		ray.origin = prd.result.pos;
		//ray.direction = reflect( ray.direction, prd.result.dir );

		float3 uz = normalize( prd.result.dir );
		float3 uy = cross_direction( uz );
		float3 ux = normalize( cross( uy, uz ) );
		uy = normalize( cross( uz, ux ) );

		float zd = sqrtf( curand_uniform( &state ) );
		float phi = 2.0f*M_PIf * curand_uniform( &state );
		float xd = cosf(phi) * zd;
		float yd = sinf(phi) * zd;
		zd = sqrtf(1.0f - zd*zd);
		ray.direction = normalize( xd*ux + yd*uy + zd*uz );
	}

	// If outdoors, there are no bounces, but we need to prevent junk data
	prd.backup.pos = make_float3( 0.0f );
	prd.backup.dir = make_float3( 0.0f );
	while ( index.z < seeds ) {
		seed_buffer[index] = prd.backup;
		index.z++;
	}
}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
	uint3 index = make_uint3( launch_index, seeds - 1u ); // record error to last segment
	seed_buffer[index].pos = exceptionToFloat3( code );
	seed_buffer[index].dir = make_float3( 0.0f );
}
