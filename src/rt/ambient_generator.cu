/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix_world.h>
#include "optix_shader_common.h"

using namespace optix;

/* Program variables */
rtDeclareVariable(unsigned int,  camera, , );
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , ); /* view.hvec */
rtDeclareVariable(float3,        V, , ); /* view.vvec */
rtDeclareVariable(float3,        W, , ); /* view.vdir */
rtDeclareVariable(float2,        fov, , );
rtDeclareVariable(float2,        shift, , );
rtDeclareVariable(float2,        clip, , );
//rtDeclareVariable(float,         dstrpix, , ); /* Pixel sample jitter (-pj) */

/* Contex variables */
rtBuffer<AmbientRecord, 3>       ambient_record_buffer; /* output */
//rtBuffer<unsigned int, 2>        rnd_seeds;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  ambient_record_ray_type, , );
rtDeclareVariable(unsigned int,  segments, , );
rtDeclareVariable(unsigned int,  level, , ) = 0u;

/* OptiX variables */
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

// Initialize the random state
RT_METHOD void init_state( PerRayData_ambient_record* prd )
{
	rand_state state;
	prd->state = &state;
	curand_init( launch_index.x + launch_dim.x * launch_index.y, 0, 0, prd->state );
}

RT_METHOD float2 get_offset( unsigned int segment )
{
	float2 offset = make_float2( 0.5f );
	float delta = 0.5f;

	for ( ; segment > 0u; segment >>= 2 ) {
		unsigned int x = segment & 1u;
		unsigned int y = ( segment >> 1 ) & 1u;
		y = x ^ y;
		if ( x ) offset.x += delta;
		if ( y ) offset.y += delta;
		delta *= -0.5f;
	}

	return offset;
}

// Pick the ray direction based on camera type as in image.c.
RT_PROGRAM void ambient_camera()
{
	PerRayData_ambient_record prd;
	init_state( &prd );
	prd.parent = NULL;
	prd.result.pos = prd.result.val = make_float3( 0.0f );
	prd.result.lvl = level;
	prd.result.weight = 1.0f;
	for ( int i = level; i--; )
		prd.result.weight *= AVGREFL; // Compute weight as in makeambient() from ambient.c

	uint3 index = make_uint3( launch_index, 0u );
	unsigned int power_of_two = 0u; // always a power of 2

	// Zero or negative aft clipping distance indicates infinity
	float aft = clip.y - clip.x;
	if (aft <= FTINY) {
		aft = RAY_END;
	}

	for ( unsigned int segment = 0u; segment < segments; segment++ ) {
		// Choose the parent
		if ( segment ) {
			// Level up if we've reached a new power of 2
			if ( !( segment & ( segment - 1u ) ) )
				power_of_two = segment;

			index.z = segment - power_of_two;
			AmbientRecord record = ambient_record_buffer[index];
			prd.parent = &record;
			// Check that the parent was computed
			if ( prd.parent->weight < FTINY )
				continue;
		}

		//float2 d = make_float2( curand_uniform( state ), curand_uniform( state ) );
		//d = 0.5f + dstrpix * ( 0.5f - d ); // this is pixjitter() from rpict.c
		float2 d = get_offset( segment );
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
				//ambient_record_buffer[launch_index] = make_float4( 0.0f );//TODO throw an exception?
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
				//ambient_record_buffer[launch_index] = make_float4( 0.0f );//TODO throw an exception?
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

		float3 ray_direction = d.x*U + d.y*V + z*W;
		ray_origin += clip.x * ray_direction;
		ray_direction = normalize(ray_direction);

		Ray ray = make_Ray(ray_origin, ray_direction, ambient_record_ray_type, 0.0f, aft);

#ifndef OLDAMB
		prd.result.rad = make_float2( 0.0f );
#else
		prd.result.rad = 0.0f;
#endif
#ifdef RAY_COUNT
		prd.result.ray_count = 0;
#endif
#ifdef HIT_COUNT
		prd.result.hit_count = 0;
#endif

		rtTrace(top_object, ray, prd);

		checkFinite(prd.result.val);

		index.z = segment;
		ambient_record_buffer[index] = prd.result;
	}
}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
	uint3 index = make_uint3( launch_index, segments - 1u ); // record error to last segment
	ambient_record_buffer[index].lvl = level;
	ambient_record_buffer[index].val = exceptionToFloat3( code );
	ambient_record_buffer[index].weight = -1.0f;
}
