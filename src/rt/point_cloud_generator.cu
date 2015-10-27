/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix_world.h>
#include "optix_shader_common.h"

using namespace optix;

/* Program variables */
rtDeclareVariable(unsigned int,  camera, , ); /* Camera type (-vt) */
rtDeclareVariable(float3,        eye, , ); /* Eye position (-vp) */
rtDeclareVariable(float3,        U, , ); /* view.hvec */
rtDeclareVariable(float3,        V, , ); /* view.vvec */
rtDeclareVariable(float3,        W, , ); /* view.vdir */
rtDeclareVariable(float2,        fov, , ); /* Field of view (-vh, -vv) */
rtDeclareVariable(float2,        shift, , ); /* Camera shift (-vs, -vl) */
rtDeclareVariable(float2,        clip, , ); /* Fore and aft clipping planes (-vo, -va) */
rtDeclareVariable(float,         dstrpix, , ); /* Pixel sample jitter (-pj) */

/* Contex variables */
rtBuffer<PointDirection, 3>      seed_buffer;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  point_cloud_ray_type, , );

/* OptiX variables */
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

RT_METHOD float3 uniform_solid_angle( float2 in )
{
	float2 d = 2.0f * in - 1.0f; // map input from [0, 1] to [-1, 1]
	float z = d.y > 0.0f ? 1.0f : -1.0f;
	if ( ( fabsf( d.x ) < FTINY ) && ( fabsf( d.y ) < FTINY ) )
		return make_float3( 0.0f, 0.0f, z );

	d.y = 2.0f * d.y - z;
	float s, x, y;
	if ( fabsf( d.x ) > fabsf( d.y ) ) {
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

	// Init random state
	rand_state* state;
	init_rand(&state, launch_index.x + launch_dim.x * launch_index.y);

	uint3 index = make_uint3( launch_index, 0u );
	const unsigned int seeds = seed_buffer.size().z;
	unsigned int loop = 2u * seeds; // Prevent infinite looping

	Ray ray;
	ray.origin = eye;
	ray.ray_type = point_cloud_ray_type;
	ray.tmin = 0.0f;

	float2 d = make_float2( curand_uniform( state ), curand_uniform( state ) );
	d = 0.5f + dstrpix * ( 0.5f - d ); // this is pixjitter() from rpict.c

	// Set initial ray direction
	if ( camera ) { // using the camera viewport
		d = shift + ( make_float2( launch_index ) + d ) / make_float2( launch_dim ) - 0.5f;
		float z = 1.0f;

		// This is adapted from viewray() in image.c.
  		if( camera == VT_PAR ) { /* parallel view */
			ray.origin += d.x*U + d.y*V;
			d = make_float2( 0.0f );
		} else if ( camera == VT_HEM ) { /* hemispherical fisheye */
			z = 1.0f - d.x*d.x * dot( U, U ) - d.y*d.y * dot( V, V );
			if (z < 0.0f)
				goto clearout;
			z = sqrtf(z);
		} else if ( camera == VT_CYL ) { /* cylindrical panorama */
			float dd = d.x * fov.x * ( M_PIf / 180.0f );
			z = cosf( dd );
			d.x = sinf( dd );
		} else if ( camera == VT_ANG ) { /* angular fisheye */
			d *= fov / 180.0f;
			float dd = length(d);
			if (dd > 1.0f)
				goto clearout;
			z = cosf( M_PIf * dd );
			d *= sqrtf( 1.0f - z*z ) / dd;
		} else if ( camera == VT_PLS ) { /* planispheric fisheye */
			d *= make_float2(length(U), length(V));
			float dd = dot( d, d );
			z = ( 1.0f - dd ) / ( 1.0f + dd );
			d *= 1.0f + z;
		}

		ray.direction = d.x*U + d.y*V + z*W;
		ray.direction += clip.x * ray.direction;
		ray.direction = normalize(ray.direction);

		// Zero or negative aft clipping distance indicates infinity
		ray.tmax = clip.y - clip.x;
		if (ray.tmax <= FTINY) {
			ray.tmax = RAY_END;
		}
	} else { // using a sphere with equal solid angle divisions
		d = ( make_float2( launch_index ) + d ) / make_float2( launch_dim );// - 0.5f;

		// Get the position and normal of the first ray
		ray.direction = uniform_solid_angle(d);
		ray.tmax = RAY_END;
	}

	while ( index.z < seeds && loop-- ) {
		clear(prd.backup);

		// Trace the current ray
		rtTrace(top_object, ray, prd);

		// Check for a valid result
		if ( isfinite( prd.result.pos ) && dot( prd.result.dir, prd.result.dir ) > FTINY ) { // NaN values will be false
			seed_buffer[index] = prd.result; // This could contain points on glass materials
			index.z++;
#ifndef AMBIENT_CELL
		} else {
			prd.result.pos = eye;
			prd.result.dir = ray.direction;
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

clearout:
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
