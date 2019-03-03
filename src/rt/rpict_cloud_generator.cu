/*
 *  rpict_cloud_generator.cu - entry point for geometry sampling for image generation on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>
#include "optix_shader_common.h"
#include "optix_point_common.h"

#define RING_BUFFER_SIZE	8

using namespace optix;

/* Contex variables */
rtDeclareVariable(unsigned int,  camera, , ); /* Camera type (-vt) */
rtDeclareVariable(float3,        eye, , ); /* Eye position (-vp) */
rtDeclareVariable(float3,        U, , ); /* view.hvec */
rtDeclareVariable(float3,        V, , ); /* view.vvec */
rtDeclareVariable(float3,        W, , ); /* view.vdir */
rtDeclareVariable(float2,        fov, , ); /* Field of view (-vh, -vv) */
rtDeclareVariable(float2,        shift, , ); /* Camera shift (-vs, -vl) */
rtDeclareVariable(float2,        clip, , ); /* Fore and aft clipping planes (-vo, -va) */
rtDeclareVariable(float,         dstrpix, , ); /* Pixel sample jitter (-pj) */

rtBuffer<PointDirection, 3>      seed_buffer;
rtDeclareVariable(rtObject,      top_object, , );

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
	init_rand(&prd.state, launch_index.x + launch_dim.x * launch_index.y);

	prd.index = make_uint3(launch_index, 0u);
	prd.seeds = seed_buffer.size().z;
	unsigned int loop = 2u * prd.seeds; // Prevent infinite looping

	float3 point_ring[RING_BUFFER_SIZE];
	float3 dir_ring[RING_BUFFER_SIZE];
	unsigned int ring_start = 0, ring_end = 0, ring_full = 0;

	Ray ray;
	ray.origin = eye;
	ray.ray_type = POINT_CLOUD_RAY;
	ray.tmin = 0.0f;

	float2 d = make_float2(curand_uniform(prd.state), curand_uniform(prd.state));
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
			d *= dd < FTINY ? M_PIf : sqrtf(1.0f - z*z) / dd;
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

	while (prd.index.z < prd.seeds && loop--) {
		prd.forward = prd.reverse = make_float3(0.0f);
#ifdef ANTIMATTER
		prd.mask = 0u;
		prd.inside = 0;
#endif

		// Trace the current ray
		rtTrace(top_object, ray, prd);

		// Add next forward ray to ring buffer
		if (isfinite(prd.point) && dot(prd.forward, prd.forward) > FTINY) { // NaN values will be false
			point_ring[ring_end] = prd.point;
			dir_ring[ring_end] = prd.forward;
			ring_end = (ring_end + 1) % RING_BUFFER_SIZE;
			ring_full = ring_start == ring_end;
		}

		// Add next reverse ray to ring buffer
		if (!ring_full && isfinite(prd.point) && dot(prd.reverse, prd.reverse) > FTINY) { // NaN values will be false
			point_ring[ring_end] = prd.point;
			dir_ring[ring_end] = prd.reverse;
			ring_end = (ring_end + 1) % RING_BUFFER_SIZE;
			ring_full = ring_start == ring_end;
		}

		if (!ring_full && ring_start == ring_end)
			break;

		// Prepare for next ray
		ray.origin = point_ring[ring_start];
		ray.direction = dir_ring[ring_start];
		ring_start = (ring_start + 1) % RING_BUFFER_SIZE;
		ring_full = 0;
		ray.tmin = ray_start(ray.origin, RAY_START);
		ray.tmax = RAY_END;
	}

clearout:
	// If outdoors, there are no bounces, but we need to prevent junk data
	while (prd.index.z < prd.seeds) {
		clear(seed_buffer[prd.index]);
		prd.index.z++;
	}
}

RT_PROGRAM void exception()
{
#ifdef PRINT_OPTIX
	rtPrintExceptionDetails();
#endif
	uint3 index = make_uint3(launch_index, seed_buffer.size().z - 1u); // record error to last segment
	seed_buffer[index].pos = exceptionToFloat3(rtGetExceptionCode());
	seed_buffer[index].dir = make_float3( 0.0f );
#ifdef AMBIENT_CELL
	seed_buffer[index].cell = make_uint2(0);
#endif
}
