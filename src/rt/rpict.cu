/*
 *  rpict.cu - entry point for image generation on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>
#include "optix_shader_ray.h"
#ifdef CONTRIB_DOUBLE
#include "optix_double.h"
#endif

using namespace optix;

/* Contex variables */
rtDeclareVariable(unsigned int,  frame, , ); /* Current frame number, starting from zero */
rtDeclareVariable(unsigned int,  camera, , ); /* Camera type (-vt) */
rtDeclareVariable(float3,        eye, , ); /* Eye position (-vp) */
rtDeclareVariable(float3,        U, , ); /* view.hvec */
rtDeclareVariable(float3,        V, , ); /* view.vvec */
rtDeclareVariable(float3,        W, , ); /* view.vdir */
rtDeclareVariable(float2,        fov, , ); /* Field of view (-vh, -vv) */
rtDeclareVariable(float2,        shift, , ); /* Camera shift (-vs, -vl) */
rtDeclareVariable(float2,        clip, , ); /* Fore and aft clipping planes (-vo, -va) */
rtDeclareVariable(float,         vdist, , ); /* Focal length */
rtDeclareVariable(float,         dstrpix, , ) = 0.0f; /* Pixel sample jitter (-pj) */
rtDeclareVariable(float,         mblur, , ) = 0.0f; /* Motion blur (-pm) */
rtDeclareVariable(float,         dblur, , ) = 0.0f; /* Depth-of-field blur (-pd) */

rtBuffer<float4, 2>              output_buffer;
#ifdef RAY_COUNT
rtBuffer<unsigned int, 2>        ray_count_buffer;
#endif
rtBuffer<RayParams, 2>           last_view_buffer;
//rtBuffer<unsigned int, 2>        rnd_seeds;
rtDeclareVariable(rtObject,      top_object, , );

/* OptiX variables */
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;

//#define TIME_VIEW


// Pick the ray direction based on camera type as in image.c.
RT_PROGRAM void ray_generator()
{
#ifdef TIME_VIEW
	clock_t t0 = clock();
	output_buffer[launch_index] = make_float4( t0 );
#endif
	PerRayData_radiance prd;
	prd.result = prd.mirror = make_float3(0.0f);
	prd.distance = prd.mirror_distance = RAY_END;
	prd.weight = 1.0f;
	prd.depth = 0;
	prd.ambient_depth = 0;
	//prd.seed = rnd_seeds[launch_index];
#ifdef CONTRIB
	prd.rcoef = make_contrib3(1.0f); //Probably not necessary
#endif
#ifdef ANTIMATTER
	prd.mask = 0u;
	prd.inside = 0;
#endif
	setupPayload(prd);

	init_rand(&prd.state, launch_index.x + launch_dim.x * launch_index.y);

	float2 d = make_float2( curand_uniform( prd.state ), curand_uniform( prd.state ) );
	d = 0.5f + dstrpix * ( 0.5f - d ); // this is pixjitter() from rpict.c
	d = shift + ( make_float2( launch_index ) + d ) / make_float2( launch_dim ) - 0.5f;
	float3 ray_origin = eye;
	float z = 1.0f;

	// This is adapted from viewray() in image.c.
  	if( camera == VT_PAR ) { /* parallel view */
		ray_origin += d.x*U + d.y*V;
		d = make_float2( 0.0f );
	} else if ( camera == VT_HEM ) { /* hemispherical fisheye */
		z = 1.0f - d.x*d.x * dot( U, U ) - d.y*d.y * dot( V, V );
		if (z < 0.0f)
			goto done;
		z = sqrtf(z);
	} else if ( camera == VT_CYL ) { /* cylindrical panorama */
		float dd = d.x * fov.x * ( M_PIf / 180.0f );
		z = cosf( dd );
		d.x = sinf( dd );
	} else if ( camera == VT_ANG ) { /* angular fisheye */
		d *= fov / 180.0f;
		float dd = length(d);
		if (dd > 1.0f)
			goto done;
		z = cosf( M_PIf * dd );
		d *= dd < FTINY ? M_PIf : sqrtf(1.0f - z*z) / dd;
	} else if ( camera == VT_PLS ) { /* planispheric fisheye */
		d *= make_float2(length(U), length(V));
		float dd = dot( d, d );
		z = ( 1.0f - dd ) / ( 1.0f + dd );
		d *= 1.0f + z;
	}

	do { // do-while for variable scoping
		float3 ray_direction = d.x*U + d.y*V + z*W;
		ray_origin += clip.x * ray_direction;
		ray_direction = normalize(ray_direction);

		// Zero or negative aft clipping distance indicates infinity
		prd.tmax = clip.y - clip.x;
		if (prd.tmax <= FTINY) {
			prd.tmax = RAY_END;
		}
		float distance = vdist;

		/* optional motion blur */
		if (mblur > FTINY) {
			RayParams next;
			next.aft = prd.tmax;
			next.origin = ray_origin;
			next.direction = ray_direction;
			next.distance = distance;

			if (frame) {
				RayParams prev = last_view_buffer[launch_index];
				z = mblur * (0.5f - curand_uniform(prd.state));

				prd.tmax = lerp(prd.tmax, prev.aft, z);
				ray_origin = lerp(ray_origin, prev.origin, z);
				ray_direction = normalize(lerp(ray_direction, prev.direction, z));
				distance = lerp(distance, prev.distance, z);
			}

			last_view_buffer[launch_index] = next;
		}

		/* optional depth-of-field */
		if (dblur > FTINY) {
			float adj = 1.0f;
			z = 0.0f;

			/* random point on disk */
			SDsquare2disk(d, curand_uniform(prd.state), curand_uniform(prd.state));
			d *= 0.5f * dblur;
			if ((camera == VT_PER) | (camera == VT_PAR)) {
				if (camera == VT_PER)
					adj /= dot(ray_direction, W);
			}
			else {			/* non-standard view case */
				z = M_PI_4f * dblur * (0.5f - curand_uniform(prd.state));
			}
			if ((camera != VT_ANG) & (camera != VT_PLS)) {
				if (camera != VT_CYL)
					d.x /= length(U);
				d.y /= length(V);
			}
			ray_origin += d.x * U + d.y * V + z * W;
			ray_direction = normalize(eye + adj * distance * ray_direction - ray_origin);
		}

		Ray ray = make_Ray(ray_origin, ray_direction, RADIANCE_RAY, 0.0f, prd.tmax);

		rtTrace(top_object, ray, prd);

		checkFinite(prd.result);
	} while (0);

done:
#ifdef TIME_VIEW
	clock_t t1 = clock();
 
	float expected_fps   = 1.0f;
	float pixel_time     = ( t1 - t0 ) * time_view_scale * expected_fps;
	output_buffer[launch_index] = make_float4( pixel_time );
#else
	output_buffer[launch_index] = make_float4(prd.result, rayDistance(prd));
#endif
#ifdef RAY_COUNT
	ray_count_buffer[launch_index] = prd.ray_count;
#endif
}

RT_PROGRAM void exception()
{
#ifdef PRINT_OPTIX
	rtPrintExceptionDetails();
#endif
#ifdef TIME_VIEW
	clock_t t1 = clock();
 
	float expected_fps   = 1.0f;
	float pixel_time     = ( t1 - output_buffer[launch_index].x ) * time_view_scale * expected_fps;
	output_buffer[launch_index] = make_float4( pixel_time );
#else
	output_buffer[launch_index] = exceptionToFloat4(rtGetExceptionCode());
#endif
}
