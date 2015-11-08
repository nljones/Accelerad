/*
* Copyright (c) 2013-2015 Nathaniel Jones
* Massachusetts Institute of Technology
*/

#include <optix_world.h>
#include "optix_shader_common.h"

#define GAMMA  2.2f
#define LUMINOUS_EFFICACY 179.0f

using namespace optix;

/* Program variables */
rtDeclareVariable(unsigned int, frame, , ); /* Current frame number, starting from zero */
rtDeclareVariable(unsigned int, camera, , ); /* Camera type (-vt) */
rtDeclareVariable(float3, eye, , ); /* Eye position (-vp) */
rtDeclareVariable(float3, U, , ); /* view.hvec */
rtDeclareVariable(float3, V, , ); /* view.vvec */
rtDeclareVariable(float3, W, , ); /* view.vdir */
rtDeclareVariable(float2, fov, , ); /* Field of view (-vh, -vv) */
rtDeclareVariable(float2, shift, , ); /* Camera shift (-vs, -vl) */
rtDeclareVariable(float2, clip, , ); /* Fore and aft clipping planes (-vo, -va) */
rtDeclareVariable(float, vdist, , ); /* Focal length */
rtDeclareVariable(float, dstrpix, , ); /* Pixel sample jitter (-pj) */
rtDeclareVariable(unsigned int, do_irrad, , ); /* Calculate irradiance (-i) */

rtDeclareVariable(float, exposure, , ) = 1.0f; /* Current exposure */
rtDeclareVariable(unsigned int, greyscale, , ) = 0u; /* Convert to monocrhome */
rtDeclareVariable(int, tonemap, , ) = RT_TEXTURE_ID_NULL; /* texture ID */
rtDeclareVariable(float, fc_scale, , ) = 1000.0f; /* Current exposure */
rtDeclareVariable(int, fc_log, , ) = 0; /* Current exposure */
rtDeclareVariable(float, fc_mask, , ) = 0.0f; /* Current exposure */

/* Contex variables */
rtBuffer<unsigned int, 2>        output_buffer;
rtBuffer<float3, 2>              direct_buffer; /* GPU storage for direct component */
rtBuffer<float3, 2>              diffuse_buffer; /* GPU storage for diffuse component */
#ifdef RAY_COUNT
rtBuffer<unsigned int, 2>        ray_count_buffer;
#endif
//rtBuffer<RayParams, 2>           last_view_buffer;
//rtBuffer<unsigned int, 2>        rnd_seeds;
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, radiance_primary_ray_type, , );
rtDeclareVariable(unsigned int, diffuse_ray_type, , );

/* OptiX variables */
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );


// Pick the ray direction based on camera type as in image.c.
RT_PROGRAM void ray_generator()
{
	PerRayData_radiance prd;
	init_rand(&prd.state, launch_index.x + launch_dim.x * (launch_index.y + launch_dim.y * frame));

	float2 d = make_float2(curand_uniform(prd.state), curand_uniform(prd.state));
	d = 0.5f + dstrpix * (0.5f - d); // this is pixjitter() from rpict.c
	d = shift + (make_float2(launch_index) + d) / make_float2(launch_dim) - 0.5f;
	float3 ray_origin = eye;
	float z = 1.0f;

	// This is adapted from viewray() in image.c.
	if (camera == VT_PAR) { /* parallel view */
		ray_origin += d.x*U + d.y*V;
		d = make_float2(0.0f);
	}
	else if (camera == VT_HEM) { /* hemispherical fisheye */
		z = 1.0f - d.x*d.x * dot(U, U) - d.y*d.y * dot(V, V);
		if (z < 0.0f) {
			output_buffer[launch_index] = 0xff000000;//TODO throw an exception?
			return;
		}
		z = sqrtf(z);
	}
	else if (camera == VT_CYL) { /* cylindrical panorama */
		float dd = d.x * fov.x * (M_PIf / 180.0f);
		z = cosf(dd);
		d.x = sinf(dd);
	}
	else if (camera == VT_ANG) { /* angular fisheye */
		d *= fov / 180.0f;
		float dd = length(d);
		if (dd > 1.0f) {
			output_buffer[launch_index] = 0xff000000;//TODO throw an exception?
			return;
		}
		z = cosf(M_PIf * dd);
		d *= sqrtf(1.0f - z*z) / dd;
	}
	else if (camera == VT_PLS) { /* planispheric fisheye */
		d *= make_float2(length(U), length(V));
		float dd = dot(d, d);
		z = (1.0f - dd) / (1.0f + dd);
		d *= 1.0f + z;
	}

	float3 ray_direction = d.x*U + d.y*V + z*W;
	ray_origin += clip.x * ray_direction;
	ray_direction = normalize(ray_direction);

	// Zero or negative aft clipping distance indicates infinity
	float aft = clip.y - clip.x;
	if (aft <= FTINY) {
		aft = RAY_END;
	}

	Ray ray = make_Ray(ray_origin, ray_direction, frame ? diffuse_ray_type : do_irrad ? radiance_primary_ray_type : radiance_ray_type, 0.0f, aft);

	prd.weight = 1.0f;
	prd.depth = 0;
	prd.ambient_depth = 0;
	//prd.seed = rnd_seeds[launch_index];
	setupPayload(prd, 1);

	rtTrace(top_object, ray, prd);

	checkFinite(prd.result);

	float3 accum;
	if (frame)
		accum = direct_buffer[launch_index] + (diffuse_buffer[launch_index] = ((frame - 1.0f) / frame) * diffuse_buffer[launch_index] + (1.0f / frame) * prd.result);
	else
		accum = direct_buffer[launch_index] = prd.result;

	/* Tone map */
	if (tonemap == RT_TEXTURE_ID_NULL) {
		accum *= exposure;
		if (greyscale)
			accum = make_float3(bright(accum));
	}
	else {
		float luminance = bright(accum) * LUMINOUS_EFFICACY;
		if (luminance < fc_mask)
			accum = make_float3(0.0f);
		else if (fc_log > 0)
			accum = make_float3(rtTex1D<float4>(tonemap, log10f(luminance / fc_scale) / fc_log + 1.0f));
		else
			accum = make_float3(rtTex1D<float4>(tonemap, luminance / fc_scale));
	}
	accum = clamp(accum * 256.0f, 0.0f, 255.0f);

	output_buffer[launch_index] = 0xff000000 |
		((int)(256.0f * powf((accum.x + 0.5f) / 256.0f, 1.0f / GAMMA)) & 0xff) << 16 |
		((int)(256.0f * powf((accum.y + 0.5f) / 256.0f, 1.0f / GAMMA)) & 0xff) << 8 |
		((int)(256.0f * powf((accum.z + 0.5f) / 256.0f, 1.0f / GAMMA)) & 0xff);

#ifdef RAY_COUNT
	if (frame)
		ray_count_buffer[launch_index] += prd.ray_count;
	else
		ray_count_buffer[launch_index] = prd.ray_count;
#endif
}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf("Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y);
	output_buffer[launch_index] = code;
}
