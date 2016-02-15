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
rtBuffer<unsigned int, 2>        color_buffer; /* Output RGBA colors */
rtBuffer<Metrics, 2>             metrics_buffer; /* Output metrics */
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


RT_METHOD float3 getViewDirection(const float &x, const float &y);
RT_METHOD int splane_normal(const float3 &e1, const float3 &e2, float3 &n);
RT_METHOD float getSolidAngle();
RT_METHOD float getPositionIndex(const float3 &dir);


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
			color_buffer[launch_index] = 0xff000000;//TODO throw an exception?
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
			color_buffer[launch_index] = 0xff000000;//TODO throw an exception?
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

	prd.result = make_float3(0.0f);
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
	const float luminance = bright(accum) * LUMINOUS_EFFICACY;
	if (tonemap == RT_TEXTURE_ID_NULL) {
		accum *= exposure;
		if (greyscale)
			accum = make_float3(bright(accum));
	}
	else {
		if (luminance < fc_mask)
			accum = make_float3(0.0f);
		else if (fc_log > 0)
			accum = make_float3(rtTex1D<float4>(tonemap, log10f(luminance / fc_scale) / fc_log + 1.0f));
		else
			accum = make_float3(rtTex1D<float4>(tonemap, luminance / fc_scale));
	}
	accum = clamp(accum * 256.0f, 0.0f, 255.0f);

	/* Save pixel color */
	color_buffer[launch_index] = 0xff000000 |
		((int)(256.0f * powf((accum.x + 0.5f) / 256.0f, 1.0f / GAMMA)) & 0xff) << 16 |
		((int)(256.0f * powf((accum.y + 0.5f) / 256.0f, 1.0f / GAMMA)) & 0xff) << 8 |
		((int)(256.0f * powf((accum.z + 0.5f) / 256.0f, 1.0f / GAMMA)) & 0xff);

	/* Calculate metrics */
	float guth = getPositionIndex(ray.direction);
	Metrics metrics;
	metrics.omega = getSolidAngle(); //TODO what if negative or bad angle?
	metrics.ev = luminance * dot(W, ray.direction * metrics.omega);
	metrics.avlum = luminance * metrics.omega;
	metrics.dgp = luminance * luminance * metrics.omega / (guth * guth);
	metrics_buffer[launch_index] = metrics;

#ifdef RAY_COUNT
	if (frame)
		ray_count_buffer[launch_index] += prd.ray_count;
	else
		ray_count_buffer[launch_index] = prd.ray_count;
#endif
}

/* From viewray() in image.c */
RT_METHOD float3 getViewDirection(const float &x, const float &y)
{
	float2 d = make_float2(x, y) - 0.5f;
	float z = 1.0f;

	if (camera == VT_PAR) { /* parallel view */
		d = make_float2(0.0f);
	}
	else if (camera == VT_HEM) { /* hemispherical fisheye */
		z = 1.0f - d.x*d.x * dot(U, U) - d.y*d.y * dot(V, V);
		if (z < 0.0f)
			return make_float3(0.0f);
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
		if (dd > 1.0f)
			return make_float3(0.0f);
		z = cosf(M_PIf * dd);
		d *= sqrtf(1.0f - z*z) / dd;
	}
	else if (camera == VT_PLS) { /* planispheric fisheye */
		d *= make_float2(length(U), length(V));
		float dd = dot(d, d);
		z = (1.0f - dd) / (1.0f + dd);
		d *= 1.0f + z;
	}

	return d.x*U + d.y*V + z*W;
}

/* From splane_normal in pictool.c */
RT_METHOD int splane_normal(const float3 &e1, const float3 &e2, float3 &n)
{
	n = cross(e1, e2 - e1);
	if (dot(n, n) == 0.0f)
		return 0;
	n = normalize(n);
	return 1;
}

/* From pict_get_sangle in pictool.c */
RT_METHOD float getSolidAngle()
{
	const float2 min = make_float2(launch_index) / make_float2(launch_dim);
	const float2 max = (make_float2(launch_index) + 1.0f) / make_float2(launch_dim);
	const float3 minmin = getViewDirection(min.x, min.y);
	const float3 minmax = getViewDirection(min.x, max.y);
	const float3 maxmin = getViewDirection(max.x, min.y);
	const float3 maxmax = getViewDirection(max.x, max.y);

	float3 n[4] = { make_float3(0.0f), make_float3(0.0f), make_float3(0.0f), make_float3(0.0f) };

	int i = splane_normal(minmin, minmax, n[0]);
	i &= splane_normal(minmax, maxmax, n[1]);
	i &= splane_normal(maxmax, maxmin, n[2]);
	i &= splane_normal(maxmin, minmin, n[3]);

	if (!i)
		return 0.0f;
	float ang = 0.0f;
	for (i = 0; i < 4; i++) {
		float a = dot(n[i], n[(i + 1) % 4]);
		ang += M_PIf - fabsf(acosf(clamp(a, -1.0f, 1.0f)));
	}
	ang = ang - 2.0f * M_PIf;
	if ((ang > (2.0f * M_PIf)) || ang < 0) {
		//fprintf(stderr, "Normal error in pict_get_sangle %f %d %d\n", ang, x, y);
		return 0.0f;
	}
	return ang;
}

/* From get_posindex in evalglare.c */
RT_METHOD float getPositionIndex(const float3 &dir)
{
	float3 up = normalize(V); // TODO Not necessarily
	float3 hv = cross(W, up);
	float phi = acosf(dot(cross(W, hv), dir)) - M_PI_2f;
	float teta = M_PI_2f - acosf(dot(hv, dir));
	float sigma = acosf(dot(W, dir));
	hv = normalize(normalize(dir) / cosf(sigma) - W);
	float tau = acosf(dot(up, hv));
	tau *= 180.0f / M_PIf;
	sigma *= 180.0f / M_PIf;

	if (phi < FTINY)
		phi = FTINY;
	if (sigma <= 0)
		sigma = -sigma;
	if (teta < FTINY)
		teta = FTINY;

	float posindex = expf((35.2f - 0.31889f * tau - 1.22f * expf(-2.0f * tau / 9.0f)) / 1000.0f * sigma + (21.0f + 0.26667f * tau - 0.002963f * tau * tau) / 100000.0f * sigma * sigma);

	/* below line of sight, using Iwata model */
	if (phi < 0.0f) {
		float fact = 0.8f;
		float d = 1.0f / tanf(phi);
		float s = tanf(teta) / tanf(phi);
		float r = sqrtf((s * s + 1.0f) / (d * d));
		if (r > 0.6f)
			fact = 1.2f;
		if (r > 3.0f)
			r = 3.0f;

		posindex = 1.0f + fact * r;
	}
	if (posindex > 16.0f)
		posindex = 16.0f;

	return posindex;
}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf("Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y);
	color_buffer[launch_index] = 0xffffffff;
	Metrics metrics;
	metrics.omega = -1.0f;
	metrics.ev = code;
	metrics.avlum = 0.0f;
	metrics.dgp = 0.0f;
	metrics_buffer[launch_index] = metrics;
}
