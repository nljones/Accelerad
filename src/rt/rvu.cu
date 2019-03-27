/*
 *  rvu.cu - entry point for progressive rendering on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>
#include "optix_shader_ray.h"
#ifdef CONTRIB_DOUBLE
#include "optix_double.h"
#endif

#define GAMMA  2.2f
#define LUMINOUS_EFFICACY 179.0f
#define VT_ODS		'o'		/* omni-directional stereo */

using namespace optix;

#define angle(a, b)	acosf(clamp(dot(a, b), -1.0f, 1.0f))

/* Contex variables */
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
rtDeclareVariable(unsigned int, do_lum, , ); /* Calculate luminance or illuminance */
#ifdef VT_ODS
rtDeclareVariable(float, ipd, , ) = 0.07f; /* inter-pupillary distance (this is between 0.055m and 0.07m on most humans) */
rtDeclareVariable(float3, gaze, , ); /* gaze direction (may be different from W) */
#endif

rtDeclareVariable(int2, task_position, , ); /* Position of task area (-T) */
rtDeclareVariable(float, task_angle, , ) = 0.0f; /* Opening angle of task area in radians (-T) */
rtDeclareVariable(int2, high_position, , ); /* Position of contrast high luminance area (-C) */
rtDeclareVariable(float, high_angle, , ) = 0.0f; /* Opening angle of contrast high luminance area (-C) */
rtDeclareVariable(int2, low_position, , ); /* Position of contrast high luminance area (-C) */
rtDeclareVariable(float, low_angle, , ) = 0.0f; /* Opening angle of contrast high luminance area (-C) */

rtDeclareVariable(float, exposure, , ) = 1.0f; /* Current exposure (-pe) */
rtDeclareVariable(unsigned int, greyscale, , ) = 0u; /* Convert to monocrhome (-b) */
rtDeclareVariable(int, tonemap, , ) = RT_TEXTURE_ID_NULL; /* texture ID */
rtDeclareVariable(float, fc_scale, , ) = 1000.0f; /* Maximum of scale for falsecolor images, zero for regular tonemapping (-s) */
rtDeclareVariable(int, fc_log, , ) = 0; /* Number of decades for log scale, zero for standard scale (-log) */
rtDeclareVariable(int, fc_base, , ) = 10; /* Base for log scale (-base) */
rtDeclareVariable(float, fc_mask, , ) = 0.0f; /* Minimum value to display in falsecolor images (-m) */
rtDeclareVariable(unsigned int, flags, , ) = 0; /* Flags for areas to highlight in image */

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

/* OptiX variables */
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );


RT_METHOD float3 getViewDirection(float2 d);
RT_METHOD int splane_normal(const float3 &e1, const float3 &e2, float3 &n);
RT_METHOD float getSolidAngle();
RT_METHOD float getPositionIndex(const float3 &dir, const float3 &forward);
RT_METHOD int inTask(const int2 &position, const float &angle, const float3 &ray_direction);
RT_METHOD void tint(unsigned int &color, const unsigned int component);


// Pick the ray direction based on camera type as in image.c.
RT_PROGRAM void ray_generator()
{
	PerRayData_radiance prd;
	init_rand(&prd.state, launch_index.x + launch_dim.x * (launch_index.y + launch_dim.y * frame));

	float2 d = make_float2(curand_uniform(prd.state), curand_uniform(prd.state));
	d = 0.5f + dstrpix * (0.5f - d); // this is pixjitter() from rpict.c
	d = shift + (make_float2(launch_index) + d) / make_float2(launch_dim) - 0.5f;
	float3 ray_origin = eye;
	if (camera == VT_PAR) /* parallel view */
		ray_origin += d.x*U + d.y*V;
	float3 ray_direction = getViewDirection(d);
#ifdef VT_ODS
	if (camera == VT_ODS) {
		float dy = ipd * (d.y < 0 ? 0.5f : -0.5f);
		float az = d.x * fov.x * (M_PIf / 180.0f);
		ray_origin.x += cosf(az) * dy;
		ray_origin.y += sinf(az) * dy;
	}
#endif

	if (dot(ray_direction, ray_direction) > 0) {
		ray_origin += clip.x * ray_direction;
		ray_direction = normalize(ray_direction);

		// Zero or negative aft clipping distance indicates infinity
		prd.tmax = clip.y - clip.x;
		if (prd.tmax <= FTINY) {
			prd.tmax = RAY_END;
			if (frame)
				prd.tmax *= 0.9999f; // Do not let diffuse rays sample the sky
		}

		Ray ray = make_Ray(ray_origin, ray_direction, RADIANCE_RAY, 0.0f, prd.tmax);

		prd.result = make_float3(0.0f);
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

		rtTrace(top_object, ray, prd);

		checkFinite(prd.result);
	}
	else
		prd.result = make_float3(0.0f);

	float3 accum;
	if (frame)
		accum = direct_buffer[launch_index] + (diffuse_buffer[launch_index] = ((frame - 1.0f) / frame) * diffuse_buffer[launch_index] + (1.0f / frame) * prd.result);
	else
		accum = direct_buffer[launch_index] = prd.result;

	/* Tone map */
	const float efficacy = do_lum ? LUMINOUS_EFFICACY : 1.0f;
	const float luminance = bright(accum) * efficacy;
	if (tonemap == RT_TEXTURE_ID_NULL) { // Natural tone mapping
		//accum *= exposure / fc_scale;
		if (greyscale)
			accum = make_float3(luminance * exposure / fc_scale);
		else
			accum *= efficacy * exposure / fc_scale;
		if (fc_log > 0)
			accum = make_float3(logf(accum.x), logf(accum.y), logf(accum.z)) / (logf(fc_base) * fc_log) + 1.0f;
	}
	else { // False color tone mapping
		if (luminance < fc_mask)
			accum = make_float3(0.0f);
		else if (fc_log > 0)
			accum = make_float3(rtTex1D<float4>(tonemap, logf(luminance / fc_scale) / (logf(fc_base) * fc_log) + 1.0f));
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
	Metrics metrics;
	metrics.omega = getSolidAngle(); //TODO what if negative or bad angle?
	metrics.ev = metrics.dgp = 0.0f;
	if (do_irrad) {
		metrics.avlum = luminance; /* In this case it is illuminance. */
	}
	else {
		metrics.avlum = luminance * metrics.omega;
#ifdef VT_ODS
		float3 gaze_dir = dot(gaze, gaze) < FTINY ? W : gaze;
#else
		float3 gaze_dir = W;
#endif
		const float WdotD = dot(gaze_dir, ray_direction);
		if (WdotD > 0.0f) {
			metrics.ev = luminance * metrics.omega * WdotD;
			if (do_lum) {
				float guth = getPositionIndex(ray_direction, gaze_dir);
				metrics.dgp = (metric)luminance * luminance * metrics.omega / (guth * guth);
			}
		}
	}

	/* Calculate contributions to task areas */
	metrics.flags = 0;
	if (task_angle > 0.0f)
		metrics.flags |= inTask(task_position, task_angle, ray_direction) & 0x1;
	if (high_angle > 0.0f)
		metrics.flags |= (inTask(high_position, high_angle, ray_direction) & 0x1) << 1;
	if (low_angle > 0.0f)
		metrics.flags |= (inTask(low_position, low_angle, ray_direction) & 0x1) << 2;

	if (flags & metrics.flags & 0x1) tint(color_buffer[launch_index], 2);
	if (flags & metrics.flags & 0x2) tint(color_buffer[launch_index], 1);
	if (flags & metrics.flags & 0x4) tint(color_buffer[launch_index], 0);

	metrics_buffer[launch_index] = metrics;

#ifdef RAY_COUNT
	if (frame)
		ray_count_buffer[launch_index] += prd.ray_count;
	else
		ray_count_buffer[launch_index] = prd.ray_count;
#endif
}

/* From viewray() in image.c */
RT_METHOD float3 getViewDirection(float2 d)
{
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
		d *= dd < FTINY ? M_PIf : sqrtf(1.0f - z*z) / dd;
	}
	else if (camera == VT_PLS) { /* planispheric fisheye */
		d *= make_float2(length(U), length(V));
		float dd = dot(d, d);
		z = (1.0f - dd) / (1.0f + dd);
		d *= 1.0f + z;
	}
#ifdef VT_ODS
	else if (camera == VT_ODS) { /* omni-directional stereo */
		d.y *= 2.0f;
		d.y += d.y < 0 ? 0.5f : -0.5f;
		d *= fov * (M_PIf / 180.0f); // d.x = azimuth, d.y = altitude
		z = cosf(d.x) * cosf(d.y);
		d.x = sinf(d.x) * cosf(d.y);
		d.y = sinf(d.y);
	}
#endif

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
	const float2 min = shift + make_float2(launch_index) / make_float2(launch_dim) - 0.5f;
	const float2 max = shift + (make_float2(launch_index) + 1.0f) / make_float2(launch_dim) - 0.5f;
	const float3 minmin = getViewDirection(min);
	const float3 minmax = getViewDirection(make_float2(min.x, max.y));
	const float3 maxmin = getViewDirection(make_float2(max.x, min.y));
	const float3 maxmax = getViewDirection(max);

	float3 n[4] = { make_float3(0.0f), make_float3(0.0f), make_float3(0.0f), make_float3(0.0f) };

	int i = splane_normal(minmin, minmax, n[0]);
	i &= splane_normal(minmax, maxmax, n[1]);
	i &= splane_normal(maxmax, maxmin, n[2]);
	i &= splane_normal(maxmin, minmin, n[3]);

	if (!i)
		return 0.0f;
	float ang = 0.0f;
	for (i = 0; i < 4; i++) {
		ang += M_PIf - fabsf(angle(n[i], n[(i + 1) % 4]));
	}
	ang = ang - 2.0f * M_PIf;
	if ((ang > (2.0f * M_PIf)) || ang < 0) {
		//fprintf(stderr, "Normal error in pict_get_sangle %f %d %d\n", ang, x, y);
		return 0.0f;
	}
	return ang;
}

/* From get_posindex in evalglare.c */
RT_METHOD float getPositionIndex(const float3 &dir, const float3 &forward)
{
	float3 up = normalize(V); // TODO Not necessarily
	float3 hv = cross(forward, up);
	float phi = angle(cross(forward, hv), dir) - M_PI_2f;
	float teta = M_PI_2f - angle(hv, dir);
	float sigma = angle(forward, dir);
	hv = normalize(normalize(dir) / cosf(sigma) - forward);
	float tau = angle(up, hv);
	tau *= 180.0f / M_PIf;
	sigma *= 180.0f / M_PIf;

	if (phi == 0.0f)
		phi = FTINY;
	if (sigma <= 0)
		sigma = -sigma;
	if (teta == 0.0f)
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

/* From get_task_lum() in evalglare.c */
RT_METHOD int inTask(const int2 &position, const float &angle, const float3 &ray_direction)
{
	float2 d = shift + make_float2(position) / make_float2(launch_dim) - 0.5f;
	float3 task_dir = getViewDirection(d);
	float r_actual = angle(task_dir, ray_direction);
	return r_actual <= angle;
}

/* Tint the color with emphasis on the component */
RT_METHOD void tint(unsigned int &color, const unsigned int component)
{
	unsigned int c = color;
	c += (0xff - (c & 0xff)) / (component == 2 ? 2 : 8);
	c += ((0xff - ((c >> 8) & 0xff)) / (component == 1 ? 2 : 8)) << 8;
	c += ((0xff - ((c >> 16) & 0xff)) / (component ? 8 : 2)) << 16;
	color = c;
}

RT_PROGRAM void exception()
{
#ifdef PRINT_OPTIX
	rtPrintExceptionDetails();
#endif
	color_buffer[launch_index] = 0xffffffff;
	if (!frame)
		direct_buffer[launch_index] = make_float3(0.0f);
	Metrics metrics;
	metrics.omega = -1.0f;
	metrics.ev = rtGetExceptionCode();
	metrics.avlum = 0.0f;
	metrics.dgp = 0.0f;
	metrics_buffer[launch_index] = metrics;
}
