/*
 *  optix_rtrace.c - routines for individual ray tracing on GPUs.
 */

#include "accelerad_copyright.h"

#include "ray.h"

#include "optix_radiance.h"

#ifdef ACCELERAD

static void getRay(RayData* data, const RAY* ray);
static void setRay(RAY* ray, const RayData* data);

#ifdef DAYSIM
/* Handles to objects used repeatedly for daylight coefficient calulation */
extern RTbuffer dc_scratch_buffer;
#endif


/**
 * Setup and run the OptiX kernel similar to RTRACE.
 */
void computeOptix(const size_t width, const size_t height, const unsigned int imm_irrad, RAY* rays)
{
	/* Primary RTAPI objects */
	RTcontext           context;
	RTbuffer            ray_buffer;
#ifdef DAYSIM_COMPATIBLE
	RTbuffer            dc_buffer;
#endif

	/* Parameters */
	size_t size, i;
	RayData* data;
#ifdef DAYSIM
	float* dc_data;
#endif

	/* Set the size */
	size = width * height;
	if (!size)
		error(USER, "Number of points is zero or not set.");

	/* Setup state */
	createContext(&context, width, height);

	/* Input/output buffer */
	createCustomBuffer2D(context, RT_BUFFER_INPUT_OUTPUT, sizeof(RayData), width, height, &ray_buffer);
	RT_CHECK_ERROR(rtBufferMap(ray_buffer, (void**)&data));
	for (i = 0u; i < size; i++) {
		getRay(data++, &rays[i]);
	}
	RT_CHECK_ERROR(rtBufferUnmap(ray_buffer));
	applyContextObject(context, "ray_buffer", ray_buffer);

#ifdef CONTRIB
	makeContribCompatible(context);
#endif
#ifdef DAYSIM_COMPATIBLE
	setupDaysim(context, &dc_buffer, width, height);
#endif

	createCamera(context, "rtrace");
	setupKernel(context, NULL, NULL, width, height, imm_irrad, NULL);

#ifdef DAYSIM
	/* Set scratch buffer size for this OptiX kernel */
	if (daysimGetCoefficients()) {
		int use_ambient = ambacc > FTINY && ambounce > 0 && ambdiv > 0;
		RT_CHECK_ERROR(rtBufferSetSize3D(dc_scratch_buffer, daysimGetCoefficients() * abs(maxdepth) * (use_ambient ? 2 : 3), width, height));
	}
#endif

	/* Run the OptiX kernel */
	runKernel2D(context, RADIANCE_ENTRY, width, height);

	RT_CHECK_ERROR(rtBufferMap(ray_buffer, (void**)&data));
#ifdef DAYSIM
	RT_CHECK_ERROR(rtBufferMap(dc_buffer, (void**)&dc_data));
#endif

	/* Copy the results to allocated memory. */
	for (i = 0u; i < size; i++) {
#ifdef DEBUG_OPTIX
		if (data->weight == -1.0f)
			logException((RTexception)data->val.x);
#endif
#ifdef RAY_COUNT
		nrays += data->ray_count;
#endif
		setRay(&rays[i], data++);
#ifdef DAYSIM
		daysimCopy(rays[i].daylightCoef, dc_data);
		dc_data += daysimGetCoefficients();
#endif
	}

	RT_CHECK_ERROR(rtBufferUnmap(ray_buffer));
#ifdef DAYSIM
	RT_CHECK_ERROR(rtBufferUnmap(dc_buffer));
#endif

#ifdef DEBUG_OPTIX
	flushExceptionLog("rtrace");
#endif

	/* Clean up */
	destroyContext(context);
}

static void getRay(RayData* data, const RAY* ray)
{
	array2cuda3(data->origin, ray->rorg);
	array2cuda3(data->dir, ray->rdir);
	array2cuda3(data->val, ray->rcol);
	array2cuda3(data->mirror, ray->mcol);
	//array2cuda3( data->contrib, ray->rcoef );
	//array2cuda3( data->extinction, ray->cext );
	//array2cuda3( data->hit, ray->rop );
	//array2cuda3( data->pnorm, ray->pert );
	//array2cuda3( data->normal, ray->ron );

	//array2cuda2(data->tex, ray->uv);
	data->max = (float)ray->rmax;
	data->weight = ray->rweight;
	data->length = (float)ray->rxt;
	data->mirrored_length = (float)ray->rmt;
	//data->t = ray->rot;
}

static void setRay(RAY* ray, const RayData* data)
{
	cuda2array3(ray->rorg, data->origin);
	cuda2array3(ray->rdir, data->dir);
	cuda2array3(ray->rcol, data->val);
	cuda2array3(ray->mcol, data->mirror);
	//cuda2array3( ray->rcoef, data->contrib );
	//cuda2array3( ray->cext, data->extinction );
	//cuda2array3( ray->rop, data->hit );
	//cuda2array3( ray->pert, data->pnorm );
	//cuda2array3( ray->ron, data->normal );

	//cuda2array2(ray->uv, data->tex);
	ray->rmax = data->max;
	ray->rweight = data->weight;
	ray->rxt = data->length;
	ray->rmt = data->mirrored_length;
	//ray->rot = data->t; //TODO setting this requires that the ray has non-null ro.
}

#endif /* ACCELERAD */
