/*
 * Copyright (c) 2013-2016 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <standard.h> /* TODO just get the includes that are required? */
#include "ray.h"
#include "lookup.h"
#include "rcontrib.h"

#include "optix_radiance.h"

#ifdef ACCELERAD

void contribOptix(const size_t width, const size_t height, const unsigned int imm_irrad, const unsigned int lim_dist, const unsigned int contrib, const double alarm, double* rays, const LUTAB* modifiers);

extern void done_contrib();


/**
 * Setup and run the OptiX kernel similar to RTRACE.
 */
void contribOptix(const size_t width, const size_t height, const unsigned int imm_irrad, const unsigned int lim_dist, const unsigned int contrib, const double alarm, double* rays, LUTAB* modifiers)
{
	/* Primary RTAPI objects */
	RTcontext           context;
	RTbuffer            origin_buffer, direction_buffer, contrib_buffer;

	/* Parameters */
	size_t size, i;
	float *origins, *directions, *contributions;
#ifdef RAY_COUNT
	RTbuffer            ray_count_buffer;
	unsigned int *ray_count_data;
#endif
	MODCONT	*mp;
	int	j, k;

	/* Set the size */
	size = width * height;
	if (!size)
		error(USER, "Number of points is zero or not set.");

	/* Setup state */
	createContext(&context, width, height, alarm);

	/* Input buffers */
	createBuffer2D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, width, height, &origin_buffer);
	RT_CHECK_ERROR(rtBufferMap(origin_buffer, (void**)&origins));

	createBuffer2D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, width, height, &direction_buffer);
	RT_CHECK_ERROR(rtBufferMap(direction_buffer, (void**)&directions));

	for (i = 0u; i < size; i++) {
		origins[0] = (float)rays[0];
		origins[1] = (float)rays[1];
		origins[2] = (float)rays[2];
		directions[0] = (float)rays[3];
		directions[1] = (float)rays[4];
		directions[2] = (float)rays[5];

		origins += 3;
		directions += 3;
		rays += 6;
	}

	RT_CHECK_ERROR(rtBufferUnmap(origin_buffer));
	applyContextObject(context, "origin_buffer", origin_buffer);

	RT_CHECK_ERROR(rtBufferUnmap(direction_buffer));
	applyContextObject(context, "direction_buffer", direction_buffer);

#ifdef DAYSIM_COMPATIBLE
	makeDaysimCompatible(context);
#endif

	createCamera(context, "rcontrib_generator");
	setupKernel(context, NULL, modifiers, width, height, imm_irrad, 0.0);
	if (lim_dist)
		applyContextVariable1ui(context, "lim_dist", lim_dist); // -ld
	if (contrib)
		applyContextVariable1ui(context, "contrib", contrib); // -V

	/* Render result buffer */
	createBuffer3D(context, RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, width, height, 1, &contrib_buffer); //TODO real depth
	applyContextObject(context, "contrib_buffer", contrib_buffer);

#ifdef RAY_COUNT
	createBuffer2D(context, RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, width, height, &ray_count_buffer); //TODO why do we collect this?
	applyContextObject(context, "ray_count_buffer", ray_count_buffer);
#endif

	/* Run the OptiX kernel */
	runKernel2D(context, RADIANCE_ENTRY, width, height);

	RT_CHECK_ERROR(rtBufferMap(contrib_buffer, (void**)&contributions));
#ifdef RAY_COUNT
	RT_CHECK_ERROR(rtBufferMap(ray_count_buffer, (void**)&ray_count_data));
#endif

	/* Copy the results to allocated memory. */
	for (i = 0u; i < size; i++) {
		for (j = 0; j < nmods; j++) {
			mp = (MODCONT *)lu_find(modifiers, modname[j])->data;
			for (k = 0; k < mp->nbins; k++) {
#ifdef DEBUG_OPTIX
				if (contributions[BLU] == -1.0f)
					logException((RTexception)contributions[RED]);
				else
#endif
				addcolor(mp->cbin[k], contributions);
				contributions += 3;
			}
		}
#ifdef RAY_COUNT
		nrays += ray_count_data[i];
#endif

		/* accumulate/output values for this ray */
		done_contrib();
	}

	RT_CHECK_ERROR(rtBufferUnmap(contrib_buffer));
#ifdef RAY_COUNT
	RT_CHECK_ERROR(rtBufferUnmap(ray_count_buffer));
#endif

#ifdef DEBUG_OPTIX
	flushExceptionLog("sensor");
#endif

	/* Clean up */
	destroyContext(context);
}

#endif /* ACCELERAD */
