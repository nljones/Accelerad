/*
 *  optix_rcontrib.c - routines for generating contribution coefficients on GPUs.
 */

#include "accelerad_copyright.h"

#include "ray.h"
#include "lookup.h"
#include "rcontrib.h"

#include "optix_radiance.h"

#ifdef ACCELERAD

extern void done_contrib();


/**
 * Setup and run the OptiX kernel similar to RTRACE.
 */
void contribOptix(const size_t width, const size_t height, const size_t ray_count, const unsigned int imm_irrad, const unsigned int lim_dist, const unsigned int contrib, const unsigned int bins, double* rays, LUTAB* modifiers)
{
	/* Primary RTAPI objects */
	RTcontext           context;
	RTbuffer            origin_buffer, direction_buffer, contrib_buffer;
	RTvariable			segment_start = NULL;

	/* Parameters */
	size_t size, start, i = 0;
	float *origins, *directions;
#ifdef CONTRIB_DOUBLE
	double *contributions;
#else
	float *contributions;
#endif
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

	/* Check size of output */
#ifdef CONTRIB_DOUBLE
	const size_t bytes_per_row = sizeof(double4) * bins * width;
#else
	const size_t bytes_per_row = sizeof(float4) * bins * width;
#endif
	const size_t rows_per_segment = bytes_per_row ? min(height, INT_MAX / bytes_per_row) : height; // Limit imposed by OptiX
	if (!rows_per_segment)
		error(USER, "Too many rays per row. Use a smaller -x.");

	/* Setup state */
	createContext(&context, width, height);

	/* Input buffers */
	createBuffer2D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, width, height, &origin_buffer);
	RT_CHECK_ERROR(rtBufferMap(origin_buffer, (void**)&origins));

	createBuffer2D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, width, height, &direction_buffer);
	RT_CHECK_ERROR(rtBufferMap(direction_buffer, (void**)&directions));

	for (i = 0u; i < ray_count; i++) {
		/* Copy valid rays to input buffers */
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
	for ( ; i < size; i++) {
		/* In case there is partial accumulation in final record, write invalid rays to fill the buffers */
		origins[0] = origins[1] = origins[2] = 0.0f;
		directions[0] = directions[1] = directions[2] = 0.0f;

		origins += 3;
		directions += 3;
	}

	RT_CHECK_ERROR(rtBufferUnmap(origin_buffer));
	applyContextObject(context, "origin_buffer", origin_buffer);

	RT_CHECK_ERROR(rtBufferUnmap(direction_buffer));
	applyContextObject(context, "direction_buffer", direction_buffer);

#ifdef DAYSIM_COMPATIBLE
	makeDaysimCompatible(context);
#endif

	createCamera(context, "rcontrib");
	setupKernel(context, NULL, modifiers, width, height, imm_irrad, NULL);

	/* Apply unique settings */
	if (lim_dist)
		applyContextVariable1ui(context, "lim_dist", lim_dist); // -ld
	if (contrib)
		applyContextVariable1ui(context, "contrib", contrib); // -V
	if (height > rows_per_segment) {
		mprintf("Processing rows in %" PRIu64 " groups of %" PRIu64 ".\n", (height - 1) / rows_per_segment + 1, rows_per_segment);
		RT_CHECK_ERROR(rtContextDeclareVariable(context, "contrib_segment", &segment_start));
	}

	/* Render result buffer */
#ifdef CONTRIB_DOUBLE
	createCustomBuffer3D(context, RT_BUFFER_OUTPUT, sizeof(double4), bins, bins ? width : 0, bins ? rows_per_segment : 0, &contrib_buffer);
#else
	createBuffer3D(context, RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, bins, bins ? width : 0, bins ? rows_per_segment : 0, &contrib_buffer);
#endif
	applyContextObject(context, "contrib_buffer", contrib_buffer);

#ifdef RAY_COUNT
	createBuffer2D(context, RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, width, rows_per_segment, &ray_count_buffer); //TODO why do we collect this?
	applyContextObject(context, "ray_count_buffer", ray_count_buffer);
#endif

	for (start = 0u; start < height; start += rows_per_segment) {
		const size_t rows = min(height - start, rows_per_segment);
		const size_t segment_size = rows * width;

		if (segment_start)
			RT_CHECK_ERROR(rtVariableSet1ui(segment_start, (unsigned int)start));

		/* Run the OptiX kernel */
		runKernel2D(context, RADIANCE_ENTRY, width, rows);

		RT_CHECK_ERROR(rtBufferMap(contrib_buffer, (void**)&contributions));
#ifdef RAY_COUNT
		RT_CHECK_ERROR(rtBufferMap(ray_count_buffer, (void**)&ray_count_data));
#endif

		/* Copy the results to allocated memory. */
		for (i = 0u; i < segment_size; i++) {
			for (j = 0; j < nmods; j++) {
				mp = (MODCONT *)lu_find(modifiers, modname[j])->data;
				for (k = 0; k < mp->nbins; k++) {
#ifdef DEBUG_OPTIX
					if (contributions[3] == -1.0f)
						logException((RTexception)contributions[RED]);
					else
#endif
					addcolor(mp->cbin[k], contributions);
					contributions += 4;
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
	}

#ifdef DEBUG_OPTIX
	flushExceptionLog("rcontrib");
#endif

	/* Clean up */
	destroyContext(context);
}

#endif /* ACCELERAD */
