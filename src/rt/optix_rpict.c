/*
 *  optix_rpict.c - routines for picture generation on GPUs.
 */

#include "accelerad_copyright.h"

#include "ray.h"

#include "optix_radiance.h"

#ifdef ACCELERAD

/* Handles to objects used repeatedly in animation */
extern unsigned int frame;
RTcontext context_handle = NULL;
RTbuffer buffer_handle = NULL;
#ifdef RAY_COUNT
RTbuffer ray_count_buffer_handle = NULL;
#endif


/**
 * Setup and run the OptiX kernel similar to RPICT.
 * This may be called repeatedly in order to render an animation.
 * After the last call, endOptix() should be called.
 */
void renderOptix(const VIEW* view, const size_t width, const size_t height, const double dstrpix, const double mblur, const double dblur, COLOR* colors, float* depths, void (*freport)(double))
{
	/* Primary RTAPI objects */
	RTcontext           context;
	RTbuffer            output_buffer, lastview;

	/* Parameters */
	size_t i, size;
	float* data;

#ifdef RAY_COUNT
	RTbuffer            ray_count_buffer;
	unsigned int *ray_count_data;
#endif

	/* Set the size */
	size = width * height;

	if (context_handle == NULL) {
		/* Setup state */
		createContext(&context, width, height);

		/* Render result buffer */
		createBuffer2D(context, RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width, height, &output_buffer);
		applyContextObject(context, "output_buffer", output_buffer);

#ifdef RAY_COUNT
		createBuffer2D(context, RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, width, height, &ray_count_buffer);
		applyContextObject(context, "ray_count_buffer", ray_count_buffer);
		ray_count_buffer_handle = ray_count_buffer;
#endif
#ifdef CONTRIB
		makeContribCompatible(context);
#endif
#ifdef DAYSIM_COMPATIBLE
		makeDaysimCompatible(context);
#endif

		/* Ray parameters buffer for motion blur effect, only needed for rendering multiple frames */
		if (mblur > FTINY)
			createCustomBuffer2D(context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, sizeof(RayParams), width, height, &lastview);
		else
			createCustomBuffer2D(context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, sizeof(RayParams), 0, 0, &lastview);
		applyContextObject(context, "last_view_buffer", lastview);

		/* Save handles to objects used in animations */
		context_handle = context;
		buffer_handle = output_buffer;

		createCamera(context, "rpict");
		setupKernel(context, view, NULL, width, height, 0u, freport);

		/* Apply unique settings */
		if (dstrpix != 0.0)
			applyContextVariable1f(context, "dstrpix", (float)dstrpix); // -pj
		if (mblur > FTINY)
			applyContextVariable1f(context, "mblur", (float)mblur); // -pm
		if (dblur > FTINY)
			applyContextVariable1f(context, "dblur", (float)dblur); // -pd
	}
	else {
		/* Retrieve handles for previously created objects */
		context = context_handle;
		output_buffer = buffer_handle;
#ifdef RAY_COUNT
		ray_count_buffer = ray_count_buffer_handle;
#endif

		/* Update the camera view for the next frame */
		frame++;
		updateCamera(context, view);
	}

	/* Run the OptiX kernel */
	runKernel2D(context, RADIANCE_ENTRY, width, height);

	RT_CHECK_ERROR(rtBufferMap(output_buffer, (void**)&data));
#ifdef RAY_COUNT
	RT_CHECK_ERROR(rtBufferMap(ray_count_buffer, (void**)&ray_count_data));
#endif

	/* Copy the results to allocated memory. */
	for (i = 0u; i < size; i++) {
#ifdef DEBUG_OPTIX
		if (data[3] == -1.0f)
			logException((RTexception)((int)data[RED]));
#endif
		copycolor(colors[i], data);
		depths[i] = data[3];
		data += 4;
#ifdef RAY_COUNT
		nrays += ray_count_data[i];
#endif
	}

	RT_CHECK_ERROR(rtBufferUnmap(output_buffer));
#ifdef RAY_COUNT
	RT_CHECK_ERROR(rtBufferUnmap(ray_count_buffer));
#endif

#ifdef DEBUG_OPTIX
	flushExceptionLog("rpict");
#endif

#ifdef PRINT_OPTIX
	/* Exit if message printing is on */
	exit(1);
#endif

	/* Clean up */
	//destroyContext(context);
}

/**
 * Destroy the OptiX context if one has been created.
 * This should be called after the last call to renderOptix().
 */
void endOptix()
{
	if (context_handle == NULL) return;

	destroyContext(context_handle);
	context_handle = NULL;
	buffer_handle = NULL;
#ifdef RAY_COUNT
	ray_count_buffer_handle = NULL;
#endif
}

#endif /* ACCELERAD */
