#include "ray.h"
#include "optix_radiance.h"

#ifdef ACCELERAD_RT
#include  "driver.h"

//#define SAVE_METRICS
#ifdef SAVE_METRICS
extern char	*octname;			/* octree name we are given */
FILE *csv;		/* metrics output */
clock_t start;	/* start for elapsed time */
#endif /* SAVE_METRICS */

#define METRICS_COUNT	6

/* Handles to objects used repeatedly in animation */
extern unsigned int frame;
RTcontext context_handle = NULL;
RTbuffer buffer_handle = NULL;
#ifdef RAY_COUNT
RTbuffer ray_count_buffer_handle = NULL;
#endif
extern RTvariable camera_frame;

extern struct driver  *dev;

/* Regions */
extern int xt, yt, xh, yh, xl, yl;
extern double omegat, omegah, omegal;

extern void qt_rvu_paint_image(int xmin, int ymin, int xmax, int ymax, const unsigned char *data);
extern void qt_rvu_update_plot(double *values);

static double calcRAMMG(const Metrics *metrics, const int width, const int height);
static int makeFalseColorMap(const RTcontext context);

/* Handles to objects used repeatedly in animation */
static RTvariable camera_exposure;
static RTbuffer metrics_buffer = NULL, direct_buffer = NULL, diffuse_buffer = NULL;

void renderOptixIterative(const VIEW* view, const int width, const int height, const int moved, const int greyscale, const double exposure, const double scale, const int decades, const double mask, const double alarm)
{
	/* Primary RTAPI objects */
	RTcontext           context;
	RTbuffer            output_buffer;

	/* Parameters */
	unsigned int i, size;
	double omega = 0.0, ev = 0.0, avlum = 0.0, dgp = 0.0, rammg = 0.0;
	double lumT = 0.0, omegaT = 0.0, lumH = 0.0, omegaH = 0.0, lumL = 0.0, omegaL = 0.0, cr = 0.0;
	int nt = 0, nh = 0, nl = 0;
	unsigned char* colors;
	Metrics *metrics;
	double *plotValues;

#ifdef RAY_COUNT
	RTbuffer            ray_count_buffer;
	//unsigned int *ray_count_data;
#endif

	/* Set the size */
	size = width * height;

	if (context_handle == NULL) {
		/* Setup state */
		createContext(&context, width, height, alarm);

		/* Render result buffer */
		createBuffer2D(context, RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, width, height, &direct_buffer);
		applyContextObject(context, "direct_buffer", direct_buffer);

		createBuffer2D(context, RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, width, height, &diffuse_buffer);
		applyContextObject(context, "diffuse_buffer", diffuse_buffer);

		createBuffer2D(context, RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, width, height, &output_buffer);
		applyContextObject(context, "color_buffer", output_buffer);

		createCustomBuffer2D(context, RT_BUFFER_OUTPUT, sizeof(Metrics), width, height, &metrics_buffer);
		applyContextObject(context, "metrics_buffer", metrics_buffer);

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

		/* Save handles to objects used in animations */
		context_handle = context;
		buffer_handle = output_buffer;

		createCamera(context, "rvu_generator");
		setupKernel(context, view, NULL, width, height, 0u, alarm);

		/* Apply unique settings */
		camera_exposure = applyContextVariable1f(context, "exposure", (float)exposure);
		if (omegat > FTINY) {
			applyContextVariable2i(context, "task_position", xt, yt);
			applyContextVariable1f(context, "task_angle", (float)omegat);
		}
		if (omegah > FTINY) {
			applyContextVariable2i(context, "high_position", xh, yh);
			applyContextVariable1f(context, "high_angle", (float)omegah);
		}
		if (omegal > FTINY) {
			applyContextVariable2i(context, "low_position", xl, yl);
			applyContextVariable1f(context, "low_angle", (float)omegal);
		}
		if (greyscale)
			applyContextVariable1ui(context, "greyscale", (unsigned int)greyscale);
		if (scale > 0)
			applyContextVariable1i(context, "tonemap", makeFalseColorMap(context));
		applyContextVariable1f(context, "fc_scale", (float)scale);
		if (decades > 0)
			applyContextVariable1i(context, "fc_log", decades);
		if (mask > 0.0)
			applyContextVariable1f(context, "fc_mask", (float)mask);
#ifdef SAVE_METRICS
		sprintf(errmsg, "%s.csv", octname);
		csv = fopen(errmsg, "w");
		fprintf(csv, "Time,Frame,AvLum,Ev,DGP,RAMMG,TaskLum,HighLum,LowLum,CR\n");
		start = clock();
#endif /* SAVE_METRICS */
	}
	else {
		/* Retrieve handles for previously created objects */
		context = context_handle;
		output_buffer = buffer_handle;
#ifdef RAY_COUNT
		ray_count_buffer = ray_count_buffer_handle;
#endif

		if (moved) {
			/* Update the camera view for the next frame */
			frame = 0u;
			updateCamera(context, view); // TODO means start over
		}
		else {
			RT_CHECK_ERROR(rtVariableSet1ui(camera_frame, ++frame));
		}
		RT_CHECK_ERROR(rtVariableSet1f(camera_exposure, (float)exposure));
	}

	/* Run the OptiX kernel */
	runKernel2D(context, RADIANCE_ENTRY, width, height);

	/* Retrieve the image */
	RT_CHECK_ERROR(rtBufferMap(output_buffer, (void**)&colors));
	qt_rvu_paint_image(0, 0, width, height, colors);
	RT_CHECK_ERROR(rtBufferUnmap(output_buffer));
	dev->flush();

	/* Calculate the metrics */
	RT_CHECK_ERROR(rtBufferMap(metrics_buffer, (void**)&metrics));

	rammg = calcRAMMG(metrics, width, height);

	for (i = 0u; i < size; i++) {
#ifdef DEBUG_OPTIX
		if (metrics->omega == -1.0f)
			logException((RTexception)((int)metrics->ev));
#endif
		if (metrics->omega >= 0.0f) {
			omega += metrics->omega;
			ev += metrics->ev;
			avlum += metrics->avlum;
			dgp += metrics->dgp;
			if (metrics->flags & 0x1) {
				lumT += metrics->avlum;
				omegaT += metrics->omega;
				nt++;
			}
			if (metrics->flags & 0x2) {
				lumH += metrics->avlum;
				omegaH += metrics->omega;
				nh++;
			}
			if (metrics->flags & 0x4) {
				lumL += metrics->avlum;
				omegaL += metrics->omega;
				nl++;
			}
		}
		metrics++;
	}
	RT_CHECK_ERROR(rtBufferUnmap(metrics_buffer));
	if (do_irrad) {
		avlum /= size;
		if (nt) lumT /= nt;
		if (nh) lumH /= nh;
		if (nl) lumL /= nl;
	}
	else {
		avlum /= omega;
		dgp = 5.87e-5 * ev + 0.0918 * log10(1 + dgp / pow(ev, 1.87)) + 0.16;
		if (dgp > 1.0) dgp = 1.0;
		if (nt) lumT /= omegaT;
		if (nh) lumH /= omegaH;
		if (nl) lumL /= omegaL;
	}
	if (nh && nl) cr = lumH / lumL;

	/* Plot results */
	plotValues = (double *)malloc(METRICS_COUNT * sizeof(double));
	if (plotValues) {
		plotValues[0] = avlum;
		plotValues[1] = ev;
		plotValues[2] = dgp;
		plotValues[3] = lumT;
		plotValues[4] = cr;
		plotValues[5] = rammg;
		qt_rvu_update_plot(plotValues);
		free(plotValues);
	}

#ifdef DEBUG_OPTIX
	flushExceptionLog("camera");
#endif
	vprintf("Solid angle:                %g\n", omega);
	if (do_irrad) {
		vprintf("Average illuminance:        %g cd/m2\n", avlum);
		if (nt) vprintf("Average task illuminance:   %g cd/m2\n", lumT);
	}
	else {
		vprintf("Vertical eye illuminance:   %g cd/m2\n", ev);
		vprintf("Average luminance:          %g lux\n", avlum);
		vprintf("Daylight glare probability: %g\n", dgp);
		if (nt) vprintf("Average task luminance:     %g lux over %i pixels\n", lumT, nt);
		if (nh && nl) vprintf("Contrast ratio:             %g over %i high and %i low pixels\n", cr, nh, nl);
	}
	vprintf("RAMMG:                      %g\n", rammg);
#ifdef SAVE_METRICS
	fprintf(csv, "%" PRIu64 ",%u,%g,%g,%g,%g,%g,%g,%g,%g\n", MILLISECONDS(clock() - start), frame, avlum, ev, dgp, rammg, lumT, lumH, lumL, cr);
#endif /* SAVE_METRICS */

	/* Clean up */
	//destroyContext(context);
}

/**
 * Destroy the OptiX context if one has been created.
 * This should be called after the last call to renderOptixIterative().
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


#define VAL(x, y)	values[(x) + (y) * xmax]

/* RAMMG Contrast Metric from Alessandro Rizzi, Thomas Algeri, Giuseppe Medeghini, Daniele Marini (2004). "A proposal for Contrast Measure in Digital Images" */
static double calcRAMMG(const Metrics *metrics, const int width, const int height)
{
	int x, y, xmax = width, ymax = height, level = 0;
	double pixelSum, levelSum = 0.0;
	float pixel, *values;

	/* Populate initial values */
	values = (float *)malloc(sizeof(float) * width * height);
	if (values == NULL) error(SYSTEM, "out of memory in calcRAMMG");
	y = width * height;
	if (do_irrad)
		for (x = 0; x < y; x++)
			values[x] = metrics[x].avlum;
	else
		for (x = 0; x < y; x++)
			values[x] = metrics[x].avlum / metrics[x].omega;

	while (xmax > 2 && ymax > 2) {
		/* Calculate contrast at this level */
		pixelSum = 0.0;
		for (x = 1; x < xmax - 1; x++)
			for (y = 1; y < ymax - 1; y++) {
				pixel = VAL(x, y);
				pixelSum += fabs(pixel - VAL(x - 1, y)) + fabs(pixel - VAL(x, y - 1)) + fabs(pixel - VAL(x + 1, y)) + fabs(pixel - VAL(x, y + 1))
					+ (fabs(pixel - VAL(x - 1, y - 1)) + fabs(pixel - VAL(x - 1, y + 1)) + fabs(pixel - VAL(x + 1, y - 1)) + fabs(pixel - VAL(x + 1, y + 1))) / sqrt(2);
			}
		levelSum += pixelSum / (xmax * ymax * (4 + 2 * sqrt(2)));

		/* Create next MIP-map level */
		for (x = 0; x < xmax - 1; x += 2)
			for (y = 0; y < ymax - 1; y += 2)
				VAL(x / 2, y / 2) = (VAL(x, y) + VAL(x, y + 1) + VAL(x + 1, y) + VAL(x + 1, y + 1)) / 4;

		xmax >>= 1;
		ymax >>= 1;
		level++;
	}

	free(values);

	return levelSum / level;
}

static int makeFalseColorMap(const RTcontext context)
{
	RTtexturesampler tex_sampler;
	RTbuffer         tex_buffer;
	float*           tex_buffer_data;

	int tex_id = RT_TEXTURE_ID_NULL;

	/* Tonemap from falsecolor2.py */
	float tonemap[92] = {
		0.18848f, 0.0009766f, 0.2666f, 1.0f,
		0.05468174f, 2.35501e-05f, 0.3638662f, 1.0f,
		0.00103547f, 0.0008966244f, 0.4770437f, 1.0f,
		8.311144e-08f, 0.0264977f, 0.5131397f, 1.0f,
		7.449763e-06f, 0.1256843f, 0.5363797f, 1.0f,
		0.0004390987f, 0.2865799f, 0.5193677f, 1.0f,
		0.001367254f, 0.4247083f, 0.4085123f, 1.0f,
		0.003076f, 0.4739468f, 0.1702815f, 1.0f,
		0.01376382f, 0.4402732f, 0.05314236f, 1.0f,
		0.06170773f, 0.3671876f, 0.05194055f, 1.0f,
		0.1739422f, 0.2629843f, 0.08564082f, 1.0f,
		0.2881156f, 0.1725325f, 0.09881395f, 1.0f,
		0.3299725f, 0.1206819f, 0.08324373f, 1.0f,
		0.3552663f, 0.07316644f, 0.06072902f, 1.0f,
		0.372552f, 0.03761026f, 0.0391076f, 1.0f,
		0.3921184f, 0.01612362f, 0.02315354f, 1.0f,
		0.4363976f, 0.004773749f, 0.01284458f, 1.0f,
		0.6102754f, 6.830967e-06f, 0.005184709f, 1.0f,
		0.7757267f, 0.00803605f, 0.001691774f, 1.0f,
		0.9087369f, 0.1008085f, 2.432735e-05f, 1.0f,
		1.0f, 0.3106831f, 1.212949e-05f, 1.0f,
		1.0f, 0.6447838f, 0.006659406f, 1.0f,
		0.9863f, 0.9707f, 0.02539f, 1.0f
	};

	/* Create buffer for texture data */
	createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, sizeof(tonemap) / (4 * sizeof(float)), &tex_buffer);

	/* Populate buffer with texture data */
	RT_CHECK_ERROR(rtBufferMap(tex_buffer, (void**)&tex_buffer_data));
	memcpy(tex_buffer_data, tonemap, sizeof(tonemap));
	RT_CHECK_ERROR(rtBufferUnmap(tex_buffer));

	/* Create texture sampler */
	RT_CHECK_ERROR(rtTextureSamplerCreate(context, &tex_sampler));
	RT_CHECK_ERROR(rtTextureSamplerSetWrapMode(tex_sampler, 0u, RT_WRAP_CLAMP_TO_EDGE));
	RT_CHECK_ERROR(rtTextureSamplerSetFilteringModes(tex_sampler, RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE));
	RT_CHECK_ERROR(rtTextureSamplerSetIndexingMode(tex_sampler, RT_TEXTURE_INDEX_NORMALIZED_COORDINATES));
	RT_CHECK_ERROR(rtTextureSamplerSetReadMode(tex_sampler, RT_TEXTURE_READ_ELEMENT_TYPE));
	RT_CHECK_ERROR(rtTextureSamplerSetMaxAnisotropy(tex_sampler, 0.0f));
	RT_CHECK_ERROR(rtTextureSamplerSetMipLevelCount(tex_sampler, 1u)); // Currently only one mipmap level supported
	RT_CHECK_ERROR(rtTextureSamplerSetArraySize(tex_sampler, 1u)); // Currently only one element supported
	RT_CHECK_ERROR(rtTextureSamplerSetBuffer(tex_sampler, 0u, 0u, tex_buffer));
	RT_CHECK_ERROR(rtTextureSamplerGetId(tex_sampler, &tex_id));

	return tex_id;
}

void retreiveOptixImage(const int width, const int height, const double exposure, COLR* colrs)
{
	unsigned int i, size;
	float *direct_data, *diffuse_data;
	RTcontext context = context_handle;

	size = width * height;

	RT_CHECK_ERROR(rtBufferMap(direct_buffer, (void**)&direct_data));
	RT_CHECK_ERROR(rtBufferMap(diffuse_buffer, (void**)&diffuse_data));

	/* Copy the results to allocated memory. */
	for (i = 0u; i < size; i++) {
		setcolr(colrs[i],
			(direct_data[RED] + diffuse_data[RED]) * exposure,
			(direct_data[GRN] + diffuse_data[GRN]) * exposure,
			(direct_data[BLU] + diffuse_data[BLU]) * exposure);
		direct_data += 3;
		diffuse_data += 3;
	}

	RT_CHECK_ERROR(rtBufferUnmap(direct_buffer));
	RT_CHECK_ERROR(rtBufferUnmap(diffuse_buffer));
}
#endif /* ACCELERAD_RT */
