/*
 *  optix_util.c - routines for interactive view generation on GPUs.
 */

#include "accelerad_copyright.h"

#include "ray.h"
#include "optix_radiance.h"

#ifdef ACCELERAD_RT
//#define SAVE_METRICS
#ifdef SAVE_METRICS
extern char	*octname;			/* octree name we are given */
FILE *csv;		/* metrics output */
clock_t start;	/* start for elapsed time */
#endif /* SAVE_METRICS */

#define METRICS_COUNT	6
#define SCALE_DEFAULT	1000 /* Default scale maximum while setting autoscale */

extern int has_diffuse_normal_closest_hit_program;	/* Flag for including rvu programs. */

/* Handles to objects used repeatedly in animation */
extern unsigned int frame;
RTcontext context = NULL;
RTbuffer output_buffer = NULL;
#ifdef RAY_COUNT
RTbuffer ray_count_buffer = NULL;
#endif
extern RTvariable camera_frame;
int auto_scale = 0;	/* flag to perform auto range scaling */

/* Parameters that can change without resetting frame count */
extern int do_lum;			/* show luminance rather than radiance */
extern double exposure;		/* exposure for scene (-pe) */
extern int greyscale;		/* map colors to brightness? (-b) */
extern int fc;				/* use falsecolor tonemapping, zero for natural tonemapping (-f) */
extern double scale;		/* maximum of scale for falsecolor images, zero auto-scaling (-s) */
extern int decades;			/* number of decades for log scale, zero for standard scale (-log) */
extern int base;			/* base for log scale (-base) */
extern double masking;		/* minimum value to display in falsecolor images (-m) */

extern double dstrpix;		/* pixel jitter (-pj) */

/* Regions */
extern int xt, yt, xh, yh, xl, yl;
extern double omegat, omegah, omegal;

static double calcRAMMG(const Metrics *metrics, const int width, const int height);
static int makeFalseColorMap(const RTcontext context);
void setScale(const double maximum);

/* Handles to objects used repeatedly in animation */
int tonemap_id = RT_TEXTURE_ID_NULL;
static RTvariable jitter_var = NULL, luminance_var = NULL, greyscale_var = NULL, exposure_var = NULL, scale_var = NULL, tonemap_var = NULL, decades_var = NULL, base_var = NULL, mask_var = NULL;
static RTvariable task_position = NULL, task_angle = NULL, high_position = NULL, high_angle = NULL, low_position = NULL, low_angle = NULL, position_flags = NULL;
#ifdef VT_ODS
static RTvariable camera_gaze = NULL;
#endif
static RTbuffer metrics_buffer = NULL, direct_buffer = NULL, diffuse_buffer = NULL;

void renderOptixIterative(const VIEW* view, const int width, const int height, const int moved, void (*fpaint)(int, int, int, int, const unsigned char *), void (*fplot)(double *, int))
{
	/* Parameters */
	unsigned int i, size;
	double omega = 0.0, ev = 0.0, avlum = 0.0, dgp = 0.0, rammg = 0.0;
	double lumT = 0.0, omegaT = 0.0, lumH = 0.0, omegaH = 0.0, lumL = 0.0, omegaL = 0.0, cr = 0.0;
	int nt = 0, nh = 0, nl = 0;
	Metrics *metrics;

	/* Set the size */
	size = width * height;

	if (context == NULL) {
		/* Do not print repetitive statements */
		verbose_output = 0;

		/* Setup state */
		createContext(&context, width, height);

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
#endif
#ifdef CONTRIB
		makeContribCompatible(context);
#endif
#ifdef DAYSIM_COMPATIBLE
		makeDaysimCompatible(context);
#endif

		has_diffuse_normal_closest_hit_program = 1;

		createCamera(context, "rvu");
		setupKernel(context, view, NULL, width, height, 0u, NULL);

		/* Apply unique settings */
		jitter_var = applyContextVariable1f(context, "dstrpix", (float)dstrpix); // -pj

		luminance_var = applyContextVariable1ui(context, "do_lum", (unsigned int)do_lum);
		exposure_var = applyContextVariable1f(context, "exposure", (float)exposure);
		greyscale_var = applyContextVariable1ui(context, "greyscale", (unsigned int)greyscale);
		if (fc)
			tonemap_id = makeFalseColorMap(context);
		tonemap_var = applyContextVariable1i(context, "tonemap", tonemap_id);
		auto_scale = scale <= 0;
		scale_var = applyContextVariable1f(context, "fc_scale", auto_scale ? SCALE_DEFAULT : (float)scale);
		decades_var = applyContextVariable1i(context, "fc_log", decades);
		base_var = applyContextVariable1i(context, "fc_base", base);
		mask_var = applyContextVariable1f(context, "fc_mask", (float)masking);

		task_position = applyContextVariable2i(context, "task_position", xt, yt);
		task_angle = applyContextVariable1f(context, "task_angle", (float)omegat);
		high_position = applyContextVariable2i(context, "high_position", xh, yh);
		high_angle = applyContextVariable1f(context, "high_angle", (float)omegah);
		low_position = applyContextVariable2i(context, "low_position", xl, yl);
		low_angle = applyContextVariable1f(context, "low_angle", (float)omegal);
		position_flags = applyContextVariable1ui(context, "flags", 0u);

#ifdef VT_ODS
		camera_gaze = applyContextVariable3f(context, "gaze", 0.0f, 0.0f, 0.0f); // gaze direction (0 to use W)
#endif

#ifdef SAVE_METRICS
		sprintf(errmsg, "%s.csv", octname);
		csv = fopen(errmsg, "w");
		fprintf(csv, "Time,Frame,AvLum,Ev,DGP,RAMMG,TaskLum,HighLum,LowLum,CR\n");
		start = clock();
#endif /* SAVE_METRICS */
	}
	else {
		if (moved) {
			/* Update the camera view for the next frame */
			frame = 0u;
			updateCamera(context, view); // TODO means start over
		}
		else {
			RT_CHECK_ERROR(rtVariableSet1ui(camera_frame, ++frame));
		}
	}

	/* Run the OptiX kernel */
	runKernel2D(context, RADIANCE_ENTRY, width, height);

	if (fpaint) {
		/* Retrieve the image */
		unsigned char* colors;
		RT_CHECK_ERROR(rtBufferMap(output_buffer, (void**)&colors));
		(*fpaint)(0, 0, width, height, colors);
		RT_CHECK_ERROR(rtBufferUnmap(output_buffer));
	}

	/* Calculate the metrics */
	RT_CHECK_ERROR(rtBufferMap(metrics_buffer, (void**)&metrics));

#ifdef SAVE_METRICS
	rammg = calcRAMMG(metrics, width, height);
#endif

	for (i = 0u; i < size; i++) {
#ifdef DEBUG_OPTIX
		if (metrics[i].omega == -1.0f)
			logException((RTexception)((int)metrics[i].ev));
#endif
		if (metrics[i].omega >= 0.0f) {
			omega += metrics[i].omega;
			ev += metrics[i].ev;
			avlum += metrics[i].avlum;
			if (metrics[i].flags & 0x1) {
				lumT += metrics[i].avlum;
				omegaT += metrics[i].omega;
				nt++;
			}
			if (metrics[i].flags & 0x2) {
				lumH += metrics[i].avlum;
				omegaH += metrics[i].omega;
				nh++;
			}
			if (metrics[i].flags & 0x4) {
				lumL += metrics[i].avlum;
				omegaL += metrics[i].omega;
				nl++;
			}
		}
	}

	if (do_irrad) {
		avlum /= size;
		if (nt) lumT /= nt;
		if (nh) lumH /= nh;
		if (nl) lumL /= nl;
	}
	else {
		avlum /= omega;
		if (nt) lumT /= omegaT;
		if (nh) lumH /= omegaH;
		if (nl) lumL /= omegaL;

		if (do_lum && ev > FTINY) { // if (ev > 380) {
			/* Calculate DGP */
			double lum_thresh = 5.0; // Max 100
			if (nt)
				lum_thresh *= lumT;
			else
				lum_thresh = 2000; // cd/m2

			for (i = 0u; i < size; i++)
				if (metrics[i].omega >= 0.0f && metrics[i].avlum > lum_thresh * metrics[i].omega)
					dgp += metrics[i].dgp;

			dgp = 5.87e-5 * ev + 0.092 * log10(1 + dgp / pow(ev, 1.87)) + 0.159;
			if (dgp > 1.0) dgp = 1.0;
			if (ev < 1000) /* low light correction */
				dgp *= exp(0.024 * ev - 4) / (1 + exp(0.024 * ev - 4));
			//dgp /= 1.1 - 0.5 * age / 100.0; /* age correction */
		}
	}
	if (nh && nl) cr = lumH / lumL;
	RT_CHECK_ERROR(rtBufferUnmap(metrics_buffer));

	/* Auto-scaling */
	int rescaled = auto_scale && avlum > 0.0;
	if (rescaled) {
		setScale(2.0 * avlum);
	}

	if (fplot) {
		/* Plot results */
		double *plotValues = (double *)malloc(METRICS_COUNT * sizeof(double));
		if (plotValues) {
			plotValues[0] = avlum;
			plotValues[1] = ev;
			plotValues[2] = dgp;
			plotValues[3] = lumT;
			plotValues[4] = cr;
			plotValues[5] = rammg;
			(*fplot)(plotValues, rescaled);
			free(plotValues);
		}
	}

#ifdef DEBUG_OPTIX
	flushExceptionLog("rvu");
#endif
#ifdef PRINT_METRICS
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
#endif /* PRINT_METRICS */
#ifdef SAVE_METRICS
	fprintf(csv, "%" PRIu64 ",%u,%g,%g,%g,%g,%g,%g,%g,%g\n", MILLISECONDS(clock() - start), frame, avlum, ev, dgp, rammg, lumT, lumH, lumL, cr);
#endif /* SAVE_METRICS */

	/* Clean up */
	//destroyContext(context);
}

void updateOctree(char* path)
{
	ray_init(path);
	updateModel(context, NULL, 0u);
}

void updateStackSize()
{
	if (context)
		RT_CHECK_ERROR(rtContextSetMaxTraceDepth(context, maxdepth ? min(abs(maxdepth) * 2, 31) : 31)); // TODO set based on lw?
}

/**
 * Destroy the OptiX context if one has been created.
 * This should be called after the last call to renderOptixIterative().
 */
void endOptix()
{
	if (context == NULL) return;

	destroyContext(context);
	context = NULL;
	output_buffer = NULL;
#ifdef RAY_COUNT
	ray_count_buffer = NULL;
#endif
#ifdef VT_ODS
	camera_gaze = NULL;
#endif
}


#define VAL(x, y)	values[(x) + (y) * xmax]

/* RAMMG Contrast Metric from Alessandro Rizzi, Thomas Algeri, Giuseppe Medeghini, Daniele Marini (2004). "A proposal for Contrast Measure in Digital Images" */
static double calcRAMMG(const Metrics *metrics, const int width, const int height)
{
	int x, y, xmax = width, ymax = height, level = 0;
	double pixelSum, levelSum = 0.0;
	metric pixel, *values;

	/* Populate initial values */
	values = (metric *)malloc(sizeof(metric) * width * height);
	if (!values) eprintf(SYSTEM, "out of memory in calcRAMMG, need %" PRIu64 " bytes", sizeof(metric) * width * height);
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

static int makeFalseColorMap(const RTcontext context)
{
	RTtexturesampler tex_sampler;
	RTbuffer         tex_buffer;
	float*           tex_buffer_data;

	int tex_id = RT_TEXTURE_ID_NULL;

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
	if (!context) return;

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

int updateIrradiance(const int irrad)
{
	int changed = setIrradiance(irrad);
	if (changed)
		updateModel(context, NULL, 0u);
	return changed;
}

void setLuminance(const int lum)
{
	do_lum = lum;
	if (context)
		RT_CHECK_ERROR(rtVariableSet1ui(luminance_var, (unsigned int)lum));
}

void setExposure(const double expose)
{
	exposure = expose;
	if (context)
		RT_CHECK_ERROR(rtVariableSet1f(exposure_var, (float)expose));
}

void setGreyscale(const int grey)
{
	greyscale = grey;
	if (context)
		RT_CHECK_ERROR(rtVariableSet1ui(greyscale_var, (unsigned int)grey));
}

void setFalseColor(const int falsecolor)
{
	fc = falsecolor;
	if (context) {
		if (falsecolor && tonemap_id == RT_TEXTURE_ID_NULL)
			tonemap_id = makeFalseColorMap(context);
		RT_CHECK_ERROR(rtVariableSet1i(tonemap_var, falsecolor ? tonemap_id : RT_TEXTURE_ID_NULL));
	}
}

void setScale(const double maximum)
{
	scale = maximum;
	auto_scale = maximum <= 0;
	if (context && !auto_scale)
		RT_CHECK_ERROR(rtVariableSet1f(scale_var, (float)maximum)); // If autoscale is true, this will be called again after the next rendering pass
}

void setDecades(const int decade)
{
	decades = decade;
	if (context)
		RT_CHECK_ERROR(rtVariableSet1i(decades_var, decade));
}

void setBase(const int log_base)
{
	base = log_base;
	if (context)
		RT_CHECK_ERROR(rtVariableSet1i(base_var, log_base));
}

void setMaskMax(const double mask)
{
	masking = mask;
	if (context)
		RT_CHECK_ERROR(rtVariableSet1f(mask_var, (float)mask));
}

void setTaskArea(const int x, const int y, const double omega)
{
	xt = x;
	yt = y;
	omegat = omega;
	if (context) {
		RT_CHECK_ERROR(rtVariableSet2i(task_position, xt, yt));
		RT_CHECK_ERROR(rtVariableSet1f(task_angle, (float)omegat));
	}
}

void setHighArea(const int x, const int y, const double omega)
{
	xh = x;
	yh = y;
	omegah = omega;
	if (context) {
		RT_CHECK_ERROR(rtVariableSet2i(high_position, xh, yh));
		RT_CHECK_ERROR(rtVariableSet1f(high_angle, (float)omegah));
	}
}

void setLowArea(const int x, const int y, const double omega)
{
	xl = x;
	yl = y;
	omegal = omega;
	if (context) {
		RT_CHECK_ERROR(rtVariableSet2i(low_position, xl, yl));
		RT_CHECK_ERROR(rtVariableSet1f(low_angle, (float)omegal));
	}
}

void setAreaFlags(const unsigned int flags)
{
	if (context)
		RT_CHECK_ERROR(rtVariableSet1ui(position_flags, flags));
}

void setGaze(const VIEW* view, double angle)
{
#ifdef VT_ODS
	if (angle > FTINY || angle < -FTINY) {
		FVECT gaze, normal;
		VCOPY(normal, view->vvec);
		normalize(normal);
		spinvector(gaze, view->vdir, normal, angle);
		if (context)
			RT_CHECK_WARN(rtVariableSet3f(camera_gaze, (float)gaze[0], (float)gaze[1], (float)gaze[2]));
	}
	else if (context)
		RT_CHECK_WARN(rtVariableSet3f(camera_gaze, 0.0f, 0.0f, 0.0f));
#endif
}

#endif /* ACCELERAD_RT */
