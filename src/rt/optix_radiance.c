/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <standard.h> /* TODO just get the includes that are required? */
#include <object.h>
#include <otypes.h>
#include "ray.h"
#include "source.h"
#include "ambient.h"
#include <face.h>
#include <cone.h>
#include <mesh.h>
#include "data.h"
#include "random.h"

#include "optix_radiance.h"

/* Needed for sleep while waiting for VCA */
#ifdef _WIN32
#include <windows.h>
#define sleep(s) Sleep((s)*1000)
#else
#include <unistd.h>
#endif

#define TRIANGULATE
#ifdef TRIANGULATE
#include "triangulate.h"
#endif /* TRIANGULATE */

#define EXPECTED_VERTICES	64
#define EXPECTED_TRIANGLES	64
#define EXPECTED_MATERIALS	8
#ifdef LIGHTS
#define EXPECTED_LIGHTS		8
#endif
#define EXPECTED_SOURCES	3

#define DEFAULT_SPHERE_STEPS	4
#define DEFAULT_CONE_STEPS		24

#define TCALNAME	"tmesh.cal"	/* the name of our auxiliary file */

#ifdef ACCELERAD

void renderOptix(const VIEW* view, const int width, const int height, const double dstrpix, const double mblur, const double dblur, const double alarm, COLOR* colors, float* depths);
void computeOptix(const int width, const int height, const unsigned int imm_irrad, const double alarm, RAY* rays);
void endOptix();

static void checkDevices();
static void checkRemoteDevice(RTremotedevice remote);
static void createContext( RTcontext* context, const int width, const int height, const double alarm );
static void destroyContext(const RTcontext context);
#ifdef DAYSIM_COMPATIBLE
static void setupDaysim(const RTcontext context, RTbuffer* dc_buffer, const int width, const int height);
#endif
static void setupKernel(const RTcontext context, const VIEW* view, const int width, const int height, const unsigned int imm_irrad, const double dstrpix, const double mblur, const double dblur, const double alarm);
static void applyRadianceSettings(const RTcontext context, const VIEW* view, const unsigned int imm_irrad, const double dstrpix, const double mblur, const double dblur);
static void createCamera( const RTcontext context, const VIEW* view );
static void updateCamera( const RTcontext context, const VIEW* view );
static void createGeometryInstance( const RTcontext context, RTgeometryinstance* instance );
static void addRadianceObject(const RTcontext context, OBJREC* rec, OBJREC* parent, const OBJECT index);
static void createFace(OBJREC* rec, OBJREC* parent);
static __inline void createTriangle(OBJREC *material, const int a, const int b, const int c);
#ifdef TRIANGULATE
static int createTriangles(const FACE *face, OBJREC* material);
static int addTriangle(const Vert2_list *tp, int a, int b, int c);
#endif /* TRIANGULATE */
static void createSphere(OBJREC* rec, OBJREC* parent);
static void createCone(OBJREC* rec, OBJREC* parent);
static void createMesh(OBJREC* rec, OBJREC* parent);
static RTmaterial createNormalMaterial(const RTcontext context, OBJREC* rec);
static RTmaterial createGlassMaterial(const RTcontext context, OBJREC* rec);
static RTmaterial createLightMaterial( const RTcontext context, OBJREC* rec );
static DistantLight createDistantLight(const RTcontext context, OBJREC* rec, OBJREC* parent);
static OBJREC* findFunction(OBJREC *o);
static int createFunction(const RTcontext context, OBJREC* rec);
static int createTexture(const RTcontext context, OBJREC* rec);
static int createTransform( XF* fxp, XF* bxp, const OBJREC* rec );
static void createAcceleration(const RTcontext context, const RTgeometryinstance instance, const unsigned int imm_irrad);
static void createIrradianceGeometry( const RTcontext context );
static void printObject(OBJREC* rec);
static void getRay( RayData* data, const RAY* ray );
static void setRay( RAY* ray, const RayData* data );


/* from ambient.c */
extern double  avsum;		/* computed ambient value sum (log) */
extern unsigned int  navsum;	/* number of values in avsum */
extern unsigned int  nambvals;	/* total number of indirect values */

/* from func.c */
extern XF  unitxf;

/* Ambient calculation flags */
static unsigned int use_ambient = 0u;
static unsigned int calc_ambient = 0u;

/* Handles to objects used in ambient calculation */
RTvariable avsum_var = NULL;
RTvariable navsum_var = NULL;

/* Handles to objects used repeatedly in animation */
static unsigned int frame = 0u;
static RTremotedevice remote_handle = NULL;
static RTcontext context_handle = NULL;
static RTbuffer buffer_handle = NULL;
#ifdef RAY_COUNT
static RTbuffer ray_count_buffer_handle = NULL;
#endif
static RTvariable camera_frame, camera_type, camera_eye, camera_u, camera_v, camera_w, camera_fov, camera_shift, camera_clip, camera_vdist;

/* Handles to buffer data */
static int*       buffer_entry_index;
static FloatArray* vertices;		/* Three entries per vertex */
static FloatArray* normals;			/* Three entries per vertex */
static FloatArray* tex_coords;		/* Two entries per vertex */
static IntArray*   vertex_indices;	/* Three entries per triangle */
static IntArray*   traingles;		/* One entry per triangle gives material of that triangle */
static MaterialArray* materials;	/* One entry per material */
static IntArray*   alt_materials;	/* Two entries per material gives alternate materials to use in place of that material */
#ifdef LIGHTS
static IntArray*   lights;			/* Three entries per triangle that is a light */
#endif
static DistantLightArray* sources;	/* One entry per source object */
static IntArray*   functions;		/* One entry per callable program */

/* Index of first vertex of current object */
static unsigned int vertex_index_0;

/* Handles to intersection program objects used by multiple materials */
static RTprogram radiance_normal_closest_hit_program, ambient_normal_closest_hit_program;
#ifndef LIGHTS
static RTprogram shadow_normal_any_hit_program;
#endif
static RTprogram radiance_glass_closest_hit_program, shadow_glass_closest_hit_program, ambient_glass_any_hit_program;
static RTprogram radiance_light_closest_hit_program, shadow_light_closest_hit_program;
static RTprogram point_cloud_closest_hit_program, point_cloud_any_hit_program;

#ifdef DAYSIM
/* Handles to objects used repeatedly for daylight coefficient calulation */
RTbuffer dc_scratch_buffer = NULL;
#endif


/**
 * Setup and run the OptiX kernel similar to RPICT.
 * This may be called repeatedly in order to render an animation.
 * After the last call, endOptix() should be called.
 */
void renderOptix(const VIEW* view, const int width, const int height, const double dstrpix, const double mblur, const double dblur, const double alarm, COLOR* colors, float* depths)
{
	/* Primary RTAPI objects */
	RTcontext           context;
	RTbuffer            output_buffer, lastview;

	/* Parameters */
	unsigned int i, size;
	float* data;

#ifdef RAY_COUNT
	RTbuffer            ray_count_buffer;
	unsigned int *ray_count_data;
#endif
#ifdef DAYSIM_COMPATIBLE
	RTbuffer            dc_buffer;
#endif

	/* Set the size */
	size = width * height;

	if ( context_handle == NULL ) {
		/* Setup state */
		createContext( &context, width, height, alarm );

		/* Render result buffer */
		createBuffer2D( context, RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width, height, &output_buffer );
		applyContextObject( context, "output_buffer", output_buffer );

#ifdef RAY_COUNT
		createBuffer2D(context, RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, width, height, &ray_count_buffer);
		applyContextObject(context, "ray_count_buffer", ray_count_buffer);
		ray_count_buffer_handle = ray_count_buffer;
#endif
#ifdef DAYSIM_COMPATIBLE
		setupDaysim(context, &dc_buffer, width, height);
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

		setupKernel(context, view, width, height, 0u, dstrpix, mblur, dblur, alarm);
	} else {
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
	runKernel2D( context, RADIANCE_ENTRY, width, height );

	RT_CHECK_ERROR( rtBufferMap( output_buffer, (void**)&data ) );
	RT_CHECK_ERROR( rtBufferUnmap( output_buffer ) );

#ifdef RAY_COUNT
	RT_CHECK_ERROR(rtBufferMap(ray_count_buffer, (void**)&ray_count_data));
	RT_CHECK_ERROR(rtBufferUnmap(ray_count_buffer));
#endif

	/* Copy the results to allocated memory. */
	for (i = 0u; i < size; i++) {
#ifdef DEBUG_OPTIX
		if (data[3] == -1.0f)
			logException((RTexception)((int)data[0]));
#endif
		copycolor( colors[i], data );
		depths[i] = data[3];
		data += 4;
#ifdef RAY_COUNT
		nrays += ray_count_data[i];
#endif
	}
#ifdef DEBUG_OPTIX
	flushExceptionLog("camera");
#endif

#ifdef PRINT_OPTIX
	/* Exit if message printing is on */
	exit(1);
#endif

	/* Clean up */
	//destroyContext(context);
}

/**
 * Setup and run the OptiX kernel similar to RTRACE.
 */
void computeOptix(const int width, const int height, const unsigned int imm_irrad, const double alarm, RAY* rays)
{
	/* Primary RTAPI objects */
	RTcontext           context;
	RTbuffer            ray_buffer;
#ifdef DAYSIM_COMPATIBLE
	RTbuffer            dc_buffer;
#endif

	/* Parameters */
	unsigned int size, i;
	RayData* data;
#ifdef DAYSIM
	float* dc_data;
#endif

	/* Set the size */
	size = width * height;
	if ( !size )
		error(USER, "Number of points is zero or not set.");

	/* Setup state */
	createContext( &context, width, height, alarm );
	
	/* Input/output buffer */
	createCustomBuffer2D( context, RT_BUFFER_INPUT_OUTPUT, sizeof(RayData), width, height, &ray_buffer );
	RT_CHECK_ERROR( rtBufferMap( ray_buffer, (void**)&data ) );
	for (i = 0u; i < size; i++) {
		getRay( data++, &rays[i] );
	}
	RT_CHECK_ERROR( rtBufferUnmap( ray_buffer ) );
	applyContextObject( context, "ray_buffer", ray_buffer );

#ifdef DAYSIM_COMPATIBLE
	setupDaysim(context, &dc_buffer, width, height);
#endif

	setupKernel(context, NULL, width, height, imm_irrad, 0.0, 0.0, 0.0, 0.0);

#ifdef DAYSIM
	/* Set scratch buffer size for this OptiX kernel */
	if (daysimGetCoefficients())
		RT_CHECK_ERROR(rtBufferSetSize3D(dc_scratch_buffer, daysimGetCoefficients() * maxdepth * 2, width, height));
#endif

	/* Run the OptiX kernel */
	runKernel2D( context, RADIANCE_ENTRY, width, height );

	RT_CHECK_ERROR( rtBufferMap( ray_buffer, (void**)&data ) );
	RT_CHECK_ERROR( rtBufferUnmap( ray_buffer ) );
#ifdef DAYSIM
	RT_CHECK_ERROR(rtBufferMap(dc_buffer, (void**)&dc_data));
	RT_CHECK_ERROR(rtBufferUnmap(dc_buffer));
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
		setRay( &rays[i], data++ );
#ifdef DAYSIM
		daysimCopy(rays[i].daylightCoef, dc_data);
		dc_data += daysimGetCoefficients();
#endif
	}
#ifdef DEBUG_OPTIX
	flushExceptionLog("sensor");
#endif

	/* Clean up */
	destroyContext(context);
}

/**
 * Destroy the OptiX context if one has been created.
 * This should be called after the last call to renderOptix().
 */
void endOptix()
{
	if ( context_handle == NULL ) return;

	destroyContext(context_handle);
	context_handle = NULL;
	buffer_handle = NULL;
#ifdef RAY_COUNT
	ray_count_buffer_handle = NULL;
#endif
	camera_frame = camera_type = camera_eye = camera_u = camera_v = camera_w = camera_fov = camera_shift = camera_clip = camera_vdist = NULL;
}

/**
 * Check for supported GPU devices.
 */
static void checkDevices()
{
	unsigned int version = 0, device_count = 0;
	unsigned int i;
	unsigned int multiprocessor_count, threads_per_block, clock_rate, texture_count, timeout_enabled, tcc_driver, cuda_device;
	unsigned int compute_capability[2];
	char device_name[128];
	RTsize memory_size;

	RT_CHECK_WARN_NO_CONTEXT(rtGetVersion(&version));
	RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetDeviceCount(&device_count)); // This will return an error if no supported devices are found
	mprintf("OptiX %d.%d.%d found %i GPU device%s:\n", version / 1000, (version % 1000) / 10, version % 10, device_count, device_count != 1 ? "s" : "");

	for (i = 0; i < device_count; i++) {
		rtDeviceGetAttribute( i, RT_DEVICE_ATTRIBUTE_NAME, sizeof(device_name), &device_name );
		rtDeviceGetAttribute( i, RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, sizeof(multiprocessor_count), &multiprocessor_count );
		rtDeviceGetAttribute( i, RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, sizeof(threads_per_block), &threads_per_block );
		rtDeviceGetAttribute( i, RT_DEVICE_ATTRIBUTE_CLOCK_RATE, sizeof(clock_rate), &clock_rate );
		rtDeviceGetAttribute( i, RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(memory_size), &memory_size );
		rtDeviceGetAttribute( i, RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT, sizeof(texture_count), &texture_count );
		rtDeviceGetAttribute( i, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(compute_capability), &compute_capability );
		rtDeviceGetAttribute( i, RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED, sizeof(timeout_enabled), &timeout_enabled );
		rtDeviceGetAttribute( i, RT_DEVICE_ATTRIBUTE_TCC_DRIVER, sizeof(tcc_driver), &tcc_driver );
		rtDeviceGetAttribute( i, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(cuda_device), &cuda_device );
		mprintf("Device %u: %s with %u multiprocessors, %u threads per block, %u kHz, %" PRIu64 " bytes global memory, %u hardware textures, compute capability %u.%u, timeout %sabled, Tesla compute cluster driver %sabled, cuda device %u.\n",
			i, device_name, multiprocessor_count, threads_per_block, clock_rate, memory_size, texture_count, compute_capability[0], compute_capability[1], timeout_enabled ? "en" : "dis", tcc_driver ? "en" : "dis", cuda_device);
	}

	mprintf("\n");
}

static void checkRemoteDevice(RTremotedevice remote)
{
	char s[256];
	unsigned int version = 0;
	int i = 0, j;
	RTsize size;

	RT_CHECK_WARN_NO_CONTEXT(rtGetVersion(&version));
	mprintf("OptiX %d.%d.%d logged into %s as %s\n", version / 1000, (version % 1000) / 10, version % 10, optix_remote_url, optix_remote_user);
	rtRemoteDeviceGetAttribute(remote, RT_REMOTEDEVICE_ATTRIBUTE_NAME, sizeof(s), &s);
	mprintf("VCA Name:                 %s\n", s);
	rtRemoteDeviceGetAttribute(remote, RT_REMOTEDEVICE_ATTRIBUTE_NUM_GPUS, sizeof(int), &i);
	mprintf("Number of GPUs:           %i\n", i);
	rtRemoteDeviceGetAttribute(remote, RT_REMOTEDEVICE_ATTRIBUTE_CLUSTER_URL, sizeof(s), &s);
	mprintf("Cluster URL:              %s\n", s);
	rtRemoteDeviceGetAttribute(remote, RT_REMOTEDEVICE_ATTRIBUTE_HEAD_NODE_URL, sizeof(s), &s);
	mprintf("Head node URL:            %s\n", s);
	rtRemoteDeviceGetAttribute(remote, RT_REMOTEDEVICE_ATTRIBUTE_NUM_CONFIGURATIONS, sizeof(int), &i);
	mprintf("Number of configurations: %i\n", i);
	for (j = 0; j < i; j++) {
		rtRemoteDeviceGetAttribute(remote, RT_REMOTEDEVICE_ATTRIBUTE_CONFIGURATIONS + j, sizeof(s), &s);
		mprintf("Configuration %i:          %s\n", j, s);
	}
	rtRemoteDeviceGetAttribute(remote, RT_REMOTEDEVICE_ATTRIBUTE_NUM_TOTAL_NODES, sizeof(int), &i);
	mprintf("Number of nodes:          %i\n", i);
	rtRemoteDeviceGetAttribute(remote, RT_REMOTEDEVICE_ATTRIBUTE_NUM_FREE_NODES, sizeof(int), &i);
	mprintf("Number of free nodes:     %i\n", i);
	rtRemoteDeviceGetAttribute(remote, RT_REMOTEDEVICE_ATTRIBUTE_NUM_RESERVED_NODES, sizeof(int), &i);
	mprintf("Number of reserved nodes: %i\n", i);
	rtRemoteDeviceGetAttribute(remote, RT_REMOTEDEVICE_ATTRIBUTE_GPU_TOTAL_MEMORY, sizeof(RTsize), &size);
	mprintf("Total memory size:        %" PRIu64 " bytes\n\n", size);
}

static void createContext( RTcontext* context, const int width, const int height, const double alarm )
{
	RTremotedevice remote;
	//RTbuffer seed_buffer;

	//unsigned int* seeds;
	//unsigned int i;
	unsigned int ray_type_count, entry_point_count;

	/* Check if irradiance cache is used */
	use_ambient = ambacc > FTINY && ambounce > 0 && ambdiv > 0;
	calc_ambient = use_ambient && nambvals == 0u;// && (ambfile == NULL || !ambfile[0]); // TODO Should really look at ambfp in ambinet.c to check that file is readable

	if ( calc_ambient ) {
		ray_type_count = RAY_TYPE_COUNT;
		entry_point_count = ENTRY_POINT_COUNT;
	} else {
		ray_type_count = 3u; /* shadow, radiance, and primary radiance */
		entry_point_count = 1u; /* Generate radiance data */
		if ( use_ambient )
			ray_type_count++; /* ambient ray */
	}

	/* Setup remote device */
	if (optix_remote_nodes > 0) {
		RTremotedevicestatus ready;
		int first = 1;

		RT_CHECK_ERROR_NO_CONTEXT(rtRemoteDeviceCreate(optix_remote_url, optix_remote_user, optix_remote_password, &remote));
		RT_CHECK_ERROR_NO_CONTEXT(rtRemoteDeviceReserve(remote, optix_remote_nodes, 0));

		// Wait until the VCA is ready
		mprintf("Waiting for %s", optix_remote_url);
		do {
			if (first)
				first = 0;
			else {
				mprintf(".");
				sleep(1); // poll once per second.
			}
			RT_CHECK_ERROR_NO_CONTEXT(rtRemoteDeviceGetAttribute(remote, RT_REMOTEDEVICE_ATTRIBUTE_STATUS, sizeof(RTremotedevicestatus), &ready));
		} while (ready != RT_REMOTEDEVICE_STATUS_READY);
		mprintf("\n");
		checkRemoteDevice(remote);
		remote_handle = remote;
	} else
		checkDevices();

	/* Setup context */
	RT_CHECK_ERROR2( rtContextCreate( context ) );
	if (optix_remote_nodes > 0) RT_CHECK_ERROR2(rtContextSetRemoteDevice(*context, remote));
	RT_CHECK_ERROR2( rtContextSetRayTypeCount( *context, ray_type_count ) );
	RT_CHECK_ERROR2( rtContextSetEntryPointCount( *context, entry_point_count ) );

	/* Set stack size for GPU threads */
	if (optix_stack_size > 0)
		RT_CHECK_ERROR2( rtContextSetStackSize( *context, optix_stack_size ) );

	/* Create a buffer of random number seeds */
	//createBuffer2D( *context, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, width, height, &seed_buffer );
	//RT_CHECK_ERROR2( rtBufferMap( seed_buffer, (void**)&seeds ) );
	//for ( i = 0; i < width * height; ++i ) 
	//	seeds[i] = rand();
	//RT_CHECK_ERROR2( rtBufferUnmap( seed_buffer ) );
	//applyContextObject( *context, "rnd_seeds", seed_buffer );

#ifdef TIMEOUT_CALLBACK
	if (optix_remote_nodes < 1 && alarm > 0)
		RT_CHECK_ERROR2( rtContextSetTimeoutCallback( *context, timeoutCallback, alarm ) );
#endif

#ifdef DEBUG_OPTIX
	/* Enable exception checking */
	RT_CHECK_ERROR2( rtContextSetExceptionEnabled( *context, RT_EXCEPTION_ALL, 1 ) );
#endif

#ifdef PRINT_OPTIX
	if (optix_remote_nodes < 1) {
		/* Enable message pringing */
		RT_CHECK_ERROR2( rtContextSetPrintEnabled( *context, 1 ) );
		RT_CHECK_ERROR2( rtContextSetPrintBufferSize( *context, 512 * width * height ) );
		//RT_CHECK_ERROR2( rtContextSetPrintLaunchIndex( *context, width / 2, height / 2, -1 ) );
	}
#endif

#ifdef REPORT_GPU_STATE
	/* Print device and context attributes */
	printCUDAProp();
	printContextInfo( *context );
#endif
}

/**
 * Destroy the OptiX context and free resources.
 */
static void destroyContext(const RTcontext context)
{
	RT_CHECK_ERROR(rtContextDestroy(context));
	if (remote_handle) {
		rtRemoteDeviceRelease(remote_handle);
		rtRemoteDeviceDestroy(remote_handle);
		remote_handle = NULL;
	}
	avsum_var = navsum_var = NULL;
}

#ifdef DAYSIM_COMPATIBLE
static void setupDaysim(const RTcontext context, RTbuffer* dc_buffer, const int width, const int height)
{
#ifndef DAYSIM
	RTbuffer            dc_scratch_buffer;
#endif

	/* Output daylight coefficient buffer */
#ifdef DAYSIM
	if (daysimGetCoefficients())
		createBuffer3D(context, RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, daysimGetCoefficients(), width, height, dc_buffer);
	else
		createBuffer3D(context, RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, 0, 0, 0, dc_buffer);
#else
	createBuffer3D(context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, 0, 0, 0, dc_buffer);
#endif
	applyContextObject(context, "dc_buffer", *dc_buffer);

	/* Scratch space */
#if defined DAYSIM && defined AMB_PARALLEL
	createBuffer3D(context, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 0, 0, 0, &dc_scratch_buffer);
#else
	createBuffer3D(context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, 0, 0, 0, &dc_scratch_buffer);
#endif
	applyContextObject(context, "dc_scratch_buffer", dc_scratch_buffer);
}
#endif /* DAYSIM_COMPATIBLE */

static void setupKernel(const RTcontext context, const VIEW* view, const int width, const int height, const unsigned int imm_irrad, const double dstrpix, const double mblur, const double dblur, const double alarm)
{
	/* Primary RTAPI objects */
	RTgeometryinstance  instance;

	/* Setup state */
	applyRadianceSettings(context, view, imm_irrad, dstrpix, mblur, dblur);
	createCamera( context, view );
	createGeometryInstance( context, &instance );
	createAcceleration(context, instance, imm_irrad);
	if ( imm_irrad )
		createIrradianceGeometry( context );

	/* Set up irradiance cache of ambient values */
	if ( use_ambient ) { // Don't bother with ambient records if -aa is set to zero
		if ( calc_ambient ) // Run pre-process only if no ambient file is available
			createAmbientRecords( context, view, width, height, alarm ); // implementation depends on current settings
		else
			setupAmbientCache( context, 0u ); // only need level 0 for final gather
	}
}

static void applyRadianceSettings(const RTcontext context, const VIEW* view, const unsigned int imm_irrad, const double dstrpix, const double mblur, const double dblur)
{
	/* Define ray types */
	applyContextVariable1ui( context, "radiance_primary_ray_type", PRIMARY_RAY );
	applyContextVariable1ui( context, "radiance_ray_type", RADIANCE_RAY );
	applyContextVariable1ui( context, "shadow_ray_type", SHADOW_RAY );

	/* Set hard coded parameters */
	applyContextVariable3f(context, "CIE_rgbf", (float)CIE_rf, (float)CIE_gf, (float)CIE_bf); // from color.h

	/* Set direct parameters */
	applyContextVariable1f(context, "dstrsrc", (float) dstrsrc); // -dj
	applyContextVariable1f(context, "srcsizerat", (float)srcsizerat); // -ds
	//applyContextVariable1f( context, "shadthresh", shadthresh ); // -dt
	//applyContextVariable1f( context, "shadcert", shadcert ); // -dc
	//applyContextVariable1i( context, "directrelay", directrelay ); // -dr
	//applyContextVariable1i( context, "vspretest", vspretest ); // -dp
	applyContextVariable1i( context, "directvis", directvis ); // -dv

	/* Set specular parameters */
	applyContextVariable1f(context, "specthresh", (float)specthresh); // -st
	applyContextVariable1f(context, "specjitter", (float)specjitter); // -ss

	/* Set ambient parameters */
	applyContextVariable3f( context, "ambval", ambval[0], ambval[1], ambval[2] ); // -av
	applyContextVariable1i( context, "ambvwt", ambvwt ); // -aw, zero by default
	applyContextVariable1i( context, "ambounce", ambounce ); // -ab
	//applyContextVariable1i( context, "ambres", ambres ); // -ar
	applyContextVariable1f(context, "ambacc", (float)ambacc); // -aa
	applyContextVariable1i( context, "ambdiv", ambdiv ); // -ad
	applyContextVariable1i(context, "ambdiv_final", ambacc > FTINY && optix_amb_fill != -1 ? optix_amb_fill : ambdiv); // -ag
	applyContextVariable1i( context, "ambssamp", ambssamp ); // -as
	applyContextVariable1f(context, "maxarad", (float)maxarad); // maximum ambient radius from ambient.h, based on armbres
	applyContextVariable1f(context, "minarad", (float)minarad); // minimum ambient radius from ambient.h, based on armbres
	avsum_var = applyContextVariable1f(context, "avsum", (float)avsum); // computed ambient value sum (log) from ambient.c
	navsum_var = applyContextVariable1ui(context, "navsum", navsum); // number of values in avsum from ambient.c

	/* Set medium parameters */
	//applyContextVariable3f( context, "cextinction", cextinction[0], cextinction[1], cextinction[2] ); // -me
	//applyContextVariable3f( context, "salbedo", salbedo[0], salbedo[1], salbedo[2] ); // -ma
	//applyContextVariable1f( context, "seccg", seccg ); // -mg
	//applyContextVariable1f( context, "ssampdist", ssampdist ); // -ms

	/* Set ray limitting parameters */
	applyContextVariable1f(context, "minweight", (float)minweight); // -lw, from ray.h
	applyContextVariable1i( context, "maxdepth", maxdepth ); // -lr, from ray.h, negative values indicate Russian roulette

	if (rand_samp)
		applyContextVariable1ui(context, "random_seed", random()); // -u

	if (view) {
		camera_frame = applyContextVariable1ui(context, "frame", frame);
		camera_type  = applyContextVariable1ui(context, "camera", view->type); // -vt
		camera_eye   = applyContextVariable3f(context, "eye", (float)view->vp[0], (float)view->vp[1], (float)view->vp[2]); // -vp
		camera_u     = applyContextVariable3f(context, "U", (float)view->hvec[0], (float)view->hvec[1], (float)view->hvec[2]);
		camera_v     = applyContextVariable3f(context, "V", (float)view->vvec[0], (float)view->vvec[1], (float)view->vvec[2]);
		camera_w     = applyContextVariable3f(context, "W", (float)view->vdir[0], (float)view->vdir[1], (float)view->vdir[2]); // -vd
		camera_fov   = applyContextVariable2f(context, "fov", (float)view->horiz, (float)view->vert); // -vh, -vv
		camera_shift = applyContextVariable2f(context, "shift", (float)view->hoff, (float)view->voff); // -vs, -vl
		camera_clip  = applyContextVariable2f(context, "clip", (float)view->vfore, (float)view->vaft); // -vo, -va
		camera_vdist = applyContextVariable1f(context, "vdist", (float)view->vdist);
		applyContextVariable1f(context, "dstrpix", (float)dstrpix); // -pj
		applyContextVariable1f(context, "mblur", (float)mblur); // -pm
		applyContextVariable1f(context, "dblur", (float)dblur); // -pd
	} else
		applyContextVariable1ui(context, "imm_irrad", imm_irrad); // -I

#ifdef DAYSIM
	/* Set daylight coefficient parameters */
	//applyContextVariable1i(context, "daysimSortMode", daysimSortMode); // -D
	applyContextVariable1ui(context, "daylightCoefficients", daysimGetCoefficients()); // -N
#endif
}

static void createCamera( const RTcontext context, const VIEW* view )
{
	RTprogram  ray_gen_program;
	RTprogram  exception_program;
	RTprogram  miss_program;
	RTprogram  miss_shadow_program;

	/* Ray generation program */
	if ( view ) { // do RPICT
		ptxFile( path_to_ptx, "camera" );
		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "image_camera", &ray_gen_program ) );
	} else { // do RTRACE
		ptxFile( path_to_ptx, "sensor" );
		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "ray_generator", &ray_gen_program ) );
	}
	applyProgramVariable1ui( context, ray_gen_program, "do_irrad", do_irrad ); // -i
	RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, RADIANCE_ENTRY, ray_gen_program ) );

	/* Exception program */
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "exception", &exception_program ) );
	RT_CHECK_ERROR( rtContextSetExceptionProgram( context, RADIANCE_ENTRY, exception_program ) );

	/* Miss program */
	ptxFile( path_to_ptx, "background" );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "miss", &miss_program ) );
	if ( do_irrad )
		RT_CHECK_ERROR( rtContextSetMissProgram( context, PRIMARY_RAY, miss_program ) );
	RT_CHECK_ERROR( rtContextSetMissProgram( context, RADIANCE_RAY, miss_program ) );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "miss_shadow", &miss_shadow_program ) );
	RT_CHECK_ERROR( rtContextSetMissProgram( context, SHADOW_RAY, miss_shadow_program ) );
#ifdef HIT_TYPE
	applyProgramVariable1ui( context, miss_program, "type", OBJ_SOURCE );
#endif
}

static void updateCamera( const RTcontext context, const VIEW* view )
{
	if (!view || !camera_frame) return; // Should really test all, but we'll assume that all are set together.

	RT_CHECK_ERROR( rtVariableSet1ui( camera_frame, frame ) );
	RT_CHECK_ERROR( rtVariableSet1ui( camera_type, view->type ) ); // -vt
	RT_CHECK_ERROR(rtVariableSet3f(camera_eye, (float)view->vp[0], (float)view->vp[1], (float)view->vp[2])); // -vp
	RT_CHECK_ERROR(rtVariableSet3f(camera_u, (float)view->hvec[0], (float)view->hvec[1], (float)view->hvec[2]));
	RT_CHECK_ERROR(rtVariableSet3f(camera_v, (float)view->vvec[0], (float)view->vvec[1], (float)view->vvec[2]));
	RT_CHECK_ERROR(rtVariableSet3f(camera_w, (float)view->vdir[0], (float)view->vdir[1], (float)view->vdir[2])); // -vd
	RT_CHECK_ERROR(rtVariableSet2f(camera_fov, (float)view->horiz, (float)view->vert)); // -vh, -vv
	RT_CHECK_ERROR(rtVariableSet2f(camera_shift, (float)view->hoff, (float)view->voff)); // -vs, -vl
	RT_CHECK_ERROR(rtVariableSet2f(camera_clip, (float)view->vfore, (float)view->vaft)); // -vo, -va
	RT_CHECK_ERROR(rtVariableSet1f(camera_vdist, (float)view->vdist));
}

static void createGeometryInstance( const RTcontext context, RTgeometryinstance* instance )
{
	RTgeometry mesh;
	RTprogram  mesh_intersection_program;
	RTprogram  mesh_bounding_box_program;
	RTbuffer   buffer;

	unsigned int i;
	OBJECT on;
	OBJREC* rec, *parent;

	/* Timers */
	clock_t geometry_clock;

	/* This array gives the OptiX buffer index of each rad file object.
	 * The OptiX buffer refered to depends on the type of object.
	 * If the object does not have an index in an OptiX buffer, -1 is given. */
	buffer_entry_index = (int *)malloc(sizeof(int) * nobjects);
	if (buffer_entry_index == NULL) goto memerr;

	geometry_clock = clock();

	/* Create buffers for storing geometry information. */
	vertices = (FloatArray *)malloc(sizeof(FloatArray));
	if (vertices == NULL) goto memerr;
	initArrayf(vertices, EXPECTED_VERTICES * 3);

	normals = (FloatArray *)malloc(sizeof(FloatArray));
	if (normals == NULL) goto memerr;
	initArrayf(normals, EXPECTED_VERTICES * 3);

	tex_coords = (FloatArray *)malloc(sizeof(FloatArray));
	if (tex_coords == NULL) goto memerr;
	initArrayf(tex_coords, EXPECTED_VERTICES * 2);

	vertex_indices = (IntArray *)malloc(sizeof(IntArray));
	if (vertex_indices == NULL) goto memerr;
	initArrayi(vertex_indices, EXPECTED_TRIANGLES * 3);

	traingles = (IntArray *)malloc(sizeof(IntArray));
	if (traingles == NULL) goto memerr;
	initArrayi(traingles, EXPECTED_TRIANGLES);

	materials = (MaterialArray *)malloc(sizeof(MaterialArray));
	if (materials == NULL) goto memerr;
	initArraym(materials, EXPECTED_MATERIALS);

	alt_materials = (IntArray *)malloc(sizeof(IntArray));
	if (alt_materials == NULL) goto memerr;
	initArrayi(alt_materials, EXPECTED_MATERIALS * 2);

	/* Create buffers for storing lighting information. */
#ifdef LIGHTS
	lights = (IntArray *)malloc(sizeof(IntArray));
	if (lights == NULL) goto memerr;
	initArrayi(lights, EXPECTED_LIGHTS * 3);
#endif

	sources = (DistantLightArray *)malloc(sizeof(DistantLightArray));
	if (sources == NULL) goto memerr;
	initArraydl(sources, EXPECTED_SOURCES);

	vertex_index_0 = 0u;

	/* Material 0 is Lambertian. */
	if ( do_irrad ) {
		insertArray2i(alt_materials, (int)materials->count, (int)materials->count);
		insertArraym(materials, createNormalMaterial(context, &Lamb));
	}

	/* Get the scene geometry as a list of triangles. */
	for (on = 0; on < nobjects; on++) {
		/* By default, no buffer entry is refered to. */
		buffer_entry_index[on] = -1;

		rec = objptr(on);
		if (rec->omod != OVOID)
			parent = objptr(rec->omod);
		else
			parent = NULL;

		addRadianceObject(context, rec, parent, on);
	}

	free( buffer_entry_index );

	/* Check for overflow */
	if (traingles->count > UINT_MAX) {
		sprintf(errmsg, "Number of triangles %" PRIu64 " is greater than maximum %u.", traingles->count, UINT_MAX);
		error(USER, errmsg);
	}
	if (materials->count > INT_MAX) {
		sprintf(errmsg, "Number of materials %" PRIu64 " is greater than maximum %u.", materials->count, INT_MAX);
		error(USER, errmsg);
	}

	/* Create the geometry reference for OptiX. */
	RT_CHECK_ERROR(rtGeometryCreate(context, &mesh));
	RT_CHECK_ERROR(rtGeometrySetPrimitiveCount(mesh, (unsigned int)traingles->count));

	ptxFile(path_to_ptx, "triangle_mesh");
	RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "mesh_bounds", &mesh_bounding_box_program));
	RT_CHECK_ERROR(rtGeometrySetBoundingBoxProgram(mesh, mesh_bounding_box_program));
	RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "mesh_intersect", &mesh_intersection_program));
	RT_CHECK_ERROR(rtGeometrySetIntersectionProgram(mesh, mesh_intersection_program));
	applyProgramVariable1ui(context, mesh_intersection_program, "backvis", backvis); // -bv

	/* Create the geometry instance containing the geometry. */
	RT_CHECK_ERROR(rtGeometryInstanceCreate(context, instance));
	RT_CHECK_ERROR(rtGeometryInstanceSetGeometry(*instance, mesh));
	RT_CHECK_ERROR(rtGeometryInstanceSetMaterialCount(*instance, (unsigned int)materials->count));

	/* Apply materials to the geometry instance. */
	vprintf("Processed %" PRIu64 " materials.\n", materials->count);
	for (i = 0u; i < materials->count; i++)
		RT_CHECK_ERROR(rtGeometryInstanceSetMaterial(*instance, i, materials->array[i]));
	freeArraym(materials);
	free(materials);

	/* Unmap and apply the geometry buffers. */
	vprintf("Processed %" PRIu64 " vertices.\n", vertices->count / 3);
	createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, vertices->count / 3, &buffer);
	copyToBufferf(context, buffer, vertices);
	//applyGeometryInstanceObject( context, *instance, "vertex_buffer", buffer );
	applyContextObject(context, "vertex_buffer", buffer);
	freeArrayf(vertices);
	free(vertices);

	createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, normals->count / 3, &buffer);
	copyToBufferf(context, buffer, normals);
	applyGeometryInstanceObject(context, *instance, "normal_buffer", buffer);
	freeArrayf(normals);
	free(normals);

	createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, tex_coords->count / 2, &buffer);
	copyToBufferf(context, buffer, tex_coords);
	applyGeometryInstanceObject(context, *instance, "texcoord_buffer", buffer);
	freeArrayf(tex_coords);
	free(tex_coords);

	createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, vertex_indices->count / 3, &buffer);
	copyToBufferi(context, buffer, vertex_indices);
	applyGeometryInstanceObject(context, *instance, "vindex_buffer", buffer);
	freeArrayi(vertex_indices);
	free(vertex_indices);

	vprintf("Processed %" PRIu64 " triangles.\n", traingles->count);
	createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_INT, traingles->count, &buffer);
	copyToBufferi(context, buffer, traingles);
	applyGeometryInstanceObject(context, *instance, "material_buffer", buffer);
	freeArrayi(traingles);
	free(traingles);

	createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_INT2, alt_materials->count / 2, &buffer);
	copyToBufferi(context, buffer, alt_materials);
	applyGeometryInstanceObject(context, *instance, "material_alt_buffer", buffer);
	freeArrayi(alt_materials);
	free(alt_materials);

	/* Unmap and apply the lighting buffers. */
#ifdef LIGHTS
	if (lights->count) vprintf("Processed %" PRIu64 " lights.\n", lights->count / 3);
	createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, lights->count / 3, &buffer);
	copyToBufferi(context, buffer, lights);
	applyContextObject(context, "lindex_buffer", buffer);
	freeArrayi(lights);
	free(lights);
#endif

	if (sources->count) vprintf("Processed %" PRIu64 " sources.\n", sources->count);
	createCustomBuffer1D(context, RT_BUFFER_INPUT, sizeof(DistantLight), sources->count, &buffer);
	copyToBufferdl(context, buffer, sources);
	applyContextObject(context, "lights", buffer);
	freeArraydl(sources);
	free(sources);

	geometry_clock = clock() - geometry_clock;
	mprintf("Geometry build time: %" PRIu64 " milliseconds for %i objects.\n", MILLISECONDS(geometry_clock), nobjects);
	return;
memerr:
	error(SYSTEM, "out of memory in createGeometryInstance");
}

static void addRadianceObject(const RTcontext context, OBJREC* rec, OBJREC* parent, const OBJECT index)
{
	switch (rec->otype) {
	case MAT_PLASTIC: // Plastic material
	case MAT_METAL: // Metal material
	case MAT_TRANS: // Translucent material
		buffer_entry_index[index] = insertArray2i(alt_materials, 0, (int)materials->count);
		insertArraym(materials, createNormalMaterial(context, rec));
		break;
	case MAT_GLASS: // Glass material
	case MAT_DIELECTRIC: // Dielectric material TODO handle separately, see dialectric.c
		buffer_entry_index[index] = insertArray2i(alt_materials, -1, (int)materials->count);
		insertArraym(materials, createGlassMaterial(context, rec));
		break;
	case MAT_LIGHT: // primary light source material, may modify a face or a source (solid angle)
	case MAT_ILLUM: // secondary light source material
	case MAT_GLOW: // Glow material
	case MAT_SPOT: // Spotlight material
		buffer_entry_index[index] = (int)materials->count;
		if (rec->otype != MAT_ILLUM)
			insertArray2i(alt_materials, (int)materials->count, (int)materials->count);
		else if (rec->oargs.nsargs && strcmp(rec->oargs.sarg[0], VOIDID)) { /* modifies another material */
			//material = objptr( lastmod( objndx(rec), rec->oargs.sarg[0] ) );
			//alt = buffer_entry_index[objndx(material)];
			int alt = buffer_entry_index[lastmod(objndx(rec), rec->oargs.sarg[0])];
			insertArray2i(alt_materials, (int)materials->count, alt);
		}
		else
			insertArray2i(alt_materials, (int)materials->count, -1);
		insertArraym(materials, createLightMaterial(context, rec));
		break;
	case OBJ_FACE: // Typical polygons
		createFace(rec, parent);
		break;
	case OBJ_SPHERE: // Sphere
	case OBJ_BUBBLE: // Inverted sphere
		createSphere(rec, parent);
		break;
	case OBJ_CONE: // Cone
	case OBJ_CUP: // Inverted cone
	case OBJ_CYLINDER: // Cylinder
	case OBJ_TUBE: // Inverted cylinder
	case OBJ_RING: // Disk
		createCone(rec, parent);
		break;
	case OBJ_MESH: // Mesh from file
		createMesh(rec, parent);
		break;
	case OBJ_SOURCE:
		insertArraydl(sources, createDistantLight(context, rec, parent));
		break;
	case PAT_BFUNC: // brightness function, used for sky brightness
		buffer_entry_index[index] = createFunction(context, rec);
		break;
	case PAT_BDATA: // brightness texture, used for IES lighting data
		buffer_entry_index[index] = createTexture(context, rec);
		break;
	case TEX_FUNC:
		if (rec->oargs.nsargs == 3) {
			if (!strcmp(rec->oargs.sarg[2], TCALNAME)) break; // Handled by face
		}
		else if (rec->oargs.nsargs >= 4) {
			if (!strcmp(rec->oargs.sarg[3], TCALNAME)) break; // Handled by face
		}
		printObject(rec);
		break;
	case MOD_ALIAS:
		if (rec->oargs.nsargs) {
			if (rec->oargs.nsargs > 1)
				objerror(rec, INTERNAL, "too many string arguments");
			addRadianceObject(context, objptr(lastmod(objndx(rec), rec->oargs.sarg[0])), objptr(rec->omod), index); // TODO necessary?
		}
		// Otherwise it's a pass-through (do nothing)
		break;
	default:
#ifdef DEBUG_OPTIX
		printObject(rec);
#endif
		break;
	}
}

static void createFace(OBJREC* rec, OBJREC* parent)
{
	int j, k;
	FACE* face = getface(rec);
	OBJREC* material = findmaterial(parent);
	if (material == NULL)
		objerror(rec, USER, "missing material");
#ifdef TRIANGULATE
	/* Triangulate the face */
	if (!createTriangles(face, material)) {
		objerror(rec, WARNING, "triangulation failed");
		goto facedone;
	}
#else /* TRIANGULATE */
	/* Triangulate the polygon as a fan */
	for (j = 2; j < face->nv; j++)
		createTriangle(material, vertex_index_0, vertex_index_0 + j - 1, vertex_index_0 + j);
#endif /* TRIANGULATE */

	/* Write the vertices to the buffers */
	material = findFunction(parent); // TODO can there be multiple parent functions?
	for (j = 0; j < face->nv; j++) {
		RREAL *va = VERTEX(face, j);
		insertArray3f(vertices, (float)va[0], (float)va[1], (float)va[2]);

		if (material && material->otype == TEX_FUNC && material->oargs.nsargs >= 4 && !strcmp(material->oargs.sarg[3], TCALNAME)) {
			/* Normal calculation from tmesh.cal */
			double bu, bv;
			FVECT v;
			XF fxp;

			k = (int)material->oargs.farg[0];
			bu = va[(k + 1) % 3];
			bv = va[(k + 2) % 3];

			v[0] = bu * material->oargs.farg[1] + bv * material->oargs.farg[2] + material->oargs.farg[3];
			v[1] = bu * material->oargs.farg[4] + bv * material->oargs.farg[5] + material->oargs.farg[6];
			v[2] = bu * material->oargs.farg[7] + bv * material->oargs.farg[8] + material->oargs.farg[9];

			/* Get transform */
			if (!(k = createTransform(&fxp, NULL, material)))
				objerror(material, USER, "bad transform");
			if (k != 1) { // not identity
				FVECT v1;
				VCOPY(v1, v);
				if (normalize(v1) != 0.0) {
					VSUB(v, v1, face->norm);
					multv3(v, v, fxp.xfm); //TODO Consider object transformation as well
					VSUM(v, face->norm, v, 1.0 / fxp.sca);
				}
			}

			if (normalize(v) == 0.0) {
				objerror(rec, WARNING, "illegal normal perturbation");
				insertArray3f(normals, (float)face->norm[0], (float)face->norm[1], (float)face->norm[2]);
			} else
				insertArray3f(normals, (float)v[0], (float)v[1], (float)v[2]);
		}
		else {
			//TODO Implement bump maps from texfunc and texdata
			// This might be done in the Intersecion program in triangle_mesh.cu rather than here
			insertArray3f(normals, (float)face->norm[0], (float)face->norm[1], (float)face->norm[2]);
		}

		if (material && material->otype == TEX_FUNC && material->oargs.nsargs == 3 && !strcmp(material->oargs.sarg[2], TCALNAME)) {
			/* Texture coordinate calculation from tmesh.cal */
			double bu, bv;

			k = (int)material->oargs.farg[0];
			bu = va[(k + 1) % 3];
			bv = va[(k + 2) % 3];

			insertArray2f(tex_coords,
				(float)(bu * material->oargs.farg[1] + bv * material->oargs.farg[2] + material->oargs.farg[3]),
				(float)(bu * material->oargs.farg[4] + bv * material->oargs.farg[5] + material->oargs.farg[6]));
		}
		else {
			//TODO Implement texture maps from colorfunc, brightfunc, colordata, brightdata, and colorpict
			// This might be done in the Intersecion program in triangle_mesh.cu rather than here
			insertArray2f(tex_coords, 0.0f, 0.0f);
		}
	}
	vertex_index_0 += face->nv;
facedone:
	freeface(rec);
}

static __inline void createTriangle(OBJREC *material, const int a, const int b, const int c)
{
	/* Write the indices to the buffers */
	insertArray3i(vertex_indices, a, b, c);
	insertArrayi(traingles, buffer_entry_index[objndx(material)]);

#ifdef LIGHTS
	if (islight(material->otype) && (material->otype != MAT_GLOW || material->oargs.farg[3] > 0))
		insertArray3i(lights, a, b, c);
#endif /* LIGHTS */
}

#ifdef TRIANGULATE
/* Generate list of triangle vertex indices from triangulation of face */
static int createTriangles(const FACE *face, OBJREC *material)
{
	if (face->nv == 3) {	/* simple case */
		createTriangle(material, vertex_index_0, vertex_index_0 + 1, vertex_index_0 + 2);
		return(1);
	}
	if (face->nv > 3) {	/* triangulation necessary */
		int i;
		Vert2_list	*v2l = polyAlloc(face->nv);
		if (v2l == NULL)	/* out of memory */
			return(0);
		if (face->norm[face->ax] > 0)	/* maintain winding direction */
			for (i = v2l->nv; i--; ) {
				v2l->v[i].mX = VERTEX(face, i)[(face->ax + 1) % 3];
				v2l->v[i].mY = VERTEX(face, i)[(face->ax + 2) % 3];
			}
		else
			for (i = v2l->nv; i--; ) {
				v2l->v[i].mX = VERTEX(face, i)[(face->ax + 2) % 3];
				v2l->v[i].mY = VERTEX(face, i)[(face->ax + 1) % 3];
			}
		v2l->p = (void *)material;
		i = polyTriangulate(v2l, addTriangle);
		polyFree(v2l);
		return(i);
	}
	return(0);	/* degenerate case */
}

/* Add triangle to polygon's list (call-back function) */
static int addTriangle( const Vert2_list *tp, int a, int b, int c )
{
	OBJREC *material = (OBJREC *)tp->p;
	createTriangle(material, vertex_index_0 + a, vertex_index_0 + b, vertex_index_0 + c);
	return(1);
}
#endif /* TRIANGULATE */

static void createSphere(OBJREC* rec, OBJREC* parent)
{
	unsigned int i, j, steps;
	int direction;
	FVECT x, y, z;
	OBJREC* material = findmaterial(parent);
	if (material == NULL)
		objerror(rec, USER, "missing material");

	// Create octahedron
	IntArray *sph_vertex_indices = (IntArray *)malloc(sizeof(IntArray));
	FloatArray *sph_vertices = (FloatArray *)malloc(sizeof(FloatArray));
	if (sph_vertex_indices == NULL || sph_vertices == NULL) goto sphmemerr;
	initArrayi(sph_vertex_indices, 24u);
	initArrayf(sph_vertices, 18u);

	insertArray3i(sph_vertex_indices, 0, 2, 4);
	insertArray3i(sph_vertex_indices, 2, 1, 4);
	insertArray3i(sph_vertex_indices, 1, 3, 4);
	insertArray3i(sph_vertex_indices, 3, 0, 4);
	insertArray3i(sph_vertex_indices, 0, 5, 2);
	insertArray3i(sph_vertex_indices, 2, 5, 1);
	insertArray3i(sph_vertex_indices, 1, 5, 3);
	insertArray3i(sph_vertex_indices, 3, 5, 0);

	insertArray3f(sph_vertices, 1.0f, 0.0f, 0.0f);
	insertArray3f(sph_vertices,-1.0f, 0.0f, 0.0f);
	insertArray3f(sph_vertices, 0.0f, 1.0f, 0.0f);
	insertArray3f(sph_vertices, 0.0f,-1.0f, 0.0f);
	insertArray3f(sph_vertices, 0.0f, 0.0f, 1.0f);
	insertArray3f(sph_vertices, 0.0f, 0.0f,-1.0f);

	if (rec->oargs.nfargs > 4)
		steps = (unsigned int)rec->oargs.farg[4];
	else
		steps = DEFAULT_SPHERE_STEPS;

	direction = rec->otype == OBJ_SPHERE ? 1 : -1; // Sphere or Bubble

	// Subdivide triangles
	for (i = 0u; i < steps; i++) {
		IntArray *new_vertex_indices = (IntArray *)malloc(sizeof(IntArray));
		if (new_vertex_indices == NULL) goto sphmemerr;
		initArrayi(new_vertex_indices, sph_vertex_indices->count * 4);

		for (j = 0u; j < sph_vertex_indices->count; j += 3) {
			VADD(x, &sph_vertices->array[sph_vertex_indices->array[j] * 3], &sph_vertices->array[sph_vertex_indices->array[j + 1] * 3]);
			VADD(y, &sph_vertices->array[sph_vertex_indices->array[j + 1] * 3], &sph_vertices->array[sph_vertex_indices->array[j + 2] * 3]);
			VADD(z, &sph_vertices->array[sph_vertex_indices->array[j + 2] * 3], &sph_vertices->array[sph_vertex_indices->array[j] * 3]);
			normalize(x);
			normalize(y);
			normalize(z);

			insertArray3f(sph_vertices, (float)x[0], (float)x[1], (float)x[2]);
			insertArray3f(sph_vertices, (float)y[0], (float)y[1], (float)y[2]);
			insertArray3f(sph_vertices, (float)z[0], (float)z[1], (float)z[2]);

			insertArray3i(new_vertex_indices, (int)sph_vertex_indices->array[j], (int)sph_vertices->count / 3 - 3, (int)sph_vertices->count / 3 - 1);
			insertArray3i(new_vertex_indices, (int)sph_vertices->count / 3 - 3, (int)sph_vertex_indices->array[j + 1], (int)sph_vertices->count / 3 - 2);
			insertArray3i(new_vertex_indices, (int)sph_vertices->count / 3 - 1, (int)sph_vertices->count / 3 - 2, (int)sph_vertex_indices->array[j + 2]);
			insertArray3i(new_vertex_indices, (int)sph_vertices->count / 3 - 3, (int)sph_vertices->count / 3 - 2, (int)sph_vertices->count / 3 - 1);
		}

		freeArrayi(sph_vertex_indices);
		free(sph_vertex_indices);

		sph_vertex_indices = new_vertex_indices;
	}

	// Add resulting traingles
	for (j = 1u; j < sph_vertex_indices->count; j += 3) {
		createTriangle(material,
			vertex_index_0 + sph_vertex_indices->array[j - direction],
			vertex_index_0 + sph_vertex_indices->array[j],
			vertex_index_0 + sph_vertex_indices->array[j + direction]);
	}

	// Add resulting vertices
	for (j = 0u; j < sph_vertices->count; j += 3) {
		insertArray3f(vertices,
			(float)(rec->oargs.farg[0] + rec->oargs.farg[3] * sph_vertices->array[j]),
			(float)(rec->oargs.farg[1] + rec->oargs.farg[3] * sph_vertices->array[j + 1]),
			(float)(rec->oargs.farg[2] + rec->oargs.farg[3] * sph_vertices->array[j + 2]));
		insertArray3f(normals,
			direction * sph_vertices->array[j],
			direction * sph_vertices->array[j + 1],
			direction * sph_vertices->array[j + 2]);
		insertArray2f(tex_coords, 0.0f, 0.0f);
	}

	vertex_index_0 += (unsigned int)sph_vertices->count / 3;

	// Free memory
	freeArrayi(sph_vertex_indices);
	free(sph_vertex_indices);
	freeArrayf(sph_vertices);
	free(sph_vertices);

	return;
sphmemerr:
	error(SYSTEM, "out of memory in createSphere");
}

static void createCone(OBJREC* rec, OBJREC* parent)
{
	unsigned int i, j, steps;
	int isCone, direction;
	double theta, sphi, cphi;
	FVECT u, v, n;
	CONE* cone = getcone(rec, 0);
	OBJREC* material = findmaterial(parent);
	if (material == NULL)
		objerror(rec, USER, "missing material");

	// Get orthonormal basis
	if (!getperpendicular(u, cone->ad, 0))
		objerror(rec, USER, "bad normal direction");
	VCROSS(v, cone->ad, u);

	isCone = rec->otype != OBJ_CYLINDER && rec->otype != OBJ_TUBE;
	direction = (rec->otype == OBJ_CUP || rec->otype == OBJ_TUBE) ? -1 : 1;

	if (rec->oargs.nfargs > 7 + isCone) // TODO oconv won't allow extra arguments
		steps = (unsigned int)rec->oargs.farg[7 + isCone];
	else
		steps = DEFAULT_CONE_STEPS;
	if (steps < 3)
		objerror(rec, USER, "resolution too low");
	
	// Check that cone is closed
	if (isCone)
		isCone = CO_R0(cone) == 0 ? 1u : CO_R1(cone) == 0 ? 2u : 0u;

	if (CO_R0(cone) == CO_R1(cone)) { // Cylinder
		sphi = 0.0;
		cphi = 1.0;
	}
	else if (cone->al) { // Cone
		double phi = atan((CO_R0(cone) - CO_R1(cone)) / cone->al);
		sphi = sin(phi);
		cphi = cos(phi);
	}
	else { // Ring
		if (CO_R0(cone) < CO_R1(cone))
			direction = -direction;
		sphi = direction;
		cphi = 0.0;
	}

	for (i = 0u; i < steps; i++) {
		// Add traingles
		if (isCone != 1)
			createTriangle(material,
				vertex_index_0 + (2 * i + 1 - direction) % (2 * steps),
				vertex_index_0 + (2 * i + 1 + direction) % (2 * steps),
				vertex_index_0 + (2 * i + 1) % (2 * steps));
		if (isCone != 2)
			createTriangle(material,
				vertex_index_0 + (2 * i + 2) % (2 * steps),
				vertex_index_0 + (2 * i + 2 + direction) % (2 * steps),
				vertex_index_0 + (2 * i + 2 - direction) % (2 * steps));

		// Add vertices
		theta = PI * 2 * i / steps;
		for (j = 0u; j < 3; j++)
			n[j] = u[j] * cos(theta) + v[j] * sin(theta);
		insertArray3f(vertices,
			(float)(CO_P0(cone)[0] + CO_R0(cone) * n[0]),
			(float)(CO_P0(cone)[1] + CO_R0(cone) * n[1]),
			(float)(CO_P0(cone)[2] + CO_R0(cone) * n[2]));
		insertArray3f(vertices,
			(float)(CO_P1(cone)[0] + CO_R1(cone) * n[0]),
			(float)(CO_P1(cone)[1] + CO_R1(cone) * n[1]),
			(float)(CO_P1(cone)[2] + CO_R1(cone) * n[2]));
		for (j = 0u; j < 3; j++)
			n[j] = direction * (sphi * cone->ad[j] + cphi * n[j]);
		insertArray3f(normals, (float)n[0], (float)n[1], (float)n[2]);
		insertArray3f(normals, (float)n[0], (float)n[1], (float)n[2]);
		insertArray2f(tex_coords, 0.0f, 0.0f);
		insertArray2f(tex_coords, 0.0f, 0.0f);
	}
	vertex_index_0 += steps * 2;
	freecone(rec);
}

static void createMesh(OBJREC* rec, OBJREC* parent)
{
	int j, k;
	OBJREC* material;
	MESHINST *meshinst = getmeshinst(rec, IO_ALL);
	unsigned int vertex_index_mesh = vertex_index_0; //TODO what if not all patches are full?
	for (j = 0; j < meshinst->msh->npatches; j++) {
		MESHPATCH *pp = &(meshinst->msh)->patch[j];
		if (parent) {
			material = findmaterial(parent);
		}
		else if (!pp->trimat) {
			OBJECT mo = pp->solemat;
			if (mo != OVOID)
				mo += meshinst->msh->mat0;
			material = findmaterial(objptr(mo));
		}

		/* Write the indices to the buffers */
		for (k = 0; k < pp->ntris; k++) {
			if (rec->omod == OVOID && pp->trimat) {
				OBJECT mo = pp->trimat[k];
				if (mo != OVOID)
					mo += meshinst->msh->mat0;
				material = findmaterial(objptr(mo));
			}
			createTriangle(material, vertex_index_0 + pp->tri[k].v1, vertex_index_0 + pp->tri[k].v2, vertex_index_0 + pp->tri[k].v3);
		}

		for (k = 0; k < pp->nj1tris; k++) {
			if (rec->omod == OVOID && pp->trimat) {
				OBJECT mo = pp->j1tri[k].mat;
				if (mo != OVOID)
					mo += meshinst->msh->mat0;
				material = findmaterial(objptr(mo));
			}
			createTriangle(material, vertex_index_mesh + pp->j1tri[k].v1j, vertex_index_0 + pp->j1tri[k].v2, vertex_index_0 + pp->j1tri[k].v3);
		}

		for (k = 0; k < pp->nj2tris; k++) {
			if (rec->omod == OVOID && pp->trimat) {
				OBJECT mo = pp->j2tri[k].mat;
				if (mo != OVOID)
					mo += meshinst->msh->mat0;
				material = findmaterial(objptr(mo));
			}
			createTriangle(material, vertex_index_mesh + pp->j2tri[k].v1j, vertex_index_mesh + pp->j2tri[k].v2j, vertex_index_0 + pp->j2tri[k].v3);
		}

		/* Write the vertices to the buffers */
		// TODO do this once per mesh and use transform for different mesh instances
		for (k = 0; k < pp->nverts; k++) {
			MESHVERT mesh_vert;
			FVECT transform;
			getmeshvert(&mesh_vert, meshinst->msh, k + (j << 8), MT_ALL);
			if (!(mesh_vert.fl & MT_V))
				objerror(rec, INTERNAL, "missing mesh vertices in createMesh");
			multp3(transform, mesh_vert.v, meshinst->x.f.xfm);
			insertArray3f(vertices, (float)transform[0], (float)transform[1], (float)transform[2]);

			if (mesh_vert.fl & MT_N) { // TODO what if normal is defined by texture function
				multv3(transform, mesh_vert.n, meshinst->x.f.xfm);
				insertArray3f(normals, (float)transform[0], (float)transform[1], (float)transform[2]);
			} else
				insertArray3f(normals, 0.0f, 0.0f, 0.0f); //TODO Can this happen?

			if (mesh_vert.fl & MT_UV)
				insertArray2f(tex_coords, (float)mesh_vert.uv[0], (float)mesh_vert.uv[1]);
			else
				insertArray2f(tex_coords, 0.0f, 0.0f);
		}

		vertex_index_0 += pp->nverts;
	}
	freemeshinst(rec);
}

static RTmaterial createNormalMaterial(const RTcontext context, OBJREC* rec)
{
	RTmaterial material;

	/* Create our hit programs to be shared among all materials */
	if ( !radiance_normal_closest_hit_program ) {
		ptxFile( path_to_ptx, "radiance_normal" );
		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "closest_hit_radiance", &radiance_normal_closest_hit_program ) );
		applyProgramVariable1ui( context, radiance_normal_closest_hit_program, "metal", MAT_METAL );
#ifndef LIGHTS
		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "any_hit_shadow", &shadow_normal_any_hit_program ) );
#endif
	}

	RT_CHECK_ERROR( rtMaterialCreate( context, &material ) );
	if ( do_irrad )
		RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( material, PRIMARY_RAY, radiance_normal_closest_hit_program ) );
	RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( material, RADIANCE_RAY, radiance_normal_closest_hit_program ) );
#ifndef LIGHTS
	RT_CHECK_ERROR( rtMaterialSetAnyHitProgram( material, SHADOW_RAY, shadow_normal_any_hit_program ) ); // Cannot use any hit program because closest might be light source
#endif

	if (calc_ambient) {
		if (!ambient_normal_closest_hit_program) {
			ptxFile(path_to_ptx, "ambient_normal");
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "closest_hit_ambient", &ambient_normal_closest_hit_program ) );

#ifdef AMBIENT_CELL
			createAmbientDynamicStorage(context, ambient_normal_closest_hit_program, 0);
#else
			createAmbientDynamicStorage(context, ambient_normal_closest_hit_program, cuda_kmeans_clusters);
#endif
		}
		RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( material, AMBIENT_RECORD_RAY, ambient_normal_closest_hit_program ) );

		if ( !point_cloud_closest_hit_program ) {
			ptxFile( path_to_ptx, "point_cloud_normal" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "closest_hit_point_cloud", &point_cloud_closest_hit_program ) );
		}
		RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( material, POINT_CLOUD_RAY, point_cloud_closest_hit_program ) );
	}

	/* Set variables to be consumed by material for this geometry instance */
	applyMaterialVariable1ui( context, material, "type", rec->otype );
	applyMaterialVariable3f(context, material, "color", (float)rec->oargs.farg[0], (float)rec->oargs.farg[1], (float)rec->oargs.farg[2]);
	applyMaterialVariable1f(context, material, "spec", (float)rec->oargs.farg[3]);
	applyMaterialVariable1f(context, material, "rough", (float)rec->oargs.farg[4]);
	if (rec->otype == MAT_TRANS) { // it's a translucent material
		applyMaterialVariable1f(context, material, "transm", (float)rec->oargs.farg[5]);
		applyMaterialVariable1f(context, material, "tspecu", (float)rec->oargs.farg[6]);
	}

	return material;
}

static RTmaterial createGlassMaterial(const RTcontext context, OBJREC* rec)
{
	RTmaterial material;

	/* Create our hit programs to be shared among all materials */
	if ( !radiance_glass_closest_hit_program || !shadow_glass_closest_hit_program ) {
		ptxFile( path_to_ptx, "radiance_glass" );
		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "closest_hit_radiance", &radiance_glass_closest_hit_program ) );
		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "closest_hit_shadow", &shadow_glass_closest_hit_program ) );
	}

	RT_CHECK_ERROR( rtMaterialCreate( context, &material ) );
	RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( material, RADIANCE_RAY, radiance_glass_closest_hit_program ) );
	RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( material, SHADOW_RAY, shadow_glass_closest_hit_program ) );

	if ( calc_ambient ) {
		if ( !ambient_glass_any_hit_program ) {
			ptxFile( path_to_ptx, "ambient_normal" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "any_hit_ambient_glass", &ambient_glass_any_hit_program ) );
		}
		RT_CHECK_ERROR( rtMaterialSetAnyHitProgram( material, AMBIENT_RECORD_RAY, ambient_glass_any_hit_program ) );

		if ( !point_cloud_any_hit_program ) {
			ptxFile( path_to_ptx, "point_cloud_normal" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "any_hit_point_cloud_glass", &point_cloud_any_hit_program ) );
		}
		RT_CHECK_ERROR( rtMaterialSetAnyHitProgram( material, POINT_CLOUD_RAY, point_cloud_any_hit_program ) );
	}

	/* Set variables to be consumed by material for this geometry instance */
#ifdef HIT_TYPE
	applyMaterialVariable1ui( context, material, "type", rec->otype );
#endif
	applyMaterialVariable3f(context, material, "color", (float)rec->oargs.farg[0], (float)rec->oargs.farg[1], (float)rec->oargs.farg[2]);
	if (rec->oargs.nfargs > 3)
		applyMaterialVariable1f(context, material, "r_index", (float)rec->oargs.farg[3]);

	return material;
}

static RTmaterial createLightMaterial( const RTcontext context, OBJREC* rec )
{
	RTmaterial material;
	OBJREC* mat;

	/* Create our hit programs to be shared among all materials */
	if ( !radiance_light_closest_hit_program || !shadow_light_closest_hit_program ) {
		ptxFile( path_to_ptx, "radiance_light" );
		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "closest_hit_radiance", &radiance_light_closest_hit_program ) );
		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "closest_hit_shadow", &shadow_light_closest_hit_program ) );
	}

	RT_CHECK_ERROR( rtMaterialCreate( context, &material ) );
	if ( do_irrad )
		RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( material, PRIMARY_RAY, radiance_light_closest_hit_program ) );
	RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( material, RADIANCE_RAY, radiance_light_closest_hit_program ) );
	RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( material, SHADOW_RAY, shadow_light_closest_hit_program ) );

	if ( calc_ambient ) {
		if ( !point_cloud_any_hit_program ) {
			ptxFile( path_to_ptx, "point_cloud_normal" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "any_hit_point_cloud_glass", &point_cloud_any_hit_program ) );
		}
		RT_CHECK_ERROR( rtMaterialSetAnyHitProgram( material, POINT_CLOUD_RAY, point_cloud_any_hit_program ) );
	}

	/* Set variables to be consumed by material for this geometry instance */
#ifdef HIT_TYPE
	applyMaterialVariable1ui( context, material, "type", rec->otype );
#endif
	applyMaterialVariable3f(context, material, "color", (float)rec->oargs.farg[0], (float)rec->oargs.farg[1], (float)rec->oargs.farg[2]);
	if (rec->otype == MAT_GLOW)
		applyMaterialVariable1f(context, material, "maxrad", (float)rec->oargs.farg[3]);
	else if (rec->otype == MAT_SPOT) {
		SPOT* spot = makespot(rec);
		applyMaterialVariable1f( context, material, "siz", spot->siz );
		applyMaterialVariable1f( context, material, "flen", spot->flen );
		applyMaterialVariable3f(context, material, "aim", (float)spot->aim[0], (float)spot->aim[1], (float)spot->aim[2]);
		free(spot);
		rec->os = NULL;
	}

	/* Check for a parent function. */
	if ((mat = findFunction(rec))) // TODO can there be multiple parent functions?
		applyMaterialVariable1i( context, material, "function", buffer_entry_index[objndx(mat)] );
	else
		applyMaterialVariable1i( context, material, "function", RT_PROGRAM_ID_NULL );

	return material;
}

static DistantLight createDistantLight(const RTcontext context, OBJREC* rec, OBJREC* parent)
{
	SRCREC source;
	OBJREC* material;
	DistantLight light;

	ssetsrc(&source, rec);
	material = findmaterial(parent);
	array2cuda3(light.color, material->oargs.farg); // TODO these are given in RGB radiance value (watts/steradian/m2)
	array2cuda3(light.pos, source.sloc);
	light.solid_angle = source.ss2;
	light.casts_shadow = material->otype != MAT_GLOW; // Glow cannot cast shadow infinitely far away

	/* Check for a parent function. */
	if ((material = findFunction(parent))) // TODO can there be multiple parent functions?
		light.function = buffer_entry_index[objndx(material)];
	else
		light.function = RT_PROGRAM_ID_NULL;

	return light;
}

static OBJREC* findFunction(OBJREC *o)
{
	while (!hasfunc(o->otype)) {
		if (o->otype == MOD_ALIAS && o->oargs.nsargs) {
			OBJECT  aobj;
			OBJREC  *ao;
			aobj = lastmod(objndx(o), o->oargs.sarg[0]);
			if (aobj < 0)
				objerror(o, USER, "bad reference");
			ao = objptr(aobj);
			if (hasfunc(ao->otype))
				return(ao);
			if (ao->otype == MOD_ALIAS) {
				o = ao;
				continue;
			}
		}
		if (o->omod == OVOID)
			return(NULL);
		o = objptr(o->omod);
	}
	return(o);		/* mixtures will return NULL */
}

static int createFunction(const RTcontext context, OBJREC* rec)
{
	RTprogram program;
	int program_id = RT_PROGRAM_ID_NULL;
	XF bxp;

	/* Get transform */
	if ( !createTransform( NULL, &bxp, rec ) )
		objerror(rec, USER, "bad transform");

	if ( rec->oargs.nsargs >= 2 ) {
		float transform[9] = {
			(float)bxp.xfm[0][0], (float)bxp.xfm[1][0], (float)bxp.xfm[2][0],
			(float)bxp.xfm[0][1], (float)bxp.xfm[1][1], (float)bxp.xfm[2][1],
			(float)bxp.xfm[0][2], (float)bxp.xfm[1][2], (float)bxp.xfm[2][2]
		};
		if ( !strcmp(rec->oargs.sarg[0], "skybr") && !strcmp(rec->oargs.sarg[1], "skybright.cal") ) {
			ptxFile( path_to_ptx, "skybright" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "sky_bright", &program ) );
			applyProgramVariable1ui(context, program, "type", (unsigned int)rec->oargs.farg[0]);
			applyProgramVariable1f(context, program, "zenith", (float)rec->oargs.farg[1]);
			applyProgramVariable1f(context, program, "ground", (float)rec->oargs.farg[2]);
			applyProgramVariable1f(context, program, "factor", (float)rec->oargs.farg[3]);
			applyProgramVariable3f(context, program, "sun", (float)rec->oargs.farg[4], (float)rec->oargs.farg[5], (float)rec->oargs.farg[6]);
			applyProgramVariable( context, program, "transform", sizeof(transform), transform );
		} else if ( !strcmp(rec->oargs.sarg[0], "skybright") && !strcmp(rec->oargs.sarg[1], "perezlum.cal") ) {
			float coef[5] = { (float)rec->oargs.farg[2], (float)rec->oargs.farg[3], (float)rec->oargs.farg[4], (float)rec->oargs.farg[5], (float)rec->oargs.farg[6] };
			ptxFile( path_to_ptx, "perezlum" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "perez_lum", &program ) );
			applyProgramVariable1f(context, program, "diffuse", (float)rec->oargs.farg[0]);
			applyProgramVariable1f(context, program, "ground", (float)rec->oargs.farg[1]);
			applyProgramVariable( context, program, "coef", sizeof(coef), coef );
			applyProgramVariable3f(context, program, "sun", (float)rec->oargs.farg[7], (float)rec->oargs.farg[8], (float)rec->oargs.farg[9]);
			applyProgramVariable( context, program, "transform", sizeof(transform), transform );
		} else if (!strcmp(rec->oargs.sarg[0], "skybright") && !strcmp(rec->oargs.sarg[1], "isotrop_sky.cal")) {
			/* Isotropic sky from daysim installation */
			ptxFile(path_to_ptx, "isotropsky");
			RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "isotrop_sky", &program));
			applyProgramVariable1f(context, program, "skybright", (float)rec->oargs.farg[0]);
			applyProgramVariable(context, program, "transform", sizeof(transform), transform);
		} else {
			printObject(rec);
			return RT_PROGRAM_ID_NULL;
		}
	} else {
		printObject(rec);
		return RT_PROGRAM_ID_NULL;
	}

	RT_CHECK_ERROR( rtProgramGetId( program, &program_id ) );
	//applyContextVariable1i( context, "func", program_id );
	//applyContextObject( context, "func", program );
	return program_id;
}

static int createTexture(const RTcontext context, OBJREC* rec)
{
	RTprogram        program;
	RTtexturesampler tex_sampler;
	RTformat         tex_format;
	RTbuffer         tex_buffer;
	float*           tex_buffer_data;

	int tex_id = RT_TEXTURE_ID_NULL;
	int i, entries;

	DATARRAY *dp;
	//COLOR color;
	XF bxp;

	/* Load texture data */
	dp = getdata(rec->oargs.sarg[1]); //TODO for Texdata and Colordata, use 3, 4, 5

	/* Get transform */
	if ( !createTransform( NULL, &bxp, rec ) )
		objerror(rec, USER, "bad transform");

	/* Create buffer for texture data */
	if (dp->type == DATATY) // floats
		tex_format = RT_FORMAT_FLOAT;
	else // colors
		tex_format = RT_FORMAT_FLOAT3;
	switch (dp->nd) {
	case 1:
		createBuffer1D( context, RT_BUFFER_INPUT, tex_format, dp->dim[0].ne, &tex_buffer );
		entries = dp->dim[0].ne;
		break;
	case 2:
		createBuffer2D( context, RT_BUFFER_INPUT, tex_format, dp->dim[1].ne, dp->dim[0].ne, &tex_buffer );
		entries = dp->dim[0].ne * dp->dim[1].ne;
		break;
	case 3:
		createBuffer3D( context, RT_BUFFER_INPUT, tex_format, dp->dim[2].ne, dp->dim[1].ne, dp->dim[0].ne, &tex_buffer );
		entries = dp->dim[0].ne * dp->dim[1].ne * dp->dim[2].ne;
		break;
	default:
		sprintf(errmsg, "bad number of dimensions %u", dp->nd);
		objerror(rec, USER, errmsg);
		return RT_PROGRAM_ID_NULL;
	}

	/* Populate buffer with texture data */
	RT_CHECK_ERROR( rtBufferMap( tex_buffer, (void**)&tex_buffer_data ) );
	if (dp->type == DATATY) // floats
		memcpy(tex_buffer_data, dp->arr.d, entries * sizeof(float));
	else // colors
		for (i = 0u; i < entries; i++) {
			colr_color(tex_buffer_data, dp->arr.c[i]);
			//colr_color(color, dp->arr.c[i]);
			//copycolor(tex_buffer_data, color);
			tex_buffer_data += 3;
		}
	RT_CHECK_ERROR( rtBufferUnmap( tex_buffer ) );

	/* Create texture sampler */
	RT_CHECK_ERROR( rtTextureSamplerCreate( context, &tex_sampler ) );
	for (i = 0u; i < dp->nd; i++) {
		RT_CHECK_ERROR( rtTextureSamplerSetWrapMode( tex_sampler, i, RT_WRAP_CLAMP_TO_EDGE ) );
	}
	RT_CHECK_ERROR( rtTextureSamplerSetFilteringModes( tex_sampler, RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE ) );
	RT_CHECK_ERROR( rtTextureSamplerSetIndexingMode( tex_sampler, RT_TEXTURE_INDEX_NORMALIZED_COORDINATES ) );
	RT_CHECK_ERROR( rtTextureSamplerSetReadMode( tex_sampler, RT_TEXTURE_READ_ELEMENT_TYPE ) );
	RT_CHECK_ERROR( rtTextureSamplerSetMaxAnisotropy( tex_sampler, 0.0f ) );
	RT_CHECK_ERROR( rtTextureSamplerSetMipLevelCount( tex_sampler, 1u ) ); // Currently only one mipmap level supported
	RT_CHECK_ERROR( rtTextureSamplerSetArraySize( tex_sampler, 1u ) ); // Currently only one element supported
	RT_CHECK_ERROR( rtTextureSamplerSetBuffer( tex_sampler, 0u, 0u, tex_buffer ) );
	RT_CHECK_ERROR( rtTextureSamplerGetId( tex_sampler, &tex_id ) );

	/* Create program to access texture sampler */
	if ( rec->oargs.nsargs >= 5 ) {
		if (!strcmp(rec->oargs.sarg[2], "source.cal")) {
			float transform[9] = {
				(float)bxp.xfm[0][0], (float)bxp.xfm[1][0], (float)bxp.xfm[2][0],
				(float)bxp.xfm[0][1], (float)bxp.xfm[1][1], (float)bxp.xfm[2][1],
				(float)bxp.xfm[0][2], (float)bxp.xfm[1][2], (float)bxp.xfm[2][2]
			};

			/* Check compatibility with existing implementation */
			if ((strcmp(rec->oargs.sarg[0], "corr") && strcmp(rec->oargs.sarg[0], "flatcorr") && strcmp(rec->oargs.sarg[0], "boxcorr") && strcmp(rec->oargs.sarg[0], "cylcorr")) || strcmp(rec->oargs.sarg[3], "src_phi") || strcmp(rec->oargs.sarg[4], "src_theta")) {
				printObject(rec);
				return RT_PROGRAM_ID_NULL;
			}

			ptxFile( path_to_ptx, "source" );
			RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, rec->oargs.sarg[0], &program));
			applyProgramVariable1i( context, program, "data", tex_id );
			applyProgramVariable1i( context, program, "type", dp->type == DATATY );
			applyProgramVariable3f( context, program, "org", dp->dim[dp->nd-1].org, dp->nd > 1 ? dp->dim[dp->nd-2].org : 0.0f, dp->nd > 2 ? dp->dim[dp->nd-3].org : 0.0f );
			applyProgramVariable3f( context, program, "siz", dp->dim[dp->nd-1].siz, dp->nd > 1 ? dp->dim[dp->nd-2].siz : 1.0f, dp->nd > 2 ? dp->dim[dp->nd-3].siz : 1.0f );
			applyProgramVariable3i(context, program, "ne", dp->dim[dp->nd - 1].ne, dp->nd > 1 ? dp->dim[dp->nd - 2].ne : 1, dp->nd > 2 ? dp->dim[dp->nd - 3].ne : 1);
			applyProgramVariable( context, program, "transform", sizeof(transform), transform );
			if (rec->oargs.nfargs > 0)
				applyProgramVariable1f(context, program, "multiplier", (float)rec->oargs.farg[0]); //TODO handle per-color channel multipliers
			if (rec->oargs.nfargs > 2)
				applyProgramVariable3f(context, program, "bounds", (float)rec->oargs.farg[1], (float)rec->oargs.farg[2], rec->oargs.nfargs > 3 ? (float)rec->oargs.farg[3] : 0.0f);
		} else {
			printObject(rec);
			return RT_PROGRAM_ID_NULL;
		}
	} else {
		printObject(rec);
		return RT_PROGRAM_ID_NULL;
	}
	RT_CHECK_ERROR( rtProgramGetId( program, &tex_id ) );

	return tex_id;
}

static int createTransform( XF* fxp, XF* bxp, const OBJREC* rec )
{
	/* Get transform - from getfunc in func.c */
	int i = 0;
	while (i < rec->oargs.nsargs && rec->oargs.sarg[i][0] != '-')
		i++;
	if (i == rec->oargs.nsargs)	{		/* no transform */
		if (fxp)
			*fxp = unitxf;
		if (bxp)
			*bxp = unitxf;
		return 1;
	}
	
	/* get transform */
	if (fxp && xf(fxp, rec->oargs.nsargs - i, rec->oargs.sarg + i) != rec->oargs.nsargs - i)
		return 0;
	if (bxp && invxf(bxp, rec->oargs.nsargs - i, rec->oargs.sarg + i) != rec->oargs.nsargs - i)
		return 0;

	if (fxp && fxp->sca < 0.0) 
		fxp->sca = -fxp->sca;
	if (bxp && bxp->sca < 0.0) 
		bxp->sca = -bxp->sca;

	return 2;
}

static void createAcceleration( const RTcontext context, const RTgeometryinstance instance, const unsigned int imm_irrad )
{
	RTgeometrygroup geometrygroup;
	RTacceleration  acceleration;

	/* Create a geometry group to hold the geometry instance.  This will be used as the top level group. */
	RT_CHECK_ERROR( rtGeometryGroupCreate( context, &geometrygroup ) );
	RT_CHECK_ERROR( rtGeometryGroupSetChildCount( geometrygroup, 1 ) );
	RT_CHECK_ERROR( rtGeometryGroupSetChild( geometrygroup, 0, instance ) );

	/* Set the geometry group as the top level object. */
	applyContextObject( context, "top_object", geometrygroup );
	if ( !use_ambient )
		applyContextObject( context, "top_ambient", geometrygroup ); // Need to define this because it is referred to by radiance_normal.cu
	if ( !imm_irrad )
		applyContextObject( context, "top_irrad", geometrygroup ); // Need to define this because it is referred to by sensor.cu, sensor_cloud_generator, and ambient_cloud_generator.cu

	/* create acceleration object for group and specify some build hints */
	RT_CHECK_ERROR( rtAccelerationCreate( context, &acceleration ) );
	RT_CHECK_ERROR( rtAccelerationSetBuilder( acceleration, "Sbvh" ) );
	RT_CHECK_ERROR( rtAccelerationSetTraverser( acceleration, "Bvh" ) );
	RT_CHECK_ERROR( rtAccelerationSetProperty( acceleration, "vertex_buffer_name", "vertex_buffer" ) ); // For Sbvh only
	RT_CHECK_ERROR( rtAccelerationSetProperty( acceleration, "index_buffer_name", "vindex_buffer" ) ); // For Sbvh only
	RT_CHECK_ERROR( rtGeometryGroupSetAcceleration( geometrygroup, acceleration ) );

	/* mark acceleration as dirty */
	RT_CHECK_ERROR( rtAccelerationMarkDirty( acceleration ) );
}

static void createIrradianceGeometry( const RTcontext context )
{
	RTgeometry         geometry;
	RTprogram          irradiance_intersection_program, irradiance_bounding_box_program;
	RTgeometryinstance instance;
	RTgeometrygroup    geometrygroup;
	RTacceleration     acceleration;

	/* Create the geometry reference for OptiX. */
	RT_CHECK_ERROR( rtGeometryCreate( context, &geometry ) );
	RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( geometry, 1 ) );

	ptxFile( path_to_ptx, "irradiance_intersect" );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "irradiance_bounds", &irradiance_bounding_box_program ) );
	RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( geometry, irradiance_bounding_box_program ) );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "irradiance_intersect", &irradiance_intersection_program ) );
	RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( geometry, irradiance_intersection_program ) );

	/* Create the geometry instance containing the geometry. */
	RT_CHECK_ERROR( rtGeometryInstanceCreate( context, &instance ) );
	RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( instance, geometry ) );

	/* Create a Lambertian material as the geometry instance's only material. */
	RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( instance, 1 ) );
	RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( instance, 0, createNormalMaterial( context, &Lamb ) ) );

	/* Create a geometry group to hold the geometry instance.  This will be used as the top level group. */
	RT_CHECK_ERROR( rtGeometryGroupCreate( context, &geometrygroup ) );
	RT_CHECK_ERROR( rtGeometryGroupSetChildCount( geometrygroup, 1 ) );
	RT_CHECK_ERROR( rtGeometryGroupSetChild( geometrygroup, 0, instance ) );

	/* Set the geometry group as the top level object. */
	applyContextObject( context, "top_irrad", geometrygroup );

	/* create acceleration object for group and specify some build hints */
	RT_CHECK_ERROR( rtAccelerationCreate( context, &acceleration ) );
	RT_CHECK_ERROR( rtAccelerationSetBuilder( acceleration, "NoAccel" ) );
	RT_CHECK_ERROR( rtAccelerationSetTraverser( acceleration, "NoAccel" ) );
	RT_CHECK_ERROR( rtGeometryGroupSetAcceleration( geometrygroup, acceleration ) );

	/* mark acceleration as dirty */
	RT_CHECK_ERROR( rtAccelerationMarkDirty( acceleration ) );
}

static void printObject(OBJREC* rec)
{
	int i;

	objerror(rec, WARNING, "no GPU support");
	mprintf(" %s(%i) %s(%i) %s(%i)\n %i", rec->omod > OVOID ? objptr(rec->omod)->oname : VOIDID, rec->omod, ofun[rec->otype].funame, rec->otype, rec->oname, objndx(rec), rec->oargs.nsargs);
	for (i = 0; i < rec->oargs.nsargs; i++)
		mprintf(" %s", rec->oargs.sarg[i]);
	mprintf("\n %i", rec->oargs.nfargs);
	for (i = 0; i < rec->oargs.nfargs; i++)
		mprintf(" %g", rec->oargs.farg[i]);
	if (rec->os)
		mprintf("\n Object structure: %s", rec->os);
	mprintf("\n");
}

static void getRay( RayData* data, const RAY* ray )
{
	array2cuda3( data->origin, ray->rorg );
	array2cuda3( data->dir, ray->rdir );
	array2cuda3( data->val, ray->rcol );
	//array2cuda3( data->contrib, ray->rcoef );
	//array2cuda3( data->extinction, ray->cext );
	//array2cuda3( data->hit, ray->rop );
	//array2cuda3( data->pnorm, ray->pert );
	//array2cuda3( data->normal, ray->ron );

	//array2cuda2(data->tex, ray->uv);
	data->max = (float)ray->rmax;
	data->weight = ray->rweight;
	data->length = (float)ray->rt;
	//data->t = ray->rot;
}

static void setRay( RAY* ray, const RayData* data )
{
	cuda2array3( ray->rorg, data->origin );
	cuda2array3( ray->rdir, data->dir );
	cuda2array3( ray->rcol, data->val );
	//cuda2array3( ray->rcoef, data->contrib );
	//cuda2array3( ray->cext, data->extinction );
	//cuda2array3( ray->rop, data->hit );
	//cuda2array3( ray->pert, data->pnorm );
	//cuda2array3( ray->ron, data->normal );

	//cuda2array2(ray->uv, data->tex);
	ray->rmax = data->max;
	ray->rweight = data->weight;
	ray->rt = data->length;
	//ray->rot = data->t; //TODO setting this requires that the ray has non-null ro.
}

#endif /* ACCELERAD */
