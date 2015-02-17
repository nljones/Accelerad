/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <standard.h> /* TODO just get the includes that are required? */
#include <object.h>
#include <otypes.h>
#include "ray.h"
#include "source.h"
#include "ambient.h"
#include <face.h>
#include <mesh.h>
#include "data.h"

#include "optix_radiance.h"

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
#define EXPECTED_FUNCTIONS	8

void renderOptix( const VIEW* view, const int width, const int height, const double alarm, COLOR* colors, float* depths );
void computeOptix( const int width, const int height, const double alarm, RAY* rays );
void endOptix();

static void checkDevices();
static void createContext( RTcontext* context, const int width, const int height, const double alarm );
static void setupKernel( const RTcontext context, const VIEW* view, const int width, const int height );
static void applyRadianceSettings( const RTcontext context );
static void createCamera( const RTcontext context, const VIEW* view );
static void updateCamera( const RTcontext context, const VIEW* view );
static void createGeometryInstance( const RTcontext context, RTgeometryinstance* instance );
static void addRadianceObject(const RTcontext context, const OBJREC* rec, const OBJREC* parent, const int index);
static void createFace(const OBJREC* rec, const OBJREC* parent);
static __inline void createTriangle(const OBJREC *material, const int a, const int b, const int c);
#ifdef TRIANGULATE
static int createTriangles(const FACE *face, const OBJREC* material);
static int addTriangle(const Vert2_list *tp, int a, int b, int c);
#endif /* TRIANGULATE */
static void createMesh(const OBJREC* rec, const OBJREC* parent);
static RTmaterial createNormalMaterial( const RTcontext context, const OBJREC* rec );
static RTmaterial createGlassMaterial( const RTcontext context, const OBJREC* rec );
static RTmaterial createLightMaterial( const RTcontext context, OBJREC* rec );
static DistantLight createDistantLight( const RTcontext context, OBJREC* rec );
static int createFunction( const RTcontext context, const OBJREC* rec, const OBJREC* parent );
static int createTexture( const RTcontext context, const OBJREC* rec );
static int createTransform( XF* bxp, const OBJREC* rec );
static void createAcceleration( const RTcontext context, const RTgeometryinstance instance );
static void createIrradianceGeometry( const RTcontext context );
static void printObect( const OBJREC* rec );
static void getRay( RayData* data, const RAY* ray );
static void setRay( RAY* ray, const RayData* data );

#define nancolor(c)	c[0]!=c[0]||c[1]!=c[1]||c[2]!=c[2]


/* from rpict.c */
extern double  dstrpix;			/* square pixel distribution */

/* from rtmain.c */
extern int  imm_irrad;			/* compute immediate irradiance? */

/* from ambient.c */
extern double  avsum;		/* computed ambient value sum (log) */
extern unsigned int  navsum;	/* number of values in avsum */
extern unsigned int  nambvals;	/* total number of indirect values */

/* from func.c */
extern XF  unitxf;

/* Ambient calculation flags */
static unsigned int use_ambient = 0u;
static unsigned int calc_ambient = 0u;

/* Handles to objects used repeatedly in animation */
static RTcontext context_handle = NULL;
static RTbuffer buffer_handle = NULL;
static RTvariable camera_type, camera_eye, camera_u, camera_v, camera_w, camera_fov, camera_shift, camera_clip;

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
#ifdef KMEANS_IC
static RTprogram point_cloud_closest_hit_program, point_cloud_any_hit_program;
#endif

#ifdef PREFER_TCC
#define MAX_TCCS	16				/* Maximum number of Tesla devices to discover */
static unsigned int tcc_count = 0u;	/* number of discovered Tesla devices */
static int tcc[MAX_TCCS];			/* indices of discovered Tesla devices */
#endif


/**
 * Setup and run the OptiX kernel similar to RPICT.
 * This may be called repeatedly in order to render an animation.
 * After the last call, endOptix() should be called.
 */
void renderOptix(const VIEW* view, const int width, const int height, const double alarm, COLOR* colors, float* depths)
{
	/* Primary RTAPI objects */
	RTcontext           context;
	RTbuffer            output_buffer;

	/* Parameters */
	unsigned int i, size;
	float* data;

	/* Set the size */
	size = width * height;

	if ( context_handle == NULL ) {
		/* Setup state */
		checkDevices();
		createContext( &context, width, height, alarm );

		/* Render result buffer */
		createBuffer2D( context, RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width, height, &output_buffer );
		applyContextObject( context, "output_buffer", output_buffer );

		/* Save handles to objects used in animations */
		context_handle = context;
		buffer_handle = output_buffer;

		setupKernel( context, view, width, height );
	} else {
		/* Retrieve handles for previously created objects */
		context = context_handle;
		output_buffer = buffer_handle;
	}

	/* Run the OptiX kernel */
	updateCamera( context, view );
	runKernel2D( context, RADIANCE_ENTRY, width, height );

	RT_CHECK_ERROR( rtBufferMap( output_buffer, (void**)&data ) );
	RT_CHECK_ERROR( rtBufferUnmap( output_buffer ) );

	/* Copy the results to allocated memory. */
	for (i = 0u; i < size; i++) {
		if ( nancolor( data ) ) {
			setcolor( colors[i], 0.0f, 0.0f, 0.0f );
			depths[i] = 0.0f;
		} else {
			copycolor( colors[i], data );
			depths[i] = data[3];
		}
		data += 4;
	}

#ifdef PRINT_OPTIX
	/* Exit if message printing is on */
	exit(1);
#endif

	/* Clean up */
	//RT_CHECK_ERROR( rtContextDestroy( context ) );
}

/**
 * Setup and run the OptiX kernel similar to RTRACE.
 */
void computeOptix(const int width, const int height, const double alarm, RAY* rays)
{
	/* Primary RTAPI objects */
	RTcontext           context;
	RTbuffer            ray_buffer;

	/* Parameters */
	unsigned int size, i;
	RayData* data;

	/* Set the size */
	size = width * height;
	if ( !size )
		error(USER, "Size is zero or not set. Both -x and -y must be positive.");

	/* Setup state */
	checkDevices();
	createContext( &context, width, height, alarm );
	
	/* Input/output buffer */
	createCustomBuffer2D( context, RT_BUFFER_INPUT_OUTPUT, sizeof(RayData), width, height, &ray_buffer );
	//createCustomBuffer2D( context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, sizeof(RayData), width, height, &ray_buffer );
	RT_CHECK_ERROR( rtBufferMap( ray_buffer, (void**)&data ) );
	for (i = 0u; i < size; i++) {
		getRay( data++, &rays[i] );
	}
	RT_CHECK_ERROR( rtBufferUnmap( ray_buffer ) );
	applyContextObject( context, "ray_buffer", ray_buffer );

	setupKernel( context, NULL, width, height );

	/* Run the OptiX kernel */
	runKernel2D( context, RADIANCE_ENTRY, width, height );

	RT_CHECK_ERROR( rtBufferMap( ray_buffer, (void**)&data ) );
	RT_CHECK_ERROR( rtBufferUnmap( ray_buffer ) );

	/* Copy the results to allocated memory. */
	for (i = 0u; i < size; i++) {
		setRay( &rays[i], data++ );
	}

	/* Clean up */
	RT_CHECK_ERROR( rtContextDestroy( context ) );
}

/**
 * Destroy the OptiX context if one has been created.
 * This should be called after the last call to renderOptix().
 */
void endOptix()
{
	/* Primary RTAPI objects */
	RTcontext context;

	if ( context_handle == NULL ) return;

	context = context_handle;
	RT_CHECK_ERROR( rtContextDestroy( context ) );
	context_handle = NULL;
	buffer_handle = NULL;
	camera_type = camera_eye = camera_u = camera_v = camera_w = camera_fov = camera_shift = camera_clip = NULL;
}

/**
 * Check for supported GPU devices.
 * This should be called prior to any work with OptiX.
 */
static void checkDevices()
{
	unsigned int version, device_count, useable_device_count;
	unsigned int i;
	unsigned int multiprocessor_count, threads_per_block, clock_rate, texture_count, timeout_enabled, tcc_driver, cuda_device;
	unsigned int compute_capability[2];
	char device_name[128];
	RTsize memory_size;

	rtGetVersion( &version );
	rtDeviceGetDeviceCount( &device_count );
	if (device_count) {
		fprintf(stderr, "OptiX %i found %i GPU device%s:\n", version, device_count, device_count > 1 ? "s" : "");
	}
	useable_device_count = device_count;

#ifdef PREFER_TCC
	tcc_count = 0u;
#endif

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
		fprintf(stderr, "Device %u: %s with %u multiprocessors, %u threads per block, %u kHz, %u bytes global memory, %u hardware textures, compute capability %u.%u, timeout %sabled, Tesla compute cluster driver %sabled, cuda device %u.\n",
			i, device_name, multiprocessor_count, threads_per_block, clock_rate, memory_size, texture_count, compute_capability[0], compute_capability[1], timeout_enabled ? "en" : "dis", tcc_driver ? "en" : "dis", cuda_device);

		if (compute_capability[0] < 2u) {
			useable_device_count--;
			sprintf(errmsg, "Device %u has insufficient compute capability for OptiX callable programs.", i);
			error(WARNING, errmsg);
			continue;
		}
#ifdef PREFER_TCC
		if (tcc_driver && (tcc_count < MAX_TCCS))
			tcc[tcc_count++] = i;
#endif
	}

	if (!useable_device_count) {
		sprintf(errmsg, "A supported GPU could not be found for OptiX %i.", version);
		error(SYSTEM, errmsg);
	}

	fprintf(stderr, "\n");
}

static void createContext( RTcontext* context, const int width, const int height, const double alarm )
{
	//RTbuffer seed_buffer;

	//unsigned int* seeds;
	//unsigned int i;
	unsigned int ray_type_count, entry_point_count;

	/* Check if irradiance cache is used */
	use_ambient = ambacc > FTINY && ambounce > 0;
	calc_ambient = use_ambient && nambvals == 0u;// && (ambfile == NULL || !ambfile[0]); // TODO Should really look at ambfp in ambinet.c to check that file is readable

	ray_type_count = 3u; /* shadow, radiance, and primary radiance */
	entry_point_count = 1u; /* Generate radiance data */

	if ( use_ambient ) {
		ray_type_count++; /* ambient ray */
		if ( ambdiv > AMB_ROW_SIZE * AMB_ROW_SIZE ) {
			sprintf(errmsg, "number of ambient divisions cannot be greater than %i (currently %i).", AMB_ROW_SIZE * AMB_ROW_SIZE, ambdiv);
			error(USER, errmsg);
		}
		if ( optix_stack_size < 124 * AMB_ROW_SIZE + 144 ) { // Based on memory requirements for samp_hemi() in ambient_normal.cu
			sprintf(errmsg, "GPU stack size must be greater than %i (currently %i).", 124 * AMB_ROW_SIZE + 144, optix_stack_size);
			error(USER, errmsg);
		}
	}
	if ( calc_ambient ) {
		ray_type_count += RAY_TYPE_COUNT;
		entry_point_count += ENTRY_POINT_COUNT;
	}

	/* Setup context */
	RT_CHECK_ERROR2( rtContextCreate( context ) );
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

#ifdef PREFER_TCC
	/* Devices to use */
	if ( tcc_count )
		RT_CHECK_ERROR2( rtContextSetDevices( *context, tcc_count, tcc ) );
#endif

#ifdef TIMEOUT_CALLBACK
	if (alarm > 0)
		RT_CHECK_ERROR2( rtContextSetTimeoutCallback( *context, timeoutCallback, alarm ) );
#endif

#ifdef DEBUG_OPTIX
	/* Enable exception checking */
	RT_CHECK_ERROR2( rtContextSetExceptionEnabled( *context, RT_EXCEPTION_ALL, 1 ) );
#endif

#ifdef PRINT_OPTIX
	/* Enable message pringing */
	RT_CHECK_ERROR2( rtContextSetPrintEnabled( *context, 1 ) );
	RT_CHECK_ERROR2( rtContextSetPrintBufferSize( *context, 512 * width * height ) );
	//RT_CHECK_ERROR2( rtContextSetPrintLaunchIndex( *context, width / 2, height / 2, -1 ) );
#endif

#ifdef REPORT_GPU_STATE
	/* Print device and context attributes */
	printCUDAProp();
	printContextInfo( *context );
#endif
}

static void setupKernel( const RTcontext context, const VIEW* view, const int width, const int height )
{
	/* Primary RTAPI objects */
	RTgeometryinstance  instance;

	/* Setup state */
	applyRadianceSettings( context );
	createCamera( context, view );
	createGeometryInstance( context, &instance );
	createAcceleration( context, instance );
	if ( imm_irrad )
		createIrradianceGeometry( context );

	/* Set up irradiance cache of ambient values */
	if ( use_ambient ) { // Don't bother with ambient records if -aa is set to zero
		if ( calc_ambient ) // Run pre-process only if no ambient file is available
			createAmbientRecords( context, view, width, height ); // implementation depends on current settings
		else
			setupAmbientCache( context, 0u ); // only need level 0 for final gather
	}
}

static void applyRadianceSettings( const RTcontext context )
{
	/* Define ray types */
	applyContextVariable1ui( context, "radiance_primary_ray_type", PRIMARY_RAY );
	applyContextVariable1ui( context, "radiance_ray_type", RADIANCE_RAY );
	applyContextVariable1ui( context, "shadow_ray_type", SHADOW_RAY );

	/* Set hard coded parameters */
	applyContextVariable3f( context, "CIE_rgbf", CIE_rf, CIE_gf, CIE_bf ); // from color.h

	/* Set direct parameters */
	applyContextVariable1f( context, "dstrsrc", dstrsrc ); // -dj
	applyContextVariable1f( context, "srcsizerat", srcsizerat ); // -ds
	//applyContextVariable1f( context, "shadthresh", shadthresh ); // -dt
	//applyContextVariable1f( context, "shadcert", shadcert ); // -dc
	//applyContextVariable1i( context, "directrelay", directrelay ); // -dr
	//applyContextVariable1i( context, "vspretest", vspretest ); // -dp
	applyContextVariable1i( context, "directvis", directvis ); // -dv

	/* Set specular parameters */
	applyContextVariable1f( context, "specthresh", specthresh ); // -st
	applyContextVariable1f( context, "specjitter", specjitter ); // -ss

	/* Set ambient parameters */
	applyContextVariable3f( context, "ambval", ambval[0], ambval[1], ambval[2] ); // -av
	applyContextVariable1i( context, "ambvwt", ambvwt ); // -aw, zero by default
	applyContextVariable1i( context, "ambounce", ambounce ); // -ab
	//applyContextVariable1i( context, "ambres", ambres ); // -ar
	applyContextVariable1f( context, "ambacc", ambacc ); // -aa
	applyContextVariable1i( context, "ambdiv", ambdiv ); // -ad
	applyContextVariable1i( context, "ambssamp", ambssamp ); // -as
	applyContextVariable1f( context, "maxarad", maxarad ); // maximum ambient radius from ambient.h, based on armbres
	applyContextVariable1f( context, "minarad", minarad ); // minimum ambient radius from ambient.h, based on armbres
	applyContextVariable1f( context, "avsum", avsum ); // computed ambient value sum (log) from ambient.c
	applyContextVariable1ui( context, "navsum", navsum ); // number of values in avsum from ambient.c

	/* Set medium parameters */
	//applyContextVariable3f( context, "cextinction", cextinction[0], cextinction[1], cextinction[2] ); // -me
	//applyContextVariable3f( context, "salbedo", salbedo[0], salbedo[1], salbedo[2] ); // -ma
	//applyContextVariable1f( context, "seccg", seccg ); // -mg
	//applyContextVariable1f( context, "ssampdist", ssampdist ); // -ms

	/* Set ray limitting parameters */
	applyContextVariable1f( context, "minweight", minweight ); // -lw, from ray.h
	applyContextVariable1i( context, "maxdepth", maxdepth ); // -lr, from ray.h, negative values indicate Russian roulette
	applyContextVariable1ui( context, "imm_irrad", imm_irrad ); // -I
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
		//applyProgramVariable1ui( context, ray_gen_program, "camera", view->type ); // -vt
		//applyProgramVariable3f( context, ray_gen_program, "eye", view->vp[0], view->vp[1], view->vp[2] ); // -vp
		//applyProgramVariable3f( context, ray_gen_program, "U", view->hvec[0], view->hvec[1], view->hvec[2] );
		//applyProgramVariable3f( context, ray_gen_program, "V", view->vvec[0], view->vvec[1], view->vvec[2] );
		//applyProgramVariable3f( context, ray_gen_program, "W", view->vdir[0], view->vdir[1], view->vdir[2] ); // -vd
		//applyProgramVariable2f( context, ray_gen_program, "fov", view->horiz, view->vert ); // -vh, -vv
		//applyProgramVariable2f( context, ray_gen_program, "shift", view->hoff, view->voff ); // -vs, -vl
		//applyProgramVariable2f( context, ray_gen_program, "clip", view->vfore, view->vaft ); // -vo, -va
		RT_CHECK_ERROR( rtProgramDeclareVariable( ray_gen_program, "camera", &camera_type ) ); // -vt
		RT_CHECK_ERROR( rtProgramDeclareVariable( ray_gen_program, "eye", &camera_eye ) ); // -vp
		RT_CHECK_ERROR( rtProgramDeclareVariable( ray_gen_program, "U", &camera_u ) );
		RT_CHECK_ERROR( rtProgramDeclareVariable( ray_gen_program, "V", &camera_v ) );
		RT_CHECK_ERROR( rtProgramDeclareVariable( ray_gen_program, "W", &camera_w ) ); // -vd
		RT_CHECK_ERROR( rtProgramDeclareVariable( ray_gen_program, "fov", &camera_fov ) ); // -vh, -vv
		RT_CHECK_ERROR( rtProgramDeclareVariable( ray_gen_program, "shift", &camera_shift ) ); // -vs, -vl
		RT_CHECK_ERROR( rtProgramDeclareVariable( ray_gen_program, "clip", &camera_clip ) ); // -vo, -va
		applyProgramVariable1f( context, ray_gen_program, "dstrpix", dstrpix ); // -pj
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
	if ( camera_type == NULL ) return; // Should really test all, but we'll assume that all are set together.

	RT_CHECK_ERROR( rtVariableSet1ui( camera_type, view->type ) ); // -vt
	RT_CHECK_ERROR( rtVariableSet3f( camera_eye, view->vp[0], view->vp[1], view->vp[2] ) ); // -vp
	RT_CHECK_ERROR( rtVariableSet3f( camera_u, view->hvec[0], view->hvec[1], view->hvec[2] ) );
	RT_CHECK_ERROR( rtVariableSet3f( camera_v, view->vvec[0], view->vvec[1], view->vvec[2] ) );
	RT_CHECK_ERROR( rtVariableSet3f( camera_w, view->vdir[0], view->vdir[1], view->vdir[2] ) ); // -vd
	RT_CHECK_ERROR( rtVariableSet2f( camera_fov, view->horiz, view->vert ) ); // -vh, -vv
	RT_CHECK_ERROR( rtVariableSet2f( camera_shift, view->hoff, view->voff ) ); // -vs, -vl
	RT_CHECK_ERROR( rtVariableSet2f( camera_clip, view->vfore, view->vaft ) ); // -vo, -va
}

static void createGeometryInstance( const RTcontext context, RTgeometryinstance* instance )
{
	RTgeometry mesh;
	RTprogram  mesh_intersection_program;
	RTprogram  mesh_bounding_box_program;
	RTbuffer   buffer;

	int i;
	OBJREC* rec, *parent;

	/* Timers */
	clock_t geometry_start_clock, geometry_end_clock;

	/* This array gives the OptiX buffer index of each rad file object.
	 * The OptiX buffer refered to depends on the type of object.
	 * If the object does not have an index in an OptiX buffer, -1 is given. */
	buffer_entry_index = (int *)malloc(sizeof(int) * nobjects);
	if (buffer_entry_index == NULL)
		error(SYSTEM, "out of memory in createGeometryInstance");

	geometry_start_clock = clock();

	/* Create buffers for storing geometry information. */
	vertices = (FloatArray *)malloc(sizeof(FloatArray));
	initArrayf(vertices, EXPECTED_VERTICES * 3);

	normals = (FloatArray *)malloc(sizeof(FloatArray));
	initArrayf(normals, EXPECTED_VERTICES * 3);

	tex_coords = (FloatArray *)malloc(sizeof(FloatArray));
	initArrayf(tex_coords, EXPECTED_VERTICES * 2);

	vertex_indices = (IntArray *)malloc(sizeof(IntArray));
	initArrayi(vertex_indices, EXPECTED_TRIANGLES * 3);

	traingles = (IntArray *)malloc(sizeof(IntArray));
	initArrayi(traingles, EXPECTED_TRIANGLES);

	materials = (MaterialArray *)malloc(sizeof(MaterialArray));
	initArraym(materials, EXPECTED_MATERIALS);

	alt_materials = (IntArray *)malloc(sizeof(IntArray));
	initArrayi(alt_materials, EXPECTED_MATERIALS * 2);

	/* Create buffers for storing lighting information. */
#ifdef LIGHTS
	lights = (IntArray *)malloc(sizeof(IntArray));
	initArrayi(lights, EXPECTED_LIGHTS * 3);
#endif

	sources = (DistantLightArray *)malloc(sizeof(DistantLightArray));
	initArraydl(sources, EXPECTED_SOURCES);

	functions = (IntArray *)malloc(sizeof(IntArray));
	initArrayi(functions, EXPECTED_FUNCTIONS);

	vertex_index_0 = 0u;

	/* Material 0 is Lambertian. */
	if ( do_irrad ) {
		insertArray2i(alt_materials, materials->count, materials->count);
		insertArraym(materials, createNormalMaterial(context, &Lamb));
	}

	/* Get the scene geometry as a list of triangles. */
	//fprintf(stderr, "Num objects %i\n", nobjects);
	for (i = 0; i < nobjects; i++) {
		/* By default, no buffer entry is refered to. */
		buffer_entry_index[i] = -1;

		rec = objptr(i);
		if (rec->omod != OVOID)
			parent = objptr(rec->omod);
		else
			parent = NULL;

		addRadianceObject(context, rec, parent, i);
	}

	free( buffer_entry_index );

	/* Create the geometry reference for OptiX. */
	RT_CHECK_ERROR(rtGeometryCreate(context, &mesh));
	RT_CHECK_ERROR(rtGeometrySetPrimitiveCount(mesh, traingles->count));

	ptxFile(path_to_ptx, "triangle_mesh");
	RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "mesh_bounds", &mesh_bounding_box_program));
	RT_CHECK_ERROR(rtGeometrySetBoundingBoxProgram(mesh, mesh_bounding_box_program));
	RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "mesh_intersect", &mesh_intersection_program));
	RT_CHECK_ERROR(rtGeometrySetIntersectionProgram(mesh, mesh_intersection_program));
	applyProgramVariable1ui(context, mesh_intersection_program, "backvis", backvis); // -bv

	/* Create the geometry instance containing the geometry. */
	RT_CHECK_ERROR(rtGeometryInstanceCreate(context, instance));
	RT_CHECK_ERROR(rtGeometryInstanceSetGeometry(*instance, mesh));
	RT_CHECK_ERROR(rtGeometryInstanceSetMaterialCount(*instance, materials->count));

	/* Apply materials to the geometry instance. */
	for (i = 0; i < materials->count; i++)
		RT_CHECK_ERROR(rtGeometryInstanceSetMaterial(*instance, i, materials->array[i]));
	freeArraym(materials);
	free(materials);

	/* Unmap and apply the geometry buffers. */
	createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, vertices->count / 3, &buffer);
	copyToBufferi(context, buffer, vertices);
	//applyGeometryInstanceObject( context, *instance, "vertex_buffer", buffer );
	applyContextObject(context, "vertex_buffer", buffer);
	freeArrayf(vertices);
	free(vertices);

	createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, normals->count / 3, &buffer);
	copyToBufferi(context, buffer, normals);
	applyGeometryInstanceObject(context, *instance, "normal_buffer", buffer);
	freeArrayf(normals);
	free(normals);

	createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, tex_coords->count / 2, &buffer);
	copyToBufferi(context, buffer, tex_coords);
	applyGeometryInstanceObject(context, *instance, "texcoord_buffer", buffer);
	freeArrayf(tex_coords);
	free(tex_coords);

	createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_INT3, vertex_indices->count / 3, &buffer);
	copyToBufferi(context, buffer, vertex_indices);
	applyGeometryInstanceObject(context, *instance, "vindex_buffer", buffer);
	freeArrayi(vertex_indices);
	free(vertex_indices);

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
	createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_INT3, lights->count / 3, &buffer);
	copyToBufferi(context, buffer, lights);
	applyContextObject(context, "lindex_buffer", buffer);
	freeArrayi(lights);
	free(lights);
#endif

	createCustomBuffer1D(context, RT_BUFFER_INPUT, sizeof(DistantLight), sources->count, &buffer);
	copyToBufferdl(context, buffer, sources);
	applyContextObject(context, "lights", buffer);
	freeArraydl(sources);
	free(sources);

	createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, functions->count, &buffer);
	copyToBufferi(context, buffer, functions);
	applyContextObject(context, "functions", buffer);
	freeArrayi(functions);
	free(functions);

	geometry_end_clock = clock();
	fprintf(stderr, "Geometry build time: %u milliseconds.\n", (geometry_end_clock - geometry_start_clock) * 1000 / CLOCKS_PER_SEC);
}

static void addRadianceObject(const RTcontext context, const OBJREC* rec, const OBJREC* parent, const int index)
{
	switch (rec->otype) {
	case MAT_PLASTIC: // Plastic material
	case MAT_METAL: // Metal material
	case MAT_TRANS: // Translucent material
		buffer_entry_index[index] = insertArray2i(alt_materials, 0, materials->count);
		insertArraym(materials, createNormalMaterial(context, rec));
		break;
	case MAT_GLASS: // Glass material
	case MAT_DIELECTRIC: // Dielectric material TODO handle separately, see dialectric.c
		buffer_entry_index[index] = insertArray2i(alt_materials, -1, materials->count);
		insertArraym(materials, createGlassMaterial(context, rec));
		break;
	case MAT_LIGHT: // primary light source material, may modify a face or a source (solid angle)
	case MAT_ILLUM: // secondary light source material
	case MAT_GLOW: // Glow material
	case MAT_SPOT: // Spotlight material
		buffer_entry_index[index] = materials->count;
		if (rec->otype != MAT_ILLUM)
			insertArray2i(alt_materials, materials->count, materials->count);
		else if (rec->oargs.nsargs && strcmp(rec->oargs.sarg[0], VOIDID)) { /* modifies another material */
			//material = objptr( lastmod( objndx(rec), rec->oargs.sarg[0] ) );
			//alt = buffer_entry_index[objndx(material)];
			int alt = buffer_entry_index[lastmod(objndx(rec), rec->oargs.sarg[0])];
			insertArray2i(alt_materials, materials->count, alt);
		}
		else
			insertArray2i(alt_materials, materials->count, -1);
		insertArraym(materials, createLightMaterial(context, rec));
		break;
	case OBJ_FACE: // Typical polygons
		createFace(rec, parent);
		break;
	case OBJ_MESH: // Mesh from file
		createMesh(rec, parent);
		break;
	case OBJ_SOURCE:
		insertArraydl(sources, createDistantLight(context, rec, parent));
		break;
	case PAT_BFUNC: // brightness function, used for sky brightness
		buffer_entry_index[index] = functions->count;
		insertArrayi(functions, createFunction(context, rec, parent));
		break;
	case PAT_BDATA: // brightness texture, used for IES lighting data
		buffer_entry_index[index] = insertArrayi(functions, createTexture(context, rec));
		break;
	case TEX_FUNC:
		if (rec->oargs.nsargs == 3) {
			if (!strcmp(rec->oargs.sarg[2], "tmesh.cal")) break; // Handled by face
		}
		else if (rec->oargs.nsargs == 4) {
			if (!strcmp(rec->oargs.sarg[3], "tmesh.cal")) break; // Handled by face
		}
		printObect(rec);
		break;
	default:
#ifdef DEBUG_OPTIX
		printObect(rec);
#endif
		break;
	}
}

static void createFace(const OBJREC* rec, const OBJREC* parent)
{
	int j, k;
	FACE* face = getface(rec);
	OBJREC* material = findmaterial(parent);
#ifdef TRIANGULATE
	/* Triangulate the face */
	if (!createTriangles(face, material)) {
		printObect(rec);
		sprintf(errmsg, "Unable to triangulate face %s", rec->oname);
		error(USER, errmsg);
	}
#else /* TRIANGULATE */
	/* Triangulate the polygon as a fan */
	for (j = 2; j < face->nv; j++)
		createTriangle(material, vertex_index_0, vertex_index_0 + j - 1, vertex_index_0 + j);
#endif /* TRIANGULATE */

	/* Write the vertices to the buffers */
	for (j = 0; j < face->nv; j++) {
		insertArray3f(vertices, face->va[3 * j], face->va[3 * j + 1], face->va[3 * j + 2]);

		if (parent->otype == TEX_FUNC && parent->oargs.nsargs == 4 && !strcmp(parent->oargs.sarg[3], "tmesh.cal")) {
			/* Normal calculation from tmesh.cal */
			double bu, bv;
			FVECT v;

			k = (int)parent->oargs.farg[0];
			bu = face->va[3 * j + (k + 1) % 3];
			bv = face->va[3 * j + (k + 2) % 3];

			v[0] = bu * parent->oargs.farg[1] + bv * parent->oargs.farg[2] + parent->oargs.farg[3];
			v[1] = bu * parent->oargs.farg[4] + bv * parent->oargs.farg[5] + parent->oargs.farg[6];
			v[2] = bu * parent->oargs.farg[7] + bv * parent->oargs.farg[8] + parent->oargs.farg[9];

			normalize(v);

			insertArray3f(normals, v[0], v[1], v[2]);
		}
		else {
			//TODO Implement bump maps from texfunc and texdata
			// This might be done in the Intersecion program in triangle_mesh.cu rather than here
			insertArray3f(normals, face->norm[0], face->norm[1], face->norm[2]);
		}

		if (parent->otype == TEX_FUNC && parent->oargs.nsargs == 3 && !strcmp(parent->oargs.sarg[2], "tmesh.cal")) {
			/* Texture coordinate calculation from tmesh.cal */
			double bu, bv;

			k = (int)parent->oargs.farg[0];
			bu = face->va[3 * j + (k + 1) % 3];
			bv = face->va[3 * j + (k + 2) % 3];

			insertArray2f(tex_coords,
				bu * parent->oargs.farg[1] + bv * parent->oargs.farg[2] + parent->oargs.farg[3],
				bu * parent->oargs.farg[4] + bv * parent->oargs.farg[5] + parent->oargs.farg[6]);
		}
		else {
			//TODO Implement texture maps from colorfunc, brightfunc, colordata, brightdata, and colorpict
			// This might be done in the Intersecion program in triangle_mesh.cu rather than here
			insertArray2f(tex_coords, 0.0f, 0.0f);
		}
	}
	vertex_index_0 += face->nv;
	free(face);
}

static __inline void createTriangle(const OBJREC *material, const int a, const int b, const int c)
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
static int createTriangles(const FACE *face, const OBJREC *material)
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
				v2l->v[i].mX = (face->va+3*i)[(face->ax + 1) % 3];
				v2l->v[i].mY = (face->va+3*i)[(face->ax + 2) % 3];
			}
		else
			for (i = v2l->nv; i--; ) {
				v2l->v[i].mX = (face->va+3*i)[(face->ax + 2) % 3];
				v2l->v[i].mY = (face->va+3*i)[(face->ax + 1) % 3];
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

static void createMesh(const OBJREC* rec, const OBJREC* parent)
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
				objerror(rec, INTERNAL, "missing mesh vertices in createGeometryInstance");
			multp3(transform, mesh_vert.v, meshinst->x.f.xfm);
			insertArray3f(vertices, transform[0], transform[1], transform[2]);

			if (mesh_vert.fl & MT_N) { // TODO what if normal is defined by texture function
				multv3(transform, mesh_vert.n, meshinst->x.f.xfm);
				insertArray3f(normals, transform[0], transform[1], transform[2]);
			} else
				insertArray3f(normals, 0.0f, 0.0f, 0.0f); //TODO Can this happen?

			if (mesh_vert.fl & MT_UV)
				insertArray2f(tex_coords, mesh_vert.uv[0], mesh_vert.uv[1]);
			else
				insertArray2f(tex_coords, 0.0f, 0.0f);
		}

		vertex_index_0 += pp->nverts;
	}
	freemeshinst(rec);
}

static RTmaterial createNormalMaterial( const RTcontext context, const OBJREC* rec )
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

	if ( calc_ambient ) {
		if ( !ambient_normal_closest_hit_program ) {
			ptxFile( path_to_ptx, "ambient_normal" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "closest_hit_ambient", &ambient_normal_closest_hit_program ) );
		}
		RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( material, AMBIENT_RECORD_RAY, ambient_normal_closest_hit_program ) );

#ifdef KMEANS_IC
		if ( !point_cloud_closest_hit_program ) {
			ptxFile( path_to_ptx, "point_cloud_normal" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "closest_hit_point_cloud", &point_cloud_closest_hit_program ) );
		}
		RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( material, POINT_CLOUD_RAY, point_cloud_closest_hit_program ) );
#endif
	}

	/* Set variables to be consumed by material for this geometry instance */
	applyMaterialVariable1ui( context, material, "type", rec->otype );
	applyMaterialVariable3f( context, material, "color", rec->oargs.farg[0], rec->oargs.farg[1], rec->oargs.farg[2] );
	applyMaterialVariable1f( context, material, "spec", rec->oargs.farg[3] );
	applyMaterialVariable1f( context, material, "rough", rec->oargs.farg[4] );
	if (rec->otype == MAT_TRANS) { // it's a translucent material
		applyMaterialVariable1f( context, material, "transm", rec->oargs.farg[5] );
		applyMaterialVariable1f( context, material, "tspecu", rec->oargs.farg[6] );
	}

	return material;
}

static RTmaterial createGlassMaterial( const RTcontext context, const OBJREC* rec )
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

#ifdef KMEANS_IC
		if ( !point_cloud_any_hit_program ) {
			ptxFile( path_to_ptx, "point_cloud_normal" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "any_hit_point_cloud_glass", &point_cloud_any_hit_program ) );
		}
		RT_CHECK_ERROR( rtMaterialSetAnyHitProgram( material, POINT_CLOUD_RAY, point_cloud_any_hit_program ) );
#endif
	}

	/* Set variables to be consumed by material for this geometry instance */
#ifdef HIT_TYPE
	applyMaterialVariable1ui( context, material, "type", rec->otype );
#endif
	applyMaterialVariable3f( context, material, "color", rec->oargs.farg[0], rec->oargs.farg[1], rec->oargs.farg[2] );
	if (rec->oargs.nfargs > 3)
		applyMaterialVariable1f( context, material, "rindex", rec->oargs.farg[3] );

	return material;
}

static RTmaterial createLightMaterial( const RTcontext context, OBJREC* rec )
{
	RTmaterial material;

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

#ifdef KMEANS_IC
	if ( calc_ambient ) {
		if ( !point_cloud_any_hit_program ) {
			ptxFile( path_to_ptx, "point_cloud_normal" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "any_hit_point_cloud_glass", &point_cloud_any_hit_program ) );
		}
		RT_CHECK_ERROR( rtMaterialSetAnyHitProgram( material, POINT_CLOUD_RAY, point_cloud_any_hit_program ) );
	}
#endif

	/* Set variables to be consumed by material for this geometry instance */
#ifdef HIT_TYPE
	applyMaterialVariable1ui( context, material, "type", rec->otype );
#endif
	applyMaterialVariable3f( context, material, "color", rec->oargs.farg[0], rec->oargs.farg[1], rec->oargs.farg[2] );
	if (rec->otype == MAT_GLOW)
		applyMaterialVariable1f( context, material, "maxrad", rec->oargs.farg[3] );
	else if (rec->otype == MAT_SPOT) {
		SPOT* spot = makespot(rec);
		applyMaterialVariable1f( context, material, "siz", spot->siz );
		applyMaterialVariable1f( context, material, "flen", spot->flen );
		applyMaterialVariable3f( context, material, "aim", spot->aim[0], spot->aim[1], spot->aim[2] );
	}

	/* Check for a parent function. */
	if (rec->omod > OVOID)
		applyMaterialVariable1i( context, material, "function", buffer_entry_index[rec->omod] );
	else
		applyMaterialVariable1i( context, material, "function", RT_PROGRAM_ID_NULL );

	return material;
}

static DistantLight createDistantLight( const RTcontext context, OBJREC* rec, OBJREC* parent )
{
	SRCREC source;
	OBJREC* material;
	DistantLight light;

	ssetsrc(&source, rec);
	material = findmaterial(parent);
	light.color.x = material->oargs.farg[0]; // TODO these are given in RGB radiance value (watts/steradian/m2)
	light.color.y = material->oargs.farg[1];
	light.color.z = material->oargs.farg[2];
	light.pos.x   = source.sloc[0];
	light.pos.y   = source.sloc[1];
	light.pos.z   = source.sloc[2];
	light.solid_angle = source.ss2;
	light.casts_shadow = material->otype != MAT_GLOW; // Glow cannot cast shadow infinitely far away

	/* Check for a parent function. */
	if (material->omod > OVOID) {
		light.function = buffer_entry_index[material->omod];
	} else {
		light.function = -1;
	}

	return light;
}

static int createFunction( const RTcontext context, const OBJREC* rec )
{
	RTprogram program;
	int program_id = RT_PROGRAM_ID_NULL;
	XF bxp;

	/* Get transform */
	if ( !createTransform( &bxp, rec ) ) {
		printObect(rec);
		sprintf(errmsg, "Function %s has bad transform.\n", rec->oname);
		error(USER, errmsg);
		return RT_PROGRAM_ID_NULL;
	}

	if ( rec->oargs.nsargs >= 2 ) {
		if ( !strcmp(rec->oargs.sarg[0], "skybr") && !strcmp(rec->oargs.sarg[1], "skybright.cal") ) {
			float transform[9] = {
				bxp.xfm[0][0], bxp.xfm[1][0], bxp.xfm[2][0],
				bxp.xfm[0][1], bxp.xfm[1][1], bxp.xfm[2][1],
				bxp.xfm[0][2], bxp.xfm[1][2], bxp.xfm[2][2]
			};
			ptxFile( path_to_ptx, "skybright" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "sky_bright", &program ) );
			applyProgramVariable1ui( context, program, "type", rec->oargs.farg[0] );
			applyProgramVariable1f( context, program, "zenith", rec->oargs.farg[1] );
			applyProgramVariable1f( context, program, "ground", rec->oargs.farg[2] );
			applyProgramVariable1f( context, program, "factor", rec->oargs.farg[3] );
			applyProgramVariable3f( context, program, "sun", rec->oargs.farg[4], rec->oargs.farg[5], rec->oargs.farg[6] );
			applyProgramVariable( context, program, "transform", sizeof(transform), transform );
		} else if ( !strcmp(rec->oargs.sarg[0], "skybright") && !strcmp(rec->oargs.sarg[1], "perezlum.cal") ) {
			float coef[5] = { rec->oargs.farg[2], rec->oargs.farg[3], rec->oargs.farg[4], rec->oargs.farg[5], rec->oargs.farg[6] };
			float transform[9] = {
				bxp.xfm[0][0], bxp.xfm[1][0], bxp.xfm[2][0],
				bxp.xfm[0][1], bxp.xfm[1][1], bxp.xfm[2][1],
				bxp.xfm[0][2], bxp.xfm[1][2], bxp.xfm[2][2]
			};
			ptxFile( path_to_ptx, "perezlum" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "perez_lum", &program ) );
			applyProgramVariable1f( context, program, "diffuse", rec->oargs.farg[0] );
			applyProgramVariable1f( context, program, "ground", rec->oargs.farg[1] );
			applyProgramVariable( context, program, "coef", sizeof(coef), coef );
			applyProgramVariable3f( context, program, "sun", rec->oargs.farg[7], rec->oargs.farg[8], rec->oargs.farg[9] );
			applyProgramVariable( context, program, "transform", sizeof(transform), transform );
		} else {
			printObect(rec);
			return RT_PROGRAM_ID_NULL;
		}
	} else {
		printObect(rec);
		return RT_PROGRAM_ID_NULL;
	}

	RT_CHECK_ERROR( rtProgramGetId( program, &program_id ) );
	//applyContextVariable1i( context, "func", program_id );
	//applyContextObject( context, "func", program );
	return program_id;
}

static int createTexture( const RTcontext context, const OBJREC* rec )
{
	RTprogram        program;
	RTtexturesampler tex_sampler;
	RTformat         tex_format;
	RTbuffer         tex_buffer;
	float*           tex_buffer_data;

	int tex_id = RT_TEXTURE_ID_NULL;
	unsigned int i, entries;

	DATARRAY *dp;
	COLORV* color;
	XF bxp;

	/* Load texture data */
	dp = getdata(rec->oargs.sarg[1]); //TODO for Texdata and Colordata, use 3, 4, 5

	/* Get transform */
	if ( !createTransform( &bxp, rec ) ) {
		printObect(rec);
		sprintf(errmsg, "Texture %s has bad transform.\n", rec->oname);
		error(USER, errmsg);
		return RT_TEXTURE_ID_NULL;
	}

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
		printObect(rec);
		sprintf(errmsg, "Texture %s has bad number of dimensions %u.", rec->oname, dp->nd);
		error(USER, errmsg);
		return RT_TEXTURE_ID_NULL;
	}

	/* Populate buffer with texture data */
	RT_CHECK_ERROR( rtBufferMap( tex_buffer, (void**)&tex_buffer_data ) );
	if (dp->type == DATATY) // floats
		memcpy(tex_buffer_data, dp->arr.d, entries * sizeof(float));
	else // colors
		for (i = 0u; i < entries; i++) {
			colr_color(color, dp->arr.c[i]);
			copycolor(tex_buffer_data, color);
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
		if ( !strcmp(rec->oargs.sarg[0], "flatcorr") && !strcmp(rec->oargs.sarg[2], "source.cal") ) {
			float transform[9] = {
				bxp.xfm[0][0], bxp.xfm[1][0], bxp.xfm[2][0],
				bxp.xfm[0][1], bxp.xfm[1][1], bxp.xfm[2][1],
				bxp.xfm[0][2], bxp.xfm[1][2], bxp.xfm[2][2]
			};
			ptxFile( path_to_ptx, "source" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "flatcorr", &program ) );
			applyProgramVariable1i( context, program, "data", tex_id );
			applyProgramVariable1i( context, program, "type", dp->type == DATATY );
			applyProgramVariable3f( context, program, "minimum", dp->dim[dp->nd-1].org, dp->nd > 1 ? dp->dim[dp->nd-2].org : 0.0f, dp->nd > 2 ? dp->dim[dp->nd-3].org : 0.0f );
			applyProgramVariable3f( context, program, "maximum", dp->dim[dp->nd-1].siz, dp->nd > 1 ? dp->dim[dp->nd-2].siz : 1.0f, dp->nd > 2 ? dp->dim[dp->nd-3].siz : 1.0f );
			applyProgramVariable( context, program, "transform", sizeof(transform), transform );
			if (rec->oargs.nfargs > 0)
				applyProgramVariable1f( context, program, "multiplier", rec->oargs.farg[0] ); //TODO handle per-color channel multipliers
		} else {
			printObect(rec);
			return RT_PROGRAM_ID_NULL;
		}
	} else {
		printObect(rec);
		return RT_PROGRAM_ID_NULL;
	}
	RT_CHECK_ERROR( rtProgramGetId( program, &tex_id ) );

	return tex_id;
}

static int createTransform( XF* bxp, const OBJREC* rec )
{
	/* Get transform - from getfunc in func.c */
	unsigned int i = 0u;
	while (i < rec->oargs.nsargs && rec->oargs.sarg[i][0] != '-')
		i++;
	if (i == rec->oargs.nsargs)	{		/* no transform */
		*bxp = unitxf;
		return 1;
	}
	
	/* get transform */
	if (invxf(bxp, rec->oargs.nsargs - i, rec->oargs.sarg + i) != rec->oargs.nsargs - i)
		return 0;

	if (bxp->sca < 0.0) 
		bxp->sca = -bxp->sca;

	return 1;
}

static void createAcceleration( const RTcontext context, const RTgeometryinstance instance )
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
	createNormalMaterial( context, instance, 0, &Lamb );

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

static void printObect( const OBJREC* rec )
{
	unsigned int i;

	fprintf(stderr, "Unsupported object\n", rec->oname);
	fprintf(stderr, "Object name: %s\n", rec->oname);
	fprintf(stderr, "Object type: %s (%i)\n", ofun[rec->otype].funame, rec->otype);
	if (rec->omod > OVOID) // does not modify void
		fprintf(stderr, "Object modifies: %s (%i)\n", objptr(rec->omod)->oname, rec->omod);
	if (rec->os)
		fprintf(stderr, "Object structure: %s\n", rec->os);

	for (i = 0u; i < rec->oargs.nsargs; i++)
		fprintf(stderr, "S Arg %i: %s\n", i, rec->oargs.sarg[i]);
	for (i = 0u; i < rec->oargs.nfargs; i++)
		fprintf(stderr, "R Arg %i: %f\n", i, rec->oargs.farg[i]);

	fprintf(stderr, "\n");
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

	//data->tex.x = ray->uv[0];
	//data->tex.y = ray->uv[1];
	data->max = ray->rmax;
	data->weight = ray->rweight;
	data->length = ray->rt;
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

	//ray->uv[0] = data->tex.x;
	//ray->uv[1] = data->tex.y;
	ray->rmax = data->max;
	ray->rweight = data->weight;
	ray->rt = data->length;
	//ray->rot = data->t; //TODO setting this requires that the ray has non-null ro.
}
