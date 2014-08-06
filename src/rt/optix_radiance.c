/*
 * Copyright (c) 2013-2014 Nathaniel Jones
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
#include "data.h"

#include "optix_radiance.h"
#include "optix_common.h"


/* Structure to track the number of each data type in the model. */
typedef struct struct_object_count
{
	unsigned int vertex, triangle, material, light, distant_light;
#ifdef CALLABLE
	unsigned int function;
#else
	unsigned int sky_bright, perez_lum, light_source;
#endif
} ObjectCount;

void renderOptix( const VIEW* view, const int width, const int height, const double alarm, COLOR* colors, float* depths );
void computeOptix( const int width, const int height, const double alarm, RAY* rays );
void endOptix();

static void printDeviceInfo();
static void createContext( RTcontext* context, const int width, const int height, const double alarm );
static void setupKernel( const RTcontext context, const VIEW* view, const int width, const int height );
static void applyRadianceSettings( const RTcontext context );
static void createCamera( const RTcontext context, const VIEW* view );
static void updateCamera( const RTcontext context, const VIEW* view );
static void countObjectsInScene( ObjectCount* count );
#ifndef CALLABLE
static int getSkyType( const OBJREC* bright_func );
#endif
static void createGeometryInstance( const RTcontext context, RTgeometryinstance* instance );
static void createNormalMaterial( const RTcontext context, const RTgeometryinstance instance, const int index, const OBJREC* rec );
static void createGlassMaterial( const RTcontext context, const RTgeometryinstance instance, const int index, const OBJREC* rec );
static void createLightMaterial( const RTcontext context, const RTgeometryinstance instance, const int index, const int* buffer_entry_index, OBJREC* rec );
static DistantLight createDistantLight( const RTcontext context, const int* buffer_entry_index, OBJREC* rec );
#ifdef CALLABLE
static int createFunction( const RTcontext context, const OBJREC* rec );
static int createTexture( const RTcontext context, const OBJREC* rec );
#else
static int createTexture( const RTcontext context, const OBJREC* rec, float* range );
#endif
static void createAcceleration( const RTcontext context, const RTgeometryinstance instance );
static void printObect( const OBJREC* rec );
static void getRay( RayData* data, const RAY* ray );
static void setRay( RAY* ray, const RayData* data );


/* from rpict.c */
extern double  dstrpix;			/* square pixel distribution */

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
#define MAX_TCCS	2				/* Maximum number of Tesla devices to discover */
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
		printDeviceInfo();
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
		copycolor( colors[i], data );
		depths[i] = data[3];
		data += 4;
	}

#ifdef PRINT_OPTIX
	/* Exit if message pringing is on */
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

	/* Setup state */
	printDeviceInfo();
	createContext( &context, width, height, alarm );
	setupKernel( context, NULL, width, height );
	
	/* Input/output buffer */
	createCustomBuffer2D( context, RT_BUFFER_INPUT_OUTPUT, sizeof(RayData), width, height, &ray_buffer );
	//createCustomBuffer2D( context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, sizeof(RayData), width, height, &ray_buffer );
	RT_CHECK_ERROR( rtBufferMap( ray_buffer, (void**)&data ) );
	for (i = 0u; i < size; i++) {
		getRay( data++, &rays[i] );
	}
	RT_CHECK_ERROR( rtBufferUnmap( ray_buffer ) );
	applyContextObject( context, "ray_buffer", ray_buffer );

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

static void printDeviceInfo()
{
	unsigned int version, device_count;
	unsigned int i;
	unsigned int multiprocessor_count, threads_per_block, clock_rate, texture_count, timeout_enabled, tcc_driver, cuda_device;
	unsigned int compute_capability[2];
	char device_name[128];
	RTsize memory_size;

	rtGetVersion( &version );
	rtDeviceGetDeviceCount( &device_count );
	if ( !device_count ) {
		fprintf(stderr, "A supported NVIDIA GPU could not be found for OptiX %i.\n", version);
		exit(1);
	}
	fprintf(stderr, "Starting OptiX %i on %i devices:\n", version, device_count);

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
#ifdef PREFER_TCC
		if ( tcc_driver && ( tcc_count < MAX_TCCS ) )
			tcc[tcc_count++] = i;
#endif
#ifdef CALLABLE
		if ( compute_capability[0] < 2u )
			fprintf(stderr, "Device %u has insufficient compute capability for OptiX callable programs.\n", i);
#endif
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

	ray_type_count = 2u; /* shadow and radiance */
	entry_point_count = 1u; /* Generate radiance data */

	if ( use_ambient )
		ray_type_count++; /* ambient ray */
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
	//RT_CHECK_ERROR2( rtContextSetPrintLaunchIndex( *context, 146, -1, -1 ) );
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
	RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, RADIANCE_ENTRY, ray_gen_program ) );

	/* Exception program */
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "exception", &exception_program ) );
	RT_CHECK_ERROR( rtContextSetExceptionProgram( context, RADIANCE_ENTRY, exception_program ) );

	/* Miss program */
	ptxFile( path_to_ptx, "background" );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "miss", &miss_program ) );
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

/* Count the number of vertices and triangles belonging to all faces in the scene. */
static void countObjectsInScene( ObjectCount* count )
{
	OBJREC* rec, *material;
	FACE* face;
#ifdef CALLABLE
	unsigned int i, v_count, t_count, m_count, l_count, dl_count, f_count;
	v_count = t_count = m_count = l_count = dl_count = f_count = 0u;
#else
	unsigned int i, v_count, t_count, m_count, l_count, dl_count, cie_count, perez_count, source_count;
	v_count = t_count = m_count = l_count = dl_count = cie_count = perez_count = source_count = 0u;
#endif

	for (i = 0; i < nobjects; i++) {
		rec = objptr(i);
		//if (issurface(rec.otype)) {
		switch (rec->otype) {
		case MAT_PLASTIC: // Plastic material
		case MAT_METAL: // Metal material
		case MAT_TRANS: // Translucent material
		case MAT_GLASS: // Glass material
		case MAT_DIELECTRIC: // Dielectric material
		case MAT_LIGHT: // primary light source material, may modify a face or a source (solid angle)
		case MAT_GLOW: // Glow material
		case MAT_SPOT: // Spotlight material
			m_count++;
			break;
		case OBJ_FACE: // Typical polygons
			face = getface(rec);
			v_count += face->nv;
			t_count += face->nv - 2;
#ifdef LIGHTS
			material = findmaterial(rec);
			if (islight(material->otype) && (material->otype != MAT_GLOW || material->oargs.farg[3] > 0))
				l_count += face->nv - 2;
#endif
			break;
		case OBJ_SOURCE: // The sun, for example
			dl_count++;
			break;
		case PAT_BFUNC: // brightness function, used for sky brightness
#ifdef CALLABLE
			f_count++;
#else
			switch (getSkyType(rec)) {
			case SKY_CIE: // CIE sky
				cie_count++;
				break;
			case SKY_PEREZ: // Perez sky
				perez_count++;
				break;
			default:
				break;
			}
#endif
			break;
		case PAT_BDATA: // brightness texture, used for IES lighting data
#ifdef CALLABLE
			f_count++;
#else
			source_count++;
#endif
			break;
		default:
			break;
		}
	}

	count->vertex = v_count;
	count->triangle = t_count;
	count->material = m_count;
#ifdef LIGHTS
	count->light = l_count;
#endif
	count->distant_light = dl_count;
#ifdef CALLABLE
	count->function = f_count;
#else
	count->sky_bright = cie_count;
	count->perez_lum = perez_count;
	count->light_source = source_count;
#endif
}

#ifndef CALLABLE
static int getSkyType( const OBJREC* rec )
{
	if ( rec->oargs.nsargs == 2 ) {
		if ( !strcmp(rec->oargs.sarg[0], "skybr") && !strcmp(rec->oargs.sarg[1], "skybright.cal") )
			return SKY_CIE;
		if ( !strcmp(rec->oargs.sarg[0], "skybright") && !strcmp(rec->oargs.sarg[1], "perezlum.cal") )
			return SKY_PEREZ;
	}
	return SKY_NONE;
}
#endif

static void createGeometryInstance( const RTcontext context, RTgeometryinstance* instance )
{
	RTgeometry mesh;
	RTprogram  mesh_intersection_program;
	RTprogram  mesh_bounding_box_program;

	RTbuffer   light_buffer;
	DistantLight* light_buffer_data;

#ifdef CALLABLE
	RTbuffer   function_buffer;
	int*       function_buffer_data;
#else
	RTbuffer   cie_buffer;
	void*      cie_buffer_data;

	RTbuffer   perez_buffer;
	void*      perez_buffer_data;

	RTbuffer   source_buffer;
	void*      source_buffer_data;
	float      range[16];
#endif

	RTbuffer   vertex_buffer;
	RTbuffer   normal_buffer;
	RTbuffer   tex_coord_buffer;
	RTbuffer   vertex_index_buffer;
	RTbuffer   normal_index_buffer;
	RTbuffer   tex_coord_index_buffer;
	RTbuffer   material_index_buffer;
	float*     vertex_buffer_data;
	float*     normal_buffer_data;
	float*     tex_coord_buffer_data;
	int*       v_index_buffer_data;
	int*       n_index_buffer_data;
	int*       t_index_buffer_data;
	unsigned int* m_index_buffer_data;

#ifdef LIGHTS
	RTbuffer   light_index_buffer;
	int*       l_index_buffer_data;
#endif

	int i, j, t;
#ifdef CALLABLE
	unsigned int vi, ni, ti, mi, vii, nii, tii, mii, li, dli, fi;
#else
	unsigned int vi, ni, ti, mi, vii, nii, tii, mii, li, dli, sbi, pli, lsi;
#endif
	ObjectCount count;
	OBJREC* rec, *parent, *material;
	FACE* face;
#ifndef CALLABLE
	SkyBright cie;
	PerezLum perez;
	Light light_source;
#endif

	/* This array gives the OptiX buffer index of each rad file object.
	 * The OptiX buffer refered to depends on the type of object.
	 * If the object does not have an index in an OptiX buffer, -1 is given. */
	int* buffer_entry_index;

	buffer_entry_index = (int *)malloc(sizeof(int) * nobjects);

	/* Count the number of each type of object in the scene. */
	countObjectsInScene(&count);

	/* Create the geometry reference for OptiX. */
	RT_CHECK_ERROR( rtGeometryCreate( context, &mesh ) );
	RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( mesh, count.triangle ) );

	ptxFile( path_to_ptx, "triangle_mesh" );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "mesh_bounds", &mesh_bounding_box_program ) );
	RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( mesh, mesh_bounding_box_program ) );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "mesh_intersect", &mesh_intersection_program ) );
	RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( mesh, mesh_intersection_program ) );
	applyProgramVariable1ui( context, mesh_intersection_program, "backvis", backvis ); // -bv

	/* Create the geometry instance containing the geometry. */
	RT_CHECK_ERROR( rtGeometryInstanceCreate( context, instance ) );
	RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( *instance, mesh ) );
	RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( *instance, count.material ) );

	/* Create buffers for storing lighting information. */
	createCustomBuffer1D( context, RT_BUFFER_INPUT, sizeof(DistantLight), count.distant_light, &light_buffer );
	RT_CHECK_ERROR( rtBufferMap( light_buffer, (void**)&light_buffer_data ) );

#ifdef CALLABLE
	createBuffer1D( context, RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, count.function, &function_buffer );
	RT_CHECK_ERROR( rtBufferMap( function_buffer, (void**)&function_buffer_data ) );
#else
	createCustomBuffer1D( context, RT_BUFFER_INPUT, sizeof(SkyBright), count.sky_bright, &cie_buffer );
	RT_CHECK_ERROR( rtBufferMap( cie_buffer, &cie_buffer_data ) );

	createCustomBuffer1D( context, RT_BUFFER_INPUT, sizeof(PerezLum), count.perez_lum, &perez_buffer );
	RT_CHECK_ERROR( rtBufferMap( perez_buffer, &perez_buffer_data ) );

	createCustomBuffer1D( context, RT_BUFFER_INPUT, sizeof(Light), count.light_source, &source_buffer );
	RT_CHECK_ERROR( rtBufferMap( source_buffer, &source_buffer_data ) );
#endif

	/* Create buffers for storing geometry information. */
	createBuffer1D( context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, count.vertex, &vertex_buffer );
	RT_CHECK_ERROR( rtBufferMap( vertex_buffer, (void**)&vertex_buffer_data ) );

	createBuffer1D( context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, count.vertex, &normal_buffer );
	RT_CHECK_ERROR( rtBufferMap( normal_buffer, (void**)&normal_buffer_data ) );

	createBuffer1D( context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, count.vertex, &tex_coord_buffer );
	RT_CHECK_ERROR( rtBufferMap( tex_coord_buffer, (void**)&tex_coord_buffer_data ) );

	createBuffer1D( context, RT_BUFFER_INPUT, RT_FORMAT_INT3, count.triangle, &vertex_index_buffer );
	RT_CHECK_ERROR( rtBufferMap( vertex_index_buffer, (void**)&v_index_buffer_data ) );

	createBuffer1D( context, RT_BUFFER_INPUT, RT_FORMAT_INT3, count.triangle, &normal_index_buffer );
	RT_CHECK_ERROR( rtBufferMap( normal_index_buffer, (void**)&n_index_buffer_data ) );

	createBuffer1D( context, RT_BUFFER_INPUT, RT_FORMAT_INT3, count.triangle, &tex_coord_index_buffer );
	RT_CHECK_ERROR( rtBufferMap( tex_coord_index_buffer, (void**)&t_index_buffer_data ) );

	createBuffer1D( context, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, count.triangle, &material_index_buffer );
	RT_CHECK_ERROR( rtBufferMap( material_index_buffer, (void**)&m_index_buffer_data ) );

#ifdef LIGHTS
	createBuffer1D( context, RT_BUFFER_INPUT, RT_FORMAT_INT3, count.light, &light_index_buffer );
	RT_CHECK_ERROR( rtBufferMap( light_index_buffer, (void**)&l_index_buffer_data ) );
#endif

#ifdef CALLABLE
	vi = ni = ti = mi = vii = nii = tii = mii = li = dli = fi = 0u;
#else
	vi = ni = ti = mi = vii = nii = tii = mii = li = dli = sbi = pli = lsi = 0u;
#endif

	/* Get the scene geometry as a list of triangles. */
	fprintf(stderr, "Num objects %i\n", nobjects);
	for (i = 0; i < nobjects; i++) {
		/* By default, no buffer entry is refered to. */
		buffer_entry_index[i] = -1;

		rec = objptr(i);
		//fprintf(stderr, "Object %i: %s\n", i, rec.oname);
		//fprintf(stderr, "Object %i modifies: %i\n", i, rec.omod);
		//fprintf(stderr, "Object %i type: %i\n", i, rec.otype);
		//fprintf(stderr, "Object %i structure: %s\n", i, rec.os);

		//if (issurface(rec.otype)) {
		switch(rec->otype) {
		case MAT_PLASTIC: // Plastic material
		case MAT_METAL: // Metal material
		case MAT_TRANS: // Translucent material
			buffer_entry_index[i] = mi;
			createNormalMaterial( context, *instance, mi++, rec );
			break;
		case MAT_GLASS: // Glass material
		case MAT_DIELECTRIC: // Dielectric material TODO handle separately, see dialectric.c
			buffer_entry_index[i] = mi;
			createGlassMaterial( context, *instance, mi++, rec );
			break;
		case MAT_LIGHT: // primary light source material, may modify a face or a source (solid angle)
		case MAT_GLOW: // Glow material
		case MAT_SPOT: // Spotlight material
			buffer_entry_index[i] = mi;
			createLightMaterial( context, *instance, mi++, buffer_entry_index, rec );
			break;
		case OBJ_FACE:
			face = getface(rec);
			parent = objptr(rec->omod);
			material = findmaterial(parent);
			t = vi / 3;

			/* Write the indices to the buffers */
			for (j = 2; j < face->nv; j++) {
				v_index_buffer_data[vii++] = t;
				v_index_buffer_data[vii++] = t + j - 1;
				v_index_buffer_data[vii++] = t + j;

				n_index_buffer_data[nii++] = t;
				n_index_buffer_data[nii++] = t + j - 1;
				n_index_buffer_data[nii++] = t + j;

				t_index_buffer_data[tii++] = t;
				t_index_buffer_data[tii++] = t + j - 1;
				t_index_buffer_data[tii++] = t + j;

				//m_index_buffer_data[mii++] = buffer_entry_index[rec.omod]; //This may be the case once texture functions are implemented
				m_index_buffer_data[mii++] = buffer_entry_index[objndx(material)];

#ifdef LIGHTS
				if (islight(material->otype) && (material->otype != MAT_GLOW || material->oargs.farg[3] > 0)) {
					l_index_buffer_data[li++] = t;
					l_index_buffer_data[li++] = t + j - 1;
					l_index_buffer_data[li++] = t + j;
				}
#endif
			}

			/* Write the vertices to the buffers */
			for (j = 0; j < face->nv; j++) {
				vertex_buffer_data[vi++] = face->va[3 * j]; //TODO this only works for convex polygons
				vertex_buffer_data[vi++] = face->va[3 * j + 1];
				vertex_buffer_data[vi++] = face->va[3 * j + 2];

				if (parent->otype == TEX_FUNC && parent->oargs.nsargs == 4 && !strcmp(parent->oargs.sarg[3], "tmesh.cal")) {
					/* Normal calculation from tmesh.cal */
					double bu, bv;
					FVECT v;

					t = (int)parent->oargs.farg[0];
					bu = face->va[3 * j + (t + 1) % 3];
					bv = face->va[3 * j + (t + 2) % 3];

					v[0] = bu * parent->oargs.farg[1] + bv * parent->oargs.farg[2] + parent->oargs.farg[3];
					v[1] = bu * parent->oargs.farg[4] + bv * parent->oargs.farg[5] + parent->oargs.farg[6];
					v[2] = bu * parent->oargs.farg[7] + bv * parent->oargs.farg[8] + parent->oargs.farg[9];

					normalize(v);

					normal_buffer_data[ni++] = v[0];
					normal_buffer_data[ni++] = v[1];
					normal_buffer_data[ni++] = v[2];
				} else {
					//TODO Implement bump maps from texfunc and texdata
					// This might be done in the Intersecion program in triangle_mesh.cu rather than here
					normal_buffer_data[ni++] = face->norm[0];
					normal_buffer_data[ni++] = face->norm[1];
					normal_buffer_data[ni++] = face->norm[2];
				}

				if (parent->otype == TEX_FUNC && parent->oargs.nsargs == 3 && !strcmp(parent->oargs.sarg[2], "tmesh.cal")) {
					/* Texture coordinate calculation from tmesh.cal */
					double bu, bv;

					t = (int)parent->oargs.farg[0];
					bu = face->va[3 * j + (t + 1) % 3];
					bv = face->va[3 * j + (t + 2) % 3];

					tex_coord_buffer_data[ti++] = bu * parent->oargs.farg[1] + bv * parent->oargs.farg[2] + parent->oargs.farg[3];
					tex_coord_buffer_data[ti++] = bu * parent->oargs.farg[4] + bv * parent->oargs.farg[5] + parent->oargs.farg[6];
				} else {
					//TODO Implement texture maps from colorfunc, brightfunc, colordata, brightdata, and colorpict
					// This might be done in the Intersecion program in triangle_mesh.cu rather than here
					tex_coord_buffer_data[ti++] = 0.0f;
					tex_coord_buffer_data[ti++] = 0.0f;
				}
			}
			break;
		case OBJ_SOURCE:
			light_buffer_data[dli++] = createDistantLight( context, buffer_entry_index, rec );
			break;
		case PAT_BFUNC: // brightness function, used for sky brightness
#ifdef CALLABLE
			buffer_entry_index[i] = fi;
			function_buffer_data[fi++] = createFunction( context, rec );
#else /* CALLLABLE */
			switch (getSkyType(rec)) {
			case SKY_CIE: // CIE sky
				buffer_entry_index[i] = sbi;

				cie.type = rec->oargs.farg[0];
				cie.zenith = rec->oargs.farg[1];
				cie.ground = rec->oargs.farg[2];
				cie.factor = rec->oargs.farg[3];
				array2cuda3(cie.sun, (rec->oargs.farg + 4));
				((SkyBright*)cie_buffer_data)[sbi++] = cie;
				break;
			case SKY_PEREZ: // Perez sky
				buffer_entry_index[i] = pli;

				perez.diffuse = rec->oargs.farg[0];
				perez.ground = rec->oargs.farg[1];
				for (j = 0; j < 5; j++) {
					perez.coef[j] = rec->oargs.farg[j + 2];
				}
				array2cuda3(perez.sun, (rec->oargs.farg + 7));
				((PerezLum*)perez_buffer_data)[pli++] = perez;
				break;
			default:
				break;
			}
#endif /* CALLLABLE */
			break;
		case PAT_BDATA: // brightness texture, used for IES lighting data
#ifdef CALLABLE
			buffer_entry_index[i] = fi;
			function_buffer_data[fi++] = createTexture( context, rec );
#else /* CALLLABLE */
			buffer_entry_index[i] = lsi;

			light_source.texture = createTexture( context, rec, range );
			array2cuda3(light_source.min, range);
			array2cuda3(light_source.max, (range + 3));
			array2cuda3(light_source.u, (range + 6));
			array2cuda3(light_source.v, (range + 9));
			array2cuda3(light_source.w, (range + 12));
			light_source.multiplier = range[15];
			((Light*)source_buffer_data)[lsi++] = light_source;
#endif /* CALLLABLE */
			break;
		case TEX_FUNC:
			if (rec->oargs.nsargs == 3) {
				if (!strcmp(rec->oargs.sarg[2], "tmesh.cal")) break; // Handled by face
			} else if (rec->oargs.nsargs == 4) {
				if (!strcmp(rec->oargs.sarg[3], "tmesh.cal")) break; // Handled by face
			}
			printObect( rec );
			break;
		default:
#ifdef DEBUG_OPTIX
			printObect( rec );
#endif
			break;
		}
		//fprintf(stderr, "\n");
	}

	free( buffer_entry_index );

	/* Unmap and apply the lighting buffer. */
	RT_CHECK_ERROR( rtBufferUnmap( light_buffer ) );
	applyContextObject( context, "lights", light_buffer );

#ifdef CALLABLE
	RT_CHECK_ERROR( rtBufferUnmap( function_buffer ) );
	applyContextObject( context, "functions", function_buffer );
#else
	RT_CHECK_ERROR( rtBufferUnmap( cie_buffer ) );
	applyContextObject( context, "sky_brights", cie_buffer );

	RT_CHECK_ERROR( rtBufferUnmap( perez_buffer ) );
	applyContextObject( context, "perez_lums", perez_buffer );

	RT_CHECK_ERROR( rtBufferUnmap( source_buffer ) );
	applyGeometryInstanceObject( context, *instance, "light_sources", source_buffer );
#endif

	/* Unmap and apply the geometry buffers. */
	RT_CHECK_ERROR( rtBufferUnmap( vertex_buffer ) );
	applyGeometryInstanceObject( context, *instance, "vertex_buffer", vertex_buffer );

	RT_CHECK_ERROR( rtBufferUnmap( normal_buffer ) );
	applyGeometryInstanceObject( context, *instance, "normal_buffer", normal_buffer );

	RT_CHECK_ERROR( rtBufferUnmap( tex_coord_buffer ) );
	applyGeometryInstanceObject( context, *instance, "texcoord_buffer", tex_coord_buffer );

	RT_CHECK_ERROR( rtBufferUnmap( vertex_index_buffer ) );
	applyGeometryInstanceObject( context, *instance, "vindex_buffer", vertex_index_buffer );

	RT_CHECK_ERROR( rtBufferUnmap( normal_index_buffer ) );
	applyGeometryInstanceObject( context, *instance, "nindex_buffer", normal_index_buffer );

	RT_CHECK_ERROR( rtBufferUnmap( tex_coord_index_buffer ) );
	applyGeometryInstanceObject( context, *instance, "tindex_buffer", tex_coord_index_buffer );

	RT_CHECK_ERROR( rtBufferUnmap( material_index_buffer ) );
	applyGeometryInstanceObject( context, *instance, "material_buffer", material_index_buffer );

#ifdef LIGHTS
	RT_CHECK_ERROR( rtBufferUnmap( light_index_buffer ) );
	applyGeometryInstanceObject( context, *instance, "lindex_buffer", light_index_buffer );
#endif
}

static void createNormalMaterial( const RTcontext context, const RTgeometryinstance instance, const int index, const OBJREC* rec )
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

	/* Apply this material to the geometry instance. */
	RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( instance, index, material ) );
}

static void createGlassMaterial( const RTcontext context, const RTgeometryinstance instance, const int index, const OBJREC* rec )
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

	/* Apply this material to the geometry instance. */
	RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( instance, index, material ) );
}

static void createLightMaterial( const RTcontext context, const RTgeometryinstance instance, const int index, const int* buffer_entry_index, OBJREC* rec )
{
	RTmaterial material;

	/* Create our hit programs to be shared among all materials */
	if ( !radiance_light_closest_hit_program || !shadow_light_closest_hit_program ) {
		ptxFile( path_to_ptx, "radiance_light" );
		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "closest_hit_radiance", &radiance_light_closest_hit_program ) );
		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "closest_hit_shadow", &shadow_light_closest_hit_program ) );
	}

	RT_CHECK_ERROR( rtMaterialCreate( context, &material ) );
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
	if (rec->omod > -1) {
#ifdef CALLABLE
		applyMaterialVariable1ui( context, material, "function", buffer_entry_index[rec->omod] );
#else
		applyMaterialVariable1i( context, material, "lindex", buffer_entry_index[rec->omod] );
#endif
	}

	/* Apply this material to the geometry instance. */
	RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( instance, index, material ) );
}

static DistantLight createDistantLight( const RTcontext context, const int* buffer_entry_index, OBJREC* rec )
{
	SRCREC source;
	OBJREC* material;
	DistantLight light;

	ssetsrc(&source, rec);
	material = findmaterial(rec);
	light.color.x = material->oargs.farg[0]; // TODO these are given in RGB radiance value (watts/steradian/m2)
	light.color.y = material->oargs.farg[1];
	light.color.z = material->oargs.farg[2];
	light.pos.x   = source.sloc[0];
	light.pos.y   = source.sloc[1];
	light.pos.z   = source.sloc[2];
	light.solid_angle = source.ss2;
	light.casts_shadow = material->otype != MAT_GLOW; // Glow cannot cast shadow infinitely far away

	/* Check for a parent function. */
	if (material->omod > -1) {
#ifndef CALLABLE
		light.type = getSkyType(objptr(material->omod));
#endif
		light.function = buffer_entry_index[material->omod];
	} else {
#ifndef CALLABLE
		light.type = SKY_NONE;
#endif
		light.function = -1;
	}

	return light;
}

#ifdef CALLABLE
static int createFunction( const RTcontext context, int* buffer_entry_index, const OBJREC* rec )
{
	RTprogram program;
	int program_id = RT_PROGRAM_ID_NULL;

	if ( rec->oargs.nsargs == 2 ) {
		if ( !strcmp(rec->oargs.sarg[0], "skybr") && !strcmp(rec->oargs.sarg[1], "skybright.cal") ) {
			ptxFile( path_to_ptx, "skybright" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "sky_bright", &program ) );
			applyProgramVariable1ui( context, program, "type", rec->oargs.farg[0] );
			applyProgramVariable1f( context, program, "zenith", rec->oargs.farg[1] );
			applyProgramVariable1f( context, program, "ground", rec->oargs.farg[2] );
			applyProgramVariable1f( context, program, "factor", rec->oargs.farg[3] );
			applyProgramVariable3f( context, program, "sun", rec->oargs.farg[4], rec->oargs.farg[5], rec->oargs.farg[6] );
		} else if ( !strcmp(rec->oargs.sarg[0], "skybright") && !strcmp(rec->oargs.sarg[1], "perezlum.cal") ) {
			ptxFile( path_to_ptx, "perezlum" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "perez_lum", &program ) );
			applyProgramVariable1f( context, program, "diffuse", rec->oargs.farg[0] );
			applyProgramVariable1f( context, program, "ground", rec->oargs.farg[1] );
			applyProgramVariable3f( context, program, "coef0", rec->oargs.farg[2], rec->oargs.farg[3], rec->oargs.farg[4] );
			applyProgramVariable2f( context, program, "coef1", rec->oargs.farg[5], rec->oargs.farg[6] );
			applyProgramVariable3f( context, program, "sun", rec->oargs.farg[7], rec->oargs.farg[8], rec->oargs.farg[9] );
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
#else /* CALLLABLE */
static int createTexture( const RTcontext context, const OBJREC* rec, float* range )
#endif /* CALLLABLE */
{
#ifdef CALLABLE
	RTprogram        program;
#endif /* CALLLABLE */
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

	/* Get transform - from getfunc in func.c */
	i = 0u;
	while (i < rec->oargs.nsargs && rec->oargs.sarg[i][0] != '-')
		i++;
	if (i == rec->oargs.nsargs) {			/* no transform */
		bxp = unitxf;
	} else {			/* get transform */
		if (invxf(&bxp, rec->oargs.nsargs - i, rec->oargs.sarg + i) != rec->oargs.nsargs - i) {
			fprintf(stderr, "Texture %s has bad transform.\n", rec->oname);
			printObect(rec);
			return RT_TEXTURE_ID_NULL;
		}
		if (bxp.sca < 0.0) 
			bxp.sca = -bxp.sca;
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
		fprintf(stderr, "Texture %s has bad number of dimensions %u.\n", rec->oname, dp->nd);
		printObect(rec);
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

#ifdef CALLABLE
	/* Create program to access texture sampler */
	if ( rec->oargs.nsargs == 5 ) {
		if ( !strcmp(rec->oargs.sarg[0], "flatcorr") && !strcmp(rec->oargs.sarg[1], "source.cal") ) {
			ptxFile( path_to_ptx, "source" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "flatcorr", &program ) );
			applyProgramObject( context, program, "data", tex_sampler );
			applyProgramVariable3f( context, program, "min", dp->dim[dp->nd-1].org, dp->nd > 1 ? dp->dim[dp->nd-2].org : 0.0f, dp->nd > 2 ? dp->dim[dp->nd-3].org : 0.0f );
			applyProgramVariable3f( context, program, "max", dp->dim[dp->nd-1].siz, dp->nd > 1 ? dp->dim[dp->nd-2].siz : 1.0f, dp->nd > 2 ? dp->dim[dp->nd-3].siz : 1.0f );
			applyProgramVariable3f( context, program, "u", bxp.xfm[0][0], bxp.xfm[1][0], bxp.xfm[2][0]);
			applyProgramVariable3f( context, program, "v", bxp.xfm[0][1], bxp.xfm[1][1], bxp.xfm[2][1]);
			applyProgramVariable3f( context, program, "w", bxp.xfm[0][2], bxp.xfm[1][2], bxp.xfm[2][2]);
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
#else /* CALLLABLE */
	RT_CHECK_ERROR( rtTextureSamplerGetId( tex_sampler, &tex_id ) );

	for (i = 0u; i < dp->nd; i++) {
		range[i]     = dp->dim[dp->nd - i - 1].org;
		range[3 + i] = dp->dim[dp->nd - i - 1].siz;
	}
	range[6]  = bxp.xfm[0][0]; range[7]  = bxp.xfm[1][0]; range[8]  = bxp.xfm[2][0];
	range[9]  = bxp.xfm[0][1]; range[10] = bxp.xfm[1][1]; range[11] = bxp.xfm[2][1];
	range[12] = bxp.xfm[0][2]; range[13] = bxp.xfm[1][2]; range[14] = bxp.xfm[2][2];
	range[15] = rec->oargs.nfargs > 0 ? rec->oargs.farg[0]: 1.0f;
#endif /* CALLLABLE */

	return tex_id;
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
	applyContextObject( context, "top_shadower", geometrygroup );
	if ( !use_ambient )
		applyContextObject( context, "top_ambient", geometrygroup ); // Need to define this because it is referred to by radiance_normal.cu

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

static void printObect( const OBJREC* rec )
{
	unsigned int i;

	fprintf(stderr, "Object name: %s\n", rec->oname);
	fprintf(stderr, "Object type: %s (%i)\n", ofun[rec->otype].funame, rec->otype);
	if (rec->omod > OVOID) // does not modify void
		fprintf(stderr, "Object modifies: %s (%i)\n", objptr(rec->omod)->oname, rec->omod);
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
	//ray->rot = data->t;
}
