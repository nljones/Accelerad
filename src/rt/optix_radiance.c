/*
 *  optix_radiance.c - routines for setting up simulations on GPUs.
 */

#include "accelerad_copyright.h"

#include "ray.h"
#include "otypes.h"
#include "source.h"
#include "ambient.h"
#include "face.h"
#include "cone.h"
#include "mesh.h"
#include "instance.h"
#include "data.h"
#include "bsdf.h"
#include "random.h"
#include "paths.h"

#include "optix_radiance.h"
#include <cuda_runtime_api.h>

#ifdef CONTRIB
#include "rcontrib.h"
#endif

#define TRIANGULATE
#ifdef TRIANGULATE
#include "triangulate.h"
#endif /* TRIANGULATE */

#define BLANK	-1	/* No index provided */

#define EXPECTED_VERTICES	64
#define EXPECTED_TRIANGLES	64
#define EXPECTED_MATERIALS	8
#ifdef LIGHTS
#define EXPECTED_LIGHTS		8
#endif
#define EXPECTED_SOURCES	3
#define EXPECTED_INSTANCES	4

#define DEFAULT_SPHERE_STEPS	4
#define DEFAULT_CONE_STEPS		24

#define TCALNAME	"tmesh.cal"	/* the name of our auxiliary file */

#ifdef ACCELERAD

/* Handles to the objects in a particular node in the scene graph. This is built once per octree and may be referenced multiple times in the scene graph. */
typedef struct SceneNode {
	char *name;						/* The name of this instance */
	RTgeometrygroup group;			/* The geometry group containing this instance */
	RTgeometryinstance instance;	/* The geometry instance kept by this scene */
	RTacceleration acceleration;	/* The acceleration structure kept by this scene */
	RTbuffer vindex_buffer;			/* Buffer containing the indices vertex indices, three entries per triangle */
	RTbuffer material_buffer;		/* Buffer containing the material indices, one entry per triangle */
	int sole_material;				/* The material index to use for all surfaces in this node, or -1 to use separate materials for each object. */
	struct SceneBranch *child;		/* First entry in linked list of child branches */
} SceneNode;

/* Handles to one instance of a node. This is built once per occurance of an octree in the scene except for the top node, which has no transform. */
typedef struct SceneBranch {
	SceneNode *node;				/* The geometry node contained in this branch */
	RTtransform transform;			/* The transformation to be applied to this node */
	struct SceneBranch *sibling;	/* Next entry in linked list of child branches */
} SceneBranch;

/* Handles to buffer data */
typedef struct {
	SceneNode*  root;				/* The current geometry instance being constructed */
	int*        buffer_entry_index;	/* This array gives the OptiX buffer index of each rad file object, or -1 if the object is not in a buffer. The OptiX buffer refered to depends on the type of object. */
	FloatArray* vertices;			/* Three entries per vertex */
	FloatArray* normals;			/* Three entries per vertex */
	FloatArray* tex_coords;			/* Two entries per vertex */
	IntArray*   vertex_indices;		/* Three entries per triangle */
	IntArray*   triangles;			/* One entry per triangle gives material of that triangle */
	PoniterArray* materials;		/* One entry per material */
	IntArray*   alt_materials;		/* Two entries per material gives alternate materials to use in place of that material */
	PoniterArray* instances;		/* One entry per instance */
#ifdef LIGHTS
	IntArray*   lights;				/* Three entries per triangle that is a light */
#endif
	DistantLightArray* sources;		/* One entry per source object */
	unsigned int vertex_index_0;	/* Index of first vertex of current object */
#ifdef CONTRIB
	LUTAB* modifiers;				/* Modifiers for contribution calculations */
#endif
	OBJREC *material;				/* Current material */
} Scene;

static void checkDevices();
static void checkRemoteDevice(RTremotedevice remote);
static void createRemoteDevice(RTremotedevice* remote);
static void applyRadianceSettings(const RTcontext context, const VIEW* view, const unsigned int imm_irrad);
static void createScene(const RTcontext context, SceneNode* root, LUTAB* modifiers);
static void createNode(const RTcontext context, Scene* scene, char* name, const OBJECT start, const OBJECT count, const int material_index, MESH* mesh);
static OBJECT addRadianceObject(const RTcontext context, OBJREC* rec, Scene* scene);
static void createFace(const RTcontext context, OBJREC* rec, Scene* scene);
static __inline void createTriangle(Scene* scene, const int a, const int b, const int c);
#ifdef TRIANGULATE
static int createTriangles(const FACE *face, Scene* scene);
static int addTriangle(const Vert2_list *tp, int a, int b, int c);
#endif /* TRIANGULATE */
static void createSphere(const RTcontext context, OBJREC* rec, Scene* scene);
static void createCone(const RTcontext context, OBJREC* rec, Scene* scene);
static void createMesh(const RTcontext context, MESH* mesh, Scene* scene);
static __inline void setMeshMaterial(const RTcontext context, const OBJECT mo, const OBJECT mat0, Scene* scene);
static void createInstance(const RTcontext context, OBJREC* rec, Scene* scene);
static RTmaterial createNormalMaterial(const RTcontext context, OBJREC* rec, Scene* scene);
static RTmaterial createGlassMaterial(const RTcontext context, OBJREC* rec, Scene* scene);
static RTmaterial createLightMaterial(const RTcontext context, OBJREC* rec, Scene* scene);
#ifdef ANTIMATTER
static RTmaterial createClipMaterial(const RTcontext context, OBJREC* rec, Scene* scene);
#endif
static void createDistantLight(const RTcontext context, OBJREC* rec, Scene* scene);
static OBJREC* findFunction(OBJREC *o);
static int createFunction(const RTcontext context, OBJREC* rec);
static int createTexture(const RTcontext context, OBJREC* rec);
static int createTransform( XF* fxp, XF* bxp, const OBJREC* rec );
static int createGenCumulativeSky(const RTcontext context, char* filename, RTprogram* program);
#ifdef CONTRIB
static char* findSymbol(EPNODE *ep);
static int createContribFunction(const RTcontext context, MODCONT *mp);
static void applyContribution(const RTcontext context, const RTmaterial material, DistantLight* light, OBJREC* rec, Scene* scene);
#endif
static void clearSceneNode(SceneNode *node);
static SceneNode* cloneSceneNode(const RTcontext context, SceneNode *node, const int material_override, Scene* scene);
static void clearSceneBranch(SceneBranch *branch);
static SceneBranch* cloneSceneBranch(const RTcontext context, SceneBranch *branch, const int material_override, Scene* scene);
static RTobject createSceneHierarchy(const RTcontext context, SceneNode* root);
static void createIrradianceGeometry( const RTcontext context );
static void printObject(OBJREC* rec);


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
unsigned int frame = 0u;
RTvariable camera_frame = NULL;
static RTvariable backvis_var = NULL, irrad_var = NULL;
static RTvariable dstrsrc_var = NULL, srcsizerat_var = NULL, directvis_var = NULL;
static RTvariable specthresh_var = NULL, specjitter_var = NULL;
static RTvariable ambounce_var = NULL;
static RTvariable minweight_var = NULL, maxdepth_var = NULL;
static RTvariable camera_type = NULL, camera_eye = NULL, camera_u = NULL, camera_v = NULL, camera_w = NULL, camera_fov = NULL, camera_shift = NULL, camera_clip = NULL, camera_vdist = NULL;
#ifdef VT_ODS
static RTvariable camera_ipd = NULL;
#endif
static RTvariable top_object = NULL, top_ambient = NULL, top_irrad = NULL;
static RTremotedevice remote_handle = NULL;
static RTbuffer vertex_buffer = NULL, normal_buffer = NULL, texcoord_buffer = NULL, material_alt_buffer = NULL, lindex_buffer = NULL, lights_buffer = NULL;
static SceneNode scene_root;

/* Handles to intersection program objects used by multiple materials */
static RTprogram radiance_normal_closest_hit_program, shadow_normal_closest_hit_program, ambient_normal_closest_hit_program, point_cloud_normal_closest_hit_program;
#ifdef ACCELERAD_RT
static RTprogram diffuse_normal_closest_hit_program;
int has_diffuse_normal_closest_hit_program = 0;	/* Flag for including rvu programs. */
#endif
static RTprogram radiance_glass_closest_hit_program, shadow_glass_closest_hit_program, ambient_glass_any_hit_program, point_cloud_glass_closest_hit_program;
static RTprogram radiance_light_closest_hit_program, shadow_light_closest_hit_program, point_cloud_light_closest_hit_program;
#ifdef ANTIMATTER
static RTprogram radiance_clip_closest_hit_program, shadow_clip_closest_hit_program, point_cloud_clip_closest_hit_program;
#endif
#ifdef BSDF
static RTprogram radiance_bsdf_closest_hit_program, shadow_bsdf_closest_hit_program, ambient_bsdf_closest_hit_program, point_cloud_bsdf_closest_hit_program;
#endif

#ifdef DAYSIM
/* Handles to objects used repeatedly for daylight coefficient calulation */
RTbuffer dc_scratch_buffer = NULL;
#endif


/**
 * Check for supported GPU devices.
 */
static void checkDevices()
{
	int driver = 0, runtime = 0;
	unsigned int version = 0, device_count = 0;
	unsigned int i;
	unsigned int multiprocessor_count, threads_per_block, clock_rate, texture_count, timeout_enabled, tcc_driver, cuda_device;
	unsigned int compute_capability[2];
	int major = 1000, minor = 10;
	char device_name[128];
	RTsize memory_size;

	/* Check driver version */
	cudaDriverGetVersion(&driver);
	cudaRuntimeGetVersion(&runtime);
	if (driver < runtime)
		eprintf(INTERNAL, "Current graphics driver %d.%d.%d does not support runtime %d.%d.%d. Update your graphics driver.",
			driver / 1000, (driver % 100) / 10, driver % 10, runtime / 1000, (runtime % 100) / 10, runtime % 10);

	RT_CHECK_WARN_NO_CONTEXT(rtGetVersion(&version));
	RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetDeviceCount(&device_count)); // This will return an error if no supported devices are found
	if (version > 4000) { // Extra digit added in OptiX 4.0.0
		major *= 10;
		minor *= 10;
	}
	mprintf("OptiX %d.%d.%d found driver %d.%d.%d and %i GPU device%s:\n",
		version / major, (version % major) / minor, version % minor, driver / 1000, (driver % 100) / 10, driver % 10,
		device_count, device_count != 1 ? "s" : "");

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
		mprintf("Device %u: %s with %u multiprocessors, %u threads per block, %u kHz, %" PRIu64 " bytes global memory, %u hardware textures, compute capability %u.%u, timeout %sabled, Tesla compute cluster driver %sabled, cuda device %u.%s",
			i, device_name, multiprocessor_count, threads_per_block, clock_rate, memory_size, texture_count, compute_capability[0], compute_capability[1],
			timeout_enabled ? "en" : "dis", tcc_driver ? "en" : "dis", cuda_device, i == device_count - 1 ? "\n\n" : "\n");
	}
}

static void checkRemoteDevice(RTremotedevice remote)
{
	char s[256];
	unsigned int version = 0;
	int major = 1000, minor = 10;
	int i = 0, j;
	RTsize size;

	RT_CHECK_WARN_NO_CONTEXT(rtGetVersion(&version));
	if (version > 4000) { // Extra digit added in OptiX 4.0.0
		major *= 10;
		minor *= 10;
	}
	mprintf("OptiX %d.%d.%d logged into %s as %s\n", version / major, (version % major) / minor, version % minor, optix_remote_url, optix_remote_user);
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
		mprintf("Configuration %i:          %s%s\n", j, s, optix_remote_config == j ? " [active]" : "");
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

static void createRemoteDevice(RTremotedevice* remote)
{
	RTremotedevicestatus ready;
	int first = 1, configs = 0;

	/* Connect to the VCA and reserve nodes */
	RT_CHECK_ERROR_NO_CONTEXT(rtRemoteDeviceCreate(optix_remote_url, optix_remote_user, optix_remote_password, remote));
	RT_CHECK_ERROR_NO_CONTEXT(rtRemoteDeviceGetAttribute(*remote, RT_REMOTEDEVICE_ATTRIBUTE_NUM_CONFIGURATIONS, sizeof(int), &configs));
	if (configs < 1)
		error(INTERNAL, "No compatible VCA configurations");
	if (optix_remote_config >= configs || optix_remote_config < 0)
		error(USER, "Invalid VCA configuration");
	RT_CHECK_ERROR_NO_CONTEXT(rtRemoteDeviceReserve(*remote, optix_remote_nodes, optix_remote_config));
	checkRemoteDevice(*remote);

	/* Wait until the VCA is ready */
	mprintf("Waiting for %s", optix_remote_url);
	do {
		if (first)
			first = 0;
		else {
			mprintf(".");
			sleep(1); // poll once per second.
		}
		RT_CHECK_ERROR_NO_CONTEXT(rtRemoteDeviceGetAttribute(*remote, RT_REMOTEDEVICE_ATTRIBUTE_STATUS, sizeof(RTremotedevicestatus), &ready));
	} while (ready != RT_REMOTEDEVICE_STATUS_READY);
	mprintf("\n");
}

void createContext(RTcontext* context, const RTsize width, const RTsize height, const double alarm)
{
	//RTbuffer seed_buffer;

	//unsigned int* seeds;
	//RTsize i;
	unsigned int ray_type_count, entry_point_count;

	/* Check if irradiance cache is used */
	use_ambient = ambacc > FTINY && ambounce > 0 && ambdiv > 0;
	calc_ambient = use_ambient && nambvals == 0u;// && (ambfile == NULL || !ambfile[0]); // TODO Should really look at ambfp in ambinet.c to check that file is readable

	if ( calc_ambient ) {
		ray_type_count = RAY_TYPE_COUNT;
		entry_point_count = ENTRY_POINT_COUNT;
	} else {
		ray_type_count = RAY_TYPE_COUNT - (use_ambient ? 2 : 3); /* leave out ambient record and point cloud ray types */
		entry_point_count = 1u; /* Generate radiance data */
	}

	/* Setup remote device */
	if (optix_remote_nodes > 0)
		createRemoteDevice(&remote_handle);
	else
		checkDevices();

	/* Setup context */
	RT_CHECK_ERROR2( rtContextCreate( context ) );
	if (remote_handle) RT_CHECK_ERROR2(rtContextSetRemoteDevice(*context, remote_handle));
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
	if (!remote_handle && alarm > 0)
		RT_CHECK_ERROR2( rtContextSetTimeoutCallback( *context, timeoutCallback, alarm ) );
#endif

#ifdef DEBUG_OPTIX
	/* Enable exception checking */
	RT_CHECK_ERROR2( rtContextSetExceptionEnabled( *context, RT_EXCEPTION_ALL, 1 ) );
#endif

#ifdef PRINT_OPTIX
	if (!remote_handle) {
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
void destroyContext(const RTcontext context)
{
	RT_CHECK_ERROR(rtContextDestroy(context));
	if (remote_handle) {
		rtRemoteDeviceRelease(remote_handle);
		rtRemoteDeviceDestroy(remote_handle);
		remote_handle = NULL;
	}
	avsum_var = navsum_var = NULL;
	camera_frame = camera_type = camera_eye = camera_u = camera_v = camera_w = camera_fov = camera_shift = camera_clip = camera_vdist = NULL;
#ifdef VT_ODS
	camera_ipd = NULL;
#endif
}

#ifdef CONTRIB
void makeContribCompatible(const RTcontext context)
{
	RTbuffer contrib_buffer;
#ifdef CONTRIB_DOUBLE
	createCustomBuffer3D(context, RT_BUFFER_OUTPUT, sizeof(double4), 0, 0, 0, &contrib_buffer);
#else
	createBuffer3D(context, RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, 0, 0, 0, &contrib_buffer);
#endif
	applyContextObject(context, "contrib_buffer", contrib_buffer);
}
#endif

#ifdef DAYSIM_COMPATIBLE
void makeDaysimCompatible(const RTcontext context)
{
	RTbuffer dc_buffer;
	setupDaysim(context, &dc_buffer, 0, 0);
}

void setupDaysim(const RTcontext context, RTbuffer* dc_buffer, const RTsize width, const RTsize height)
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

void setupKernel(const RTcontext context, const VIEW* view, LUTAB* modifiers, const RTsize width, const RTsize height, const unsigned int imm_irrad, const double alarm)
{
	/* Setup state */
	applyRadianceSettings(context, view, imm_irrad);
	clearSceneNode(&scene_root);
	createScene(context, &scene_root, modifiers);
	RTobject top = createSceneHierarchy(context, &scene_root);

	/* Set the geometry group as the top level object. */
	top_object = applyContextObject(context, "top_object", top);
	if (!use_ambient)
		top_ambient = applyContextObject(context, "top_ambient", top); // Need to define this because it is referred to by radiance_normal.cu
	if (!imm_irrad)
		top_irrad = applyContextObject(context, "top_irrad", top); // Need to define this because it is referred to by sensor.cu, sensor_cloud_generator, and ambient_cloud_generator.cu

	if ( imm_irrad )
		createIrradianceGeometry( context );

	/* Set up irradiance cache of ambient values */
	if ( use_ambient ) { // Don't bother with ambient records if -aa is set to zero
		if ( calc_ambient ) // Run pre-process only if no ambient file is available
			createAmbientRecords( context, view, width, height, alarm ); // implementation depends on current settings
		else
			setupAmbientCache( context, 0u ); // only need level 0 for final gather
	}
	else
		RT_CHECK_ERROR(rtContextSetEntryPointCount(context, 1u));
}

void updateModel(const RTcontext context, LUTAB* modifiers, const unsigned int imm_irrad)
{
	if (!context) return;
	scene_root.child = NULL; // TODO delete subscenes
	createScene(context, &scene_root, modifiers);
	RTobject top = createSceneHierarchy(context, &scene_root);

	/* Set the geometry group as the top level object. */
	RT_CHECK_ERROR(rtVariableSetObject(top_object, top));
	if (!use_ambient)
		RT_CHECK_ERROR(rtVariableSetObject(top_ambient, top));
	if (!imm_irrad)
		RT_CHECK_ERROR(rtVariableSetObject(top_irrad, top));
}

static void applyRadianceSettings(const RTcontext context, const VIEW* view, const unsigned int imm_irrad)
{
	/* Define ray types */
	applyContextVariable1ui( context, "radiance_primary_ray_type", PRIMARY_RAY );
	applyContextVariable1ui( context, "radiance_ray_type", RADIANCE_RAY );
#ifdef ACCELERAD_RT
	applyContextVariable1ui(context, "diffuse_primary_ray_type", DIFFUSE_PRIMARY_RAY);
	applyContextVariable1ui(context, "diffuse_ray_type", DIFFUSE_RAY);
#endif
	applyContextVariable1ui( context, "shadow_ray_type", SHADOW_RAY );

	/* Set hard coded parameters */
	applyContextVariable3f(context, "CIE_rgbf", (float)CIE_rf, (float)CIE_gf, (float)CIE_bf); // from color.h

	/* Set direct parameters */
	dstrsrc_var = applyContextVariable1f(context, "dstrsrc", (float)dstrsrc); // -dj
	srcsizerat_var = applyContextVariable1f(context, "srcsizerat", (float)srcsizerat); // -ds
	//applyContextVariable1f( context, "shadthresh", shadthresh ); // -dt
	//applyContextVariable1f( context, "shadcert", shadcert ); // -dc
	//applyContextVariable1i( context, "directrelay", directrelay ); // -dr
	//applyContextVariable1i( context, "vspretest", vspretest ); // -dp
	directvis_var = applyContextVariable1i(context, "directvis", directvis); // -dv

	/* Set specular parameters */
	specthresh_var = applyContextVariable1f(context, "specthresh", (float)specthresh); // -st
	specjitter_var = applyContextVariable1f(context, "specjitter", (float)specjitter); // -ss

	/* Set ambient parameters */
	applyContextVariable3f( context, "ambval", ambval[0], ambval[1], ambval[2] ); // -av
	applyContextVariable1i( context, "ambvwt", ambvwt ); // -aw, zero by default
	ambounce_var = applyContextVariable1i(context, "ambounce", ambounce); // -ab
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
	if (maxdepth <= 0 && minweight <= 0.0) // check found in rayorigin() from raytrace.c
		error(USER, "zero ray weight in Russian roulette");
	minweight_var = applyContextVariable1f(context, "minweight", (float)minweight); // -lw, from ray.h
	maxdepth_var = applyContextVariable1i(context, "maxdepth", maxdepth); // -lr, from ray.h, negative values indicate Russian roulette

	if (rand_samp)
		applyContextVariable1ui(context, "random_seed", (unsigned int)random()); // -u

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
#ifdef VT_ODS
		camera_ipd   = applyContextVariable1f(context, "ipd", (float)view->ipd); // -vi
#endif
	}
	else if (imm_irrad)
		applyContextVariable1ui(context, "imm_irrad", imm_irrad); // -I

#ifdef DAYSIM
	/* Set daylight coefficient parameters */
	//applyContextVariable1i(context, "daysimSortMode", daysimSortMode); // -D
	applyContextVariable1ui(context, "daylightCoefficients", daysimGetCoefficients()); // -N
#endif
}

void createCamera(const RTcontext context, const char* ptx_name)
{
	RTprogram program;

	/* Ray generation program */
	ptxFile(path_to_ptx, ptx_name);
	RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "ray_generator", &program));
	irrad_var = applyProgramVariable1ui(context, program, "do_irrad", (unsigned int)do_irrad); // -i
	RT_CHECK_ERROR(rtContextSetRayGenerationProgram(context, RADIANCE_ENTRY, program));

	/* Exception program */
	RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "exception", &program));
	RT_CHECK_ERROR(rtContextSetExceptionProgram(context, RADIANCE_ENTRY, program));

	/* Miss program */
	ptxFile( path_to_ptx, "background" );
	RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "miss", &program));
	RT_CHECK_ERROR(rtContextSetMissProgram(context, PRIMARY_RAY, program)); // only needed when do_irrad is true
	RT_CHECK_ERROR(rtContextSetMissProgram(context, RADIANCE_RAY, program));
#ifdef HIT_TYPE
	applyProgramVariable1ui(context, program, "type", OBJ_SOURCE);
#endif

	/* Miss program for shadow rays */
	RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "miss_shadow", &program));
	RT_CHECK_ERROR(rtContextSetMissProgram(context, SHADOW_RAY, program));
}

void updateCamera( const RTcontext context, const VIEW* view )
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
#ifdef VT_ODS
	RT_CHECK_ERROR(rtVariableSet1f(camera_ipd, (float)view->ipd)); // -vi
#endif
}

static void createScene(const RTcontext context, SceneNode* root, LUTAB* modifiers)
{
	Scene scene;

	/* Timers */
	clock_t geometry_clock = clock();

	/* Set the existing instance and transform for this geometry instance (or null for no instance or transform) */
	scene.root = root;
	scene.buffer_entry_index = NULL;

	/* Create buffers for storing geometry information. */
	scene.vertices = initArrayf(EXPECTED_VERTICES * 3);
	scene.normals = initArrayf(EXPECTED_VERTICES * 3);
	scene.tex_coords = initArrayf(EXPECTED_VERTICES * 2);
	scene.vertex_indices = NULL; //initArrayi(EXPECTED_TRIANGLES * 3);
	scene.triangles = NULL; //initArrayi(EXPECTED_TRIANGLES);
	scene.materials = initArrayp(EXPECTED_MATERIALS);
	scene.alt_materials = initArrayi(EXPECTED_MATERIALS * 2);
	scene.instances = initArrayp(EXPECTED_INSTANCES);

	/* Create buffers for storing lighting information. */
#ifdef LIGHTS
	scene.lights = initArrayi(EXPECTED_LIGHTS * 3);
#endif
	scene.sources = initArraydl(EXPECTED_SOURCES);

	scene.vertex_index_0 = 0u;
#ifdef CONTRIB
	scene.modifiers = modifiers;
#endif
	scene.material = NULL; // TODO Change this for modified instances

	/* Material 0 is Lambertian. */
	if ( do_irrad ) {
		insertArray2i(scene.alt_materials, (int)scene.materials->count, (int)scene.materials->count);
		insertArrayp(scene.materials, createNormalMaterial(context, &Lamb, NULL));
	}

	/* Create the top node of the scene graph. */
	createNode(context, &scene, octname, 0, nobjects, OVOID, NULL);

	/* Free resources used in creating the scene graph. */
	free(scene.buffer_entry_index);
	freeArrayp(scene.instances); //TODO keep this if the instances are not going to change
	freeArrayp(scene.materials);

	/* Apply the geometry buffers. */
	vprintf("Processed %" PRIu64 " vertices.\n", scene.vertices->count / 3);
	if (!vertex_buffer) {
		createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, scene.vertices->count / 3, &vertex_buffer);
		applyContextObject(context, "vertex_buffer", vertex_buffer);
	}
	else
		RT_CHECK_ERROR(rtBufferSetSize1D(vertex_buffer, scene.vertices->count / 3));
	copyToBufferf(context, vertex_buffer, scene.vertices);
	freeArrayf(scene.vertices);

	if (!normal_buffer) {
		createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, scene.normals->count / 3, &normal_buffer);
		applyContextObject(context, "normal_buffer", normal_buffer);
	}
	else
		RT_CHECK_ERROR(rtBufferSetSize1D(normal_buffer, scene.normals->count / 3));
	copyToBufferf(context, normal_buffer, scene.normals);
	freeArrayf(scene.normals);

	if (!texcoord_buffer) {
		createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, scene.tex_coords->count / 2, &texcoord_buffer);
		applyContextObject(context, "texcoord_buffer", texcoord_buffer);
	}
	else
		RT_CHECK_ERROR(rtBufferSetSize1D(texcoord_buffer, scene.tex_coords->count / 2));
	copyToBufferf(context, texcoord_buffer, scene.tex_coords);
	freeArrayf(scene.tex_coords);

	if (!material_alt_buffer) {
		createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_INT2, scene.alt_materials->count / 2, &material_alt_buffer);
		applyContextObject(context, "material_alt_buffer", material_alt_buffer);
	}
	else
		RT_CHECK_ERROR(rtBufferSetSize1D(material_alt_buffer, scene.alt_materials->count / 2));
	copyToBufferi(context, material_alt_buffer, scene.alt_materials);
	freeArrayi(scene.alt_materials);

	/* Apply the lighting buffers. */
#ifdef LIGHTS
	if (scene.lights->count) vprintf("Processed %" PRIu64 " lights.\n", scene.lights->count / 3);
	if (!lindex_buffer) {
		createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, scene.lights->count / 3, &lindex_buffer);
		applyContextObject(context, "lindex_buffer", lindex_buffer);
	}
	else
		RT_CHECK_ERROR(rtBufferSetSize1D(lindex_buffer, scene.lights->count));
	copyToBufferi(context, lindex_buffer, scene.lights);
	freeArrayi(scene.lights);
#endif

	if (scene.sources->count) vprintf("Processed %" PRIu64 " sources.\n", scene.sources->count);
	if (!lights_buffer) {
		createCustomBuffer1D(context, RT_BUFFER_INPUT, sizeof(DistantLight), scene.sources->count, &lights_buffer);
		applyContextObject(context, "lights", lights_buffer);
	}
	else
		RT_CHECK_ERROR(rtBufferSetSize1D(lights_buffer, scene.sources->count));
	copyToBufferdl(context, lights_buffer, scene.sources);
	freeArraydl(scene.sources);

	geometry_clock = clock() - geometry_clock;
	mprintf("Geometry build time: %" PRIu64 " milliseconds for %i objects.\n", MILLISECONDS(geometry_clock), nobjects);
}

static void createNode(const RTcontext context, Scene* scene, char* name, const OBJECT start, const OBJECT count, const int material_index, MESH* mesh)
{
	unsigned int i;
	const OBJECT end = start + count;
	OBJECT on;
	OBJREC *rec;
	RTgeometry geometry = NULL;

	/* Allocate or expand space for entry buffer. */
	if (count > 0) {
		scene->buffer_entry_index = (int *)realloc(scene->buffer_entry_index, sizeof(int) * end);
		if (!scene->buffer_entry_index) eprintf(SYSTEM, "out of memory in createNode, need %" PRIu64 " bytes", sizeof(int) * end);
	}

	/* Temporarily store for parent's lists. */
	IntArray* vertex_indices = scene->vertex_indices;
	scene->vertex_indices = initArrayi(EXPECTED_TRIANGLES * 3);
	IntArray* triangles = scene->triangles;
	scene->triangles = initArrayi(EXPECTED_TRIANGLES);

	/* Create this level of the geometry tree. */
	SceneNode* node = scene->root;
	if (!node) {
		scene->root = node = (SceneNode*)malloc(sizeof(SceneNode));
		if (!node) eprintf(SYSTEM, "out of memory in createNode, need %" PRIu64 " bytes", sizeof(SceneNode));
		clearSceneNode(node);
	}
	node->name = savestr(name);
	insertArrayp(scene->instances, node);
	node->sole_material = material_index;

	/* Get the scene geometry as a list of triangles. */
	for (on = start; on < end; on++) {
		/* By default, no buffer entry is refered to. */
		scene->buffer_entry_index[on] = OVOID;

		rec = objptr(on);
		if (!ismodifier(rec->otype))
			addRadianceObject(context, rec, scene);
	}
	if (mesh)
		createMesh(context, mesh, scene);

	/* Check for overflow */
	if (scene->triangles->count > UINT_MAX)
		eprintf(USER, "Number of triangles %" PRIu64 " in %s is greater than maximum %u.", scene->triangles->count, node->name, UINT_MAX);
	if (scene->materials->count > INT_MAX)
		eprintf(USER, "Number of materials %" PRIu64 " in %s is greater than maximum %u.", scene->materials->count, node->name, INT_MAX);

	/* Create the geometry instance containing the geometry. */
	if (!node->instance) {
		RTprogram program;

		RT_CHECK_ERROR(rtGeometryCreate(context, &geometry));

		ptxFile(path_to_ptx, "triangle_mesh");
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "mesh_bounds", &program));
		RT_CHECK_ERROR(rtGeometrySetBoundingBoxProgram(geometry, program));
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "mesh_intersect", &program));
		RT_CHECK_ERROR(rtGeometrySetIntersectionProgram(geometry, program));
		backvis_var = applyProgramVariable1ui(context, program, "backvis", (unsigned int)backvis); // -bv

		RT_CHECK_ERROR(rtGeometryInstanceCreate(context, &node->instance));
		RT_CHECK_ERROR(rtGeometryInstanceSetGeometry(node->instance, geometry));
	}
	else {
		RT_CHECK_ERROR(rtGeometryInstanceGetGeometry(node->instance, &geometry));
	}
	RT_CHECK_ERROR(rtGeometrySetPrimitiveCount(geometry, (unsigned int)scene->triangles->count));

	/* Check that there is at least one material for context validation. */
	if (!scene->materials->count) {
		RTmaterial null_material;
		RT_CHECK_ERROR(rtMaterialCreate(context, &null_material));
		insertArrayp(scene->materials, null_material);
		use_ambient = calc_ambient = 0u;
	}
	RT_CHECK_ERROR(rtGeometryInstanceSetMaterialCount(node->instance, (unsigned int)scene->materials->count));

	/* Apply materials to the geometry instance. */
	vprintf("Processed %" PRIu64 " materials for %s.\n", scene->materials->count, node->name);
	for (i = 0u; i < scene->materials->count; i++)
		RT_CHECK_ERROR(rtGeometryInstanceSetMaterial(node->instance, i, (RTmaterial)scene->materials->array[i]));

	if (!node->vindex_buffer) {
		createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, scene->vertex_indices->count / 3, &node->vindex_buffer);
		applyGeometryObject(context, geometry, "vindex_buffer", node->vindex_buffer);
	}
	else
		RT_CHECK_ERROR(rtBufferSetSize1D(node->vindex_buffer, scene->vertex_indices->count / 3));
	copyToBufferi(context, node->vindex_buffer, scene->vertex_indices);
	freeArrayi(scene->vertex_indices);
	scene->vertex_indices = vertex_indices;

	vprintf("Processed %" PRIu64 " triangles for %s.\n", scene->triangles->count, node->name);
	if (!node->material_buffer) {
		createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_INT, scene->triangles->count, &node->material_buffer);
		applyGeometryInstanceObject(context, node->instance, "material_buffer", node->material_buffer);
	}
	else
		RT_CHECK_ERROR(rtBufferSetSize1D(node->material_buffer, scene->triangles->count));
	copyToBufferi(context, node->material_buffer, scene->triangles);
	freeArrayi(scene->triangles);
	scene->triangles = triangles;

	/* Set material override */
	if (node->sole_material != OVOID)
		applyGeometryInstanceVariable1i(context, node->instance, "sole_material", node->sole_material);

	/* Create a geometry group to hold the geometry instance. */
	if (!node->group) {
		RT_CHECK_ERROR(rtGeometryGroupCreate(context, &node->group));
		RT_CHECK_ERROR(rtGeometryGroupSetChildCount(node->group, 1));
		RT_CHECK_ERROR(rtGeometryGroupSetChild(node->group, 0, node->instance));
	}

	/* create acceleration object for group and specify some build hints */
	if (!node->acceleration) {
		RT_CHECK_ERROR(rtAccelerationCreate(context, &node->acceleration));
		RT_CHECK_ERROR(rtAccelerationSetBuilder(node->acceleration, "Sbvh"));
		RT_CHECK_ERROR(rtAccelerationSetTraverser(node->acceleration, "Bvh"));
		RT_CHECK_ERROR(rtAccelerationSetProperty(node->acceleration, "vertex_buffer_name", "vertex_buffer")); // For Sbvh only
		RT_CHECK_ERROR(rtAccelerationSetProperty(node->acceleration, "index_buffer_name", "vindex_buffer")); // For Sbvh only
		RT_CHECK_ERROR(rtGeometryGroupSetAcceleration(node->group, node->acceleration));
	}

	/* mark acceleration as dirty */
	RT_CHECK_ERROR(rtAccelerationMarkDirty(node->acceleration));
}

static OBJECT addRadianceObject(const RTcontext context, OBJREC* rec, Scene* scene)
{
	const OBJECT index = objndx(rec);
	int alternate = OVOID;

	if (scene->buffer_entry_index[index] != OVOID) return index; /* Already done */

	switch (rec->otype) {
	case MAT_PLASTIC: // Plastic material
	case MAT_METAL: // Metal material
	case MAT_TRANS: // Translucent material
		scene->buffer_entry_index[index] = insertArray2i(scene->alt_materials, 0, (int)scene->materials->count);
		insertArrayp(scene->materials, createNormalMaterial(context, rec, scene));
		break;
	case MAT_GLASS: // Glass material
	case MAT_DIELECTRIC: // Dielectric material TODO handle separately, see dialectric.c
		scene->buffer_entry_index[index] = insertArray2i(scene->alt_materials, OVOID, (int)scene->materials->count);
		insertArrayp(scene->materials, createGlassMaterial(context, rec, scene));
		break;
	case MAT_LIGHT: // primary light source material, may modify a face or a source (solid angle)
	case MAT_ILLUM: // secondary light source material
	case MAT_GLOW: // Glow material
	case MAT_SPOT: // Spotlight material
		scene->buffer_entry_index[index] = (int)scene->materials->count;
		if (rec->otype != MAT_ILLUM)
			alternate = (int)scene->materials->count;
		else if (rec->oargs.nsargs && strcmp(rec->oargs.sarg[0], VOIDID)) { /* modifies another material */
			//material = objptr( lastmod( objndx(rec), rec->oargs.sarg[0] ) );
			//alternate = buffer_entry_index[objndx(material)];
			alternate = scene->buffer_entry_index[lastmod(objndx(rec), rec->oargs.sarg[0])];
		}
		insertArray2i(scene->alt_materials, (int)scene->materials->count, alternate);
		insertArrayp(scene->materials, createLightMaterial(context, rec, scene));
		break;
#ifdef BSDF
	case MAT_BSDF: // BSDF
		scene->buffer_entry_index[index] = insertArray2i(scene->alt_materials, 0, (int)scene->materials->count);
		insertArrayp(scene->materials, createBSDFMaterial(context, rec, scene));
		break;
#endif
#ifdef ANTIMATTER
	case MAT_CLIP: // Antimatter
		scene->buffer_entry_index[index] = insertArray2i(scene->alt_materials, (int)scene->materials->count, (int)scene->materials->count);
		insertArrayp(scene->materials, createClipMaterial(context, rec, scene));
		break;
#endif
	case OBJ_FACE: // Typical polygons
		createFace(context, rec, scene);
		break;
	case OBJ_SPHERE: // Sphere
	case OBJ_BUBBLE: // Inverted sphere
		createSphere(context, rec, scene);
		break;
	case OBJ_CONE: // Cone
	case OBJ_CUP: // Inverted cone
	case OBJ_CYLINDER: // Cylinder
	case OBJ_TUBE: // Inverted cylinder
	case OBJ_RING: // Disk
		createCone(context, rec, scene);
		break;
	case OBJ_MESH: // Mesh from file
	case OBJ_INSTANCE: // octree instance
		createInstance(context, rec, scene);
		break;
	case OBJ_SOURCE:
		createDistantLight(context, rec, scene);
		break;
	//case MAT_TFUNC: // bumpmap function
	case PAT_BFUNC: // brightness function, used for sky brightness
	case PAT_CFUNC: // color function, used for sky chromaticity
		scene->buffer_entry_index[index] = createFunction(context, rec);
		break;
	//case MAT_TDATA: // bumpmap
	case PAT_BDATA: // brightness texture, used for IES lighting data
	case PAT_CDATA: // color texture
	case PAT_CPICT: // color picture, used as texture map
		scene->buffer_entry_index[index] = createTexture(context, rec);
		break;
	case TEX_FUNC:
		if (rec->oargs.nsargs == 3) {
			if (!strcmp(filename(rec->oargs.sarg[2]), TCALNAME)) break; // Handled by face
		}
		else if (rec->oargs.nsargs >= 4) {
			if (!strcmp(filename(rec->oargs.sarg[3]), TCALNAME)) break; // Handled by face
		}
		printObject(rec);
		break;
	case MOD_ALIAS:
		//if (rec->oargs.nsargs) {
		//	if (rec->oargs.nsargs > 1)
		//		objerror(rec, USER, "too many string arguments");
		//	fprintf(stderr, "Got alias %s %s\n", rec->oname, rec->oargs.sarg[0]);
		//	addRadianceObject(context, objptr(lastmod(objndx(rec), rec->oargs.sarg[0])), objptr(rec->omod), scene); // TODO necessary?
		//}
		// Otherwise it's a pass-through (do nothing)
		break;
	default:
#ifdef DEBUG_OPTIX
		printObject(rec);
#endif
		break;
	}
	return index;
}

static void createFace(const RTcontext context, OBJREC* rec, Scene* scene)
{
	int j, k;
	FACE* face = getface(rec);
	if (face->area == 0.0)
		goto facedone;
	scene->material = findmaterial(rec);
	if (!scene->material) {
		objerror(rec, WARNING, "missing material");
		goto facedone;
	}
	addRadianceObject(context, scene->material, scene);
#ifdef TRIANGULATE
	/* Triangulate the face */
	if (!createTriangles(face, scene)) {
		objerror(rec, WARNING, "triangulation failed");
		goto facedone;
	}
#else /* TRIANGULATE */
	/* Triangulate the polygon as a fan */
	for (j = 2; j < face->nv; j++)
		createTriangle(scene, vertex_index_0, vertex_index_0 + j - 1, vertex_index_0 + j);
#endif /* TRIANGULATE */

	/* Write the vertices to the buffers */
	OBJREC *function = findFunction(rec); // TODO can there be multiple parent functions?
	if (function)
		addRadianceObject(context, function, scene);
	for (j = 0; j < face->nv; j++) {
		RREAL *va = VERTEX(face, j);
		insertArray3f(scene->vertices, (float)va[0], (float)va[1], (float)va[2]);

		if (function && function->otype == TEX_FUNC && function->oargs.nsargs >= 4 && !strcmp(filename(function->oargs.sarg[3]), TCALNAME)) {
			/* Normal calculation from tmesh.cal */
			double bu, bv;
			FVECT v;
			XF fxp;

			k = (int)function->oargs.farg[0];
			bu = va[(k + 1) % 3];
			bv = va[(k + 2) % 3];

			v[0] = bu * function->oargs.farg[1] + bv * function->oargs.farg[2] + function->oargs.farg[3];
			v[1] = bu * function->oargs.farg[4] + bv * function->oargs.farg[5] + function->oargs.farg[6];
			v[2] = bu * function->oargs.farg[7] + bv * function->oargs.farg[8] + function->oargs.farg[9];

			/* Get transform */
			if (!(k = createTransform(&fxp, NULL, function)))
				objerror(function, USER, "bad transform");
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
				insertArray3f(scene->normals, (float)face->norm[0], (float)face->norm[1], (float)face->norm[2]);
			} else
				insertArray3f(scene->normals, (float)v[0], (float)v[1], (float)v[2]);
		}
		else {
			//TODO Implement bump maps from texfunc and texdata
			// This might be done in the Intersecion program in triangle_mesh.cu rather than here
			insertArray3f(scene->normals, (float)face->norm[0], (float)face->norm[1], (float)face->norm[2]);
		}

		if (function && function->otype == TEX_FUNC && function->oargs.nsargs == 3 && !strcmp(filename(function->oargs.sarg[2]), TCALNAME)) {
			/* Texture coordinate calculation from tmesh.cal */
			double bu, bv;

			k = (int)function->oargs.farg[0];
			bu = va[(k + 1) % 3];
			bv = va[(k + 2) % 3];

			insertArray2f(scene->tex_coords,
				(float)(bu * function->oargs.farg[1] + bv * function->oargs.farg[2] + function->oargs.farg[3]),
				(float)(bu * function->oargs.farg[4] + bv * function->oargs.farg[5] + function->oargs.farg[6]));
		}
		else {
			//TODO Implement texture maps from colorfunc, brightfunc, colordata, brightdata, and colorpict
			// This might be done in the Intersecion program in triangle_mesh.cu rather than here
			insertArray2f(scene->tex_coords, 0.0f, 0.0f);
		}
	}
	scene->vertex_index_0 += face->nv;
facedone:
	freeface(rec);
}

static __inline void createTriangle(Scene *scene, const int a, const int b, const int c)
{
	/* Write the indices to the buffers */
	insertArray3i(scene->vertex_indices, a, b, c);
	if (scene->material) {
		insertArrayi(scene->triangles, scene->buffer_entry_index[objndx(scene->material)]);

#ifdef LIGHTS
		if (islight(scene->material->otype) && (scene->material->otype != MAT_GLOW || scene->material->oargs.farg[3] > 0))
			insertArray3i(scene->lights, a, b, c);
#endif /* LIGHTS */
	}
	else {
		insertArrayi(scene->triangles, OVOID);
	}
}

#ifdef TRIANGULATE
/* Generate list of triangle vertex indices from triangulation of face */
static int createTriangles(const FACE *face, Scene *scene)
{
	if (face->nv == 3) {	/* simple case */
		createTriangle(scene, scene->vertex_index_0, scene->vertex_index_0 + 1, scene->vertex_index_0 + 2);
		return(1);
	}
	if (face->nv > 3) {	/* triangulation necessary */
		int i;
		size_t vertex_index_count = scene->vertex_indices->count;
		size_t triangle_count = scene->triangles->count;
#ifdef LIGHTS
		size_t light_count = scene->lights->count;
#endif /* LIGHTS */
		Vert2_list	*v2l = polyAlloc(face->nv);
		if (v2l == NULL)	/* out of memory */
			error(SYSTEM, "out of memory in polyAlloc");
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
		v2l->p = (void *)scene;
		i = polyTriangulate(v2l, addTriangle);
		polyFree(v2l);
		if (!i) { /* triangulation failed */
			scene->vertex_indices->count = vertex_index_count;
			scene->triangles->count = triangle_count;
#ifdef LIGHTS
			scene->lights->count = light_count;
#endif /* LIGHTS */
		}
		return(i);
	}
	return(0);	/* degenerate case */
}

/* Add triangle to polygon's list (call-back function) */
static int addTriangle( const Vert2_list *tp, int a, int b, int c )
{
	Scene *scene = (Scene *)tp->p;
	createTriangle(scene, scene->vertex_index_0 + a, scene->vertex_index_0 + b, scene->vertex_index_0 + c);
	return(1);
}
#endif /* TRIANGULATE */

static void createSphere(const RTcontext context, OBJREC* rec, Scene* scene)
{
	unsigned int i, j, steps;
	int direction = rec->otype == OBJ_SPHERE ? 1 : -1; // Sphere or Bubble
	double radius;
	FVECT x, y, z;
	scene->material = findmaterial(rec);
	if (!scene->material) {
		objerror(rec, WARNING, "missing material");
		return;
	}
	addRadianceObject(context, scene->material, scene);

	// Check radius
	if (rec->oargs.nfargs < 4)
		objerror(rec, USER, "bad # arguments");
	radius = rec->oargs.farg[3];
	if (radius < -FTINY) {
		objerror(rec, WARNING, "negative radius");
		radius = -radius;
		direction = -direction;
	}
	else if (radius <= FTINY) {
		objerror(rec, WARNING, "zero radius");
		return;
	}

	// Create octahedron
	IntArray *sph_vertex_indices = initArrayi(24u);
	FloatArray *sph_vertices = initArrayf(18u);

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

	if (rec->oargs.nfargs > 4) {
		const double numsteps = rec->oargs.farg[4];
		if (numsteps < 0)
			objerror(rec, USER, "resolution too low");
		steps = (unsigned int)numsteps;
	}
	else
		steps = DEFAULT_SPHERE_STEPS;

	// Subdivide triangles
	for (i = 0u; i < steps; i++) {
		IntArray *new_vertex_indices = initArrayi(sph_vertex_indices->count * 4);

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
		sph_vertex_indices = new_vertex_indices;
	}

	// Add resulting triangles
	for (j = 1u; j < sph_vertex_indices->count; j += 3) {
		createTriangle(scene,
			scene->vertex_index_0 + sph_vertex_indices->array[j - direction],
			scene->vertex_index_0 + sph_vertex_indices->array[j],
			scene->vertex_index_0 + sph_vertex_indices->array[j + direction]);
	}

	// Add resulting vertices
	for (j = 0u; j < sph_vertices->count; j += 3) {
		insertArray3f(scene->vertices,
			(float)(rec->oargs.farg[0] + radius * sph_vertices->array[j]),
			(float)(rec->oargs.farg[1] + radius * sph_vertices->array[j + 1]),
			(float)(rec->oargs.farg[2] + radius * sph_vertices->array[j + 2]));
		insertArray3f(scene->normals,
			direction * sph_vertices->array[j],
			direction * sph_vertices->array[j + 1],
			direction * sph_vertices->array[j + 2]);
		insertArray2f(scene->tex_coords, 0.0f, 0.0f);
	}

	scene->vertex_index_0 += (unsigned int)sph_vertices->count / 3;

	// Free memory
	freeArrayi(sph_vertex_indices);
	freeArrayf(sph_vertices);
}

static void createCone(const RTcontext context, OBJREC* rec, Scene* scene)
{
	unsigned int i, j, steps;
	int isCone, direction;
	double theta, sphi, cphi;
	FVECT u, v, n;
	CONE* cone = getcone(rec, 0);
	if (!cone) return;
	scene->material = findmaterial(rec);
	if (!scene->material) {
		objerror(rec, WARNING, "missing material");
		goto conedone;
	}
	addRadianceObject(context, scene->material, scene);

	// Get orthonormal basis
	if (!getperpendicular(u, cone->ad, 0))
		objerror(rec, USER, "bad normal direction");
	VCROSS(v, cone->ad, u);

	isCone = rec->otype != OBJ_CYLINDER && rec->otype != OBJ_TUBE;
	direction = (rec->otype == OBJ_CUP || rec->otype == OBJ_TUBE) ? -1 : 1;

	if (rec->oargs.nfargs > 7 + isCone) { // TODO oconv won't allow extra arguments
		const double numsteps = rec->oargs.farg[7 + isCone];
		if (numsteps < 3)
			objerror(rec, USER, "resolution too low");
		steps = (unsigned int)numsteps;
	}
	else
		steps = DEFAULT_CONE_STEPS;
	
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
		// Add triangles
		if (isCone != 1)
			createTriangle(scene,
				scene->vertex_index_0 + (2 * i + 1 - direction) % (2 * steps),
				scene->vertex_index_0 + (2 * i + 1 + direction) % (2 * steps),
				scene->vertex_index_0 + (2 * i + 1) % (2 * steps));
		if (isCone != 2)
			createTriangle(scene,
				scene->vertex_index_0 + (2 * i + 2) % (2 * steps),
				scene->vertex_index_0 + (2 * i + 2 + direction) % (2 * steps),
				scene->vertex_index_0 + (2 * i + 2 - direction) % (2 * steps));

		// Add vertices
		theta = PI * 2 * i / steps;
		for (j = 0u; j < 3; j++)
			n[j] = u[j] * cos(theta) + v[j] * sin(theta);
		insertArray3f(scene->vertices,
			(float)(CO_P0(cone)[0] + CO_R0(cone) * n[0]),
			(float)(CO_P0(cone)[1] + CO_R0(cone) * n[1]),
			(float)(CO_P0(cone)[2] + CO_R0(cone) * n[2]));
		insertArray3f(scene->vertices,
			(float)(CO_P1(cone)[0] + CO_R1(cone) * n[0]),
			(float)(CO_P1(cone)[1] + CO_R1(cone) * n[1]),
			(float)(CO_P1(cone)[2] + CO_R1(cone) * n[2]));
		for (j = 0u; j < 3; j++)
			n[j] = direction * (sphi * cone->ad[j] + cphi * n[j]);
		insertArray3f(scene->normals, (float)n[0], (float)n[1], (float)n[2]);
		insertArray3f(scene->normals, (float)n[0], (float)n[1], (float)n[2]);
		insertArray2f(scene->tex_coords, 0.0f, 0.0f);
		insertArray2f(scene->tex_coords, 0.0f, 0.0f);
	}
	scene->vertex_index_0 += steps * 2;
conedone:
	freecone(rec);
}

static void createMesh(const RTcontext context, MESH* mesh, Scene* scene)
{
	int j, k;
	unsigned int vertex_index_mesh = scene->vertex_index_0; //TODO what if not all patches are full?
	for (j = 0; j < mesh->npatches; j++) {
		MESHPATCH *pp = &mesh->patch[j];
		scene->material = NULL;
		if (!pp->trimat)
			setMeshMaterial(context, pp->solemat, mesh->mat0, scene);

		/* Write the indices to the buffers */
		for (k = 0; k < pp->ntris; k++) {
			if (pp->trimat)
				setMeshMaterial(context, pp->trimat[k], mesh->mat0, scene);
			createTriangle(scene,
				scene->vertex_index_0 + pp->tri[k].v1,
				scene->vertex_index_0 + pp->tri[k].v2,
				scene->vertex_index_0 + pp->tri[k].v3);
		}

		for (k = 0; k < pp->nj1tris; k++) {
			if (pp->trimat)
				setMeshMaterial(context, pp->j1tri[k].mat, mesh->mat0, scene);
			createTriangle(scene,
				vertex_index_mesh + pp->j1tri[k].v1j,
				scene->vertex_index_0 + pp->j1tri[k].v2,
				scene->vertex_index_0 + pp->j1tri[k].v3);
		}

		for (k = 0; k < pp->nj2tris; k++) {
			if (pp->trimat)
				setMeshMaterial(context, pp->j2tri[k].mat, mesh->mat0, scene);
			createTriangle(scene,
				vertex_index_mesh + pp->j2tri[k].v1j,
				vertex_index_mesh + pp->j2tri[k].v2j,
				scene->vertex_index_0 + pp->j2tri[k].v3);
		}

		/* Write the vertices to the buffers */
		for (k = 0; k < pp->nverts; k++) {
			MESHVERT mesh_vert;
			getmeshvert(&mesh_vert, mesh, k + (j << 8), MT_ALL);
			if (!(mesh_vert.fl & MT_V))
				eprintf(INTERNAL, "missing mesh vertices in %s", mesh->name);
			insertArray3f(scene->vertices, (float)mesh_vert.v[0], (float)mesh_vert.v[1], (float)mesh_vert.v[2]);

			if (mesh_vert.fl & MT_N) // TODO what if normal is defined by texture function
				insertArray3f(scene->normals, (float)mesh_vert.n[0], (float)mesh_vert.n[1], (float)mesh_vert.n[2]);
			else
				insertArray3f(scene->normals, 0.0f, 0.0f, 0.0f); //TODO Can this happen?

			if (mesh_vert.fl & MT_UV)
				insertArray2f(scene->tex_coords, (float)mesh_vert.uv[0], (float)mesh_vert.uv[1]);
			else
				insertArray2f(scene->tex_coords, 0.0f, 0.0f);
		}

		scene->vertex_index_0 += pp->nverts;
	}
}

static __inline void setMeshMaterial(const RTcontext context, const OBJECT mo, const OBJECT mat0, Scene* scene)
{
	if (mo != OVOID) {
		scene->material = findmaterial(objptr(mo + mat0));
		addRadianceObject(context, scene->material, scene);
	}
}

static void createInstance(const RTcontext context, OBJREC* rec, Scene* scene)
{
	INSTANCE *inst = NULL;
	MESHINST *meshinst = NULL;
	FULLXF *trans; // Transform for this instance
	char *name; // File name
	SceneNode *twin = NULL;
	int material_override, i;

	SceneBranch *branch = (SceneBranch*)malloc(sizeof(SceneBranch)); //TODO what can be reused?
	if (!branch) eprintf(SYSTEM, "Out of memory in createInstance, need %" PRIu64 " bytes", sizeof(SceneBranch));
	clearSceneBranch(branch);

	/* Read the name and transform of the instance */
	if (rec->otype == OBJ_INSTANCE) {
		inst = getinstance(rec, IO_CHECK);
		trans = &inst->x;
		name = inst->obj->name;
	}
	else {
		meshinst = getmeshinst(rec, IO_CHECK);
		trans = &meshinst->x;
		name = meshinst->msh->name;
	}

	/* Get the material for the instance, if any */
	material_override = scene->root->sole_material;
	if (material_override == OVOID) {
		scene->material = findmaterial(rec);
		if (scene->material)
			material_override = scene->buffer_entry_index[addRadianceObject(context, scene->material, scene)];
	}

	/* Check if instance has already been loaded */
	for (i = 0; i < scene->instances->count; i++) {
		SceneNode *node = (SceneNode*)scene->instances->array[i];
		if (!strcmp(name, node->name)) {
			twin = node;
			if (node->sole_material == material_override) {
				branch->node = node;
				break;
			}
		}
	}
	if (!twin) {
		// Create a new instance
		SceneNode *root = scene->root;
		scene->root = NULL;
		if (inst) {
			inst = getinstance(rec, IO_SCENE); // Load the octree for the instance
			createNode(context, scene, name, inst->obj->firstobj, inst->obj->nobjs, material_override, NULL);
		}
		else {
			meshinst = getmeshinst(rec, IO_SCENE | IO_BOUNDS); // Load the mesh file for the instance
			createNode(context, scene, name, meshinst->msh->mat0, meshinst->msh->nmats, material_override, meshinst->msh);
		}
		branch->node = scene->root;
		scene->root = root;
	}
	else if (!branch->node) {
		// Create a copy with different materials
		branch->node = cloneSceneNode(context, twin, material_override, scene);
	}

	// Get the transform of the new instance
	if (rec->oargs.nsargs > 1) {
		const float ft[16] = {
			(float)trans->f.xfm[0][0], (float)trans->f.xfm[1][0], (float)trans->f.xfm[2][0], (float)trans->f.xfm[3][0],
			(float)trans->f.xfm[0][1], (float)trans->f.xfm[1][1], (float)trans->f.xfm[2][1], (float)trans->f.xfm[3][1],
			(float)trans->f.xfm[0][2], (float)trans->f.xfm[1][2], (float)trans->f.xfm[2][2], (float)trans->f.xfm[3][2],
			(float)trans->f.xfm[0][3], (float)trans->f.xfm[1][3], (float)trans->f.xfm[2][3], (float)trans->f.xfm[3][3]
		};
		const float bt[16] = {
			(float)trans->b.xfm[0][0], (float)trans->b.xfm[1][0], (float)trans->b.xfm[2][0], (float)trans->b.xfm[3][0],
			(float)trans->b.xfm[0][1], (float)trans->b.xfm[1][1], (float)trans->b.xfm[2][1], (float)trans->b.xfm[3][1],
			(float)trans->b.xfm[0][2], (float)trans->b.xfm[1][2], (float)trans->b.xfm[2][2], (float)trans->b.xfm[3][2],
			(float)trans->b.xfm[0][3], (float)trans->b.xfm[1][3], (float)trans->b.xfm[2][3], (float)trans->b.xfm[3][3]
		};
		RT_CHECK_ERROR(rtTransformCreate(context, &branch->transform));
		RT_CHECK_ERROR(rtTransformSetMatrix(branch->transform, 0, ft, bt));
	}

	// Add the child node to the tree structure
	if (!scene->root->child)
		scene->root->child = branch;
	else {
		SceneBranch *insertion = scene->root->child;
		while (insertion->sibling)
			insertion = insertion->sibling;
		insertion->sibling = branch;
	}

	if (inst)
		freeinstance(rec);
	else
		freemeshinst(rec);
}

#ifdef BSDF
static RTmaterial createBSDFMaterial(const RTcontext context, OBJREC* rec, Scene* scene)
{
	RTmaterial material;
	MFUNC *mf;
	SDData *sd;
	double thick;
	FVECT upvec;

	if (rec->oargs.nsargs < 6 || rec->oargs.nfargs > 9 || rec->oargs.nfargs % 3)
		objerror(rec, USER, "bad number of arguments");
	if (strcmp(rec->oargs.sarg[5], ".")) {
		printObject(rec);
		return NULL; // TODO accept null ouptut
	}
	mf = getfunc(rec, 5, 0x1d, 1);
	sd = loadBSDF(rec->oargs.sarg[1]);
	thick = evalue(mf->ep[0]);
	upvec[0] = evalue(mf->ep[1]);
	upvec[1] = evalue(mf->ep[2]);
	upvec[2] = evalue(mf->ep[3]);
	if (-FTINY <= thick && thick <= FTINY)
		thick = 0.0;

	/* return to world coords */
	if (mf->fxp != &unitxf) {
		multv3(upvec, upvec, mf->fxp->xfm);
		thick *= mf->fxp->sca;
	}

	/* Create our material */
	RT_CHECK_ERROR(rtMaterialCreate(context, &material));

	/* Set variables to be consumed by material for this geometry instance */
	applyMaterialVariable1f(context, material, "thick", (float)thick);
	applyMaterialVariable3f(context, material, "up", (float)upvec[0], (float)upvec[1], (float)upvec[2]);
	if (rec->oargs.nfargs >= 3)
		applyMaterialVariable3f(context, material, "front", (float)rec->oargs.farg[0], (float)rec->oargs.farg[1], (float)rec->oargs.farg[2]);
	if (rec->oargs.nfargs >= 6)
		applyMaterialVariable3f(context, material, "back", (float)rec->oargs.farg[3], (float)rec->oargs.farg[4], (float)rec->oargs.farg[5]);
	if (rec->oargs.nfargs == 9)
		applyMaterialVariable3f(context, material, "trans", (float)rec->oargs.farg[6], (float)rec->oargs.farg[7], (float)rec->oargs.farg[8]);
#ifdef CONTRIB
	applyContribution(context, material, NULL, rec, scene);
#endif

	/* Create our hit programs to be shared among all normal materials */
	if (!radiance_bsdf_closest_hit_program || (!shadow_bsdf_closest_hit_program && thick == 0.0))
		ptxFile(path_to_ptx, "radiance_bsdf");

	//TODO shadow program only if thickness == 0
	if (!radiance_bsdf_closest_hit_program)
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_radiance", &radiance_bsdf_closest_hit_program));
	RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, RADIANCE_RAY, radiance_bsdf_closest_hit_program));

	if (thick == 0.0) {
		if (do_irrad)
			RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, PRIMARY_RAY, radiance_bsdf_closest_hit_program));
		if (!shadow_bsdf_closest_hit_program)
			RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_shadow", &shadow_bsdf_closest_hit_program));
		RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, SHADOW_RAY, shadow_bsdf_closest_hit_program));
	}

//#ifdef ACCELERAD_RT
//	if (!has_diffuse_normal_closest_hit_program) // Don't create the program if it won't be used
//		diffuse_bsdf_closest_hit_program = radiance_bsdf_closest_hit_program;
//	if (!diffuse_bsdf_closest_hit_program) {
//		ptxFile(path_to_ptx, "diffuse_bsdf");
//		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_radiance", &diffuse_bsdf_closest_hit_program));
//	}
//	if (do_irrad)
//		RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, DIFFUSE_PRIMARY_RAY, diffuse_bsdf_closest_hit_program));
//	RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, DIFFUSE_RAY, diffuse_bsdf_closest_hit_program));
//#endif

	if (calc_ambient) {
		if (!ambient_bsdf_closest_hit_program) {
			ptxFile(path_to_ptx, "ambient_bsdf");
			RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_ambient", &ambient_bsdf_closest_hit_program));

#ifdef AMBIENT_CELL
			createAmbientDynamicStorage(context, ambient_bsdf_closest_hit_program, 0);
#else
			createAmbientDynamicStorage(context, ambient_bsdf_closest_hit_program, cuda_kmeans_clusters);
#endif
		}
		RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, AMBIENT_RECORD_RAY, ambient_bsdf_closest_hit_program));

		if (thick == 0.0) {
			if (!point_cloud_bsdf_closest_hit_program) {
				ptxFile(path_to_ptx, "point_cloud_bsdf");
				RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_point_cloud_normal", &point_cloud_bsdf_closest_hit_program));
			}
			RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, POINT_CLOUD_RAY, point_cloud_bsdf_closest_hit_program));
		}
	}

	return material;
}
#endif /* BSDF */

static RTmaterial createNormalMaterial(const RTcontext context, OBJREC* rec, Scene* scene)
{
	RTmaterial material;

	/* Create our material */
	RT_CHECK_ERROR(rtMaterialCreate(context, &material));

	/* Set variables to be consumed by material for this geometry instance */
	applyMaterialVariable1ui(context, material, "type", rec->otype);
	applyMaterialVariable3f(context, material, "color", (float)rec->oargs.farg[0], (float)rec->oargs.farg[1], (float)rec->oargs.farg[2]);
	applyMaterialVariable1f(context, material, "spec", (float)rec->oargs.farg[3]);
	applyMaterialVariable1f(context, material, "rough", (float)rec->oargs.farg[4]);
	if (rec->otype == MAT_TRANS) { // it's a translucent material
		applyMaterialVariable1f(context, material, "transm", (float)rec->oargs.farg[5]);
		applyMaterialVariable1f(context, material, "tspecu", (float)rec->oargs.farg[6]);
	}
#ifdef CONTRIB
	applyContribution(context, material, NULL, rec, scene);
#endif

	/* As a shortcut, exclude black materials from ambient calculation */
	if (rec->oargs.farg[0] == 0.0 && rec->oargs.farg[1] == 0.0 && rec->oargs.farg[2] == 0.0 && (rec->otype != MAT_TRANS || rec->oargs.farg[5] == 0.0))
		applyMaterialVariable1ui(context, material, "ambincl", 0u);

	/* Check ambient include/exclude list */
	else if (ambincl != -1) {
		char **amblp;
		int in_set = 0;
		for (amblp = amblist; *amblp != NULL; amblp++)
			if (!strcmp(rec->oname, *amblp)) {
				in_set = 1;
				break;
			}
		applyMaterialVariable1ui(context, material, "ambincl", in_set == ambincl);
	}

	/* Create our hit programs to be shared among all normal materials */
	if (!radiance_normal_closest_hit_program || !shadow_normal_closest_hit_program)
		ptxFile(path_to_ptx, "radiance_normal");

	if (!radiance_normal_closest_hit_program) {
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_radiance", &radiance_normal_closest_hit_program));
		applyProgramVariable1ui( context, radiance_normal_closest_hit_program, "metal", MAT_METAL );
	}
	if (do_irrad)
		RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, PRIMARY_RAY, radiance_normal_closest_hit_program));
	RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, RADIANCE_RAY, radiance_normal_closest_hit_program));

	if (!shadow_normal_closest_hit_program)
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_shadow", &shadow_normal_closest_hit_program));
	RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, SHADOW_RAY, shadow_normal_closest_hit_program));

#ifdef ACCELERAD_RT
	if (!has_diffuse_normal_closest_hit_program) // Don't create the program if it won't be used
		diffuse_normal_closest_hit_program = radiance_normal_closest_hit_program;
	if (!diffuse_normal_closest_hit_program) {
		ptxFile(path_to_ptx, "diffuse_normal");
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_radiance", &diffuse_normal_closest_hit_program));
	}
	if (do_irrad)
		RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, DIFFUSE_PRIMARY_RAY, diffuse_normal_closest_hit_program));
	RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, DIFFUSE_RAY, diffuse_normal_closest_hit_program));
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

		if (!point_cloud_normal_closest_hit_program) {
			ptxFile( path_to_ptx, "point_cloud_normal" );
			RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_point_cloud_normal", &point_cloud_normal_closest_hit_program));
		}
		RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, POINT_CLOUD_RAY, point_cloud_normal_closest_hit_program));
	}

	return material;
}

static RTmaterial createGlassMaterial(const RTcontext context, OBJREC* rec, Scene* scene)
{
	RTmaterial material;

	/* Create our material */
	RT_CHECK_ERROR(rtMaterialCreate(context, &material));

	/* Set variables to be consumed by material for this geometry instance */
#ifdef HIT_TYPE
	applyMaterialVariable1ui(context, material, "type", rec->otype);
#endif
	applyMaterialVariable3f(context, material, "color", (float)rec->oargs.farg[0], (float)rec->oargs.farg[1], (float)rec->oargs.farg[2]);
	if (rec->oargs.nfargs > 3)
		applyMaterialVariable1f(context, material, "r_index", (float)rec->oargs.farg[3]);
#ifdef CONTRIB
	applyContribution(context, material, NULL, rec, scene);
#endif

	/* Create our hit programs to be shared among all glass materials */
	if (!radiance_glass_closest_hit_program || !shadow_glass_closest_hit_program)
		ptxFile(path_to_ptx, "radiance_glass");

	if (!radiance_glass_closest_hit_program)
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_radiance", &radiance_glass_closest_hit_program));
	RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, RADIANCE_RAY, radiance_glass_closest_hit_program));
#ifdef ACCELERAD_RT
	RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, DIFFUSE_RAY, radiance_glass_closest_hit_program));
#endif

	if (!shadow_glass_closest_hit_program)
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_shadow", &shadow_glass_closest_hit_program));
	RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( material, SHADOW_RAY, shadow_glass_closest_hit_program ) );

	if ( calc_ambient ) {
		if ( !ambient_glass_any_hit_program ) {
			ptxFile( path_to_ptx, "ambient_normal" );
			RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "any_hit_ambient_glass", &ambient_glass_any_hit_program ) );
		}
		RT_CHECK_ERROR( rtMaterialSetAnyHitProgram( material, AMBIENT_RECORD_RAY, ambient_glass_any_hit_program ) );

		if (!point_cloud_glass_closest_hit_program) {
			ptxFile( path_to_ptx, "point_cloud_normal" );
			RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_point_cloud_glass", &point_cloud_glass_closest_hit_program));
		}
		RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, POINT_CLOUD_RAY, point_cloud_glass_closest_hit_program));
	}

	return material;
}

static RTmaterial createLightMaterial(const RTcontext context, OBJREC* rec, Scene* scene)
{
	RTmaterial material;
	OBJREC* mat;

	/* Create our material */
	RT_CHECK_ERROR(rtMaterialCreate(context, &material));

	/* Set variables to be consumed by material for this geometry instance */
#ifdef HIT_TYPE
	applyMaterialVariable1ui(context, material, "type", rec->otype);
#endif
	applyMaterialVariable3f(context, material, "color", (float)rec->oargs.farg[0], (float)rec->oargs.farg[1], (float)rec->oargs.farg[2]);
	if (rec->otype == MAT_GLOW)
		applyMaterialVariable1f(context, material, "maxrad", (float)rec->oargs.farg[3]);
	else if (rec->otype == MAT_SPOT) {
		SPOT* spot = makespot(rec);
		applyMaterialVariable1f(context, material, "siz", spot->siz);
		applyMaterialVariable1f(context, material, "flen", spot->flen);
		applyMaterialVariable3f(context, material, "aim", (float)spot->aim[0], (float)spot->aim[1], (float)spot->aim[2]);
		free(spot);
		rec->os = NULL;
	}
#ifdef CONTRIB
	applyContribution(context, material, NULL, rec, scene);
#endif

	/* Check for a parent function. */
	if ((mat = findFunction(rec))) // TODO can there be multiple parent functions?
		applyMaterialVariable1i(context, material, "function", scene->buffer_entry_index[addRadianceObject(context, mat, scene)]);
	else
		applyMaterialVariable1i(context, material, "function", RT_PROGRAM_ID_NULL);

	/* Create our hit programs to be shared among all light materials */
	if (!radiance_light_closest_hit_program || !shadow_light_closest_hit_program)
		ptxFile(path_to_ptx, "radiance_light");

	if (!radiance_light_closest_hit_program)
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_radiance", &radiance_light_closest_hit_program));
	if (do_irrad)
		RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, PRIMARY_RAY, radiance_light_closest_hit_program));
	RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, RADIANCE_RAY, radiance_light_closest_hit_program));

	if (!shadow_light_closest_hit_program)
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_shadow", &shadow_light_closest_hit_program));
	RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, SHADOW_RAY, shadow_light_closest_hit_program));

	if (calc_ambient) {
		if (!point_cloud_light_closest_hit_program) {
			ptxFile(path_to_ptx, "point_cloud_normal");
			RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_point_cloud_light", &point_cloud_light_closest_hit_program));
		}
		RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, POINT_CLOUD_RAY, point_cloud_light_closest_hit_program));
	}

	return material;
}

#ifdef ANTIMATTER
static RTmaterial createClipMaterial(const RTcontext context, OBJREC* rec, Scene* scene)
{
	RTmaterial material;
	OBJECT mod;
	unsigned int mask = 0u;
	int i, index;

	/* Determine the material mask */
	for (i = 0; i < rec->oargs.nsargs; i++) {
		if (!strcmp(rec->oargs.sarg[i], VOIDID))
			continue;
		if ((mod = lastmod(objndx(rec), rec->oargs.sarg[i])) == OVOID) {
			sprintf(errmsg, "unknown modifier \"%s\"", rec->oargs.sarg[i]);
			objerror(rec, WARNING, errmsg);
			continue;
		}
		if ((index = scene->buffer_entry_index[mod]) > CHAR_BIT * sizeof(mask)) {
			sprintf(errmsg, "out of range modifier \"%s\"", rec->oargs.sarg[i]);
			objerror(rec, WARNING, errmsg);
			continue;
		}
		if (mask & (1 << index)) {
			objerror(rec, WARNING, "duplicate modifier");
			continue;
		}
		mask |= 1 << index;
	}
	if (!mask)
		objerror(rec, USER, "no modifiers clipped");

	/* Create our material */
	RT_CHECK_ERROR(rtMaterialCreate(context, &material));

	/* Set variables to be consumed by material for this geometry instance */
#ifdef HIT_TYPE
	applyMaterialVariable1ui(context, material, "type", rec->otype);
#endif
	applyMaterialVariable1ui(context, material, "mask", mask);

	/* Create our hit programs to be shared among all materials */
	if (!radiance_clip_closest_hit_program || !shadow_clip_closest_hit_program)
		ptxFile(path_to_ptx, "clip");

	if (!radiance_clip_closest_hit_program)
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_radiance", &radiance_clip_closest_hit_program));
	if (do_irrad)
		RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, PRIMARY_RAY, radiance_clip_closest_hit_program));
	RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, RADIANCE_RAY, radiance_clip_closest_hit_program));
#ifdef ACCELERAD_RT
	if (do_irrad)
		RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, DIFFUSE_PRIMARY_RAY, radiance_clip_closest_hit_program));
	RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, DIFFUSE_RAY, radiance_clip_closest_hit_program));
#endif

	if (!shadow_clip_closest_hit_program)
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_shadow", &shadow_clip_closest_hit_program));
	RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, SHADOW_RAY, shadow_clip_closest_hit_program));

	if (calc_ambient) {
		if (!point_cloud_clip_closest_hit_program)
			RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_point_cloud", &point_cloud_clip_closest_hit_program));
		RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(material, POINT_CLOUD_RAY, point_cloud_clip_closest_hit_program));
	}

	return material;
}
#endif

static void createDistantLight(const RTcontext context, OBJREC* rec, Scene* scene)
{
	SRCREC source;
	DistantLight light;
	OBJREC* material = findmaterial(rec);
	if (!material) {
		objerror(rec, WARNING, "missing material");
		return;
	}

	ssetsrc(&source, rec);
	array2cuda3(light.color, material->oargs.farg); // TODO these are given in RGB radiance value (watts/steradian/m2)
	array2cuda3(light.pos, source.sloc);
	light.solid_angle = source.ss2;
	light.casts_shadow = material->otype != MAT_GLOW; // Glow cannot cast shadow infinitely far away

#ifdef CONTRIB
	applyContribution(context, NULL, &light, material, scene);
#endif /* CONTRIB */

	/* Check for a parent function. */
	if ((material = findFunction(rec))) // TODO can there be multiple parent functions?
		light.function = scene->buffer_entry_index[addRadianceObject(context, material, scene)];
	else
		light.function = RT_PROGRAM_ID_NULL;

	insertArraydl(scene->sources, light);
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
		if (!strcmp(rec->oargs.sarg[0], "skybright")) {
			if (!strcmp(filename(rec->oargs.sarg[1]), "perezlum.cal")) {
				float coef[5] = { (float)rec->oargs.farg[2], (float)rec->oargs.farg[3], (float)rec->oargs.farg[4], (float)rec->oargs.farg[5], (float)rec->oargs.farg[6] };
				ptxFile(path_to_ptx, "perezlum");
				RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, rec->oargs.sarg[0], &program));
				applyProgramVariable1f(context, program, "diffuse", (float)rec->oargs.farg[0]);
				applyProgramVariable1f(context, program, "ground", (float)rec->oargs.farg[1]);
				applyProgramVariable(context, program, "coef", sizeof(coef), coef);
				applyProgramVariable3f(context, program, "sun", (float)rec->oargs.farg[7], (float)rec->oargs.farg[8], (float)rec->oargs.farg[9]);
			}
			else if (!strcmp(filename(rec->oargs.sarg[1]), "isotrop_sky.cal")) {
				/* Isotropic sky from daysim installation */
				ptxFile(path_to_ptx, "isotropsky");
				RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, rec->oargs.sarg[0], &program));
				applyProgramVariable1f(context, program, "radiance", (float)rec->oargs.farg[0]);
			}
			else if (!createGenCumulativeSky(context, rec->oargs.sarg[1], &program)) {
				printObject(rec);
				return RT_PROGRAM_ID_NULL;
			}
		}
		else if (!strcmp(rec->oargs.sarg[0], "skybr")) {
			if (!strcmp(filename(rec->oargs.sarg[1]), "skybright.cal")) {
				ptxFile(path_to_ptx, "skybright");
				RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, rec->oargs.sarg[0], &program));
				applyProgramVariable1ui(context, program, "type", (unsigned int)rec->oargs.farg[0]);
				applyProgramVariable1f(context, program, "zenith", (float)rec->oargs.farg[1]);
				applyProgramVariable1f(context, program, "ground", (float)rec->oargs.farg[2]);
				applyProgramVariable1f(context, program, "factor", (float)rec->oargs.farg[3]);
				applyProgramVariable3f(context, program, "sun", (float)rec->oargs.farg[4], (float)rec->oargs.farg[5], (float)rec->oargs.farg[6]);
			}
			else if (!strcmp(filename(rec->oargs.sarg[1]), "utah.cal")) {
				/* Preetham sky brightness from Mark Stock */
				ptxFile(path_to_ptx, "utah");
				RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, rec->oargs.sarg[0], &program));
				applyProgramVariable1ui(context, program, "monochrome", 1u);
				applyProgramVariable1f(context, program, "turbidity", (float)rec->oargs.farg[0]);
				applyProgramVariable3f(context, program, "sun", (float)rec->oargs.farg[1], (float)rec->oargs.farg[2], (float)rec->oargs.farg[3]);
			}
			else {
				printObject(rec);
				return RT_PROGRAM_ID_NULL;
			}
		}
		else if (rec->oargs.nsargs >= 4 && !strcmp(rec->oargs.sarg[0], "skyr") && !strcmp(rec->oargs.sarg[1], "skyg") && !strcmp(rec->oargs.sarg[2], "skyb") && !strcmp(filename(rec->oargs.sarg[3]), "utah.cal")) {
			/* Preetham sky from Mark Stock */
			ptxFile(path_to_ptx, "utah");
			RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "skybr", &program));
			applyProgramVariable1f(context, program, "turbidity", (float)rec->oargs.farg[0]);
			applyProgramVariable3f(context, program, "sun", (float)rec->oargs.farg[1], (float)rec->oargs.farg[2], (float)rec->oargs.farg[3]);
		}
		else {
			printObject(rec);
			return RT_PROGRAM_ID_NULL;
		}
		applyProgramVariable(context, program, "transform", sizeof(transform), transform);
	}
	else {
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

	DATARRAY *dp, *dpg = NULL, *dpb = NULL;
	XF bxp;

	/* Load texture data */
	if (rec->otype == PAT_BDATA)
		dp = getdata(rec->oargs.sarg[1]);
	else if (rec->otype == MAT_TDATA || rec->otype == PAT_CDATA) {
		dp = getdata(rec->oargs.sarg[3]); // red
		dpg = getdata(rec->oargs.sarg[4]); // green
		dpb = getdata(rec->oargs.sarg[5]); // blue
	}
	else if (rec->otype == PAT_CPICT)
		dp = getpict(rec->oargs.sarg[3]);
	else {
		printObject(rec);
		return RT_PROGRAM_ID_NULL;
	}

	/* Get transform */
	if ( !createTransform( NULL, &bxp, rec ) )
		objerror(rec, USER, "bad transform");

	/* Create buffer for texture data */
	if (dp->type == DATATY) // floats
		tex_format = RT_FORMAT_FLOAT;
	else // colors
		tex_format = RT_FORMAT_FLOAT4;
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
	else if (dpg != NULL && dpb != NULL) // colors from separate files
		for (i = 0u; i < entries; i++) {
			tex_buffer_data[0] = 1.0f; // Set alpha
			tex_buffer_data[1] = dp->arr.d[i]; // Set red
			tex_buffer_data[2] = dpg->arr.d[i]; // Set green
			tex_buffer_data[3] = dpb->arr.d[i]; // Set blue
			tex_buffer_data += 4;
		}
	else // colors from image
		for (i = 0u; i < entries; i++) {
			tex_buffer_data[0] = 1.0f; // Set alpha
			colr_color(tex_buffer_data + 1, dp->arr.c[i]);
			tex_buffer_data += 4;
		}
	RT_CHECK_ERROR( rtBufferUnmap( tex_buffer ) );

	/* Create texture sampler */
	RT_CHECK_ERROR( rtTextureSamplerCreate( context, &tex_sampler ) );
	for (i = 0u; i < dp->nd; i++) {
		if (dp->dim[i].p)
			objerror(rec, WARNING, "ignoring point locations");
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
		float transform[9] = {
			(float)bxp.xfm[0][0], (float)bxp.xfm[1][0], (float)bxp.xfm[2][0],
			(float)bxp.xfm[0][1], (float)bxp.xfm[1][1], (float)bxp.xfm[2][1],
			(float)bxp.xfm[0][2], (float)bxp.xfm[1][2], (float)bxp.xfm[2][2]
		};
		if (!strcmp(filename(rec->oargs.sarg[2]), "source.cal")) {
			const char *src_phi = "src_phi", *src_theta = "src_theta";
			const size_t length_phi = strlen(src_phi);
			int transpose = 0;
			double symmetry = 0.0;
			char *sym_str;

			/* Check compatibility with existing implementation */
			if (strcmp(rec->oargs.sarg[0], "corr") && strcmp(rec->oargs.sarg[0], "flatcorr") && strcmp(rec->oargs.sarg[0], "boxcorr") && strcmp(rec->oargs.sarg[0], "cylcorr")) {
				printObject(rec);
				return RT_PROGRAM_ID_NULL;
			}
			if (strncmp(rec->oargs.sarg[3], src_phi, length_phi) || strcmp(rec->oargs.sarg[4], src_theta)) {
				if (strncmp(rec->oargs.sarg[4], src_phi, length_phi) || strcmp(rec->oargs.sarg[3], src_theta)) {
					printObject(rec);
					return RT_PROGRAM_ID_NULL;
				}
				transpose = 1;
			}
			sym_str = &rec->oargs.sarg[3 + transpose][length_phi];
			if (!strcmp(sym_str, "2"))
				symmetry = PI;
			else if (!strcmp(sym_str, "4"))
				symmetry = PI / 2;
			else if (strlen(sym_str)) {
				printObject(rec);
				return RT_PROGRAM_ID_NULL;
			}

			ptxFile( path_to_ptx, "source" );
			RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, rec->oargs.sarg[0], &program));
			applyProgramVariable1i( context, program, "data", tex_id );
			applyProgramVariable1i( context, program, "type", dp->type == DATATY );
			applyProgramVariable3f( context, program, "org", dp->dim[dp->nd-1].org, dp->nd > 1 ? dp->dim[dp->nd-2].org : 0.0f, dp->nd > 2 ? dp->dim[dp->nd-3].org : 0.0f );
			applyProgramVariable3f( context, program, "siz", dp->dim[dp->nd-1].siz, dp->nd > 1 ? dp->dim[dp->nd-2].siz : 1.0f, dp->nd > 2 ? dp->dim[dp->nd-3].siz : 1.0f );
			if (transpose)
				applyProgramVariable1i(context, program, "transpose", transpose);
			if (symmetry > 0.0)
				applyProgramVariable1f(context, program, "symmetry", (float)symmetry);
			if (rec->oargs.nfargs > 0)
				applyProgramVariable1f(context, program, "multiplier", (float)rec->oargs.farg[0]); //TODO handle per-color channel multipliers
			if (rec->oargs.nfargs > 2)
				applyProgramVariable3f(context, program, "bounds", (float)rec->oargs.farg[1], (float)rec->oargs.farg[2], rec->oargs.nfargs > 3 ? (float)rec->oargs.farg[3] : 0.0f);
		}
		else if (rec->oargs.nsargs >= 7 && !strcmp(rec->oargs.sarg[0], "clip_r") && !strcmp(rec->oargs.sarg[1], "clip_g") && !strcmp(rec->oargs.sarg[2], "clip_b") && !strcmp(filename(rec->oargs.sarg[4]), "fisheye.cal") && !strcmp(rec->oargs.sarg[5], "u") && !strcmp(rec->oargs.sarg[6], "v")) {
			/* Sigma fisheye projection from Nathaniel Jones */
			ptxFile(path_to_ptx, "fisheye");
			RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "fisheye", &program));
			applyProgramVariable1i(context, program, "data", tex_id);
			applyProgramVariable1i(context, program, "type", dp->type == DATATY);
		}
		else {
			printObject(rec);
			return RT_PROGRAM_ID_NULL;
		}
		applyProgramVariable(context, program, "transform", sizeof(transform), transform);
	}
	else {
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

static int createGenCumulativeSky(const RTcontext context, char* filename, RTprogram* program)
{
	char  *fname, *line;
	FILE  *fp;
	int success = 0;
	char header[] = "{ This .cal file was generated automatically by GenCumulativeSky }";
	size_t length = strlen(header);

	if ((fname = getpath(filename, getrlibpath(), R_OK)) == NULL) {
		//eprintf(SYSTEM, "cannot find function file \"%s\"", filename);
		return 0;
	}
	if ((fp = fopen(fname, "r")) == NULL) {
		//eprintf(SYSTEM, "cannot open file \"%s\"", filename);
		return 0;
	}

	line = (char*)malloc((length + 1) * sizeof(char));
	if (!line) eprintf(SYSTEM, "out of memory in createGenCumulativeSky, need %" PRIu64 " bytes", (length + 1) * sizeof(char));

	if (fgets(line, (int)length + 1, fp) && !strncmp(header, line, length)) { // It's a GenCumulativeSky file
		RTtexturesampler tex_sampler;
		RTbuffer         tex_buffer;
		float*           tex_buffer_data;
		int              i = 0, j, tex_id = RT_TEXTURE_ID_NULL;

		/* Populate buffer with texture data */
		createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 145, &tex_buffer);
		RT_CHECK_ERROR(rtBufferMap(tex_buffer, (void**)&tex_buffer_data));

		for (j = 0; j < 6; j++) while (fgets(line, 2, fp) && line[0] != '\n'); // Skip 5 lines
		while (i < 30) if(!fscanf(fp, "%f,", tex_buffer_data + i++)) goto gencumerr; // Read 1st row (30 entries)
	
		for (j = 0; j < 4; j++) while (fgets(line, 2, fp) && line[0] != '\n'); // Skip 3 lines
		while (i < 60) if (!fscanf(fp, "%f,", tex_buffer_data + i++)) goto gencumerr; // Read 2nd row (30 entries)

		for (j = 0; j < 4; j++) while (fgets(line, 2, fp) && line[0] != '\n'); // Skip 3 lines
		while (i < 84) if (!fscanf(fp, "%f,", tex_buffer_data + i++)) goto gencumerr; // Read 3rd row (24 entries)

		for (j = 0; j < 4; j++) while (fgets(line, 2, fp) && line[0] != '\n'); // Skip 3 lines
		while (i < 108) if (!fscanf(fp, "%f,", tex_buffer_data + i++)) goto gencumerr; // Read 4th row (24 entries)

		for (j = 0; j < 4; j++) while (fgets(line, 2, fp) && line[0] != '\n'); // Skip 3 lines
		while (i < 126) if (!fscanf(fp, "%f,", tex_buffer_data + i++)) goto gencumerr; // Read 5th row (18 entries)

		for (j = 0; j < 4; j++) while (fgets(line, 2, fp) && line[0] != '\n'); // Skip 3 lines
		while (i < 138) if (!fscanf(fp, "%f,", tex_buffer_data + i++)) goto gencumerr; // Read 6th row (12 entries)

		for (j = 0; j < 4; j++) while (fgets(line, 2, fp) && line[0] != '\n'); // Skip 3 lines
		while (i < 144) if (!fscanf(fp, "%f,", tex_buffer_data + i++)) goto gencumerr; // Read 7th row (6 entries)

		for (j = 0; j < 3; j++) while (fgets(line, 2, fp) && line[0] != '\n'); // Skip 2 lines
		while (fgets(line, 2, fp) && line[0] != ','); // Read past comma
		if (!fscanf(fp, "%f,", tex_buffer_data + i)) goto gencumerr; // Read last row (1 entry)

		RT_CHECK_ERROR(rtBufferUnmap(tex_buffer));

		/* Create texture sampler */
		RT_CHECK_ERROR(rtTextureSamplerCreate(context, &tex_sampler));
		RT_CHECK_ERROR(rtTextureSamplerSetWrapMode(tex_sampler, 0, RT_WRAP_CLAMP_TO_EDGE));
		RT_CHECK_ERROR(rtTextureSamplerSetFilteringModes(tex_sampler, RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE));
		RT_CHECK_ERROR(rtTextureSamplerSetIndexingMode(tex_sampler, RT_TEXTURE_INDEX_ARRAY_INDEX));
		RT_CHECK_ERROR(rtTextureSamplerSetReadMode(tex_sampler, RT_TEXTURE_READ_ELEMENT_TYPE));
		RT_CHECK_ERROR(rtTextureSamplerSetMaxAnisotropy(tex_sampler, 0.0f));
		RT_CHECK_ERROR(rtTextureSamplerSetMipLevelCount(tex_sampler, 1u)); // Currently only one mipmap level supported
		RT_CHECK_ERROR(rtTextureSamplerSetArraySize(tex_sampler, 1u)); // Currently only one element supported
		RT_CHECK_ERROR(rtTextureSamplerSetBuffer(tex_sampler, 0u, 0u, tex_buffer));
		RT_CHECK_ERROR(rtTextureSamplerGetId(tex_sampler, &tex_id));

		/* Create program to access texture sampler */
		ptxFile(path_to_ptx, "gencumulativesky");
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "gencumsky", program));
		applyProgramVariable1i(context, *program, "data", tex_id);

		success = 1;
	}

	free(line);
	fclose(fp);

	return success;

gencumerr:
	eprintf(USER, "bad format in \"%s\"", filename);
	return 0;
}

#ifdef CONTRIB
static char* findSymbol(EPNODE *ep)
{
	static EPNODE  *curdef = NULL;
	EPNODE  *ep1 = NULL;
	char *found = NULL;
	static char num[50];

	switch (ep->type) {
	case VAR:
		if (ep->v.ln->def)
			found = findSymbol(ep->v.ln->def);
		else
			found = ep->v.ln->name;
		break;

	case NUM:
		sprintf(num, "\"%g\"", ep->v.num);
		found = num;
		break;

	case SYM:
		found = ep->v.name;
		break;

	case FUNC:
		found = findSymbol(ep->v.kid);
		break;

	case ARG:
		if (curdef == NULL || curdef->v.kid->type != FUNC || (ep1 = ekid(curdef->v.kid, ep->v.chan)) == NULL) {
			return NULL;
		}
		found = findSymbol(ep1);
		break;

	case UMINUS:
		found = findSymbol(ep->v.kid);
		break;

	case '=':
	case ':':
		ep1 = curdef;
		curdef = ep;
		found = findSymbol(ep->v.kid);
		if (!found)
			found = findSymbol(ep->v.kid->sibling);
		curdef = ep1;
		break;

	case '+':
	case '-':
	case '*':
	case '/':
	case '^':
		found = findSymbol(ep->v.kid);
		if (!found)
			found = findSymbol(ep->v.kid->sibling);
		break;
	}
	return found;
}

static int createContribFunction(const RTcontext context, MODCONT *mp)
{
	RTprogram program;
	int program_id = RT_PROGRAM_ID_NULL;
	char *bin_func;

	if (mp->binv->type == NUM) // Guessing that no program is needed for a single bin
		return RT_PROGRAM_ID_NULL;

	/* Set current definitions */
	set_eparams(mp->params);

	//eprint(mp->binv, stderr); fprintf(stderr, "\n");
	bin_func = findSymbol(mp->binv);
	if (!bin_func) {
		eprintf(WARNING, "Undefined bin function for modifier %s\n", mp->modname);
		return RT_PROGRAM_ID_NULL;
	}

	/* Special case for uniform sampling */
	if (!strcmp(bin_func, "if")) { // It's probably uniform sampling
		FVECT normal;
		normal[0] = normal[1] = normal[2] = 0;
		if (mp->binv->v.kid && mp->binv->v.kid->sibling && mp->binv->v.kid->sibling->v.kid) {
			EPNODE *child = mp->binv->v.kid->sibling->v.kid;
			while (child) {
				if (child->v.kid && child->v.kid->sibling) {
					char *sym = findSymbol(child->v.kid);
					if (!strcmp(sym, "Dx")) normal[0] = child->v.kid->sibling->v.num;
					else if (!strcmp(sym, "Dy")) normal[1] = child->v.kid->sibling->v.num;
					else if (!strcmp(sym, "Dz")) normal[2] = child->v.kid->sibling->v.num;
				}
				child = child->sibling;
			}
		}

		if (normal[0] == 0 && normal[1] == 0 && normal[2] == 0) {
			eprintf(WARNING, "Unrecognized bin function \"%s\" for modifier %s\n", bin_func, mp->modname);
			return RT_PROGRAM_ID_NULL;
		}

		/* Create the bin selection program */
		ptxFile(path_to_ptx, "uniform");
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "front", &program));
		applyProgramVariable3f(context, program, "normal", (float)normal[0], (float)normal[1], (float)normal[2]); // Normal direction
		RT_CHECK_ERROR(rtProgramGetId(program, &program_id));
		return program_id;
	}

	/* Expect PTX file with same name as CAL file */
	if (!mp->file) {
		eprintf(WARNING, "Could not determine function file for modifier %s\n", mp->modname);
		return RT_PROGRAM_ID_NULL;
	}
	ptxFile(path_to_ptx, mp->file);

	/* Create the bin selection program */
	if (mp->nbins == 146 && !strcmp(bin_func, "tbin")) { // It's probably tregenza.cal
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, bin_func, &program));
	}
	else if (!strcmp(bin_func, "scbin")) { // It's probably disk2square.cal
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, bin_func, &program));
		applyProgramVariable1i(context, program, "SCdim", evali("SCdim", 1)); // Side length of square
		applyProgramVariable3f(context, program, "normal", evalf("rNx", 0.0f), evalf("rNy", 0.0f), evalf("rNz", -1.0f)); // Normal direction
		applyProgramVariable3f(context, program, "up", evalf("Ux", 0.0f), evalf("Uy", 1.0f), evalf("Uz", 0.0f)); // Up direction
		applyProgramVariable1i(context, program, "RHS", evali("RHS", 1)); // Coordinate system handedness
	}
	else if (!strcmp(bin_func, "rbin")) { // It's probably reinhart.cal or reinhartb.cal
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, bin_func, &program));
		applyProgramVariable1i(context, program, "mf", evali("MF", 1)); // Number of divisions per Tregenza patch
		if (!strcmp(mp->file, "reinhartb")) {
			applyProgramVariable3f(context, program, "normal", evalf("rNx", 0.0f), evalf("rNy", 0.0f), evalf("rNz", -1.0f)); // Normal direction
			applyProgramVariable3f(context, program, "up", evalf("Ux", 0.0f), evalf("Uy", 1.0f), evalf("Uz", 0.0f)); // Up direction
			applyProgramVariable1i(context, program, "RHS", evali("RHS", 1)); // Coordinate system handedness
		}
	}
	else if (bin_func[0] == 'k') { // It's probably klems_full.cal, klems_half.cal, or klems_quarter.cal
		float orientation[6] = {
			0.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f
		};
		if (mp->binv->type == FUNC) { // It's a function
			EPNODE *child = mp->binv->v.kid;
			int i = 0;
			const size_t count = sizeof(orientation) / sizeof(float);
			while ((child = child->sibling) != NULL && i < count) {
				orientation[i++] = (float)(evalue(child));
			}
			if (child != NULL || i != count) {
				eprintf(WARNING, "Bad arguments to bin function \"%s\" for modifier %s\n", bin_func, mp->modname);
				return RT_PROGRAM_ID_NULL;
			}
		}
		else switch (bin_func[4]) { // Variables from klems_full.cal
		case 'N': orientation[1] = -1.0f; orientation[5] = 1.0f; break;
		case 'E': orientation[0] = -1.0f; orientation[5] = 1.0f; break;
		case 'S': orientation[1] = 1.0f; orientation[5] = 1.0f; break;
		case 'W': orientation[0] = 1.0f; orientation[5] = 1.0f; break;
		case 'D': orientation[2] = -1.0f; orientation[4] = 1.0f; break;
		default:
			eprintf(WARNING, "Undefined bin function \"%s\" for modifier %s\n", bin_func, mp->modname);
			return RT_PROGRAM_ID_NULL;
		}

		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "kbin", &program));
		applyProgramVariable3f(context, program, "normal", orientation[0], orientation[1], orientation[2]); // Normal direction
		applyProgramVariable3f(context, program, "up", orientation[3], orientation[4], orientation[5]); // Up direction
		applyProgramVariable1i(context, program, "RHS", evali("RHS", 1)); // Coordinate system handedness
	}
	else {
		eprintf(WARNING, "Unrecognized bin function \"%s\" for modifier %s\n", bin_func, mp->modname);
		return RT_PROGRAM_ID_NULL;
	}

	RT_CHECK_ERROR(rtProgramGetId(program, &program_id));
	return program_id;
}

static void applyContribution(const RTcontext context, const RTmaterial material, DistantLight* light, OBJREC* rec, Scene* scene)
{
	static char cfunc[] = "contrib_function";

	/* Check for a call-back function. */
	if (scene && scene->modifiers) {
		MODCONT	*mp;
		if ((mp = (MODCONT *)lu_find(scene->modifiers, rec->oname)->data)) {
			if (material) {
				applyMaterialVariable1i(context, material, "contrib_index", mp->start_bin);
				applyMaterialVariable1i(context, material, cfunc, createContribFunction(context, mp));
			}
			else if (light) {
				/* Check for a existing program. */
				int mat_index = scene->buffer_entry_index[objndx(rec)];
				light->contrib_index = mp->start_bin;
				if (mat_index != OVOID) { /* In case this material has also already been used for a surface, we can reuse the function here */
					RTvariable var;
					RT_CHECK_ERROR(rtMaterialQueryVariable((RTmaterial)scene->materials->array[mat_index], cfunc, &var));
					RT_CHECK_ERROR(rtVariableGet1i(var, &light->contrib_function));
				}
				else
					light->contrib_function = createContribFunction(context, mp);
			}
			return;
		}
	}

	/* No call-back function. */
	if (material) {
		//applyMaterialVariable1i(context, material, "contrib_index", -1);
		applyMaterialVariable1i(context, material, cfunc, RT_PROGRAM_ID_NULL);
	}
	else if (light) {
		light->contrib_index = -1;
		light->contrib_function = RT_PROGRAM_ID_NULL;
	}
}
#endif /* CONTRIB */

static void clearSceneNode(SceneNode *node)
{
	node->name = NULL;
	node->group = NULL;
	node->instance = NULL;
	node->acceleration = NULL;
	node->vindex_buffer = NULL;
	node->material_buffer = NULL;
	node->sole_material = OVOID;
	node->child = NULL;
}

static SceneNode* cloneSceneNode(const RTcontext context, SceneNode *node, const int material_override, Scene* scene)
{
	if (!node) return NULL;

	SceneNode *twin = (SceneNode*)malloc(sizeof(SceneNode));
	if (!twin) eprintf(SYSTEM, "out of memory in cloneSceneNode, need %" PRIu64 " bytes", sizeof(SceneNode));
	insertArrayp(scene->instances, twin);

	twin->name = node->name;
	twin->acceleration = node->acceleration;
	twin->vindex_buffer = node->vindex_buffer;
	twin->material_buffer = node->material_buffer;
	twin->sole_material = material_override;

	if (node->instance) {
		RTgeometry geometry;
		unsigned int i;
		RT_CHECK_ERROR(rtGeometryInstanceCreate(context, &twin->instance));
		RT_CHECK_ERROR(rtGeometryInstanceGetGeometry(node->instance, &geometry));
		RT_CHECK_ERROR(rtGeometryInstanceSetGeometry(twin->instance, geometry));
		applyGeometryInstanceObject(context, twin->instance, "material_buffer", twin->material_buffer);

		RT_CHECK_ERROR(rtGeometryInstanceSetMaterialCount(twin->instance, (unsigned int)scene->materials->count));

		/* Apply materials to the geometry instance. */
		vprintf("Processed %" PRIu64 " materials for %s.\n", scene->materials->count, twin->name);
		for (i = 0u; i < scene->materials->count; i++)
			RT_CHECK_ERROR(rtGeometryInstanceSetMaterial(twin->instance, i, (RTmaterial)scene->materials->array[i]));

		if (material_override != OVOID)
			applyGeometryInstanceVariable1i(context, twin->instance, "sole_material", material_override);
	}
	else
		twin->instance = NULL;

	if (node->group) {
		RT_CHECK_ERROR(rtGeometryGroupCreate(context, &twin->group));
		RT_CHECK_ERROR(rtGeometryGroupSetChildCount(twin->group, 1));
		RT_CHECK_ERROR(rtGeometryGroupSetChild(twin->group, 0, twin->instance));
		RT_CHECK_ERROR(rtGeometryGroupSetAcceleration(twin->group, twin->acceleration));
	}
	else
		twin->group = NULL;

	twin->child = cloneSceneBranch(context, node->child, material_override, scene);

	return twin;
}

static void clearSceneBranch(SceneBranch *branch)
{
	branch->node = NULL;
	branch->transform = NULL;
	branch->sibling = NULL;
}

static SceneBranch* cloneSceneBranch(const RTcontext context, SceneBranch *branch, const int material_override, Scene* scene)
{
	if (!branch) return NULL;

	int i;
	SceneBranch *twin = (SceneBranch*)malloc(sizeof(SceneBranch));
	if (!twin) eprintf(SYSTEM, "Out of memory in cloneSceneBranch, need %" PRIu64 " bytes", sizeof(SceneBranch));

	/* Check if instance has already been loaded */
	twin->node = NULL;
	if (branch->node) {
		for (i = 0; i < scene->instances->count; i++) {
			SceneNode *node = (SceneNode*)scene->instances->array[i];
			if (node->sole_material == material_override && !strcmp(branch->node->name, node->name)) {
				twin->node = node;
				break;
			}
		}

		if (!twin->node)
			twin->node = cloneSceneNode(context, branch->node, material_override, scene);
	}
	twin->transform = branch->transform;
	twin->sibling = cloneSceneBranch(context, branch->sibling, material_override, scene);

	return twin;
}

static RTobject createSceneHierarchy(const RTcontext context, SceneNode* node)
{
	RTgroup group;
	RTacceleration groupAccel;
	SceneBranch* child = node->child;
	unsigned int num_children = 0;
	RTsize num_triangles;

	if (!child) {
		/* This is a leaf node. Create a geometry group to hold the geometry instance. */
		return node->group;
	}

	/* Create a group to hold the geometry group and any children. */
	RT_CHECK_ERROR(rtBufferGetSize1D(node->material_buffer, &num_triangles));
	if (num_triangles > 0) num_children = 1;
	while (child) {
		num_children++;
		child = child->sibling;
	}
	RT_CHECK_ERROR(rtGroupCreate(context, &group));
	RT_CHECK_ERROR(rtGroupSetChildCount(group, num_children));
	if (num_triangles > 0)
		RT_CHECK_ERROR(rtGroupSetChild(group, 0, node->group));

	/* Set remaining group children. */
	child = node->child;
	num_children = num_triangles > 0 ? 1 : 0;
	while (child) {
		RTobject childGroup = createSceneHierarchy(context, child->node);
		if (child->transform) {
			RT_CHECK_ERROR(rtTransformSetChild(child->transform, childGroup));
			childGroup = child->transform;
		}
		RT_CHECK_ERROR(rtGroupSetChild(group, num_children++, childGroup));

		child = child->sibling;
	}

	/* create acceleration object for group and specify some build hints */
	RT_CHECK_ERROR(rtAccelerationCreate(context, &groupAccel));
	RT_CHECK_ERROR(rtAccelerationSetBuilder(groupAccel, "Sbvh"));
	RT_CHECK_ERROR(rtAccelerationSetTraverser(groupAccel, "Bvh"));
	RT_CHECK_ERROR(rtGroupSetAcceleration(group, groupAccel));

	/* mark acceleration as dirty */
	RT_CHECK_ERROR(rtAccelerationMarkDirty(groupAccel));

	return group;
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
	RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( instance, 0, createNormalMaterial( context, &Lamb, NULL ) ) );

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
	mprintf(" %s(%i) %s(%i) %s(%i) %i", rec->omod > OVOID ? objptr(rec->omod)->oname : VOIDID, rec->omod, ofun[rec->otype].funame, rec->otype, rec->oname, objndx(rec), rec->oargs.nsargs);
	for (i = 0; i < rec->oargs.nsargs; i++)
		mprintf(" %s", rec->oargs.sarg[i]);
	mprintf(" 0 %i", rec->oargs.nfargs);
	for (i = 0; i < rec->oargs.nfargs; i++)
		mprintf(" %g", rec->oargs.farg[i]);
	mprintf("\n");
}

int setBackfaceVisibility(const int back)
{
	int changed;
	if ((changed = (backvis != back))) {
		backvis = back;
		RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1ui(backvis_var, (unsigned int)back));
	}
	return changed;
}

int setIrradiance(const int irrad)
{
	int changed;
	if ((changed = (do_irrad != irrad))) {
		do_irrad = irrad;
		RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1ui(irrad_var, (unsigned int)irrad));
	}
	return changed;
}

int setDirectJitter(const double jitter)
{
	int changed;
	if ((changed = (dstrsrc != jitter))) {
		dstrsrc = jitter;
		RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1f(dstrsrc_var, (float)jitter));
	}
	return changed;
}

int setDirectSampling(const double ratio)
{
	int changed;
	if ((changed = (srcsizerat != ratio))) {
		srcsizerat = ratio;
		RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1f(srcsizerat_var, (float)ratio));
	}
	return changed;
}

int setDirectVisibility(const int vis)
{
	int changed;
	if ((changed = (directvis != vis))) {
		directvis = vis;
		RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1i(directvis_var, vis));
	}
	return changed;
}

int setSpecularThreshold(const double threshold)
{
	int changed;
	if ((changed = (specthresh != threshold))) {
		specthresh = threshold;
		RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1f(specthresh_var, (float)threshold));
	}
	return changed;
}

int setSpecularJitter(const double jitter)
{
	int changed;
	if ((changed = (specjitter != jitter))) {
		specjitter = jitter;
		RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1f(specjitter_var, (float)jitter));
	}
	return changed;
}

int setAmbientBounces(const int bounces)
{
	int changed;
	if ((changed = (ambounce != bounces))) {
		ambounce = bounces;
		RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1i(ambounce_var, bounces));
	}
	return changed;
}

int setMinWeight(const double weight)
{
	int changed;
	if ((changed = (minweight != weight))) {
		minweight = weight;
		RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1f(minweight_var, (float)weight));
	}
	return changed;
}

int setMaxDepth(const int depth)
{
	int changed;
	if ((changed = (maxdepth != depth))) {
		maxdepth = depth;
		RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1i(maxdepth_var, depth));
	}
	return changed;
}

#endif /* ACCELERAD */
