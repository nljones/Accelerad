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
#include "otspecial.h"

#include "optix_radiance.h"
#include <cuda_runtime_api.h>

#ifdef CONTRIB
#include "rcontrib.h"
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
#define EXPECTED_INSTANCES	4

#define DEFAULT_SPHERE_STEPS	4
#define DEFAULT_CONE_STEPS		24

#define TCALNAME	"tmesh.cal"	/* the name of our auxiliary file */
#define RINDEX		1.52f		/* refractive index of glass */

#ifdef ACCELERAD

#ifdef RTX
#define RTX_TRIANGLES
#endif

/* Handles to the objects in a particular node in the scene graph. This is built once per octree and may be referenced multiple times in the scene graph. */
typedef struct SceneNode {
	char *name;						/* The name of this instance */
	RTgeometrygroup group;			/* The geometry group containing this instance */
	RTgeometryinstance instance;	/* The geometry instance kept by this scene */
	RTacceleration acceleration;	/* The acceleration structure kept by this scene */
	IntArray* vertex_indices;		/* Three entries per triangle */
	IntArray* triangles;			/* One entry per triangle gives material of that triangle */
	size_t triangle_count;			/* Number of triangles in this instance */
	size_t offset;					/* Offset of first triangle into global buffer */
	RTvariable sole_material;		/* Variable containing the material index to use for all surfaces in this node, or -1 to use separate materials for each object. */
	int sole_material_id;				/* The material index to use for all surfaces in this node, or -1 to use separate materials for each object. */
	struct SceneBranch *child;		/* First entry in linked list of child branches */
	struct SceneNode *twin;			/* Older twin to copy from */
} SceneNode;

/* Handles to one instance of a node. This is built once per occurance of an octree in the scene except for the top node, which has no transform. */
typedef struct SceneBranch {
	SceneNode *node;				/* The geometry node contained in this branch */
	int material_id;				/* The material index to use for all surfaces in this node if there is no sole material, or -1 to use separate materials for each object. */
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
	MaterialDataArray* material_data;		/* One entry per material */
	PointerArray* instances;		/* One entry per instance */
#ifdef LIGHTS
	IntArray*   lights;				/* Three entries per triangle that is a light */
#endif
	DistantLightArray* sources;		/* One entry per source object */
	unsigned int vertex_index_0;	/* Index of first vertex of current object */
#ifdef ANTIMATTER
	unsigned int clip_count;		/* Antimatter material counter */
#endif
#ifdef CONTRIB
	LUTAB* modifiers;				/* Modifiers for contribution calculations */
#endif
	OBJREC *material;				/* Current material */
	unsigned int unhandled[NUMOTYPE];	/* Track unknown object types */
} Scene;

/* Material types */
typedef enum
{
	M_NORMAL = 0,	/* Normal material type */
	M_GLASS,		/* Glass material type */
	M_LIGHT,		/* Light material type */
#ifdef ACCELERAD_RT
	M_DIFFUSE,		/* Diffuse material type for progressive rendering of normal materials */
#endif

	M_COUNT			/* Number of material types */
} MaterialTypes;

char *ray_type_names[RAY_TYPE_COUNT] = {
	"radiance",		/* Radiance ray type */
	"shadow",		/* Shadow ray type */
	"ambient",		/* Ray into ambient cache */
	"ambient_record",	/* Ray to create ambient record */
	"point_cloud"	/* Ray to create point cloud */
};


static void checkDevices();
#ifdef REMOTE_VCA
static void checkRemoteDevice(RTremotedevice remote);
static void createRemoteDevice(RTremotedevice* remote);
#endif
static void applyRadianceSettings(const RTcontext context, const VIEW* view, const unsigned int imm_irrad);
static void createScene(const RTcontext context, SceneNode* root, LUTAB* modifiers, const unsigned int imm_irrad);
static void createNode(const RTcontext context, Scene* scene, char* name, const OBJECT start, const OBJECT count, MESH* mesh, const int material_id);
static void createGeometryInstance(const RTcontext context, SceneNode *node, const size_t vertex_count);
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
static void createMaterial(const RTcontext context);
static void createMaterialPrograms(const RTcontext context);
static int createNormalMaterial(const RTcontext context, OBJREC* rec, Scene* scene);
static int createGlassMaterial(const RTcontext context, OBJREC* rec, Scene* scene);
static int createLightMaterial(const RTcontext context, OBJREC* rec, Scene* scene);
#ifdef ANTIMATTER
static int createClipMaterial(const RTcontext context, OBJREC* rec, Scene* scene);
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
static void applyContribution(const RTcontext context, MaterialData* material, DistantLight* light, OBJREC* rec, Scene* scene);
#endif
static void clearSceneNode(SceneNode *node);
static SceneNode* cloneSceneNode(const RTcontext context, SceneNode *node, const int material_override, Scene* scene);
static void clearSceneBranch(SceneBranch *branch);
static SceneBranch* cloneSceneBranch(const RTcontext context, SceneBranch *branch, const int material_override, Scene* scene);
static RTobject createSceneHierarchy(const RTcontext context, SceneNode* root);
static void createIrradianceGeometry( const RTcontext context );
static void unhandledObject(OBJREC* rec, Scene *scene);
static void printObject(OBJREC* rec);


/* from ambient.c */
extern double  avsum;		/* computed ambient value sum (log) */
extern unsigned int  navsum;	/* number of values in avsum */
extern unsigned int  nambvals;	/* total number of indirect values */

/* from func.c */
extern XF  unitxf;

/* Ambient calculation flags */
#define AMBIENT_REQUESTED			0x4	/* irradiance caching should be used according to program arguments */
#define AMBIENT_CALCULATION_NEEDED	0x2	/* irradiance cahce must be calculated */
#define AMBIENT_SURFACES_PRESENT	0x1	/* materials that use irradiance caching are present in model */
static unsigned int ambient_flags = 0u;

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
#ifdef REMOTE_VCA
static RTremotedevice remote_handle = NULL;
#endif
static RTbuffer vindex_buffer = NULL, vertex_buffer = NULL, normal_buffer = NULL, texcoord_buffer = NULL, material_data_buffer = NULL, material_buffer = NULL, lindex_buffer = NULL, lights_buffer = NULL;
static SceneNode scene_root;

/* Handles to intersection program objects used by multiple materials */
static RTmaterial generic_material = NULL;
static RTprogram any_hit_program = NULL, any_hit_ambient_record_program = NULL;
static RTprogram closest_hit_programs[RAY_TYPE_COUNT];
static int closest_hit_callable_programs[RAY_TYPE_COUNT][M_COUNT];
#ifdef ACCELERAD_RT
int has_diffuse_normal_closest_hit_program = 0;	/* Flag for including rvu programs. */
#endif
#ifdef RTX_TRIANGLES
static RTprogram attribute_program = NULL;
#else
static RTprogram intersect_program = NULL, bbox_program = NULL;
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
	unsigned int display_major = 0, display_minor = 0;
	unsigned int i;
	unsigned int multiprocessor_count, threads_per_block, clock_rate, texture_count, timeout_enabled, tcc_driver, cuda_device;
	unsigned int compute_capability[2];
	int major = 1000, minor = 10;
	char device_name[128], pci[13];
	RTsize memory_size;

	/* Check driver version */
	cudaDriverGetVersion(&driver);
	cudaRuntimeGetVersion(&runtime);
	if (driver < runtime)
		eprintf(INTERNAL, "Current graphics driver %d.%d.%d does not support runtime %d.%d.%d. Update your graphics driver.",
			driver / 1000, (driver % 100) / 10, driver % 10, runtime / 1000, (runtime % 100) / 10, runtime % 10);

	rtGetVersion(&version);
	if (!version)
		eprintf(INTERNAL, "Error reading OptiX library. Update your graphics driver.");
	if (version > 4000) { // Extra digit added in OptiX 4.0.0
		major *= 10;
		minor *= 10;
	}
	RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetDeviceCount(&device_count)); // This will return an error if no supported devices are found
	rtGlobalGetAttribute(RT_GLOBAL_ATTRIBUTE_DISPLAY_DRIVER_VERSION_MAJOR, sizeof(display_major), &display_major);
	rtGlobalGetAttribute(RT_GLOBAL_ATTRIBUTE_DISPLAY_DRIVER_VERSION_MINOR, sizeof(display_minor), &display_minor);
	mprintf("OptiX %d.%d.%d found display driver %d.%02d, CUDA driver %d.%d.%d, and %i GPU device%s:\n",
		version / major, (version % major) / minor, version % minor, display_major, display_minor, driver / 1000, (driver % 100) / 10, driver % 10,
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
		rtDeviceGetAttribute( i, RT_DEVICE_ATTRIBUTE_PCI_BUS_ID, sizeof(pci), &pci );
		mprintf("Device %u: %s with %u multiprocessors, %u threads per block, %u Hz, %" PRIu64 " bytes global memory, %u hardware textures, compute capability %u.%u, timeout %sabled, Tesla compute cluster driver %sabled, PCI %s.%s",
			cuda_device, device_name, multiprocessor_count, threads_per_block, clock_rate, memory_size, texture_count, compute_capability[0], compute_capability[1],
			timeout_enabled ? "en" : "dis", tcc_driver ? "en" : "dis", pci, i == device_count - 1 ? "\n\n" : "\n");
	}
}

#ifdef REMOTE_VCA
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
#endif

void createContext(RTcontext* context, const RTsize width, const RTsize height)
{
	//RTbuffer seed_buffer;

	//unsigned int* seeds;
	//RTsize i;
	unsigned int ray_type_count, entry_point_count;

	/* Check if irradiance cache is used */
	ambient_flags = (ambacc > FTINY && ambounce > 0 && ambdiv > 0) ? AMBIENT_REQUESTED : 0u;
	if (ambient_flags && nambvals == 0u) { // && (ambfile == NULL || !ambfile[0]); // TODO Should really look at ambfp in ambinet.c to check that file is readable
		ambient_flags |= AMBIENT_CALCULATION_NEEDED;
		ray_type_count = RAY_TYPE_COUNT;
		entry_point_count = ENTRY_POINT_COUNT;
	} else {
		ray_type_count = RAY_TYPE_COUNT - (ambient_flags ? 2 : 3); /* leave out ambient record and point cloud ray types */
		entry_point_count = 1u; /* Generate radiance data */
	}

	/* Setup remote device */
#ifdef REMOTE_VCA
	if (optix_remote_nodes > 0)
		createRemoteDevice(&remote_handle);
	else
#endif
		checkDevices();

	/* Setup context */
	RT_CHECK_ERROR2( rtContextCreate( context ) );
#ifdef REMOTE_VCA
	if (remote_handle) RT_CHECK_ERROR2(rtContextSetRemoteDevice(*context, remote_handle));
#endif
	RT_CHECK_ERROR2( rtContextSetRayTypeCount( *context, ray_type_count ) );
	RT_CHECK_ERROR2( rtContextSetEntryPointCount( *context, entry_point_count ) );

#ifdef RTX
	/* Set recursion depths */
	RT_CHECK_ERROR2(rtContextSetMaxTraceDepth(*context, maxdepth ? min(abs(maxdepth) * 2, 31) : 31)); // TODO set based on lw?
	RT_CHECK_ERROR2(rtContextSetMaxCallableProgramDepth(*context, 1));
#else
	/* Set stack size for GPU threads */
	if (optix_stack_size > 0)
		RT_CHECK_ERROR2( rtContextSetStackSize( *context, optix_stack_size ) );
#endif

	/* Create a buffer of random number seeds */
	//createBuffer2D( *context, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, width, height, &seed_buffer );
	//RT_CHECK_ERROR2( rtBufferMap( seed_buffer, (void**)&seeds ) );
	//for ( i = 0; i < width * height; ++i ) 
	//	seeds[i] = rand();
	//RT_CHECK_ERROR2( rtBufferUnmap( seed_buffer ) );
	//applyContextObject( *context, "rnd_seeds", seed_buffer );

#ifdef TIMEOUT_CALLBACK
#ifdef REMOTE_VCA
	if (!remote_handle)
#endif
	if (alarm > 0) {
		verbose_output = 1;
		RT_CHECK_ERROR2(rtContextSetTimeoutCallback(*context, timeoutCallback, alarm));
	}
#endif

#ifdef DEBUG_OPTIX
#ifdef RTX
	/* Enable debugging callbacks */
	RT_CHECK_ERROR2(rtContextSetUsageReportCallback(*context, reportCallback, min(max(optix_verbosity - 1, 0), 3), NULL));

	if (optix_verbosity)
#endif
	/* Enable exception checking */
	RT_CHECK_ERROR2(rtContextSetExceptionEnabled(*context, RT_EXCEPTION_ALL, 1));
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
#ifdef REMOTE_VCA
	if (remote_handle) {
		rtRemoteDeviceRelease(remote_handle);
		rtRemoteDeviceDestroy(remote_handle);
		remote_handle = NULL;
	}
#endif
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

void setupKernel(const RTcontext context, const VIEW* view, LUTAB* modifiers, const RTsize width, const RTsize height, const unsigned int imm_irrad, void (*freport)(double))
{
	/* Setup state */
	applyRadianceSettings(context, view, imm_irrad);
	createMaterial(context);
	clearSceneNode(&scene_root);
	createScene(context, &scene_root, modifiers, imm_irrad);
	RTobject top = createSceneHierarchy(context, &scene_root);

	/* Set the geometry group as the top level object. */
	top_object = applyContextObject(context, "top_object", top);
	if (!(ambient_flags & AMBIENT_SURFACES_PRESENT))
		top_ambient = applyContextObject(context, "top_ambient", top); // Need to define this because it is referred to by material_normal.cu
	if (!imm_irrad)
		top_irrad = applyContextObject(context, "top_irrad", top); // Need to define this because it is referred to by rtrace.cu, rtrace_cloud_generator.cu, and ambient_cloud_generator.cu

	if ( imm_irrad )
		createIrradianceGeometry( context );

	/* Set up irradiance cache of ambient values */
	if (ambient_flags & AMBIENT_SURFACES_PRESENT) { // Don't bother with ambient records if no surfaces will use them
		if (ambient_flags & AMBIENT_CALCULATION_NEEDED) // Run pre-process only if no ambient file is available
			createAmbientRecords( context, view, width, height, freport ); // implementation depends on current settings
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
	createScene(context, &scene_root, modifiers, imm_irrad);
	RTobject top = createSceneHierarchy(context, &scene_root);

	/* Set the geometry group as the top level object. */
	RT_CHECK_ERROR(rtVariableSetObject(top_object, top));
	if (!(ambient_flags & AMBIENT_SURFACES_PRESENT)) //TODO If ambient surfaces become present, the irradiance cache must be built and the number of entry points needs to change
		RT_CHECK_ERROR(rtVariableSetObject(top_ambient, top));
	if (!imm_irrad)
		RT_CHECK_ERROR(rtVariableSetObject(top_irrad, top));
}

static void applyRadianceSettings(const RTcontext context, const VIEW* view, const unsigned int imm_irrad)
{
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
	irrad_var = applyContextVariable1ui(context, "do_irrad", (unsigned int)do_irrad); // -i

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
	char *ptx = ptxString(ptx_name);
	RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "ray_generator", &program));
	RT_CHECK_ERROR(rtContextSetRayGenerationProgram(context, RADIANCE_ENTRY, program));

	/* Exception program */
	RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "exception", &program));
	RT_CHECK_ERROR(rtContextSetExceptionProgram(context, RADIANCE_ENTRY, program));
	free(ptx);

	/* Miss program */
	ptx = ptxString("background");
	RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "miss", &program));
	RT_CHECK_ERROR(rtContextSetMissProgram(context, RADIANCE_RAY, program));

	/* Miss program for shadow rays */
	RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "miss_shadow", &program));
	RT_CHECK_ERROR(rtContextSetMissProgram(context, SHADOW_RAY, program));
	free(ptx);
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

static void createScene(const RTcontext context, SceneNode* root, LUTAB* modifiers, const unsigned int imm_irrad)
{
	unsigned int i;
	int *vdata, *mdata;
	Scene scene;

	/* Timers */
	clock_t geometry_clock = clock();

	/* Set the existing instance and transform for this geometry instance (or null for no instance or transform) */
	scene.root = root;
	scene.buffer_entry_index = NULL;
#ifdef ANTIMATTER
	scene.clip_count = 0u;
#endif

	/* Create buffers for storing geometry information. */
	scene.vertices = initArrayf(EXPECTED_VERTICES * 3);
	scene.normals = initArrayf(EXPECTED_VERTICES * 3);
	scene.tex_coords = initArrayf(EXPECTED_VERTICES * 2);
	scene.material_data = initArraym(EXPECTED_MATERIALS);
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
	for (i = 0; i < NUMOTYPE; i++) {
		scene.unhandled[i] = 0;
	}

	/* Material 0 is Lambertian. */
	if (do_irrad || imm_irrad) {
		createNormalMaterial(context, &Lamb, &scene);
	}

	/* Create the top node of the scene graph. */
	createNode(context, &scene, octname, 0, nsceneobjs, NULL, OVOID);

	/* Check for overflow */
	if (scene.material_data->count > UINT_MAX)
		eprintf(USER, "Number of materials %" PRIu64 " is greater than maximum %u.", scene.material_data->count, UINT_MAX);

	/* Count the triangles that were found. */
	size_t triangle_count = 0;
	for (i = 0; i < scene.instances->count; i++) {
		SceneNode *node = (SceneNode*)scene.instances->array[i];
		if (node->twin) {
			node->triangle_count = node->twin->triangle_count;
			node->offset = node->twin->offset;
		}
		else {
			node->triangle_count = node->triangles->count;
			if (node->triangle_count > UINT_MAX)
				eprintf(USER, "Number of triangles %" PRIu64 " in %s is greater than maximum %u.", node->triangle_count, node->name, UINT_MAX);
			node->offset = triangle_count;
			if (node->offset > UINT_MAX)
				eprintf(USER, "Offset of triangles %" PRIu64 " in %s is greater than maximum %u.", node->offset, node->name, UINT_MAX);
			triangle_count += node->triangle_count;
		}
	}

	/* Create the closest and any hit programs. */
	createMaterialPrograms(context);

	/* Apply the geometry buffers. */
	vprintf("Processed %" PRIu64 " triangles.\n", triangle_count);
	if (!vindex_buffer) {
		createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, triangle_count, &vindex_buffer);
		applyContextObject(context, "vindex_buffer", vindex_buffer);
	}
	else
		RT_CHECK_ERROR(rtBufferSetSize1D(vindex_buffer, triangle_count));
	//copyToBufferi(context, vindex_buffer, scene->vertex_indices);

	size_t vertex_count = scene.vertices->count / 3;
	vprintf("Processed %" PRIu64 " vertices.\n", vertex_count);
	if (!vertex_buffer) {
		createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, vertex_count, &vertex_buffer);
		applyContextObject(context, "vertex_buffer", vertex_buffer);
	}
	else
		RT_CHECK_ERROR(rtBufferSetSize1D(vertex_buffer, vertex_count));
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

	if (!material_buffer) {
		createBuffer1D(context, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, triangle_count, &material_buffer);
		applyContextObject(context, "material_buffer", material_buffer);
	}
	else
		RT_CHECK_ERROR(rtBufferSetSize1D(material_buffer, triangle_count));
	//copyToBufferi(context, material_buffer, scene->triangles);

	vprintf("Processed %" PRIu64 " materials.\n", scene.material_data->count);
	if (!material_data_buffer) {
		createCustomBuffer1D(context, RT_BUFFER_INPUT, sizeof(MaterialData), scene.material_data->count, &material_data_buffer);
		applyContextObject(context, "material_data", material_data_buffer);
	}
	else
		RT_CHECK_ERROR(rtBufferSetSize1D(material_data_buffer, scene.material_data->count));
	copyToBufferm(context, material_data_buffer, scene.material_data);
	freeArraym(scene.material_data);

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

	/* Apply per-instance buffers */
	RT_CHECK_ERROR(rtBufferMap(vindex_buffer, (void**)&vdata));
	RT_CHECK_ERROR(rtBufferMap(material_buffer, (void**)&mdata));
	if (!triangle_count)
		createGeometryInstance(context, scene.root, vertex_count); // Create an empty geometry group
	else for (i = 0; i < scene.instances->count; i++) {
		SceneNode *node = (SceneNode*)scene.instances->array[i];
		if (!node->twin) {
			memcpy(vdata + node->offset * 3, node->vertex_indices->array, node->vertex_indices->count*sizeof(int));
			freeArrayi(node->vertex_indices);
			memcpy(mdata + node->offset, node->triangles->array, node->triangles->count*sizeof(int));
			freeArrayi(node->triangles);
		}
		if (node->triangle_count)
			createGeometryInstance(context, node, vertex_count);
	}
	RT_CHECK_ERROR(rtBufferUnmap(vindex_buffer));
	RT_CHECK_ERROR(rtBufferUnmap(material_buffer));

	/* Free resources used in creating the scene graph. */
	free(scene.buffer_entry_index);
	freeArrayp(scene.instances); //TODO keep this if the instances are not going to change

	/* Warn about unhandled object types */
	for (i = 0; i < NUMOTYPE; i++) {
		if (scene.unhandled[i])
			eprintf(WARNING, "no GPU support for %d instance%s of %s", scene.unhandled[i], scene.unhandled[i] != 1 ? "s" : "", ofun[i].funame);
	}

	tprintf(clock() - geometry_clock, "Geometry build for %i objects", nsceneobjs);
}

static void createNode(const RTcontext context, Scene* scene, char* name, const OBJECT start, const OBJECT count, MESH* mesh, const int material_id)
{
	const OBJECT end = start + count;
	OBJECT on;
	OBJREC *rec;

	/* Allocate or expand space for entry buffer. */
	if (count > 0) {
		scene->buffer_entry_index = (int *)realloc(scene->buffer_entry_index, sizeof(int) * end);
		if (!scene->buffer_entry_index) eprintf(SYSTEM, "out of memory in createNode, need %" PRIu64 " bytes", sizeof(int) * end);
	}

	/* Create this level of the geometry tree. */
	if (!scene->root) {
		scene->root = (SceneNode*)malloc(sizeof(SceneNode));
		if (!scene->root) eprintf(SYSTEM, "out of memory in createNode, need %" PRIu64 " bytes", sizeof(SceneNode));
		clearSceneNode(scene->root);
	}
	insertArrayp(scene->instances, scene->root);
	scene->root->name = savestr(name);
	scene->root->vertex_indices = initArrayi(EXPECTED_TRIANGLES * 3);
	scene->root->triangles = initArrayi(EXPECTED_TRIANGLES);
	scene->root->sole_material_id = material_id;

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
}

static void createGeometryInstance(const RTcontext context, SceneNode *node, const size_t vertex_count)
{
#ifdef RTX_TRIANGLES
	RTgeometrytriangles geometry = NULL;
#else
	RTgeometry geometry = NULL;
#endif

#ifdef RTX_TRIANGLES
	/* Create the geometry instance containing the geometry. */
	if (!node->instance) {
		if (node->twin && node->twin->instance)
			RT_CHECK_ERROR(rtGeometryInstanceGetGeometryTriangles(node->twin->instance, &geometry));
		else {
			RT_CHECK_ERROR(rtGeometryTrianglesCreate(context, &geometry));
			RT_CHECK_ERROR(rtGeometryTrianglesSetMaterialCount(geometry, 1u));
			//RT_CHECK_ERROR(rtGeometryTrianglesSetFlagsPerMaterial(geometry, 0, RT_GEOMETRY_FLAG_DISABLE_ANYHIT));

			if (!attribute_program) {
				ptxFile(path_to_ptx, "triangle_mesh");
				RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "mesh_attribute", &attribute_program));
			}
			RT_CHECK_ERROR(rtGeometryTrianglesSetAttributeProgram(geometry, attribute_program));
		}

		RT_CHECK_ERROR(rtGeometryInstanceCreate(context, &node->instance));
		RT_CHECK_ERROR(rtGeometryInstanceSetGeometryTriangles(node->instance, geometry));
	}
	else {
		RT_CHECK_ERROR(rtGeometryInstanceGetGeometryTriangles(node->instance, &geometry));
	}
	RT_CHECK_ERROR(rtGeometryTrianglesSetPrimitiveCount(geometry, (unsigned int)node->triangle_count));
	RT_CHECK_ERROR(rtGeometryTrianglesSetPrimitiveIndexOffset(geometry, (unsigned int)node->offset));

	if (vertex_count > UINT_MAX)
		eprintf(USER, "Number of vertices %" PRIu64 " is greater than maximum %u.", vertex_count, UINT_MAX);
	RT_CHECK_ERROR(rtGeometryTrianglesSetTriangleIndices(geometry, vindex_buffer, node->offset * sizeof(uint3), sizeof(uint3), RT_FORMAT_UNSIGNED_INT3)); //TODO not sure if this offset is needed
	//RT_CHECK_ERROR(rtGeometryTrianglesSetMaterialIndices(geometry, material_buffer, node->offset * sizeof(int), sizeof(int), RT_FORMAT_UNSIGNED_INT));
	RT_CHECK_ERROR(rtGeometryTrianglesSetVertices(geometry, (unsigned int)vertex_count, vertex_buffer, 0, sizeof(float3), RT_FORMAT_FLOAT3));
#else
	/* Create the geometry instance containing the geometry. */
	if (!node->instance) {
		if (node->twin && node->twin->instance)
			RT_CHECK_ERROR(rtGeometryInstanceGetGeometry(node->twin->instance, &geometry));
		else {
			RT_CHECK_ERROR(rtGeometryCreate(context, &geometry));
			//RT_CHECK_ERROR(rtGeometrySetFlags(geometry, RT_GEOMETRY_FLAG_DISABLE_ANYHIT));

			if (!intersect_program) {
				char *ptx = ptxString("triangle_mesh");
				RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "mesh_bounds", &bbox_program));
				RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "mesh_intersect", &intersect_program));
				free(ptx);
				backvis_var = applyProgramVariable1ui(context, intersect_program, "backvis", (unsigned int)backvis); // -bv
			}
			RT_CHECK_ERROR(rtGeometrySetBoundingBoxProgram(geometry, bbox_program));
			RT_CHECK_ERROR(rtGeometrySetIntersectionProgram(geometry, intersect_program));
		}

		RT_CHECK_ERROR(rtGeometryInstanceCreate(context, &node->instance));
		RT_CHECK_ERROR(rtGeometryInstanceSetGeometry(node->instance, geometry));
	}
	else {
		RT_CHECK_ERROR(rtGeometryInstanceGetGeometry(node->instance, &geometry));
	}
	RT_CHECK_ERROR(rtGeometrySetPrimitiveCount(geometry, (unsigned int)node->triangle_count));
	RT_CHECK_ERROR(rtGeometrySetPrimitiveIndexOffset(geometry, (unsigned int)node->offset));
#endif

	/* Apply materials to the geometry instance. */
	RT_CHECK_ERROR(rtGeometryInstanceSetMaterialCount(node->instance, 1u));
	RT_CHECK_ERROR(rtGeometryInstanceSetMaterial(node->instance, 0, generic_material));
	if (!node->sole_material)
		RT_CHECK_ERROR(rtGeometryInstanceDeclareVariable(node->instance, "sole_material", &node->sole_material));
	RT_CHECK_ERROR(rtVariableSet1i(node->sole_material, node->sole_material_id));

	/* Create a geometry group to hold the geometry instance. */
	if (!node->group) {
		RT_CHECK_ERROR(rtGeometryGroupCreate(context, &node->group));
		RT_CHECK_ERROR(rtGeometryGroupSetChildCount(node->group, 1));
		RT_CHECK_ERROR(rtGeometryGroupSetChild(node->group, 0, node->instance));
	}

	/* create acceleration object for group and specify some build hints */
	if (!node->acceleration) {
		if (node->twin && node->twin->acceleration)
			node->acceleration = node->twin->acceleration;
		else {
			RT_CHECK_ERROR(rtAccelerationCreate(context, &node->acceleration));
			RT_CHECK_ERROR(rtAccelerationSetBuilder(node->acceleration, "Trbvh"));
#ifndef RTX
			RT_CHECK_ERROR(rtAccelerationSetTraverser(node->acceleration, "Bvh"));
#endif
#ifndef RTX_TRIANGLES
			RT_CHECK_ERROR(rtAccelerationSetProperty(node->acceleration, "vertex_buffer_name", "vertex_buffer")); // For Sbvh only
			RT_CHECK_ERROR(rtAccelerationSetProperty(node->acceleration, "index_buffer_name", "vindex_buffer")); // For Sbvh only
#endif
		}
		RT_CHECK_ERROR(rtGeometryGroupSetAcceleration(node->group, node->acceleration));
	}

	/* mark acceleration as dirty */
	RT_CHECK_ERROR(rtAccelerationMarkDirty(node->acceleration));
}

static OBJECT addRadianceObject(const RTcontext context, OBJREC* rec, Scene* scene)
{
	const OBJECT index = objndx(rec);

	if (scene->buffer_entry_index[index] != OVOID) return index; /* Already done */

	switch (rec->otype) {
	case MAT_PLASTIC: // Plastic material
	case MAT_METAL: // Metal material
	case MAT_TRANS: // Translucent material
		scene->buffer_entry_index[index] = createNormalMaterial(context, rec, scene);
		break;
	case MAT_GLASS: // Glass material
	case MAT_DIELECTRIC: // Dielectric material TODO handle separately, see dialectric.c
		scene->buffer_entry_index[index] = createGlassMaterial(context, rec, scene);
		break;
	case MAT_LIGHT: // primary light source material, may modify a face or a source (solid angle)
	case MAT_ILLUM: // secondary light source material
	case MAT_GLOW: // Glow material
	case MAT_SPOT: // Spotlight material
		scene->buffer_entry_index[index] = createLightMaterial(context, rec, scene);
		break;
#ifdef BSDF
	case MAT_BSDF: // BSDF
		scene->buffer_entry_index[index] = createBSDFMaterial(context, rec, scene);
		break;
#endif
#ifdef ANTIMATTER
	case MAT_CLIP: // Antimatter
		scene->buffer_entry_index[index] = createClipMaterial(context, rec, scene);
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
		if (RT_PROGRAM_ID_NULL == (scene->buffer_entry_index[index] = createFunction(context, rec)))
			unhandledObject(rec, scene);
		break;
	//case MAT_TDATA: // bumpmap
	case PAT_BDATA: // brightness texture, used for IES lighting data
	case PAT_CDATA: // color texture
	case PAT_CPICT: // color picture, used as texture map
		if (RT_PROGRAM_ID_NULL == (scene->buffer_entry_index[index] = createTexture(context, rec)))
			unhandledObject(rec, scene);
		break;
	case TEX_FUNC:
		if (rec->oargs.nsargs == 3) {
			if (!strcmp(filename(rec->oargs.sarg[2]), TCALNAME)) break; // Handled by face
		}
		else if (rec->oargs.nsargs >= 4) {
			if (!strcmp(filename(rec->oargs.sarg[3]), TCALNAME)) break; // Handled by face
		}
		unhandledObject(rec, scene);
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
		unhandledObject(rec, scene);
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
	insertArray3i(scene->root->vertex_indices, a, b, c);
	if (scene->material) {
		insertArrayi(scene->root->triangles, scene->buffer_entry_index[objndx(scene->material)]);

#ifdef LIGHTS
		if (islight(scene->material->otype) && (scene->material->otype != MAT_GLOW || scene->material->oargs.farg[3] > 0))
			insertArray3i(scene->lights, a, b, c);
#endif /* LIGHTS */
	}
	else {
		insertArrayi(scene->root->triangles, OVOID);
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
		size_t vertex_index_count = scene->root->vertex_indices->count;
		size_t triangle_count = scene->root->triangles->count;
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
			scene->root->vertex_indices->count = vertex_index_count;
			scene->root->triangles->count = triangle_count;
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
	scene->material = findmaterial(rec);
	if (scene->material) {
		branch->material_id = scene->buffer_entry_index[addRadianceObject(context, scene->material, scene)];
	}
	material_override = scene->root->sole_material_id;
	if (material_override == OVOID) {
		material_override = branch->material_id;
	}

	/* Check if instance has already been loaded */
	for (i = 0; i < scene->instances->count; i++) {
		SceneNode *node = (SceneNode*)scene->instances->array[i];
		if (!strcmp(name, node->name)) {
			twin = node;
			if (node->sole_material_id == material_override) {
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
			createNode(context, scene, name, inst->obj->firstobj, inst->obj->nobjs, NULL, material_override);
		}
		else {
			meshinst = getmeshinst(rec, IO_SCENE | IO_BOUNDS); // Load the mesh file for the instance
			createNode(context, scene, name, meshinst->msh->mat0, meshinst->msh->nmats, meshinst->msh, material_override);
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

static void createMaterial(const RTcontext context)
{
	unsigned int i, j;

	if (!generic_material) {
		/* Create our material */
		RT_CHECK_ERROR(rtMaterialCreate(context, &generic_material));
	}

	for (i = 0; i < RAY_TYPE_COUNT; i++) {
		closest_hit_programs[i] = NULL;
		for (j = 0; j < M_COUNT; j++) {
			closest_hit_callable_programs[i][j] = RT_PROGRAM_ID_NULL;
		}
	}
}

static void createMaterialPrograms(const RTcontext context)
{
	unsigned int i, j;
	char name[100];

	char *ptx = ptxString("material_intersect");

	if (!any_hit_program) {
		RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "any_hit", &any_hit_program));
		backvis_var = applyProgramVariable1ui(context, any_hit_program, "backvis", (unsigned int)backvis); // -bv
	}

	for (i = 0; i < RAY_TYPE_COUNT; i++) {
		int program_ids[M_COUNT];
		int program_id_count = 0;
		for (j = 0; j < M_COUNT; j++) {
			if (closest_hit_callable_programs[i][j] != RT_PROGRAM_ID_NULL) {
				program_ids[program_id_count] = closest_hit_callable_programs[i][j];
				program_id_count++;
			}
		}
		if (program_id_count) {
			if (!closest_hit_programs[i]) {
				/* If any material type will use this program, then create the program */
				sprintf(name, "closest_hit_%s", ray_type_names[i]);

				RT_CHECK_ERROR(rtMaterialSetAnyHitProgram(generic_material, i, any_hit_program));

				RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, name, &closest_hit_programs[i]));
				RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(generic_material, i, closest_hit_programs[i]));
			}
			sprintf(name, "closest_hit_%s_call_site", ray_type_names[i]);
			RT_CHECK_ERROR(rtProgramCallsiteSetPotentialCallees(closest_hit_programs[i], name, program_ids, program_id_count)); // TODO need to remove any that are not used?
		}
	}
	free(ptx);
	if ((ambient_flags & AMBIENT_CALCULATION_NEEDED) && !any_hit_ambient_record_program) {
		ptx = ptxString("ambient_normal");
		RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "any_hit_ambient", &any_hit_ambient_record_program));
		RT_CHECK_ERROR(rtMaterialSetAnyHitProgram(generic_material, AMBIENT_RECORD_RAY, any_hit_ambient_record_program));

		RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "closest_hit_ambient", &closest_hit_programs[AMBIENT_RECORD_RAY]));
		RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(generic_material, AMBIENT_RECORD_RAY, closest_hit_programs[AMBIENT_RECORD_RAY]));
		free(ptx);

#ifdef AMBIENT_CELL
		createAmbientDynamicStorage(context, closest_hit_programs[AMBIENT_RECORD_RAY], 0);
#else
		createAmbientDynamicStorage(context, closest_hit_programs[AMBIENT_RECORD_RAY], cuda_kmeans_clusters);
#endif
	}
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
		unhandledObject(rec, scene);
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

	if (ambient_flags & AMBIENT_CALCULATION_NEEDED) {
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

static int createNormalMaterial(const RTcontext context, OBJREC* rec, Scene* scene)
{
	MaterialData matData;

	vprintf("Reading normal material %s\n", rec->oname);
	matData.type = rec->otype;
	array2cuda3(matData.color, rec->oargs.farg); // Color is the first three entries in farg
	matData.params.n.spec = (float)rec->oargs.farg[3];
	matData.params.n.rough = (float)rec->oargs.farg[4];
	matData.params.n.trans = rec->otype == MAT_TRANS ? (float)rec->oargs.farg[5] : 0.0f;
	matData.params.n.tspec = rec->otype == MAT_TRANS ? (float)rec->oargs.farg[6] : 0.0f;
	matData.mask = 0u;
#ifdef CONTRIB
	applyContribution(context, &matData, NULL, rec, scene);
#endif

	/* As a shortcut, exclude black materials from ambient calculation */
	if (rec->oargs.farg[0] == 0.0 && rec->oargs.farg[1] == 0.0 && rec->oargs.farg[2] == 0.0 && (rec->otype != MAT_TRANS || rec->oargs.farg[5] == 0.0))
		matData.params.n.ambincl = 0u;

	/* Check ambient include/exclude list */
	else if (ambincl != -1) {
		char **amblp;
		int in_set = 0;
		for (amblp = amblist; *amblp != NULL; amblp++)
			if (!strcmp(rec->oname, *amblp)) {
				in_set = 1;
				break;
			}
		matData.params.n.ambincl = in_set == ambincl;
	}
	else
		matData.params.n.ambincl = 1u;
	if ((ambient_flags & AMBIENT_REQUESTED) && matData.params.n.ambincl)
		ambient_flags |= AMBIENT_SURFACES_PRESENT;

	/* Check that material programs exist. */
	if (closest_hit_callable_programs[RADIANCE_RAY][M_NORMAL] == RT_PROGRAM_ID_NULL) {
		RTprogram program;

		/* Programs have not been created yet. */
		char *ptx = ptxString("material_normal");

		RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "closest_hit_normal_radiance", &program));
		RT_CHECK_ERROR(rtProgramGetId(program, &closest_hit_callable_programs[RADIANCE_RAY][M_NORMAL]));

		RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "closest_hit_normal_shadow", &program));
		RT_CHECK_ERROR(rtProgramGetId(program, &closest_hit_callable_programs[SHADOW_RAY][M_NORMAL]));
		free(ptx);

#ifdef ACCELERAD_RT
		if (has_diffuse_normal_closest_hit_program) { // Don't create the program if it won't be used
			ptxFile(path_to_ptx, "material_diffuse");
			RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_diffuse_radiance", &program));
			RT_CHECK_ERROR(rtProgramGetId(program, &closest_hit_callable_programs[RADIANCE_RAY][M_DIFFUSE]));
		}
#endif

		if (ambient_flags & AMBIENT_CALCULATION_NEEDED) {
			ptxFile(path_to_ptx, "point_cloud_normal");
			RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_normal_point_cloud", &program));
			RT_CHECK_ERROR(rtProgramGetId(program, &closest_hit_callable_programs[POINT_CLOUD_RAY][M_NORMAL]));
		}
	}

	/* Assign programs. */
	matData.radiance_program_id = closest_hit_callable_programs[RADIANCE_RAY][M_NORMAL];
#ifdef ACCELERAD_RT
	matData.diffuse_program_id = closest_hit_callable_programs[RADIANCE_RAY][M_DIFFUSE];
#endif
	matData.shadow_program_id = closest_hit_callable_programs[SHADOW_RAY][M_NORMAL];
	matData.point_cloud_program_id = closest_hit_callable_programs[POINT_CLOUD_RAY][M_NORMAL];

	insertArraym(scene->material_data, matData);
	return (int)scene->material_data->count - 1;
}

static int createGlassMaterial(const RTcontext context, OBJREC* rec, Scene* scene)
{
	MaterialData matData;

	vprintf("Reading glass material %s\n", rec->oname);
	matData.type = rec->otype;
	array2cuda3(matData.color, rec->oargs.farg); // Color is the first three entries in farg
	matData.params.r_index = rec->oargs.nfargs > 3 ? (float)rec->oargs.farg[3] : RINDEX;		/* refractive index of glass */
	matData.mask = 0u;
#ifdef CONTRIB
	applyContribution(context, &matData, NULL, rec, scene);
#endif

	/* Check that material programs exist. */
	if (closest_hit_callable_programs[RADIANCE_RAY][M_GLASS] == RT_PROGRAM_ID_NULL) {
		RTprogram program;

		/* Programs have not been created yet. */
		char *ptx = ptxString("material_glass");

		RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "closest_hit_glass_radiance", &program));
		RT_CHECK_ERROR(rtProgramGetId(program, &closest_hit_callable_programs[RADIANCE_RAY][M_GLASS]));

		RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "closest_hit_glass_shadow", &program));
		RT_CHECK_ERROR(rtProgramGetId(program, &closest_hit_callable_programs[SHADOW_RAY][M_GLASS]));
		free(ptx);

		if (ambient_flags & AMBIENT_CALCULATION_NEEDED) {
			ptxFile(path_to_ptx, "point_cloud_normal");
			RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "closest_hit_glass_point_cloud", &program));
			RT_CHECK_ERROR(rtProgramGetId(program, &closest_hit_callable_programs[POINT_CLOUD_RAY][M_GLASS]));
		}
	}

	/* Assign programs. */
	matData.radiance_program_id = closest_hit_callable_programs[RADIANCE_RAY][M_GLASS];
#ifdef ACCELERAD_RT
	matData.diffuse_program_id = matData.radiance_program_id;
#endif
	matData.shadow_program_id = closest_hit_callable_programs[SHADOW_RAY][M_GLASS];
	matData.point_cloud_program_id = closest_hit_callable_programs[POINT_CLOUD_RAY][M_GLASS];

	insertArraym(scene->material_data, matData);
	return (int)scene->material_data->count - 1;
}

static int createLightMaterial(const RTcontext context, OBJREC* rec, Scene* scene)
{
	MaterialData matData;
	OBJREC* mat;

	vprintf("Reading light material %s\n", rec->oname);
	matData.type = rec->otype;
	array2cuda3(matData.color, rec->oargs.farg); // Color is the first three entries in farg
	matData.mask = 0u;

	matData.params.l.maxrad = rec->otype == MAT_GLOW ? (float)rec->oargs.farg[3] : (float)FHUGE;
	if (rec->otype == MAT_SPOT) {
		SPOT* spot = makespot(rec);
		matData.params.l.siz = spot->siz;
		matData.params.l.flen = spot->flen;
		array2cuda3(matData.params.l.aim, spot->aim);
		free(spot);
		rec->os = NULL;
	}
	else if (rec->otype == MAT_ILLUM) {
		/* Check for a proxy material for direct views. */
		if (rec->oargs.nsargs && strcmp(rec->oargs.sarg[0], VOIDID)) {	/* modifies another material */
			mat = findmaterial(objptr(lastmod(objndx(rec), rec->oargs.sarg[0])));
			if (!mat) {
				objerror(rec, WARNING, "missing material");
				return OVOID;
			}
			matData.proxy = scene->buffer_entry_index[addRadianceObject(context, mat, scene)];
		}
		else
			matData.proxy = OVOID;
	}
#ifdef CONTRIB
	applyContribution(context, &matData, NULL, rec, scene);
#endif

	/* Check for a parent function. */
	if ((mat = findFunction(rec))) // TODO can there be multiple parent functions?
		matData.params.l.function = scene->buffer_entry_index[addRadianceObject(context, mat, scene)];
	else
		matData.params.l.function = RT_PROGRAM_ID_NULL;

	/* Check that material programs exist. */
	if (closest_hit_callable_programs[RADIANCE_RAY][M_LIGHT] == RT_PROGRAM_ID_NULL) {
		RTprogram program;

		/* Programs have not been created yet. */
		char *ptx = ptxString("material_light");

		RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "closest_hit_light_radiance", &program));
		RT_CHECK_ERROR(rtProgramGetId(program, &closest_hit_callable_programs[RADIANCE_RAY][M_LIGHT]));

		RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "closest_hit_light_shadow", &program));
		RT_CHECK_ERROR(rtProgramGetId(program, &closest_hit_callable_programs[SHADOW_RAY][M_LIGHT]));
		free(ptx);
	}

	/* Assign programs. */
	matData.radiance_program_id = closest_hit_callable_programs[RADIANCE_RAY][M_LIGHT];
#ifdef ACCELERAD_RT
	matData.diffuse_program_id = matData.radiance_program_id;
#endif
	matData.shadow_program_id = closest_hit_callable_programs[SHADOW_RAY][M_LIGHT];
	matData.point_cloud_program_id = closest_hit_callable_programs[POINT_CLOUD_RAY][M_LIGHT];

	insertArraym(scene->material_data, matData);
	return (int)scene->material_data->count - 1;
}

#ifdef ANTIMATTER
static int createClipMaterial(const RTcontext context, OBJREC* rec, Scene* scene)
{
	MaterialData matData;
	OBJECT mod;
	OBJREC* mat;
	int i, index;

	vprintf("Reading clipping material %s\n", rec->oname);
	matData.type = rec->otype;
	matData.mask = 1 << scene->clip_count++;
	matData.proxy = OVOID;

	/* Determine the material mask */
	if (!matData.mask)
		objerror(rec, WARNING, "too many antimatter materials");
	else for (i = 0; i < rec->oargs.nsargs; i++) {
		if (!strcmp(rec->oargs.sarg[i], VOIDID))
			continue;
		if ((mod = lastmod(objndx(rec), rec->oargs.sarg[i])) == OVOID) {
			sprintf(errmsg, "unknown modifier \"%s\"", rec->oargs.sarg[i]);
			objerror(rec, WARNING, errmsg);
			continue;
		}
		mat = findmaterial(objptr(mod));
		if (!mat) {
			sprintf(errmsg, "missing material for modifier \"%s\"", rec->oargs.sarg[i]);
			objerror(rec, WARNING, errmsg);
			continue;
		}
		index = scene->buffer_entry_index[addRadianceObject(context, mat, scene)];
		if (index == OVOID)
			continue;
		if (scene->material_data->array[index].mask & matData.mask) {
			objerror(rec, WARNING, "duplicate modifier");
			continue;
		}
		scene->material_data->array[index].mask |= matData.mask; // This will lead to strange behavior if antimatter clips another antimatter material
		if (!i)
			matData.proxy = index;
	}
#ifdef CONTRIB
	applyContribution(context, &matData, NULL, rec, scene);
#endif

	/* Assign programs. */
	matData.radiance_program_id = RT_PROGRAM_ID_NULL;
#ifdef ACCELERAD_RT
	matData.diffuse_program_id = matData.radiance_program_id;
#endif
	matData.shadow_program_id = RT_PROGRAM_ID_NULL;
	matData.point_cloud_program_id = RT_PROGRAM_ID_NULL;

	insertArraym(scene->material_data, matData);
	return (int)scene->material_data->count - 1;
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
			return RT_PROGRAM_ID_NULL;
		}
		applyProgramVariable(context, program, "transform", sizeof(transform), transform);
	}
	else {
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
#ifndef RTX
	RT_CHECK_ERROR( rtTextureSamplerSetMipLevelCount( tex_sampler, 1u ) ); // Currently only one mipmap level supported
	RT_CHECK_ERROR( rtTextureSamplerSetArraySize( tex_sampler, 1u ) ); // Currently only one element supported
#endif
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
				return RT_PROGRAM_ID_NULL;
			}
			if (strncmp(rec->oargs.sarg[3], src_phi, length_phi) || strcmp(rec->oargs.sarg[4], src_theta)) {
				if (strncmp(rec->oargs.sarg[4], src_phi, length_phi) || strcmp(rec->oargs.sarg[3], src_theta)) {
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
			return RT_PROGRAM_ID_NULL;
		}
		applyProgramVariable(context, program, "transform", sizeof(transform), transform);
	}
	else {
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
#ifndef RTX
		RT_CHECK_ERROR(rtTextureSamplerSetMipLevelCount(tex_sampler, 1u)); // Currently only one mipmap level supported
		RT_CHECK_ERROR(rtTextureSamplerSetArraySize(tex_sampler, 1u)); // Currently only one element supported
#endif
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

static void applyContribution(const RTcontext context, MaterialData* material, DistantLight* light, OBJREC* rec, Scene* scene)
{
	static char cfunc[] = "contrib_function";

	/* Check for a call-back function. */
	if (scene && scene->modifiers) {
		MODCONT	*mp;
		if ((mp = (MODCONT *)lu_find(scene->modifiers, rec->oname)->data)) {
			if (material) {
				material->contrib_index = mp->start_bin;
				material->contrib_function = createContribFunction(context, mp);
			}
			else if (light) {
				/* Check for a existing program. */
				int mat_index = scene->buffer_entry_index[objndx(rec)];
				light->contrib_index = mp->start_bin;
				if (mat_index != OVOID) /* In case this material has also already been used for a surface, we can reuse the function here */
					light->contrib_function = scene->material_data->array[mat_index].contrib_function;
				else
					light->contrib_function = createContribFunction(context, mp);
			}
			return;
		}
	}

	/* No call-back function. */
	if (material) {
		material->contrib_index = -1;
		material->contrib_function = RT_PROGRAM_ID_NULL;
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
	node->triangle_count = 0;
	node->offset = 0;
	node->sole_material = NULL;
	node->sole_material_id = OVOID;
	node->child = NULL;
	node->twin = NULL;
}

static SceneNode* cloneSceneNode(const RTcontext context, SceneNode *node, const int material_override, Scene* scene)
{
	if (!node) return NULL;

	SceneNode *twin = (SceneNode*)malloc(sizeof(SceneNode));
	if (!twin) eprintf(SYSTEM, "out of memory in cloneSceneNode, need %" PRIu64 " bytes", sizeof(SceneNode));
	insertArrayp(scene->instances, twin);

	twin->name = node->name;
	twin->acceleration = node->acceleration;
	twin->triangle_count = node->triangle_count;
	twin->offset = node->offset;
	twin->sole_material_id = material_override;
	twin->twin = node;

	if (node->instance) {
#ifdef RTX_TRIANGLES
		RTgeometrytriangles geometry;
#else
		RTgeometry geometry;
#endif

		RT_CHECK_ERROR(rtGeometryInstanceCreate(context, &twin->instance));
#ifdef RTX_TRIANGLES
		RT_CHECK_ERROR(rtGeometryInstanceGetGeometryTriangles(node->instance, &geometry));
		RT_CHECK_ERROR(rtGeometryInstanceSetGeometryTriangles(twin->instance, geometry));
#else
		RT_CHECK_ERROR(rtGeometryInstanceGetGeometry(node->instance, &geometry));
		RT_CHECK_ERROR(rtGeometryInstanceSetGeometry(twin->instance, geometry));
#endif

		/* Apply materials to the geometry instance. */
		RT_CHECK_ERROR(rtGeometryInstanceSetMaterialCount(twin->instance, 1u));
		RT_CHECK_ERROR(rtGeometryInstanceSetMaterial(twin->instance, 0, generic_material));
		RT_CHECK_ERROR(rtGeometryInstanceDeclareVariable(twin->instance, "sole_material", &twin->sole_material));
		RT_CHECK_ERROR(rtVariableSet1i(twin->sole_material, material_override));

		vprintf("Duplicated instance %s\n", node->name);
	}
	else {
		twin->instance = NULL;
		twin->sole_material = NULL;
	}

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
	branch->material_id = OVOID;
	branch->transform = NULL;
	branch->sibling = NULL;
}

static SceneBranch* cloneSceneBranch(const RTcontext context, SceneBranch *branch, const int material_override, Scene* scene)
{
	if (!branch) return NULL;

	int i;
	const int material_id = material_override == OVOID ? branch->material_id : material_override;
	SceneBranch *twin = (SceneBranch*)malloc(sizeof(SceneBranch));
	if (!twin) eprintf(SYSTEM, "Out of memory in cloneSceneBranch, need %" PRIu64 " bytes", sizeof(SceneBranch));
	twin->material_id = branch->material_id;

	/* Check if instance has already been loaded */
	twin->node = NULL;
	if (branch->node) {
		for (i = 0; i < scene->instances->count; i++) {
			SceneNode *node = (SceneNode*)scene->instances->array[i];
			if (node->sole_material_id == material_id && !strcmp(branch->node->name, node->name)) {
				twin->node = node;
				break;
			}
		}

		if (!twin->node)
			twin->node = cloneSceneNode(context, branch->node, material_id, scene);
	}

	/* Copy the transform */
	if (branch->transform) {
		float ft[16], bt[16];
		RT_CHECK_ERROR(rtTransformCreate(context, &twin->transform));
		RT_CHECK_ERROR(rtTransformGetMatrix(branch->transform, 0, ft, bt));
		RT_CHECK_ERROR(rtTransformSetMatrix(twin->transform, 0, ft, bt));
	}
	else
		twin->transform = NULL;

	/* Clone the next branch */
	twin->sibling = cloneSceneBranch(context, branch->sibling, material_override, scene);

	return twin;
}

static RTobject createSceneHierarchy(const RTcontext context, SceneNode* node)
{
	RTgroup group;
	RTacceleration groupAccel;
	SceneBranch* child = node->child;
	unsigned int num_children = node->triangle_count ? 1 : 0;

	if (!child) {
		/* This is a leaf node. Create a geometry group to hold the geometry instance. */
		return node->group;
	}

	/* Create a group to hold the geometry group and any children. */
	while (child) {
		num_children++;
		child = child->sibling;
	}
	RT_CHECK_ERROR(rtGroupCreate(context, &group));
	RT_CHECK_ERROR(rtGroupSetChildCount(group, num_children));
	if (node->triangle_count)
		RT_CHECK_ERROR(rtGroupSetChild(group, 0, node->group));

	/* Set remaining group children. */
	child = node->child;
	num_children = node->triangle_count ? 1 : 0;
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
	RT_CHECK_ERROR(rtAccelerationSetBuilder(groupAccel, "Bvh"));
#ifndef RTX
	RT_CHECK_ERROR(rtAccelerationSetTraverser(groupAccel, "Bvh"));
#endif
	RT_CHECK_ERROR(rtGroupSetAcceleration(group, groupAccel));

	/* mark acceleration as dirty */
	RT_CHECK_ERROR(rtAccelerationMarkDirty(groupAccel));

	return group;
}

static void createIrradianceGeometry( const RTcontext context )
{
	RTgeometry         geometry;
	RTprogram          program;
	RTgeometryinstance instance;
	RTgeometrygroup    geometrygroup;
	RTacceleration     acceleration;
	char *ptx;

	/* Create the geometry reference for OptiX. */
	RT_CHECK_ERROR( rtGeometryCreate( context, &geometry ) );
	RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( geometry, 1 ) );

	ptx = ptxString("irradiance_intersect" );
	RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "irradiance_bounds", &program));
	RT_CHECK_ERROR(rtGeometrySetBoundingBoxProgram(geometry, program));
	RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "irradiance_intersect", &program));
	RT_CHECK_ERROR(rtGeometrySetIntersectionProgram(geometry, program));
	free(ptx);

	/* Create the geometry instance containing the geometry. */
	RT_CHECK_ERROR( rtGeometryInstanceCreate( context, &instance ) );
	RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( instance, geometry ) );

	/* Create a Lambertian material as the geometry instance's only material. */
	RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( instance, 1 ) );
	RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( instance, 0, generic_material ) );

	/* Create a geometry group to hold the geometry instance.  This will be used as the top level group. */
	RT_CHECK_ERROR( rtGeometryGroupCreate( context, &geometrygroup ) );
	RT_CHECK_ERROR( rtGeometryGroupSetChildCount( geometrygroup, 1 ) );
	RT_CHECK_ERROR( rtGeometryGroupSetChild( geometrygroup, 0, instance ) );

	/* Set the geometry group as the top level object. */
	applyContextObject( context, "top_irrad", geometrygroup );

	/* create acceleration object for group and specify some build hints */
	RT_CHECK_ERROR( rtAccelerationCreate( context, &acceleration ) );
	RT_CHECK_ERROR( rtAccelerationSetBuilder( acceleration, "NoAccel" ) );
#ifndef RTX
	RT_CHECK_ERROR( rtAccelerationSetTraverser( acceleration, "NoAccel" ) );
#endif
	RT_CHECK_ERROR( rtGeometryGroupSetAcceleration( geometrygroup, acceleration ) );

	/* mark acceleration as dirty */
	RT_CHECK_ERROR( rtAccelerationMarkDirty( acceleration ) );
}

static void unhandledObject(OBJREC* rec, Scene *scene)
{
	if (!(scene->unhandled[rec->otype]++)) {
		objerror(rec, WARNING, "no GPU support"); // Print error only on the first time the object is discovered
		printObject(rec);
	}
}

static void printObject(OBJREC* rec)
{
	int i;

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
		if (backvis_var) RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1ui(backvis_var, (unsigned int)back));
	}
	return changed;
}

int setIrradiance(const int irrad)
{
	int changed;
	if ((changed = (do_irrad != irrad))) {
		do_irrad = irrad;
		if (irrad_var) RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1ui(irrad_var, (unsigned int)irrad));
	}
	return changed;
}

int setDirectJitter(const double jitter)
{
	int changed;
	if ((changed = (dstrsrc != jitter))) {
		dstrsrc = jitter;
		if (dstrsrc_var) RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1f(dstrsrc_var, (float)jitter));
	}
	return changed;
}

int setDirectSampling(const double ratio)
{
	int changed;
	if ((changed = (srcsizerat != ratio))) {
		srcsizerat = ratio;
		if (srcsizerat_var) RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1f(srcsizerat_var, (float)ratio));
	}
	return changed;
}

int setDirectVisibility(const int vis)
{
	int changed;
	if ((changed = (directvis != vis))) {
		directvis = vis;
		if (directvis_var) RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1i(directvis_var, vis));
	}
	return changed;
}

int setSpecularThreshold(const double threshold)
{
	int changed;
	if ((changed = (specthresh != threshold))) {
		specthresh = threshold;
		if (specthresh_var) RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1f(specthresh_var, (float)threshold));
	}
	return changed;
}

int setSpecularJitter(const double jitter)
{
	int changed;
	if ((changed = (specjitter != jitter))) {
		specjitter = jitter;
		if (specjitter_var) RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1f(specjitter_var, (float)jitter));
	}
	return changed;
}

int setAmbientBounces(const int bounces)
{
	int changed;
	if ((changed = (ambounce != bounces))) {
		ambounce = bounces;
		if (ambounce_var) RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1i(ambounce_var, bounces));
	}
	return changed;
}

int setMinWeight(const double weight)
{
	int changed;
	if ((changed = (minweight != weight))) {
		minweight = weight;
		if (minweight_var) RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1f(minweight_var, (float)weight));
	}
	return changed;
}

int setMaxDepth(const int depth)
{
	int changed;
	if ((changed = (maxdepth != depth))) {
		maxdepth = depth;
		if (maxdepth_var) RT_CHECK_WARN_NO_CONTEXT(rtVariableSet1i(maxdepth_var, depth));
	}
	return changed;
}

#endif /* ACCELERAD */
