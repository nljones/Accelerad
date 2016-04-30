/*
 * Copyright (c) 2013-2016 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <fvect.h>
#include <resolu.h>
#include <view.h>
#include <color.h>
#include <inttypes.h>

#include <optix_world.h>
#include "optix_common.h"
#include "optix_ambient_common.h"


#define  array2cuda2(c,a)	((c).x=(a)[0],(c).y=(a)[1])
#define  cuda2array2(a,c)	((a)[0]=(c).x,(a)[1]=(c).y)
#define  array2cuda3(c,a)	((c).x=(float)(a)[0],(c).y=(float)(a)[1],(c).z=(float)(a)[2])
#define  cuda2array3(a,c)	((a)[0]=(c).x,(a)[1]=(c).y,(a)[2]=(c).z)

/* Enable features on CPU side */
#define TIMEOUT_CALLBACK /* Interupt OptiX kernel periodically to refresh screen */
//#define CUMULTATIVE_TIME /* Track cumulative timing of OptiX kernel functions */
#define DEBUG_OPTIX /* Catch unexptected OptiX exceptions */
//#define PRINT_OPTIX /* Enable OptiX rtPrintf statements to standard out */
//#define REPORT_GPU_STATE /* Report verbose GPU details */
#define VERBOSE_OUTPUT /* Print extra statements */
#define ITERATIVE_IC /* Iterative irradiance cache calculation */

/* Entry points */
typedef enum
{
	RADIANCE_ENTRY = 0,			/* Generate radiance data */
	AMBIENT_ENTRY,				/* Generate ambient records */
	POINT_CLOUD_ENTRY,			/* Generate point cloud */
#ifdef ITERATIVE_IC
	HEMISPHERE_SAMPLING_ENTRY,	/* Generate point cloud from hemisphere */
#endif
#ifdef AMB_PARALLEL
	AMBIENT_SAMPLING_ENTRY,		/* Generate ambient samples for irradiance caching */
#endif

	ENTRY_POINT_COUNT			/* Entry point count for ambient calculation */
} RTentry;

/* Ray types */
typedef enum
{
	PRIMARY_RAY = 0,	/* Radiance primary ray type for irradiance calculation */
	RADIANCE_RAY,		/* Radiance ray type */
#ifdef ACCELERAD_RT
	DIFFUSE_PRIMARY_RAY,/* Radiance primary ray type for irradiance calculation sampling only diffuse paths */
	DIFFUSE_RAY,		/* Radiance ray type sampling only diffuse paths */
#endif
	SHADOW_RAY,			/* Shadow ray type */
	AMBIENT_RAY,		/* Ray into ambient cache */
	AMBIENT_RECORD_RAY,	/* Ray to create ambient record */
	POINT_CLOUD_RAY,	/* Ray to create point cloud */

	RAY_TYPE_COUNT		/* Entry point count for ambient calculation */
} RTraytype;

/* Error handling */
#ifdef DEBUG_OPTIX
/* assumes current scope has Context variable named 'context' */
#define RT_CHECK_ERROR( func ) do {	\
	RTresult code = func;			\
	if( code != RT_SUCCESS )		\
		handleError( context, code, __FILE__, __LINE__, INTERNAL ); } while(0)
#define RT_CHECK_WARN( func ) do {	\
	RTresult code = func;			\
	if( code != RT_SUCCESS )		\
		handleError( context, code, __FILE__, __LINE__, WARNING ); } while(0)

/* assumes current scope has Context pointer variable named 'context' */
#define RT_CHECK_ERROR2( func ) do {	\
	RTresult code = func;				\
	if( code != RT_SUCCESS )			\
		handleError( *context, code, __FILE__, __LINE__, INTERNAL ); } while(0)
#define RT_CHECK_WARN2( func ) do {	\
	RTresult code = func;			\
	if( code != RT_SUCCESS )		\
		handleError( *context, code, __FILE__, __LINE__, WARNING ); } while(0)

/* assumes that there is no context, just print to stderr */
#define RT_CHECK_ERROR_NO_CONTEXT( func ) do {	\
	RTresult code = func;						\
	if( code != RT_SUCCESS )					\
		handleError( NULL, code, __FILE__, __LINE__, INTERNAL ); } while(0)
#define RT_CHECK_WARN_NO_CONTEXT( func ) do {	\
	RTresult code = func;						\
	if( code != RT_SUCCESS )					\
		handleError( NULL, code, __FILE__, __LINE__, WARNING ); } while(0)
#else
/* When debugging is off, do nothing extra. */
#define RT_CHECK_ERROR( func )				func
#define RT_CHECK_WARN( func )				func
#define RT_CHECK_ERROR2( func )				func
#define RT_CHECK_WARN2( func )				func
#define RT_CHECK_ERROR_NO_CONTEXT( func )	func
#define RT_CHECK_WARN_NO_CONTEXT( func )	func
#endif

/* Print statements */
#define mprintf(format, ...) \
	do { if (erract[WARNING].pf) fprintf(stderr, format, ##__VA_ARGS__); } while(0)

#ifdef VERBOSE_OUTPUT
#define vprintf(format, ...)	mprintf(format, ##__VA_ARGS__)
#else
#define vprintf(format, ...)	/* Ignore */
#endif

/* TIMING */
#ifdef _WIN64
#define MILLISECONDS(c)	(c) * 1000uLL / CLOCKS_PER_SEC
#else
#define MILLISECONDS(c)	(c) * 1000uL / CLOCKS_PER_SEC
#endif

/* Resizeable array structures */
typedef struct {
	int *array;
	size_t count;
	size_t size;
} IntArray;

typedef struct {
	float *array;
	size_t count;
	size_t size;
} FloatArray;

typedef struct {
	RTmaterial *array;
	size_t count;
	size_t size;
} MaterialArray;

typedef struct {
	DistantLight *array;
	size_t count;
	size_t size;
} DistantLightArray;


char path_to_ptx[512];     /* The path to the PTX file. */

/* in optix_radiance.c */
void createContext(RTcontext* context, const int width, const int height, const double alarm);
void destroyContext(const RTcontext context);
#ifdef DAYSIM_COMPATIBLE
void setupDaysim(const RTcontext context, RTbuffer* dc_buffer, const int width, const int height);
#endif
void setupKernel(const RTcontext context, const VIEW* view, const int width, const int height, const unsigned int imm_irrad, const double dstrpix, const double mblur, const double dblur, const double alarm);
void createCamera(const RTcontext context, const char* ptx_name);
void updateCamera(const RTcontext context, const VIEW* view);

/* in optix_ambient.c */
void createAmbientRecords( const RTcontext context, const VIEW* view, const int width, const int height, const double alarm );
void setupAmbientCache( const RTcontext context, const unsigned int level );
void createAmbientDynamicStorage(const RTcontext context, const RTprogram program, const RTsize size);

/* in optix_util.c */
#ifdef REPORT_GPU_STATE
void printContextInfo( const RTcontext context );
extern void printCUDAProp();
#endif
void runKernel1D(const RTcontext context, const unsigned int entry, const RTsize size);
void runKernel2D(const RTcontext context, const unsigned int entry, const RTsize width, const RTsize height);
void runKernel3D(const RTcontext context, const unsigned int entry, const RTsize width, const RTsize height, const RTsize depth);
RTvariable applyContextVariable1i(const RTcontext context, const char* name, const int value);
RTvariable applyContextVariable2i(const RTcontext context, const char* name, const int x, const int y);
RTvariable applyContextVariable1ui(const RTcontext context, const char* name, const unsigned int value);
RTvariable applyContextVariable1f(const RTcontext context, const char* name, const float value);
RTvariable applyContextVariable2f(const RTcontext context, const char* name, const float x, const float y);
RTvariable applyContextVariable3f(const RTcontext context, const char* name, const float x, const float y, const float z);
RTvariable applyProgramVariable1i(const RTcontext context, const RTprogram program, const char* name, const int value);
RTvariable applyProgramVariable3i(const RTcontext context, const RTprogram program, const char* name, const int x, const int y, const int z);
RTvariable applyProgramVariable1ui(const RTcontext context, const RTprogram program, const char* name, const unsigned int value);
RTvariable applyProgramVariable1f(const RTcontext context, const RTprogram program, const char* name, const float value);
RTvariable applyProgramVariable2f(const RTcontext context, const RTprogram program, const char* name, const float x, const float y);
RTvariable applyProgramVariable3f(const RTcontext context, const RTprogram program, const char* name, const float x, const float y, const float z);
RTvariable applyProgramVariable(const RTcontext context, const RTprogram program, const char* name, const unsigned int size, const void* data);
RTvariable applyMaterialVariable1i(const RTcontext context, const RTmaterial material, const char* name, const int value);
RTvariable applyMaterialVariable1ui(const RTcontext context, const RTmaterial material, const char* name, const unsigned int value);
RTvariable applyMaterialVariable1f(const RTcontext context, const RTmaterial material, const char* name, const float value);
RTvariable applyMaterialVariable3f(const RTcontext context, const RTmaterial material, const char* name, const float x, const float y, const float z);
void createBuffer1D(const RTcontext context, const RTbuffertype type, const RTformat format, const RTsize element_count, RTbuffer* buffer);
void createCustomBuffer1D(const RTcontext context, const RTbuffertype type, const RTsize element_size, const RTsize element_count, RTbuffer* buffer);
void createBuffer2D(const RTcontext context, const RTbuffertype type, const RTformat format, const RTsize x_count, const RTsize y_count, RTbuffer* buffer);
void createCustomBuffer2D(const RTcontext context, const RTbuffertype type, const RTsize element_size, const RTsize x_count, const RTsize y_count, RTbuffer* buffer);
void createBuffer3D(const RTcontext context, const RTbuffertype type, const RTformat format, const RTsize x_count, const RTsize y_count, const RTsize z_count, RTbuffer* buffer);
void createCustomBuffer3D(const RTcontext context, const RTbuffertype type, const RTsize element_size, const RTsize x_count, const RTsize y_count, const RTsize z_count, RTbuffer* buffer);
RTvariable applyContextObject(const RTcontext context, const char* name, const RTobject object);
RTvariable applyProgramObject(const RTcontext context, const RTprogram program, const char* name, const RTobject object);
RTvariable applyGeometryObject(const RTcontext context, const RTgeometry geometry, const char* name, const RTobject object);
RTvariable applyGeometryInstanceObject(const RTcontext context, const RTgeometryinstance instance, const char* name, const RTobject object);
void copyToBufferi(const RTcontext context, const RTbuffer buffer, const IntArray *a);
void copyToBufferf(const RTcontext context, const RTbuffer buffer, const FloatArray *a);
void copyToBufferdl(const RTcontext context, const RTbuffer buffer, const DistantLightArray *a);
void initArrayi(IntArray *a, const size_t initialSize);
int insertArrayi(IntArray *a, const int element);
int insertArray2i(IntArray *a, const int x, const int y);
int insertArray3i(IntArray *a, const int x, const int y, const int z);
void freeArrayi(IntArray *a);
void initArrayf(FloatArray *a, const size_t initialSize);
float insertArrayf(FloatArray *a, const float element);
float insertArray2f(FloatArray *a, const float x, const float y);
float insertArray3f(FloatArray *a, const float x, const float y, const float z);
void freeArrayf(FloatArray *a);
void initArraym(MaterialArray *a, const size_t initialSize);
RTmaterial insertArraym(MaterialArray *a, const RTmaterial element);
void freeArraym(MaterialArray *a);
void initArraydl(DistantLightArray *a, const size_t initialSize);
DistantLight insertArraydl(DistantLightArray *a, const DistantLight element);
void freeArraydl(DistantLightArray *a);
void handleError( const RTcontext context, const RTresult code, const char* file, const int line, const int etype );
#ifdef DEBUG_OPTIX
void logException(const RTexception type);
void flushExceptionLog(const char* location);
void printException(const RTexception type, const int index, const char* location);
#endif
void ptxFile( char* path, const char* name );
char* filename(char *path);
void reportProgress( const double progress, const double alarm );
#ifdef TIMEOUT_CALLBACK
int timeoutCallback(void);
#endif
