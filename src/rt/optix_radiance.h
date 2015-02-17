/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <fvect.h>
#include <resolu.h>
#include <view.h>
#include <color.h>

#include <optix_world.h>
#include "optix_common.h"


#define  array2cuda2(c,a)	(c.x=a[0],c.y=a[1])
#define  cuda2array2(a,c)	(a[0]=c.x,a[1]=c.y)
#define  array2cuda3(c,a)	(c.x=a[0],c.y=a[1],c.z=a[2])
#define  cuda2array3(a,c)	(a[0]=c.x,a[1]=c.y,a[2]=c.z)

/* Enable features on CPU side */
#define PREFER_TCC /* When Tesla devices are present, use them exclusively */
#define TIMEOUT_CALLBACK /* Interupt OptiX kernel periodically to refresh screen */
#define CUMULTATIVE_TIME /* Track cumulative timing of OptiX kernel functions */
#define DEBUG_OPTIX /* Catch unexptected OptiX exceptions */
//#define PRINT_OPTIX /* Enable OptiX rtPrintf statements to standard out */
//#define REPORT_GPU_STATE /* Report verbose GPU details */
#define KMEANS_IC /* K-Means irradiance cache calculation */
#ifdef KMEANS_IC
#define ITERATIVE_KMEANS_IC /* Iterative K-Means irradiance cache calculation */
#endif

/* Entry points */
#define RADIANCE_ENTRY		0u	/* Generate radiance data */
#define AMBIENT_ENTRY		1u	/* Generate ambient records */
#ifdef KMEANS_IC
#define POINT_CLOUD_ENTRY	2u	/* Generate point cloud */
#ifdef ITERATIVE_KMEANS_IC
#define HEMISPHERE_SAMPLING_ENTRY	3u	/* Generate point cloud from hemisphere */
#endif
#endif

/* Entry point count for ambient calculation */
#ifdef ITERATIVE_KMEANS_IC
#define ENTRY_POINT_COUNT	3u	/* Generate ambient records, point cloud, and hemispherical sampling */
#elif defined KMEANS_IC
#define ENTRY_POINT_COUNT	2u	/* Generate ambient records and point cloud */
#else
#define ENTRY_POINT_COUNT	1u	/* Generate ambient records */
#endif

/* Ray types */
#define PRIMARY_RAY			0u	/* Radiance primary ray type for irradiance calculation */
#define RADIANCE_RAY		1u	/* Radiance ray type */
#define SHADOW_RAY			2u	/* Shadow ray type */
#define AMBIENT_RAY			3u	/* Ray into ambient cache */
#define AMBIENT_RECORD_RAY	4u	/* Ray to create ambient record */
#ifdef KMEANS_IC
#define POINT_CLOUD_RAY		5u	/* Ray to create point cloud */
#endif

/* Ray type count for ambient calculation */
#ifdef KMEANS_IC
#define RAY_TYPE_COUNT		2u	/* ambient record ray and point cloud ray */
#else
#define RAY_TYPE_COUNT		1u	/* ambient record ray */
#endif

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

/* in optix_ambient.c */
void createAmbientRecords( const RTcontext context, const VIEW* view, const int width, const int height );
void setupAmbientCache( const RTcontext context, const unsigned int level );

/* in optix_util.c */
#ifdef REPORT_GPU_STATE
void printContextInfo( const RTcontext context );
extern int printCUDAProp();
#endif
void runKernel1D( const RTcontext context, const unsigned int entry, const int size );
void runKernel2D( const RTcontext context, const unsigned int entry, const int width, const int height );
void runKernel3D( const RTcontext context, const unsigned int entry, const int width, const int height, const int depth );
void applyContextVariable1i( const RTcontext context, const char* name, const int value );
void applyContextVariable1ui( const RTcontext context, const char* name, const unsigned int value );
void applyContextVariable1f( const RTcontext context, const char* name, const float value );
void applyContextVariable3f( const RTcontext context, const char* name, const float x, const float y, const float z );
void applyProgramVariable1i( const RTcontext context, const RTprogram program, const char* name, const int value );
void applyProgramVariable1ui( const RTcontext context, const RTprogram program, const char* name, const unsigned int value );
void applyProgramVariable1f( const RTcontext context, const RTprogram program, const char* name, const float value );
void applyProgramVariable2f( const RTcontext context, const RTprogram program, const char* name, const float x, const float y );
void applyProgramVariable3f( const RTcontext context, const RTprogram program, const char* name, const float x, const float y, const float z );
void applyProgramVariable( const RTcontext context, const RTprogram program, const char* name, const unsigned int size, const void* data );
void applyMaterialVariable1i( const RTcontext context, const RTmaterial material, const char* name, const int value );
void applyMaterialVariable1ui( const RTcontext context, const RTmaterial material, const char* name, const unsigned int value );
void applyMaterialVariable1f( const RTcontext context, const RTmaterial material, const char* name, const float value );
void applyMaterialVariable3f( const RTcontext context, const RTmaterial material, const char* name, const float x, const float y, const float z );
void createBuffer1D( const RTcontext context, const RTbuffertype type, const RTformat format, const int element_count, RTbuffer* buffer );
void createCustomBuffer1D( const RTcontext context, const RTbuffertype type, const int element_size, const int element_count, RTbuffer* buffer );
void createBuffer2D( const RTcontext context, const RTbuffertype type, const RTformat format, const int x_count, const int y_count, RTbuffer* buffer );
void createCustomBuffer2D( const RTcontext context, const RTbuffertype type, const int element_size, const int x_count, const int y_count, RTbuffer* buffer );
void createBuffer3D( const RTcontext context, const RTbuffertype type, const RTformat format, const int x_count, const int y_count, const int z_count, RTbuffer* buffer );
void createCustomBuffer3D( const RTcontext context, const RTbuffertype type, const int element_size, const int x_count, const int y_count, const int z_count, RTbuffer* buffer );
void applyContextObject( const RTcontext context, const char* name, const RTobject object );
void applyProgramObject( const RTcontext context, const RTprogram program, const char* name, const RTobject object );
void applyGeometryObject( const RTcontext context, const RTgeometry geometry, const char* name, const RTobject object );
void applyGeometryInstanceObject( const RTcontext context, const RTgeometryinstance instance, const char* name, const RTobject object );
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
void printException( const float3 code, const char* location, const int index );
void ptxFile( char* path, const char* name );
#ifdef TIMEOUT_CALLBACK
int timeoutCallback(void);
#endif
