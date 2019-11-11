/*
 *  optix_radiance.h - declarations for simulations on GPUs.
 */

#pragma once

#include "accelerad_copyright.h"

#include "fvect.h"
#include "view.h"
#include "color.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "lookup.h"
#include "rterror.h"

#include <optix_world.h>
#include "optix_common.h"
#include "optix_ambient_common.h"
#include "optix_point_common.h"


#define  array2cuda2(c,a)	((c).x=(a)[0],(c).y=(a)[1])
#define  cuda2array2(a,c)	((a)[0]=(c).x,(a)[1]=(c).y)
#define  array2cuda3(c,a)	((c).x=(float)(a)[0],(c).y=(float)(a)[1],(c).z=(float)(a)[2])
#define  cuda2array3(a,c)	((a)[0]=(c).x,(a)[1]=(c).y,(a)[2]=(c).z)

/* Enable features on CPU side */
#ifndef RTX
#define TIMEOUT_CALLBACK /* Interupt OptiX kernel periodically to refresh screen */
#endif
//#define CUMULTATIVE_TIME /* Track cumulative timing of OptiX kernel functions */
#define DEBUG_OPTIX /* Catch unexptected OptiX exceptions. Creates a big slow down with OptiX 4.0.0 and above. */
//#define REPORT_GPU_STATE /* Report verbose GPU details */
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

/* Print formatted error */
#define eprintf(etype, format, ...) do {	\
	sprintf(errmsg, format, ##__VA_ARGS__);	\
	error(etype, errmsg); } while(0)
/* Print formatted message */
#define mprintf(format, ...) do {	\
	if (erract[WARNING].pf) {		\
		sprintf(errmsg, format, ##__VA_ARGS__);	\
		(*erract[WARNING].pf)(errmsg); } } while(0)

#ifdef ACCELERAD_DEBUG
/* Print extra statements */
#define vprintf(format, ...)	mprintf(format, ##__VA_ARGS__)
#else
#define vprintf(format, ...)	/* Ignore */
#endif

/* TIMING */
#ifdef _WIN64
#define MILLISECONDS(c)	((c) * 1000uLL / CLOCKS_PER_SEC)
#else
#define MILLISECONDS(c)	((c) * 1000uL / CLOCKS_PER_SEC)
#endif

/* Print elapsed time for the operation */
#define tprint(clock, label) do {	\
	if (erract[WARNING].pf) {		\
		const size_t milliseconds = MILLISECONDS(clock);	\
		sprintf(errmsg, "%s: %" PRIu64 ".%03" PRIu64 " seconds.\n", label, milliseconds / 1000, milliseconds % 1000);	\
		(*erract[WARNING].pf)(errmsg); } } while(0)
/* Print formatted elapsed time for the operation */
#define tprintf(clock, format, ...) do {	\
	if (erract[WARNING].pf) {		\
		const size_t milliseconds = MILLISECONDS(clock);	\
		sprintf(errmsg, format ": %" PRIu64 ".%03" PRIu64 " seconds.\n", ##__VA_ARGS__, milliseconds / 1000, milliseconds % 1000);	\
		(*erract[WARNING].pf)(errmsg); } } while(0)

/* Math */
#ifndef min
#define min(a,b)	(((a)<(b))?(a):(b))
#endif
#ifndef max
#define max(a,b)	(((a)>(b))?(a):(b))
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
	void **array;
	size_t count;
	size_t size;
} PointerArray;

typedef struct {
	MaterialData *array;
	size_t count;
	size_t size;
} MaterialDataArray;

typedef struct {
	DistantLight *array;
	size_t count;
	size_t size;
} DistantLightArray;


char path_to_ptx[512];     /* The path to the PTX file. */

/* in optix_radiance.c */
void createContext(RTcontext* context, const RTsize width, const RTsize height);
void destroyContext(const RTcontext context);
#ifdef CONTRIB
void makeContribCompatible(const RTcontext context);
#endif
#ifdef DAYSIM_COMPATIBLE
void makeDaysimCompatible(const RTcontext context);
void setupDaysim(const RTcontext context, RTbuffer* dc_buffer, const RTsize width, const RTsize height);
#endif
void setupKernel(const RTcontext context, const VIEW* view, LUTAB* modifiers, const RTsize width, const RTsize height, const unsigned int imm_irrad, void (*freport)(double));
void updateModel(const RTcontext context, LUTAB* modifiers, const unsigned int imm_irrad);
void createCamera(const RTcontext context, const char* ptx_name);
void updateCamera(const RTcontext context, const VIEW* view);
int setIrradiance(const int irrad);

/* in optix_ambient.c */
void createAmbientRecords(const RTcontext context, const VIEW* view, const RTsize width, const RTsize height, void (*freport)(double));
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
RTvariable applyContextVariable(const RTcontext context, const char* name, const RTsize size, const void* data);
RTvariable applyProgramVariable1i(const RTcontext context, const RTprogram program, const char* name, const int value);
RTvariable applyProgramVariable3i(const RTcontext context, const RTprogram program, const char* name, const int x, const int y, const int z);
RTvariable applyProgramVariable1ui(const RTcontext context, const RTprogram program, const char* name, const unsigned int value);
RTvariable applyProgramVariable1f(const RTcontext context, const RTprogram program, const char* name, const float value);
RTvariable applyProgramVariable2f(const RTcontext context, const RTprogram program, const char* name, const float x, const float y);
RTvariable applyProgramVariable3f(const RTcontext context, const RTprogram program, const char* name, const float x, const float y, const float z);
RTvariable applyProgramVariable(const RTcontext context, const RTprogram program, const char* name, const RTsize size, const void* data);
void createBuffer1D(const RTcontext context, const RTbuffertype type, const RTformat format, const RTsize element_count, RTbuffer* buffer);
void createCustomBuffer1D(const RTcontext context, const RTbuffertype type, const RTsize element_size, const RTsize element_count, RTbuffer* buffer);
void createBuffer2D(const RTcontext context, const RTbuffertype type, const RTformat format, const RTsize x_count, const RTsize y_count, RTbuffer* buffer);
void createCustomBuffer2D(const RTcontext context, const RTbuffertype type, const RTsize element_size, const RTsize x_count, const RTsize y_count, RTbuffer* buffer);
void createBuffer3D(const RTcontext context, const RTbuffertype type, const RTformat format, const RTsize x_count, const RTsize y_count, const RTsize z_count, RTbuffer* buffer);
void createCustomBuffer3D(const RTcontext context, const RTbuffertype type, const RTsize element_size, const RTsize x_count, const RTsize y_count, const RTsize z_count, RTbuffer* buffer);
RTvariable applyContextObject(const RTcontext context, const char* name, const RTobject object);
RTvariable applyProgramObject(const RTcontext context, const RTprogram program, const char* name, const RTobject object);
void copyToBufferi(const RTcontext context, const RTbuffer buffer, const IntArray *a);
void copyToBufferf(const RTcontext context, const RTbuffer buffer, const FloatArray *a);
void copyToBufferm(const RTcontext context, const RTbuffer buffer, const MaterialDataArray *a);
void copyToBufferdl(const RTcontext context, const RTbuffer buffer, const DistantLightArray *a);
IntArray* initArrayi(const size_t initialSize);
int insertArrayi(IntArray *a, const int element);
int insertArray2i(IntArray *a, const int x, const int y);
int insertArray3i(IntArray *a, const int x, const int y, const int z);
void freeArrayi(IntArray *a);
FloatArray* initArrayf(const size_t initialSize);
float insertArrayf(FloatArray *a, const float element);
float insertArray2f(FloatArray *a, const float x, const float y);
float insertArray3f(FloatArray *a, const float x, const float y, const float z);
void freeArrayf(FloatArray *a);
PointerArray* initArrayp(const size_t initialSize);
void* insertArrayp(PointerArray *a, void *element);
void freeArrayp(PointerArray *a);
MaterialDataArray* initArraym(const size_t initialSize);
MaterialData insertArraym(MaterialDataArray *a, const MaterialData element);
void freeArraym(MaterialDataArray *a);
DistantLightArray* initArraydl(const size_t initialSize);
DistantLight insertArraydl(DistantLightArray *a, const DistantLight element);
void freeArraydl(DistantLightArray *a);
int evali(char *name, const int val);
float evalf(char *name, const float val);
void handleError( const RTcontext context, const RTresult code, const char* file, const int line, const int etype );
#ifdef DEBUG_OPTIX
void logException(const RTexception type);
void flushExceptionLog(const char* location);
void printException(const RTexception code, const int index, const char* location);
#endif
void ptxFile( char* path, const char* name );
char* ptxString(const char* name);
char* filename(char *path);
#ifdef TIMEOUT_CALLBACK
int timeoutCallback(void);
#endif
#ifdef RTX
void reportCallback(int verbosity, const char* tag, const char* message, void* payload);
#endif
extern int verbose_output;	/* Print repetitive outputs. */
