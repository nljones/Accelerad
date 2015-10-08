/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <stdio.h>

#include <standard.h> /* TODO just get the includes that are required? */
#include "paths.h" /* Required for R_OK argument to getpath() */

#include "optix_radiance.h"


/* Kernel dimension */
typedef enum
{
	LAUNCH_1D,	/* Call rtContextLaunch1D */
	LAUNCH_2D,	/* Call rtContextLaunch2D */
	LAUNCH_3D	/* Call rtContextLaunch3D */
} RTdimension;

static void runKernelImpl(const RTcontext context, const unsigned int entry, const RTsize width, const RTsize height, const RTsize depth, const RTdimension dim);

#ifdef TIMEOUT_CALLBACK
static clock_t last_callback_time;  /* time of last callback from GPU */
#endif

#ifdef CUMULTATIVE_TIME
static size_t cumulative_millis = 0;  /* cumulative timing of kernel fuctions */
#endif

/* from rpict.c */
extern void report(int);
extern double  pctdone;			/* percentage done */


#ifdef REPORT_GPU_STATE
void printContextInfo( const RTcontext context )
{
	int value, device_count = 0, i;
	int* devices;
	RTsize size;

	if ( !context ) return;

	RT_CHECK_WARN(rtContextGetDeviceCount(context, &device_count));
	devices = (int*)malloc(sizeof(int) * device_count);
	if (device_count && !devices)
		error(SYSTEM, "out of memory in printContextInfo");
	RT_CHECK_WARN(rtContextGetDevices(context, devices));

	//RT_CHECK_ERROR( rtContextGetRunningState( context, &value ) );
	//mprintf("OptiX kernel running:             %s\n", value ? "Yes" : "No");
	RT_CHECK_ERROR( rtContextGetAttribute( context, RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT, sizeof(int), &value ) );
	mprintf("OptiX maximum textures:           %i\n", value);
	RT_CHECK_ERROR( rtContextGetAttribute( context, RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS, sizeof(int), &value ) ); // This can be set
	mprintf("OptiX host CPU threads:           %i\n", value);
	RT_CHECK_ERROR( rtContextGetAttribute( context, RT_CONTEXT_ATTRIBUTE_USED_HOST_MEMORY, sizeof(RTsize), &size ) );
	mprintf("OptiX host memory allocated:      %" PRIu64 " bytes\n", size);
	for (i = 0; i < device_count; i++) {
		value = devices[i];
		RT_CHECK_ERROR(rtContextGetAttribute(context, RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY + value, sizeof(RTsize), &size));
		mprintf("OptiX free memory on device %i:    %" PRIu64 " bytes\n", value, size);
	}
	RT_CHECK_ERROR( rtContextGetAttribute( context, RT_CONTEXT_ATTRIBUTE_GPU_PAGING_ACTIVE, sizeof(int), &value ) );
	mprintf("OptiX software paging:            %s\n", value ? "Yes" : "No");
	RT_CHECK_ERROR( rtContextGetAttribute( context, RT_CONTEXT_ATTRIBUTE_GPU_PAGING_FORCED_OFF, sizeof(int), &value ) ); // This can be set
	mprintf("OptiX software paging prohibited: %s\n", value ? "Yes" : "No");
	free(devices);
}
#endif

void runKernel1D(const RTcontext context, const unsigned int entry, const RTsize size)
{
	runKernelImpl(context, entry, size, 0, 0, LAUNCH_1D);
}

void runKernel2D(const RTcontext context, const unsigned int entry, const RTsize width, const RTsize height)
{
	runKernelImpl(context, entry, width, height, 0, LAUNCH_2D);
}

void runKernel3D(const RTcontext context, const unsigned int entry, const RTsize width, const RTsize height, const RTsize depth)
{
	runKernelImpl(context, entry, width, height, depth, LAUNCH_3D);
}

static void runKernelImpl(const RTcontext context, const unsigned int entry, const RTsize width, const RTsize height, const RTsize depth, const RTdimension dim)
{
	/* Timers */
	time_t kernel_time; // Timer in seconds for long jobs
	clock_t kernel_clock; // Timer in clock cycles for short jobs

#ifdef REPORT_GPU_STATE
	/* Print context attributes before */
	printContextInfo( context );
#endif

	/* Validate and compile if necessary */
	RT_CHECK_ERROR( rtContextValidate( context ) );
	kernel_clock = clock();
	RT_CHECK_ERROR( rtContextCompile( context ) ); // This should happen automatically when necessary.
	kernel_clock = clock() - kernel_clock;
	if (kernel_clock > 1)
		vprintf("OptiX compile time: %" PRIu64 " milliseconds.\n", MILLISECONDS(kernel_clock));

	/* Start timers */
	kernel_time = time((time_t *)NULL);
	kernel_clock = clock();
#ifdef TIMEOUT_CALLBACK
	last_callback_time = kernel_clock;
#endif

	/* Run */
	switch (dim) {
	case LAUNCH_1D:
		RT_CHECK_ERROR(rtContextLaunch1D(context, entry, width));
		break;
	case LAUNCH_2D:
		RT_CHECK_ERROR(rtContextLaunch2D(context, entry, width, height));
		break;
	case LAUNCH_3D:
		RT_CHECK_ERROR(rtContextLaunch3D(context, entry, width, height, depth));
		break;
	}

	/* Stop timers */
	kernel_clock = clock() - kernel_clock;
	kernel_time = time((time_t *)NULL) - kernel_time;
	mprintf("OptiX kernel time: %" PRIu64 " milliseconds (%" PRIu64 " seconds).\n", MILLISECONDS(kernel_clock), kernel_time);
#ifdef CUMULTATIVE_TIME
	cumulative_millis += MILLISECONDS(kernel_clock);
	mprintf("OptiX kernel cumulative time: %" PRIu64 " milliseconds.\n", cumulative_millis);
#endif

#ifdef REPORT_GPU_STATE
	/* Print context attributes after */
	printContextInfo( context );
#endif
}

void printRayTracingTime(const time_t time, const clock_t clock)
{
	/* Check if warnings are printed */
	if (!erract[WARNING].pf)
		return;

	/* Print the given elapsed time for ray tracing */
	sprintf(errmsg, "ray tracing time: %" PRIu64 " milliseconds (%" PRIu64 " seconds).\n", MILLISECONDS(clock), time);
	(*erract[WARNING].pf)(errmsg);
}

/* Helper functions */

RTvariable applyContextVariable1i(const RTcontext context, const char* name, const int value)
{
	RTvariable var;
	RT_CHECK_ERROR( rtContextDeclareVariable( context, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1i( var, value ) );
	return var;
}

RTvariable applyContextVariable1ui(const RTcontext context, const char* name, const unsigned int value)
{
	RTvariable var;
	RT_CHECK_ERROR( rtContextDeclareVariable( context, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1ui( var, value ) );
	return var;
}

RTvariable applyContextVariable1f(const RTcontext context, const char* name, const float value)
{
	RTvariable var;
	RT_CHECK_ERROR( rtContextDeclareVariable( context, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1f( var, value ) );
	return var;
}

RTvariable applyContextVariable2f(const RTcontext context, const char* name, const float x, const float y)
{
	RTvariable var;
	RT_CHECK_ERROR(rtContextDeclareVariable(context, name, &var));
	RT_CHECK_ERROR(rtVariableSet2f(var, x, y));
	return var;
}

RTvariable applyContextVariable3f(const RTcontext context, const char* name, const float x, const float y, const float z)
{
	RTvariable var;
	RT_CHECK_ERROR( rtContextDeclareVariable( context, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet3f( var, x, y, z ) );
	return var;
}

RTvariable applyProgramVariable1i(const RTcontext context, const RTprogram program, const char* name, const int value)
{
	RTvariable var;
	RT_CHECK_ERROR( rtProgramDeclareVariable( program, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1i( var, value ) );
	return var;
}

RTvariable applyProgramVariable3i(const RTcontext context, const RTprogram program, const char* name, const int x, const int y, const int z)
{
	RTvariable var;
	RT_CHECK_ERROR(rtProgramDeclareVariable(program, name, &var));
	RT_CHECK_ERROR(rtVariableSet3i(var, x, y, z));
	return var;
}

RTvariable applyProgramVariable1ui(const RTcontext context, const RTprogram program, const char* name, const unsigned int value)
{
	RTvariable var;
	RT_CHECK_ERROR( rtProgramDeclareVariable( program, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1ui( var, value ) );
	return var;
}

RTvariable applyProgramVariable1f(const RTcontext context, const RTprogram program, const char* name, const float value)
{
	RTvariable var;
	RT_CHECK_ERROR( rtProgramDeclareVariable( program, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1f( var, value ) );
	return var;
}

RTvariable applyProgramVariable2f(const RTcontext context, const RTprogram program, const char* name, const float x, const float y)
{
	RTvariable var;
	RT_CHECK_ERROR( rtProgramDeclareVariable( program, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet2f( var, x, y ) );
	return var;
}

RTvariable applyProgramVariable3f(const RTcontext context, const RTprogram program, const char* name, const float x, const float y, const float z)
{
	RTvariable var;
	RT_CHECK_ERROR( rtProgramDeclareVariable( program, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet3f( var, x, y, z ) );
	return var;
}

RTvariable applyProgramVariable(const RTcontext context, const RTprogram program, const char* name, const unsigned int size, const void* data)
{
	RTvariable var;
	RT_CHECK_ERROR( rtProgramDeclareVariable( program, name, &var ) );
	RT_CHECK_ERROR( rtVariableSetUserData( var, size, data ) );
	return var;
}

RTvariable applyMaterialVariable1i(const RTcontext context, const RTmaterial material, const char* name, const int value)
{
	RTvariable var;
	RT_CHECK_ERROR( rtMaterialDeclareVariable( material, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1i( var, value ) );
	return var;
}

RTvariable applyMaterialVariable1ui(const RTcontext context, const RTmaterial material, const char* name, const unsigned int value)
{
	RTvariable var;
	RT_CHECK_ERROR( rtMaterialDeclareVariable( material, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1ui( var, value ) );
	return var;
}

RTvariable applyMaterialVariable1f(const RTcontext context, const RTmaterial material, const char* name, const float value)
{
	RTvariable var;
	RT_CHECK_ERROR( rtMaterialDeclareVariable( material, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1f( var, value ) );
	return var;
}

RTvariable applyMaterialVariable3f(const RTcontext context, const RTmaterial material, const char* name, const float x, const float y, const float z)
{
	RTvariable var;
	RT_CHECK_ERROR( rtMaterialDeclareVariable( material, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet3f( var, x, y, z ) );
	return var;
}

void createBuffer1D(const RTcontext context, const RTbuffertype type, const RTformat format, const RTsize element_count, RTbuffer* buffer)
{
	RT_CHECK_ERROR( rtBufferCreate( context, type, buffer ) );
	RT_CHECK_ERROR( rtBufferSetFormat( *buffer, format ) );
	RT_CHECK_ERROR( rtBufferSetSize1D( *buffer, element_count ) );
}

void createCustomBuffer1D(const RTcontext context, const RTbuffertype type, const RTsize element_size, const RTsize element_count, RTbuffer* buffer)
{
	RT_CHECK_ERROR( rtBufferCreate( context, type, buffer ) );
	RT_CHECK_ERROR( rtBufferSetFormat( *buffer, RT_FORMAT_USER ) );
	RT_CHECK_ERROR( rtBufferSetElementSize( *buffer, element_size ) );
	RT_CHECK_ERROR( rtBufferSetSize1D( *buffer, element_count ) );
}

void createBuffer2D(const RTcontext context, const RTbuffertype type, const RTformat format, const RTsize x_count, const RTsize y_count, RTbuffer* buffer)
{
	RT_CHECK_ERROR( rtBufferCreate( context, type, buffer ) );
	RT_CHECK_ERROR( rtBufferSetFormat( *buffer, format ) );
	RT_CHECK_ERROR( rtBufferSetSize2D( *buffer, x_count, y_count ) );
}

void createCustomBuffer2D(const RTcontext context, const RTbuffertype type, const RTsize element_size, const RTsize x_count, const RTsize y_count, RTbuffer* buffer)
{
	RT_CHECK_ERROR( rtBufferCreate( context, type, buffer ) );
	RT_CHECK_ERROR( rtBufferSetFormat( *buffer, RT_FORMAT_USER ) );
	RT_CHECK_ERROR( rtBufferSetElementSize( *buffer, element_size ) );
	RT_CHECK_ERROR( rtBufferSetSize2D( *buffer, x_count, y_count ) );
}

void createBuffer3D(const RTcontext context, const RTbuffertype type, const RTformat format, const RTsize x_count, const RTsize y_count, const RTsize z_count, RTbuffer* buffer)
{
	RT_CHECK_ERROR( rtBufferCreate( context, type, buffer ) );
	RT_CHECK_ERROR( rtBufferSetFormat( *buffer, format ) );
	RT_CHECK_ERROR( rtBufferSetSize3D( *buffer, x_count, y_count, z_count ) );
}

void createCustomBuffer3D(const RTcontext context, const RTbuffertype type, const RTsize element_size, const RTsize x_count, const RTsize y_count, const RTsize z_count, RTbuffer* buffer)
{
	RT_CHECK_ERROR( rtBufferCreate( context, type, buffer ) );
	RT_CHECK_ERROR( rtBufferSetFormat( *buffer, RT_FORMAT_USER ) );
	RT_CHECK_ERROR( rtBufferSetElementSize( *buffer, element_size ) );
	RT_CHECK_ERROR( rtBufferSetSize3D( *buffer, x_count, y_count, z_count ) );
}

RTvariable applyContextObject(const RTcontext context, const char* name, const RTobject object)
{
	RTvariable var;
	RT_CHECK_ERROR( rtContextDeclareVariable( context, name, &var ) );
	RT_CHECK_ERROR( rtVariableSetObject( var, object ) );
	return var;
}

RTvariable applyProgramObject(const RTcontext context, const RTprogram program, const char* name, const RTobject object)
{
	RTvariable var;
	RT_CHECK_ERROR( rtProgramDeclareVariable( program, name, &var ) );
	RT_CHECK_ERROR( rtVariableSetObject( var, object ) );
	return var;
}

RTvariable applyGeometryObject(const RTcontext context, const RTgeometry geometry, const char* name, const RTobject object)
{
	RTvariable var;
	RT_CHECK_ERROR( rtGeometryDeclareVariable( geometry, name, &var ) );
	RT_CHECK_ERROR( rtVariableSetObject( var, object ) );
	return var;
}

RTvariable applyGeometryInstanceObject(const RTcontext context, const RTgeometryinstance instance, const char* name, const RTobject object)
{
	RTvariable var;
	RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( instance, name, &var ) );
	RT_CHECK_ERROR( rtVariableSetObject( var, object ) );
	return var;
}

void copyToBufferi(const RTcontext context, const RTbuffer buffer, const IntArray *a)
{
	int *data;

	RT_CHECK_ERROR(rtBufferMap(buffer, (void**)&data));
	memcpy(data, a->array, a->count*sizeof(int));
	RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void copyToBufferf(const RTcontext context, const RTbuffer buffer, const FloatArray *a)
{
	float *data;

	RT_CHECK_ERROR(rtBufferMap(buffer, (void**)&data));
	memcpy(data, a->array, a->count*sizeof(float));
	RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void copyToBufferdl(const RTcontext context, const RTbuffer buffer, const DistantLightArray *a)
{
	DistantLight *data;

	RT_CHECK_ERROR(rtBufferMap(buffer, (void**)&data));
	memcpy(data, a->array, a->count*sizeof(DistantLight));
	RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void initArrayi(IntArray *a, const size_t initialSize)
{
	a->array = (int *)malloc(initialSize * sizeof(int));
	if (a->array == NULL)
		error(SYSTEM, "out of memory in initArrayi");
	a->count = 0;
	a->size = initialSize;
}

int insertArrayi(IntArray *a, const int element)
{
	if (a->count == a->size) {
		a->size *= 2;
		a->array = (int *)realloc(a->array, a->size * sizeof(int));
		if (a->array == NULL)
			error(SYSTEM, "out of memory in insertArrayi");
	}
	return a->array[a->count++] = element;
}

int insertArray2i(IntArray *a, const int x, const int y)
{
	if (a->count + 1 >= a->size) {
		a->size *= 2;
		a->array = (int *)realloc(a->array, a->size * sizeof(int));
		if (a->array == NULL)
			error(SYSTEM, "out of memory in insertArray2i");
	}
	a->array[a->count++] = x;
	return a->array[a->count++] = y;
}

int insertArray3i(IntArray *a, const int x, const int y, const int z)
{
	if (a->count + 2 >= a->size) {
		a->size *= 2;
		a->array = (int *)realloc(a->array, a->size * sizeof(int));
		if (a->array == NULL)
			error(SYSTEM, "out of memory in insertArray3i");
	}
	a->array[a->count++] = x;
	a->array[a->count++] = y;
	return a->array[a->count++] = z;
}

void freeArrayi(IntArray *a)
{
	free(a->array);
	a->array = NULL;
	a->count = a->size = 0;
}

void initArrayf(FloatArray *a, const size_t initialSize)
{
	a->array = (float *)malloc(initialSize * sizeof(float));
	if (a->array == NULL)
		error(SYSTEM, "out of memory in initArrayi");
	a->count = 0;
	a->size = initialSize;
}

float insertArrayf(FloatArray *a, const float element)
{
	if (a->count == a->size) {
		a->size *= 2;
		a->array = (float *)realloc(a->array, a->size * sizeof(float));
		if (a->array == NULL)
			error(SYSTEM, "out of memory in insertArrayf");
	}
	return a->array[a->count++] = element;
}

float insertArray2f(FloatArray *a, const float x, const float y)
{
	if (a->count + 1 >= a->size) {
		a->size *= 2;
		a->array = (float *)realloc(a->array, a->size * sizeof(float));
		if (a->array == NULL)
			error(SYSTEM, "out of memory in insertArray2f");
	}
	a->array[a->count++] = x;
	return a->array[a->count++] = y;
}

float insertArray3f(FloatArray *a, const float x, const float y, const float z)
{
	if (a->count + 2 >= a->size) {
		a->size *= 2;
		a->array = (float *)realloc(a->array, a->size * sizeof(float));
		if (a->array == NULL)
			error(SYSTEM, "out of memory in insertArray3f");
	}
	a->array[a->count++] = x;
	a->array[a->count++] = y;
	return a->array[a->count++] = z;
}

void freeArrayf(FloatArray *a)
{
	free(a->array);
	a->array = NULL;
	a->count = a->size = 0;
}

void initArraym(MaterialArray *a, const size_t initialSize)
{
	a->array = (RTmaterial *)malloc(initialSize * sizeof(RTmaterial));
	if (a->array == NULL)
		error(SYSTEM, "out of memory in initArraym");
	a->count = 0;
	a->size = initialSize;
}

RTmaterial insertArraym(MaterialArray *a, const RTmaterial element)
{
	if (a->count == a->size) {
		a->size *= 2;
		a->array = (RTmaterial *)realloc(a->array, a->size * sizeof(RTmaterial));
		if (a->array == NULL)
			error(SYSTEM, "out of memory in insertArraym");
	}
	return a->array[a->count++] = element;
}

void freeArraym(MaterialArray *a)
{
	free(a->array);
	a->array = NULL;
	a->count = a->size = 0;
}

void initArraydl(DistantLightArray *a, const size_t initialSize)
{
	a->array = (DistantLight *)malloc(initialSize * sizeof(DistantLight));
	if (a->array == NULL)
		error(SYSTEM, "out of memory in initArraydl");
	a->count = 0;
	a->size = initialSize;
}

DistantLight insertArraydl(DistantLightArray *a, const DistantLight element)
{
	if (a->count == a->size) {
		a->size *= 2;
		a->array = (DistantLight *)realloc(a->array, a->size * sizeof(DistantLight));
		if (a->array == NULL)
			error(SYSTEM, "out of memory in insertArraydl");
	}
	return a->array[a->count++] = element;
}

void freeArraydl(DistantLightArray *a)
{
	free(a->array);
	a->array = NULL;
	a->count = a->size = 0;
}

void handleError( const RTcontext context, const RTresult code, const char* file, const int line, const int etype )
{
	const char* message;
	rtContextGetErrorString( context, code, &message ); // This function allows context to be null.
	sprintf( errmsg, "%s\n(%s:%d)", message, file, line );
	error( etype, errmsg );
}

#ifdef DEBUG_OPTIX
static IntArray *error_log; // Keep track of error types and occurance frequencies as ordered pairs

void logException(const RTexception type)
{
	unsigned int i;

	if (!error_log) {
		error_log = (IntArray *)malloc(sizeof(IntArray));
		if (!error_log)
			error(SYSTEM, "out of memory in logException");
		initArrayi(error_log, (RT_EXCEPTION_USER - RT_EXCEPTION_PROGRAM_ID_INVALID) * 2);
	}

	for (i = 0u; i < error_log->count; i += 2u)
		if (type == error_log->array[i]) {
			error_log->array[i + 1]++;
			return;
		}

	insertArray2i(error_log, type, 1);
}

void flushExceptionLog(const char* location)
{
	unsigned int i;

	if (!error_log) return; // No errors!

	for (i = 0u; i < error_log->count; i += 2u)
		printException((RTexception)error_log->array[i], error_log->array[i + 1], location);

	freeArrayi(error_log);
	free(error_log);
	error_log = NULL;
}

void printException(const RTexception type, const int count, const char* location)
{
	char times[16];
	
	if (count < 1)
		return;

	if (count > 1)
		sprintf(times, " %i times", count);
	else
		times[0] = '\0'; // Empty string

	if (type < RT_EXCEPTION_USER) {
		char* msg;
		switch (type) {
		case RT_EXCEPTION_INF:
			msg = "Infinite result";			break;
		case RT_EXCEPTION_NAN:
			msg = "NAN result";					break;
		case RT_EXCEPTION_PROGRAM_ID_INVALID:
			msg = "Program ID not valid";		break;
		case RT_EXCEPTION_TEXTURE_ID_INVALID:
			msg = "Texture ID not valid";		break;
		case RT_EXCEPTION_BUFFER_ID_INVALID:
			msg = "Buffer ID not valid";		break;
		case RT_EXCEPTION_INDEX_OUT_OF_BOUNDS:
			msg = "Index out of bounds";		break;
		case RT_EXCEPTION_STACK_OVERFLOW:
			msg = "Stack overflow";				break;
		case RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS:
			msg = "Buffer index out of bounds";	break;
		case RT_EXCEPTION_INVALID_RAY:
			msg = "Invalid ray";				break;
		case RT_EXCEPTION_INTERNAL_ERROR:
			msg = "Internal error";				break;
		default:
			msg = "Unknown error";				break;
		}
		sprintf(errmsg, "%s occurred%s in %s", msg, times, location);
	} else {
		sprintf(errmsg, "Exception %i occurred%s in %s", type - RT_EXCEPTION_USER, times, location);
	}
	error(WARNING, errmsg);
}
#endif /* DEBUG_OPTIX */

//void programCreateFromPTX( RTcontext context, const char* ptx, const char* program_name, RTprogram* program )
//{
//#ifdef PTX_STRING
//	RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, program_name, program ) );
//#else
//	if ( prev_ptx != ptx ) {
//		prev_ptx = (char*)ptx; // cast to avoid const warning
//		ptxFile( path_to_ptx, ptx );
//	}
//	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, program_name, program ) );
//#endif
//}

void ptxFile( char* path, const char* name )
{
	char* fp;
	sprintf( path, "cuda_compile_ptx_generated_%s.cu.ptx", name );
	fp = getpath( path, getenv("RAYPATH"), R_OK );
	if ( fp )
		sprintf(path, "%s", fp);
	else {
		sprintf(errmsg, "File %s not found in RAYPATH.", path );
		error(SYSTEM, errmsg);
	}
	//sprintf( path, "%s/cuda_compile_ptx_generated_%s.cu.ptx", optix_ptx_dir, name );
	//mprintf("Referencing %s\n", path);
}

void reportProgress( const double progress, const double alarm )
{
	pctdone = progress;
	if (alarm > 0)
		report(0);
}

#ifdef TIMEOUT_CALLBACK
// Enable the use of OptiX timeout callbacks to help reduce the likelihood
// of kernel timeouts when rendering on GPUs that are also used for display
int timeoutCallback(void)
{
	int earlyexit = 0;
	clock_t callback_time = clock();
	//mprintf("OptiX kernel running...\n");
	mprintf("OptiX kernel running: %" PRIu64 " milliseconds since last callback.\n", MILLISECONDS(callback_time - last_callback_time));
	last_callback_time = callback_time;
	return earlyexit; 
}
#endif
