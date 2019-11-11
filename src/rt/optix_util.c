/*
 *  optix_util.c - common routines for simulations on GPUs.
 */

#include "accelerad_copyright.h"

#include "rtio.h"
#include "paths.h" /* Required for R_OK argument to getpath() */
#include "calcomp.h"

#include "accelerad.h"
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
static clock_t cumulative_time = 0;  /* cumulative timing of kernel fuctions */
#endif

int verbose_output = 1;	/* Print repetitive outputs. */

#ifdef REPORT_GPU_STATE
void printContextInfo( const RTcontext context )
{
	int value, device_count = 0, i;
	int* devices;
	RTsize size;
	char *path;
	RTsize sizes[2];

	if ( !context ) return;

	RT_CHECK_WARN(rtContextGetDeviceCount(context, &device_count));
	devices = (int*)malloc(sizeof(int) * device_count);
	if (device_count && !devices)
		error(INTERNAL, "out of memory in printContextInfo");
	RT_CHECK_WARN(rtContextGetDevices(context, devices));

	//RT_CHECK_ERROR( rtContextGetRunningState( context, &value ) );
	//mprintf("OptiX kernel running:             %s\n", value ? "Yes" : "No");
	RT_CHECK_WARN(rtContextGetAttribute(context, RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT, sizeof(value), &value));
	mprintf("OptiX maximum textures:           %i\n", value);
	RT_CHECK_WARN(rtContextGetAttribute(context, RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS, sizeof(value), &value)); // This can be set
	mprintf("OptiX host CPU threads:           %i\n", value);
	RT_CHECK_WARN(rtContextGetAttribute(context, RT_CONTEXT_ATTRIBUTE_USED_HOST_MEMORY, sizeof(size), &size));
	mprintf("OptiX host memory allocated:      %" PRIu64 " bytes\n", size);
	for (i = 0; i < device_count; i++) {
		value = devices[i];
		RT_CHECK_WARN(rtContextGetAttribute(context, RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY + value, sizeof(size), &size));
		mprintf("OptiX free memory on device %i:    %" PRIu64 " bytes\n", value, size);
	}
	RT_CHECK_WARN(rtContextGetAttribute(context, RT_CONTEXT_ATTRIBUTE_GPU_PAGING_ACTIVE, sizeof(value), &value));
	mprintf("OptiX software paging:            %s\n", value ? "Yes" : "No");
	RT_CHECK_WARN(rtContextGetAttribute(context, RT_CONTEXT_ATTRIBUTE_GPU_PAGING_FORCED_OFF, sizeof(value), &value)); // This can be set
	mprintf("OptiX software paging prohibited: %s\n", value ? "Yes" : "No");
#ifdef RTX
	RT_CHECK_WARN(rtContextGetAttribute(context, RT_CONTEXT_ATTRIBUTE_DISK_CACHE_ENABLED, sizeof(value), &value)); // This can be set
	mprintf("OptiX disk cache enabled:         %s\n", value ? "Yes" : "No");
	RT_CHECK_WARN(rtContextGetAttribute(context, RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION, sizeof(path), &path)); // This can be set
	mprintf("OptiX disk cache location:        %s\n", path);
	RT_CHECK_WARN(rtContextGetAttribute(context, RT_CONTEXT_ATTRIBUTE_DISK_CACHE_MEMORY_LIMITS, sizeof(sizes), &sizes)); // This can be set
	mprintf("OptiX disk cache limits:          %" PRIu64 " to %" PRIu64 "\n", sizes[0], sizes[1]);
#endif
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
	clock_t kernel_clock; // Timer in clock cycles for short jobs

#ifdef REPORT_GPU_STATE
	/* Print context attributes before */
	printContextInfo( context );
#endif

	/* Validate and compile if necessary */
	RT_CHECK_ERROR( rtContextValidate( context ) );
#if defined(DEBUG_OPTIX) && !defined(RTX)
	kernel_clock = clock();
	RT_CHECK_ERROR( rtContextCompile( context ) ); // This should happen automatically when necessary.
	kernel_clock = clock() - kernel_clock;
	if (kernel_clock > 1)
		tprintf(kernel_clock, "OptiX compile %u", entry);
#endif

	/* Start timers */
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
	if (verbose_output) {
		kernel_clock = clock() - kernel_clock;
		tprintf(kernel_clock, "OptiX kernel %u", entry);
#ifdef CUMULTATIVE_TIME
		cumulative_time += kernel_clock;
		tprint(cumulative_time, "OptiX kernel cumulative");
#endif
	}

#ifdef REPORT_GPU_STATE
	/* Print context attributes after */
	printContextInfo( context );
#endif
}

void printRayTracingTime(const clock_t clock)
{
	/* Print the given elapsed time for ray tracing */
	tprint(clock, "Ray tracing total");
}

/* Helper functions */

RTvariable applyContextVariable1i(const RTcontext context, const char* name, const int value)
{
	RTvariable var;
	RT_CHECK_ERROR( rtContextDeclareVariable( context, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1i( var, value ) );
	return var;
}

RTvariable applyContextVariable2i(const RTcontext context, const char* name, const int x, const int y)
{
	RTvariable var;
	RT_CHECK_ERROR(rtContextDeclareVariable(context, name, &var));
	RT_CHECK_ERROR(rtVariableSet2i(var, x, y));
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

RTvariable applyContextVariable(const RTcontext context, const char* name, const RTsize size, const void* data)
{
	RTvariable var;
	RT_CHECK_ERROR(rtContextDeclareVariable(context, name, &var));
	RT_CHECK_ERROR(rtVariableSetUserData(var, size, data));
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

RTvariable applyProgramVariable(const RTcontext context, const RTprogram program, const char* name, const RTsize size, const void* data)
{
	RTvariable var;
	RT_CHECK_ERROR( rtProgramDeclareVariable( program, name, &var ) );
	RT_CHECK_ERROR( rtVariableSetUserData( var, size, data ) );
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

void copyToBufferm(const RTcontext context, const RTbuffer buffer, const MaterialDataArray *a)
{
	MaterialData *data;

	RT_CHECK_ERROR(rtBufferMap(buffer, (void**)&data));
	memcpy(data, a->array, a->count*sizeof(MaterialData));
	RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void copyToBufferdl(const RTcontext context, const RTbuffer buffer, const DistantLightArray *a)
{
	DistantLight *data;

	RT_CHECK_ERROR(rtBufferMap(buffer, (void**)&data));
	memcpy(data, a->array, a->count*sizeof(DistantLight));
	RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

IntArray* initArrayi(const size_t initialSize)
{
	IntArray *a = (IntArray *)malloc(sizeof(IntArray));
	int *array = (int *)malloc(initialSize * sizeof(int));
	if (!a || !array)
		eprintf(SYSTEM, "out of memory in initArrayi, need %" PRIu64 " bytes", sizeof(IntArray) + initialSize * sizeof(int));
	a->array = array;
	a->count = 0;
	a->size = initialSize;
	return a;
}

int insertArrayi(IntArray *a, const int element)
{
	if (a->count == a->size) {
		a->size *= 2;
		a->array = (int *)realloc(a->array, a->size * sizeof(int));
		if (a->array == NULL)
			eprintf(SYSTEM, "out of memory in insertArrayi, need %" PRIu64 " bytes", a->size * sizeof(int));
	}
	return a->array[a->count++] = element;
}

int insertArray2i(IntArray *a, const int x, const int y)
{
	if (a->count + 1 >= a->size) {
		a->size *= 2;
		a->array = (int *)realloc(a->array, a->size * sizeof(int));
		if (a->array == NULL)
			eprintf(SYSTEM, "out of memory in insertArray2i, need %" PRIu64 " bytes", a->size * sizeof(int));
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
			eprintf(SYSTEM, "out of memory in insertArray3i, need %" PRIu64 " bytes", a->size * sizeof(int));
	}
	a->array[a->count++] = x;
	a->array[a->count++] = y;
	return a->array[a->count++] = z;
}

void freeArrayi(IntArray *a)
{
	free(a->array);
	free(a);
}

FloatArray* initArrayf(const size_t initialSize)
{
	FloatArray *a = (FloatArray *)malloc(sizeof(FloatArray));
	float *array = (float *)malloc(initialSize * sizeof(float));
	if (!a || !array)
		eprintf(SYSTEM, "out of memory in initArrayf, need %" PRIu64 " bytes", sizeof(FloatArray) + initialSize * sizeof(float));
	a->array = array;
	a->count = 0;
	a->size = initialSize;
	return a;
}

float insertArrayf(FloatArray *a, const float element)
{
	if (a->count == a->size) {
		a->size *= 2;
		a->array = (float *)realloc(a->array, a->size * sizeof(float));
		if (a->array == NULL)
			eprintf(SYSTEM, "out of memory in insertArrayf, need %" PRIu64 " bytes", a->size * sizeof(float));
	}
	return a->array[a->count++] = element;
}

float insertArray2f(FloatArray *a, const float x, const float y)
{
	if (a->count + 1 >= a->size) {
		a->size *= 2;
		a->array = (float *)realloc(a->array, a->size * sizeof(float));
		if (a->array == NULL)
			eprintf(SYSTEM, "out of memory in insertArray2f, need %" PRIu64 " bytes", a->size * sizeof(float));
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
			eprintf(SYSTEM, "out of memory in insertArray3f, need %" PRIu64 " bytes", a->size * sizeof(float));
	}
	a->array[a->count++] = x;
	a->array[a->count++] = y;
	return a->array[a->count++] = z;
}

void freeArrayf(FloatArray *a)
{
	free(a->array);
	free(a);
}

PointerArray* initArrayp(const size_t initialSize)
{
	PointerArray *a = (PointerArray *)malloc(sizeof(PointerArray));
	void **array = (void**)malloc(initialSize * sizeof(void*));
	if (!a || !array)
		eprintf(SYSTEM, "out of memory in initArrayp, need %" PRIu64 " bytes", sizeof(PointerArray) + initialSize * sizeof(void*));
	a->array = array;
	a->count = 0;
	a->size = initialSize;
	return a;
}

void* insertArrayp(PointerArray *a, void *element)
{
	if (a->count == a->size) {
		a->size *= 2;
		a->array = (void**)realloc(a->array, a->size * sizeof(void*));
		if (a->array == NULL)
			eprintf(SYSTEM, "out of memory in insertArrayp, need %" PRIu64 " bytes", a->size * sizeof(void*));
	}
	return a->array[a->count++] = element;
}

void freeArrayp(PointerArray *a)
{
	free(a->array);
	free(a);
}

MaterialDataArray* initArraym(const size_t initialSize)
{
	MaterialDataArray *a = (MaterialDataArray *)malloc(sizeof(MaterialDataArray));
	MaterialData *array = (MaterialData *)malloc(initialSize * sizeof(MaterialData));
	if (!a || !array)
		eprintf(SYSTEM, "out of memory in initArraym, need %" PRIu64 " bytes", sizeof(MaterialDataArray) + initialSize * sizeof(MaterialData));
	a->array = array;
	a->count = 0;
	a->size = initialSize;
	return a;
}

MaterialData insertArraym(MaterialDataArray *a, const MaterialData element)
{
	if (a->count == a->size) {
		a->size *= 2;
		a->array = (MaterialData *)realloc(a->array, a->size * sizeof(MaterialData));
		if (a->array == NULL)
			eprintf(SYSTEM, "out of memory in insertArraym, need %" PRIu64 " bytes", a->size * sizeof(MaterialData));
	}
	return a->array[a->count++] = element;
}

void freeArraym(MaterialDataArray *a)
{
	free(a->array);
	free(a);
}

DistantLightArray* initArraydl(const size_t initialSize)
{
	DistantLightArray *a = (DistantLightArray *)malloc(sizeof(DistantLightArray));
	DistantLight *array = (DistantLight *)malloc(initialSize * sizeof(DistantLight));
	if (!a || !array)
		eprintf(SYSTEM, "out of memory in initArraydl, need %" PRIu64 " bytes", sizeof(DistantLightArray) + initialSize * sizeof(DistantLight));
	a->array = array;
	a->count = 0;
	a->size = initialSize;
	return a;
}

DistantLight insertArraydl(DistantLightArray *a, const DistantLight element)
{
	if (a->count == a->size) {
		a->size *= 2;
		a->array = (DistantLight *)realloc(a->array, a->size * sizeof(DistantLight));
		if (a->array == NULL)
			eprintf(SYSTEM, "out of memory in insertArraydl, need %" PRIu64 " bytes", a->size * sizeof(DistantLight));
	}
	return a->array[a->count++] = element;
}

void freeArraydl(DistantLightArray *a)
{
	free(a->array);
	free(a);
}

int evali(char *name, const int val)
{
	if (!varlookup(name)) {
		eprintf(WARNING, "function %s not found, default value %i used", name, val);
		return val;
	}
	return (int)(eval(name) + 0.5);
}

float evalf(char *name, const float val)
{
	if (!varlookup(name)) {
		eprintf(WARNING, "function %s not found, default value %g used", name, val);
		return val;
	}
	return (float)eval(name);
}

void handleError( const RTcontext context, const RTresult code, const char* file, const int line, const int etype )
{
	const char* message;
	rtContextGetErrorString( context, code, &message ); // This function allows context to be null.
	eprintf(etype, "%s\n(%s:%d)", message, file, line);
}

#ifdef DEBUG_OPTIX
static IntArray *error_log = NULL; // Keep track of error types and occurance frequencies as ordered pairs

void logException(const RTexception type)
{
	unsigned int i;

	if (!type) return; /* Not an error */

	if (!error_log)
		error_log = initArrayi((RT_EXCEPTION_USER - RT_EXCEPTION_PROGRAM_ID_INVALID) * 2);

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
	error_log = NULL;
}

void printException(const RTexception code, const int count, const char* location)
{
	char times[16];
	const RTexception type = code & ~RT_RETHROWN_EXCEPTION;
	
	if (count < 1)
		return;

	if (count > 1)
		sprintf(times, " %i times", count);
	else
		times[0] = '\0'; // Empty string

	if (type < RT_EXCEPTION_CUSTOM) {
		char *msg, unknown[30];
		switch (type) {
		case RT_EXCEPTION_INF:
			msg = "Infinite result";			break;
		case RT_EXCEPTION_NAN:
			msg = "NAN result";					break;
#ifdef RTX
		case RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS:
			msg = "Payload access out of bounds";					break;
		case RT_EXCEPTION_USER_EXCEPTION_CODE_OUT_OF_BOUNDS:
			msg = "Exception code of user exception out of bounds";	break;
		case RT_EXCEPTION_TRACE_DEPTH_EXCEEDED:
			msg = "Trace depth exceeded";		break;
#endif
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
			sprintf(unknown, "Unknown error (0x%03X)", type);
			msg = unknown;				        break;
		}
		sprintf(errmsg, "%s %s%s in %s", msg, code & RT_RETHROWN_EXCEPTION ? "inherited" : "occurred", times, location);
	} else {
		sprintf(errmsg, "Exception %i %s%s in %s", type - RT_EXCEPTION_CUSTOM, code & RT_RETHROWN_EXCEPTION ? "inherited" : "occurred", times, location);
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
	//sprintf( path, "cuda_compile_ptx_generated_%s.cu.ptx", name );
	sprintf(path, "%s.ptx", name);
	fp = getpath(path, getenv("RAYPATH"), R_OK);
	if ( fp )
		sprintf(path, "%s", fp);
	else
		eprintf(SYSTEM, "File %s not found in RAYPATH.", path);
	//sprintf( path, "%s/cuda_compile_ptx_generated_%s.cu.ptx", optix_ptx_dir, name );
	//mprintf("Referencing %s\n", path);
}

char* ptxString(const char* name)
{
	char *buffer = NULL;
	size_t length;
	FILE *f;

	ptxFile(path_to_ptx, name);

	if ((f = fopen(path_to_ptx, "rb")) == NULL)
		eprintf(SYSTEM, "cannot find file %s", path_to_ptx);

	fseek(f, 0, SEEK_END);
	length = ftell(f);
	fseek(f, 0, SEEK_SET);

	// 1 GiB; best not to load a hole large file in one string
	if (length > 1073741824) {
		eprintf(SYSTEM, "file %s is too large", path_to_ptx);
	}

	buffer = (char *)malloc(length + 1);

	if (length) {
		size_t read_length = fread(buffer, 1, length, f);

		if (length != read_length) {
			eprintf(SYSTEM, "failure reading %s", path_to_ptx);
		}
	}

	fclose(f);

	buffer[length] = '\0';

	return buffer;
}

/* Extract file name from full path. */
char* filename(char *path)
{
	char  *cp = path, *separator = NULL;

	while (*cp) {
		if (*cp == '\\' || *cp == '/') /* remove directory */
			separator = cp;
		//else
		//	*cp = tolower(*cp);
		cp++;
	}
	if (separator) {
		/* make sure the original pointer remains the same */
		//memmove(path, separator + 1, cp - separator);
		return separator + 1;
	}
	return path;
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

#ifdef RTX
void reportCallback(int verbosity, const char* tag, const char* message, void* payload)
{
	mprintf("%d %s: %s", verbosity, tag, message);
}
#endif
