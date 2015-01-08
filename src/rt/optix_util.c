/*
 * Copyright (c) 2013-2014 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <stdio.h>

#include <standard.h> /* TODO just get the includes that are required? */
#include "paths.h" /* Required for R_OK argument to getpath() */

#include "optix_radiance.h"


#ifdef TIMEOUT_CALLBACK
static clock_t last_callback_time;  /* time of last callback from GPU */
#endif

#ifdef CUMULTATIVE_TIME
static unsigned long cumulative_millis = 0uL;  /* cumulative timing of kernel fuctions */
#endif


#ifdef REPORT_GPU_STATE
void printContextInfo( const RTcontext context )
{
	int value;

	if ( !context ) return;

	//RT_CHECK_ERROR( rtContextGetRunningState( context, &value ) );
	//fprintf(stderr, "OptiX kernel running:             %s\n", value ? "Yes" : "No");
	RT_CHECK_ERROR( rtContextGetAttribute( context, RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT, sizeof(int), &value ) );
	fprintf(stderr, "OptiX maximum textures:           %i\n", value);
	RT_CHECK_ERROR( rtContextGetAttribute( context, RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS, sizeof(int), &value ) ); // This can be set
	fprintf(stderr, "OptiX host CPU threads:           %i\n", value);
	RT_CHECK_ERROR( rtContextGetAttribute( context, RT_CONTEXT_ATTRIBUTE_USED_HOST_MEMORY, sizeof(RTsize), &value ) );
	fprintf(stderr, "OptiX host memory allocated:      %u bytes\n", value);
	RT_CHECK_ERROR( rtContextGetAttribute( context, RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY, sizeof(RTsize), &value ) );
	fprintf(stderr, "OptiX free device memory:         %u bytes\n", value);
	RT_CHECK_ERROR( rtContextGetAttribute( context, RT_CONTEXT_ATTRIBUTE_GPU_PAGING_ACTIVE, sizeof(int), &value ) );
	fprintf(stderr, "OptiX software paging:            %s\n", value ? "Yes" : "No");
	RT_CHECK_ERROR( rtContextGetAttribute( context, RT_CONTEXT_ATTRIBUTE_GPU_PAGING_FORCED_OFF, sizeof(int), &value ) ); // This can be set
	fprintf(stderr, "OptiX software paging prohibited: %s\n", value ? "Yes" : "No");
}
#endif

void runKernel1D( const RTcontext context, const unsigned int entry, const int size )
{
	runKernel3D( context, entry, size, 0, 0 );
}

void runKernel2D( const RTcontext context, const unsigned int entry, const int width, const int height )
{
	runKernel3D( context, entry, width, height, 0 );
}

void runKernel3D( const RTcontext context, const unsigned int entry, const int width, const int height, const int depth )
{
	/* Timers */
	time_t kernel_start_time, kernel_end_time; // Timer in seconds for long jobs
	clock_t kernel_start_clock, kernel_end_clock; // Timer in clock cycles for short jobs

#ifdef REPORT_GPU_STATE
	/* Print context attributes before */
	printContextInfo( context );
#endif

	/* Validate and compile if necessary */
	RT_CHECK_ERROR( rtContextValidate( context ) );
	kernel_start_clock = clock();
	RT_CHECK_ERROR( rtContextCompile( context ) ); // This should happen automatically when necessary.
	kernel_end_clock = clock();
	if ( kernel_end_clock - kernel_start_clock )
		fprintf(stderr, "OptiX compile time: %u milliseconds.\n", (kernel_end_clock - kernel_start_clock) * 1000 / CLOCKS_PER_SEC);

	/* Start timers */
	kernel_start_time = time((time_t *)NULL);
	kernel_start_clock = clock();
#ifdef TIMEOUT_CALLBACK
	last_callback_time = kernel_start_clock;
#endif

	/* Run */
	if ( depth )
		RT_CHECK_ERROR( rtContextLaunch3D( context, entry, width, height, depth ) );
	else if ( height )
		RT_CHECK_ERROR( rtContextLaunch2D( context, entry, width, height ) );
	else
		RT_CHECK_ERROR( rtContextLaunch1D( context, entry, width ) );

	/* Stop timers */
	kernel_end_clock = clock();
	kernel_end_time = time((time_t *)NULL);
	fprintf(stderr, "OptiX kernel time: %u milliseconds (%u seconds).\n", (kernel_end_clock - kernel_start_clock) * 1000 / CLOCKS_PER_SEC, kernel_end_time - kernel_start_time);
#ifdef CUMULTATIVE_TIME
	cumulative_millis += (kernel_end_clock - kernel_start_clock) * 1000 / CLOCKS_PER_SEC;
	fprintf(stderr, "OptiX kernel cumulative time: %u milliseconds.\n", cumulative_millis);
#endif

#ifdef REPORT_GPU_STATE
	/* Print context attributes after */
	printContextInfo( context );
#endif
}

/* Helper functions */

void applyContextVariable1i( const RTcontext context, const char* name, const int value )
{
	RTvariable var;
	RT_CHECK_ERROR( rtContextDeclareVariable( context, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1i( var, value ) );
}

void applyContextVariable1ui( const RTcontext context, const char* name, const unsigned int value )
{
	RTvariable var;
	RT_CHECK_ERROR( rtContextDeclareVariable( context, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1ui( var, value ) );
}

void applyContextVariable1f( const RTcontext context, const char* name, const float value )
{
	RTvariable var;
	RT_CHECK_ERROR( rtContextDeclareVariable( context, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1f( var, value ) );
}

void applyContextVariable3f( const RTcontext context, const char* name, const float x, const float y, const float z )
{
	RTvariable var;
	RT_CHECK_ERROR( rtContextDeclareVariable( context, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet3f( var, x, y, z ) );
}

void applyProgramVariable1i( const RTcontext context, const RTprogram program, const char* name, const int value )
{
	RTvariable var;
	RT_CHECK_ERROR( rtProgramDeclareVariable( program, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1i( var, value ) );
}

void applyProgramVariable1ui( const RTcontext context, const RTprogram program, const char* name, const unsigned int value )
{
	RTvariable var;
	RT_CHECK_ERROR( rtProgramDeclareVariable( program, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1ui( var, value ) );
}

void applyProgramVariable1f( const RTcontext context, const RTprogram program, const char* name, const float value )
{
	RTvariable var;
	RT_CHECK_ERROR( rtProgramDeclareVariable( program, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1f( var, value ) );
}

void applyProgramVariable2f( const RTcontext context, const RTprogram program, const char* name, const float x, const float y )
{
	RTvariable var;
	RT_CHECK_ERROR( rtProgramDeclareVariable( program, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet2f( var, x, y ) );
}

void applyProgramVariable3f( const RTcontext context, const RTprogram program, const char* name, const float x, const float y, const float z )
{
	RTvariable var;
	RT_CHECK_ERROR( rtProgramDeclareVariable( program, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet3f( var, x, y, z ) );
}

void applyProgramVariable( const RTcontext context, const RTprogram program, const char* name, const unsigned int size, const void* data )
{
	RTvariable var;
	RT_CHECK_ERROR( rtProgramDeclareVariable( program, name, &var ) );
	RT_CHECK_ERROR( rtVariableSetUserData( var, size, data ) );
}

void applyMaterialVariable1i( const RTcontext context, const RTmaterial material, const char* name, const int value )
{
	RTvariable var;
	RT_CHECK_ERROR( rtMaterialDeclareVariable( material, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1i( var, value ) );
}

void applyMaterialVariable1ui( const RTcontext context, const RTmaterial material, const char* name, const unsigned int value )
{
	RTvariable var;
	RT_CHECK_ERROR( rtMaterialDeclareVariable( material, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1ui( var, value ) );
}

void applyMaterialVariable1f( const RTcontext context, const RTmaterial material, const char* name, const float value )
{
	RTvariable var;
	RT_CHECK_ERROR( rtMaterialDeclareVariable( material, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet1f( var, value ) );
}

void applyMaterialVariable3f( const RTcontext context, const RTmaterial material, const char* name, const float x, const float y, const float z )
{
	RTvariable var;
	RT_CHECK_ERROR( rtMaterialDeclareVariable( material, name, &var ) );
	RT_CHECK_ERROR( rtVariableSet3f( var, x, y, z ) );
}

void createBuffer1D( const RTcontext context, const RTbuffertype type, const RTformat format, const int element_count, RTbuffer* buffer )
{
	RT_CHECK_ERROR( rtBufferCreate( context, type, buffer ) );
	RT_CHECK_ERROR( rtBufferSetFormat( *buffer, format ) );
	RT_CHECK_ERROR( rtBufferSetSize1D( *buffer, element_count ) );
}

void createCustomBuffer1D( const RTcontext context, const RTbuffertype type, const int element_size, const int element_count, RTbuffer* buffer )
{
	RT_CHECK_ERROR( rtBufferCreate( context, type, buffer ) );
	RT_CHECK_ERROR( rtBufferSetFormat( *buffer, RT_FORMAT_USER ) );
	RT_CHECK_ERROR( rtBufferSetElementSize( *buffer, element_size ) );
	RT_CHECK_ERROR( rtBufferSetSize1D( *buffer, element_count ) );
}

void createBuffer2D( const RTcontext context, const RTbuffertype type, const RTformat format, const int x_count, const int y_count, RTbuffer* buffer )
{
	RT_CHECK_ERROR( rtBufferCreate( context, type, buffer ) );
	RT_CHECK_ERROR( rtBufferSetFormat( *buffer, format ) );
	RT_CHECK_ERROR( rtBufferSetSize2D( *buffer, x_count, y_count ) );
}

void createCustomBuffer2D( const RTcontext context, const RTbuffertype type, const int element_size, const int x_count, const int y_count, RTbuffer* buffer )
{
	RT_CHECK_ERROR( rtBufferCreate( context, type, buffer ) );
	RT_CHECK_ERROR( rtBufferSetFormat( *buffer, RT_FORMAT_USER ) );
	RT_CHECK_ERROR( rtBufferSetElementSize( *buffer, element_size ) );
	RT_CHECK_ERROR( rtBufferSetSize2D( *buffer, x_count, y_count ) );
}

void createBuffer3D( const RTcontext context, const RTbuffertype type, const RTformat format, const int x_count, const int y_count, const int z_count, RTbuffer* buffer )
{
	RT_CHECK_ERROR( rtBufferCreate( context, type, buffer ) );
	RT_CHECK_ERROR( rtBufferSetFormat( *buffer, format ) );
	RT_CHECK_ERROR( rtBufferSetSize3D( *buffer, x_count, y_count, z_count ) );
}

void createCustomBuffer3D( const RTcontext context, const RTbuffertype type, const int element_size, const int x_count, const int y_count, const int z_count, RTbuffer* buffer )
{
	RT_CHECK_ERROR( rtBufferCreate( context, type, buffer ) );
	RT_CHECK_ERROR( rtBufferSetFormat( *buffer, RT_FORMAT_USER ) );
	RT_CHECK_ERROR( rtBufferSetElementSize( *buffer, element_size ) );
	RT_CHECK_ERROR( rtBufferSetSize3D( *buffer, x_count, y_count, z_count ) );
}

void applyContextObject( const RTcontext context, const char* name, const RTobject object )
{
	RTvariable var;
	RT_CHECK_ERROR( rtContextDeclareVariable( context, name, &var ) );
	RT_CHECK_ERROR( rtVariableSetObject( var, object ) );
}

void applyProgramObject( const RTcontext context, const RTprogram program, const char* name, const RTobject object )
{
	RTvariable var;
	RT_CHECK_ERROR( rtProgramDeclareVariable( program, name, &var ) );
	RT_CHECK_ERROR( rtVariableSetObject( var, object ) );
}

void applyGeometryObject( const RTcontext context, const RTgeometry geometry, const char* name, const RTobject object )
{
	RTvariable var;
	RT_CHECK_ERROR( rtGeometryDeclareVariable( geometry, name, &var ) );
	RT_CHECK_ERROR( rtVariableSetObject( var, object ) );
}

void applyGeometryInstanceObject( const RTcontext context, const RTgeometryinstance instance, const char* name, const RTobject object )
{
	RTvariable var;
	RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( instance, name, &var ) );
	RT_CHECK_ERROR( rtVariableSetObject( var, object ) );
}

void handleError( const RTcontext context, const RTresult code, const char* file, const int line, const int etype )
{
	const char* message;
	rtContextGetErrorString( context, code, &message ); // This function allows context to be null.
	sprintf( errmsg, "%s\n(%s:%d)", message, file, line );
	error( etype, errmsg );
}

void printException( const float3 code, const char* location, const int index )
{
	if ( code.x == 0.0f ) {
		char* msg;
		switch ( (int)( RT_EXCEPTION_USER - code.y ) ) {
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
		}
		fprintf(stderr, "Exception on %s %i: %s\n", location, index, msg );
	} else {
		fprintf(stderr, "Exception on %s %i: User exception %g\n", location, index, code.y );
	}
}

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
		sprintf( path, fp );
	else {
		sprintf(errmsg, "File %s not found in RAYPATH.", path );
		error(SYSTEM, errmsg);
	}
	//sprintf( path, "%s/cuda_compile_ptx_generated_%s.cu.ptx", optix_ptx_dir, name );
	//fprintf( stderr, "Referencing %s\n", path );
}

#ifdef TIMEOUT_CALLBACK
// Enable the use of OptiX timeout callbacks to help reduce the likelihood
// of kernel timeouts when rendering on GPUs that are also used for display
int timeoutCallback(void)
{
	int earlyexit = 0;
	clock_t callback_time = clock();
	//fprintf(stderr, "OptiX kernel running...\n");
	fprintf(stderr, "OptiX kernel running: %u milliseconds since last callback.\n", (callback_time - last_callback_time) * 1000 / CLOCKS_PER_SEC);
	last_callback_time = callback_time;
	return earlyexit; 
}
#endif
