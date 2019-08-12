/*
 *  ambient_cloud_generator.cu - entry point for geometry sampling on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>
#include "optix_shader_ray.h"
#include "optix_point_common.h"

using namespace optix;

#ifdef DAYSIM_COMPATIBLE
#define threadIndex()	((launch_index.x + launch_dim.x * launch_index.y) / stride + segment_offset)
#else
#define threadIndex()	((launch_index.x + launch_dim.x * launch_index.y) / stride)
#endif

/* Program variables */
rtDeclareVariable(unsigned int,  stride, , ) = 1u; /* Spacing between used threads in warp. */

/* Contex variables */
//rtBuffer<PointDirection, 1>      cluster_buffer; /* input */
rtDeclareVariable(PointDirectionBuffer, cluster_buffer, , ); /* input */
rtBuffer<AmbientRecord, 1>       ambient_record_buffer; /* ambient record output */
#ifdef DAYSIM_COMPATIBLE
rtBuffer<DC, 2>                  ambient_dc_buffer; /* daylight coefficient output */
#endif
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(rtObject,      top_irrad, , );
rtDeclareVariable(unsigned int,  level, , ) = 0u;
#ifdef DAYSIM_COMPATIBLE
rtDeclareVariable(unsigned int,  segment_offset, , ) = 0u; /* Offset into data if computed with multiple segments */
#endif /* DAYSIM_COMPATIBLE */
rtDeclareVariable(unsigned int,  imm_irrad, , ) = 0u; /* Immediate irradiance (-I) */

/* OptiX variables */
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
//rtDeclareVariable(unsigned int, launch_index, rtLaunchIndex, );
//rtDeclareVariable(unsigned int, launch_dim,   rtLaunchDim, );


RT_PROGRAM void ambient_cloud_camera()
{
	// Check stride
	if ((launch_index.x + launch_dim.x * launch_index.y) % stride)
		return;
	const unsigned int index = threadIndex();
	if (index >= cluster_buffer.size())
		return;

	PerRayData_ambient_record prd;
	init_rand(&prd.state, launch_index.x + launch_dim.x * (launch_index.y + launch_dim.y * level));
	prd.result.pos = prd.result.val = make_float3( 0.0f );
	prd.result.lvl = level;
	prd.result.weight = 1.0f;
	for ( int i = level; i--; )
		prd.result.weight *= AVGREFL; // Compute weight as in makeambient() from ambient.c
	prd.result.rad = make_float2( 0.0f );
	prd.result.udir = 0; // Initialize in case something goes wrong
#ifdef DAYSIM_COMPATIBLE
	prd.dc = make_uint3(0, 0, index - segment_offset);
	daysimSet(prd.dc, 0.0f);
#endif
#ifdef RAY_COUNT
	prd.result.ray_count = 1;
#endif
#ifdef HIT_COUNT
	prd.result.hit_count = 0;
#endif

	// Get the position and normal of the ambient record to be created
	PointDirection cluster = cluster_buffer[index];

	if ( dot( cluster.dir, cluster.dir ) > FTINY ) { // Check that this is a valid ray
		float3 ray_direction = -normalize( cluster.dir ); // Ray will face opposite the normal direction
		const float tmax = ray_start( cluster.pos, RAY_START );
		if (imm_irrad && !level) {
			Ray ray = make_Ray(cluster.pos, ray_direction, AMBIENT_RECORD_RAY, 0.0f, tmax); // For rtrace, the position is already offset
			rtTrace(top_irrad, ray, prd);
		}
		else {
			Ray ray = make_Ray(cluster.pos - ray_direction * tmax, ray_direction, AMBIENT_RECORD_RAY, 0.0f, 2.0f * tmax);
			rtTrace(top_object, ray, prd);
		}
	}

	checkFinite(prd.result.val);
	checkFinite(prd.result.gdir);

	ambient_record_buffer[index] = prd.result;
#ifdef DAYSIM_COMPATIBLE
	if (ambient_dc_buffer.size().x)
		daysimCopy(&ambient_dc_buffer[make_uint2(0, index)], prd.dc);
#endif
}

RT_PROGRAM void exception()
{
	// Check stride
	if ((launch_index.x + launch_dim.x * launch_index.y) % stride)
		return;
	const unsigned int index = threadIndex();
	if (index >= ambient_record_buffer.size())
		return;

#ifdef PRINT_OPTIX
	rtPrintExceptionDetails();
#endif
	ambient_record_buffer[index].lvl = level;
	ambient_record_buffer[index].val = exceptionToFloat3(rtGetExceptionCode());
	ambient_record_buffer[index].weight = -1.0f;
}
