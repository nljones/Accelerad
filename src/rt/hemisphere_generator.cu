/*
 *  hemisphere_generator.cu - entry point for geometry sampling after the first bounce on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>
#include "optix_shader_common.h"
#include "optix_point_common.h"

using namespace optix;

/* Contex variables */
//rtBuffer<PointDirection, 1>      cluster_buffer; /* input */
rtDeclareVariable(PointDirectionBuffer, cluster_buffer, , ); /* input */
rtBuffer<PointDirection, 3>      seed_buffer; /* output */
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  segment_offset, , ) = 0u; /* Offset into data if computed with multiple segments */

/* OptiX variables */
rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim,   rtLaunchDim, );

RT_PROGRAM void hemisphere_camera()
{
	PerRayData_point_cloud prd;
	clear(seed_buffer[launch_index]);

	PointDirection eye = cluster_buffer[launch_index.z + segment_offset];

	// Check for valid input
	if ( isfinite( eye.pos ) && isfinite( eye.dir ) && dot( eye.dir, eye.dir ) > FTINY ) { // NaN values will be false
		// Init random state
		init_rand(&prd.state, launch_index.x + launch_dim.x * (launch_index.y + launch_dim.y * launch_index.z));

		// Make axes
		float3 uz = normalize(eye.dir);
		float3 ux = getperpendicular(uz, prd.state);
		float3 uy = cross(uz, ux);
						/* avoid coincident samples */
		float2 spt = 0.1f + 0.8f * make_float2(curand_uniform(prd.state), curand_uniform(prd.state));
		SDsquare2disk(spt, (launch_index.y + spt.y) / launch_dim.y, (launch_index.x + spt.x) / launch_dim.x);
		float zd = sqrtf(1.0f - dot(spt, spt));
		float3 rdir = normalize(spt.x * ux + spt.y * uy + zd * uz);

		prd.index = launch_index;
		prd.seeds = launch_index.z + 1;
#ifdef ANTIMATTER
		prd.mask = 0u;
		prd.inside = 0;
#endif

		// Trace the current ray
		Ray ray = make_Ray(eye.pos, rdir, POINT_CLOUD_RAY, ray_start( eye.pos, rdir, uz, RAY_START ), RAY_END);
		rtTrace(top_object, ray, prd);
	}
}

RT_PROGRAM void exception()
{
#ifdef PRINT_OPTIX
	rtPrintExceptionDetails();
#endif
	seed_buffer[launch_index].pos = exceptionToFloat3(rtGetExceptionCode());
	seed_buffer[launch_index].dir = make_float3( 0.0f );
#ifdef AMBIENT_CELL
	seed_buffer[launch_index].cell = make_uint2(0);
#endif
}
