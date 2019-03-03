/*
 *  rtrace_cloud_generator.cu - entry point for geometry sampling for individual ray tracing on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>
#include "optix_shader_common.h"
#include "optix_point_common.h"

#define RING_BUFFER_SIZE	8

using namespace optix;

/* Contex variables */
rtBuffer<RayData, 2>             ray_buffer;
rtBuffer<PointDirection, 3>      seed_buffer;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(rtObject,      top_irrad, , );
rtDeclareVariable(unsigned int,  imm_irrad, , ) = 0u; /* Immediate irradiance (-I) */

/* OptiX variables */
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

RT_PROGRAM void cloud_generator()
{
	PerRayData_point_cloud prd;

	// Init random state
	init_rand(&prd.state, launch_index.x + launch_dim.x * launch_index.y);

	prd.index = make_uint3(launch_index, 0u);
	prd.seeds = seed_buffer.size().z;
	unsigned int loop = 2u * prd.seeds; // Prevent infinite looping

	float3 point_ring[RING_BUFFER_SIZE];
	float3 dir_ring[RING_BUFFER_SIZE];
	unsigned int ring_start = 0, ring_end = 0, ring_full = 0;

	float tmin = ray_start(ray_buffer[launch_index].origin, RAY_START);
	float tmax;
	if ( imm_irrad ) {
		tmax = 2.0f * tmin;
		tmin = 0.0f;
	} else {
		// Zero or negative aft clipping distance indicates infinity
		tmax = ray_buffer[launch_index].max;
		if (tmax <= FTINY) {
			tmax = RAY_END;
		}
	}

	Ray ray = make_Ray(ray_buffer[launch_index].origin, ray_buffer[launch_index].dir, POINT_CLOUD_RAY, tmin, tmax);

	while (prd.index.z < prd.seeds && loop--) {
		prd.forward = prd.reverse = make_float3(0.0f);
#ifdef ANTIMATTER
		prd.mask = 0u;
		prd.inside = 0;
#endif

		// Trace the current ray
		if ( imm_irrad )
			rtTrace(top_irrad, ray, prd);
		else
			rtTrace(top_object, ray, prd);

		// Add next forward ray to ring buffer
		if (isfinite(prd.point) && dot(prd.forward, prd.forward) > FTINY) { // NaN values will be false
			point_ring[ring_end] = prd.point;
			dir_ring[ring_end] = prd.forward;
			ring_end = (ring_end + 1) % RING_BUFFER_SIZE;
			ring_full = ring_start == ring_end;
		}

		// Add next reverse ray to ring buffer
		if (!ring_full && isfinite(prd.point) && dot(prd.reverse, prd.reverse) > FTINY) { // NaN values will be false
			point_ring[ring_end] = prd.point;
			dir_ring[ring_end] = prd.reverse;
			ring_end = (ring_end + 1) % RING_BUFFER_SIZE;
			ring_full = ring_start == ring_end;
		}

		if (!ring_full && ring_start == ring_end)
			break;

		// Prepare for next ray
		ray.origin = point_ring[ring_start];
		ray.direction = dir_ring[ring_start];
		ring_start = (ring_start + 1) % RING_BUFFER_SIZE;
		ring_full = 0;
		ray.tmin = ray_start(ray.origin, RAY_START);
		ray.tmax = RAY_END;
	}

	// If outdoors, there are no bounces, but we need to prevent junk data
	while (prd.index.z < prd.seeds) {
		clear(seed_buffer[prd.index]);
		prd.index.z++;
	}
}

RT_PROGRAM void exception()
{
#ifdef PRINT_OPTIX
	rtPrintExceptionDetails();
#endif
	uint3 index = make_uint3(launch_index, seed_buffer.size().z - 1u); // record error to last segment
	seed_buffer[index].pos = exceptionToFloat3(rtGetExceptionCode());
	seed_buffer[index].dir = make_float3( 0.0f );
#ifdef AMBIENT_CELL
	seed_buffer[index].cell = make_uint2(0);
#endif
}
