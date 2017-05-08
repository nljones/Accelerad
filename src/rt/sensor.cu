/*
 *  sensor.cu - entry point for individual ray tracing on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>
#include "optix_shader_common.h"
#include "optiX_shader_ray.h"
#ifdef CONTRIB_DOUBLE
#include "optix_double.h"
#endif

using namespace optix;

/* Program variables */
rtDeclareVariable(unsigned int,  do_irrad, , ) = 0u; /* Calculate irradiance (-i) */

/* Contex variables */
rtBuffer<RayData, 2>             ray_buffer;
#ifdef DAYSIM_COMPATIBLE
rtBuffer<DC, 3>                  dc_buffer;
#endif
//rtBuffer<unsigned int, 2>        rnd_seeds;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(rtObject,      top_irrad, , );
rtDeclareVariable(unsigned int,  radiance_ray_type, , );
rtDeclareVariable(unsigned int,  radiance_primary_ray_type, , );
rtDeclareVariable(unsigned int,  imm_irrad, , ) = 0u; /* Immediate irradiance (-I) */

/* OptiX variables */
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;

//#define TIME_VIEW


RT_PROGRAM void ray_generator()
{
#ifdef TIME_VIEW
	clock_t t0 = clock();
	ray_buffer[launch_index].val = make_float3( t0 );
#endif
	PerRayData_radiance prd;
	init_rand(&prd.state, launch_index.x + launch_dim.x * launch_index.y);
	prd.weight = 1.0f;
	prd.depth = 0;
	prd.ambient_depth = 0;
	//prd.seed = rnd_seeds[launch_index];
#ifdef CONTRIB
	prd.rcoef = make_contrib3(1.0f); //Probably not necessary
#endif
#ifdef ANTIMATTER
	prd.mask = 0u;
	prd.inside = 0;
#endif
#ifdef DAYSIM_COMPATIBLE
	prd.dc = make_uint3(0, launch_index.x, launch_index.y);
#endif
	setupPayload(prd);

	const float tmin = ray_start( ray_buffer[launch_index].origin, RAY_START );
	if ( imm_irrad ) {
		Ray ray = make_Ray(ray_buffer[launch_index].origin, ray_buffer[launch_index].dir, radiance_ray_type, -tmin, tmin);
		rtTrace(top_irrad, ray, prd);
	} else {
		// Zero or negative aft clipping distance indicates infinity
		float aft = ray_buffer[launch_index].max;
		if (aft <= FTINY) {
			aft = RAY_END;
		}

		Ray ray = make_Ray(ray_buffer[launch_index].origin, ray_buffer[launch_index].dir, do_irrad ? radiance_primary_ray_type : radiance_ray_type, tmin, aft);
		rtTrace(top_object, ray, prd);
	}

	checkFinite(prd.result);

#ifdef TIME_VIEW
	clock_t t1 = clock();
 
	float expected_fps   = 1.0f;
	float pixel_time     = ( t1 - t0 ) * time_view_scale * expected_fps;
	ray_buffer[launch_index].val = make_float3( pixel_time );
#else
	ray_buffer[launch_index].val = prd.result;
#endif
	//ray_buffer[launch_index].hit = ray_buffer[launch_index].origin + prd.distance * ray_buffer[launch_index].dir;
	ray_buffer[launch_index].weight = prd.weight;
	ray_buffer[launch_index].length = prd.distance;
	//ray_buffer[launch_index].t = prd.distance;
#ifdef RAY_COUNT
	ray_buffer[launch_index].ray_count = prd.ray_count;
#endif
#ifdef DAYSIM_COMPATIBLE
	if (dc_buffer.size().x)
		daysimCopy(&dc_buffer[prd.dc], prd.dc);
#endif
}

RT_PROGRAM void exception()
{
#ifdef PRINT_OPTIX
	rtPrintExceptionDetails();
#endif
#ifdef TIME_VIEW
	clock_t t1 = clock();
 
	float expected_fps   = 1.0f;
	float ray_time       = ( t1 - ray_buffer[launch_index].val.x ) * time_view_scale * expected_fps;
	ray_buffer[index].val = make_float3( ray_time );
#else
	ray_buffer[launch_index].val = exceptionToFloat3(rtGetExceptionCode());
	ray_buffer[launch_index].weight = -1.0f;
#endif
}
