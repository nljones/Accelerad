/*
 *  material_antimatter.cu - hit programs for the antimatter materials on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_ray.h"
#include "optix_point_common.h"

using namespace optix;


/* Context variables */
rtDeclareVariable(rtObject, top_object, , );


RT_CALLABLE_PROGRAM PerRayData_radiance closest_hit_antimatter_radiance(IntersectData const&data, PerRayData_radiance prd)
{
	/* Continue the ray */
	Ray new_ray = make_Ray(data.hit, data.ray_direction, RADIANCE_RAY, ray_start(data.hit, data.ray_direction, data.world_geometric_normal, RAY_START), prd.tmax);
	rtTrace(top_object, new_ray, prd);
	prd.distance += data.t;
	prd.mirror_distance += data.t;
	return prd;
}

RT_CALLABLE_PROGRAM PerRayData_shadow closest_hit_antimatter_shadow(IntersectData const&data, PerRayData_shadow prd)
{
	/* Continue the ray */
	Ray new_ray = make_Ray(data.hit, data.ray_direction, SHADOW_RAY, ray_start(data.hit, data.ray_direction, data.world_geometric_normal, RAY_START), RAY_END);
	rtTrace(top_object, new_ray, prd);
	return prd;
}

RT_CALLABLE_PROGRAM PerRayData_point_cloud closest_hit_antimatter_point_cloud(IntersectData const&data, PerRayData_point_cloud prd)
{
	/* Continue the ray */
	Ray new_ray = make_Ray(data.hit, data.ray_direction, POINT_CLOUD_RAY, ray_start(data.hit, data.ray_direction, data.world_geometric_normal, RAY_START), RAY_END);
	rtTrace(top_object, new_ray, prd);
	return prd;
}
