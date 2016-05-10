/*
* Copyright (c) 2013-2016 Nathaniel Jones
* Massachusetts Institute of Technology
*/

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_common.h"
#include "optix_point_common.h"

using namespace optix;

#ifdef ANTIMATTER

/* Material variables */
#ifdef HIT_TYPE
rtDeclareVariable(unsigned int, type, , ); /* The material type representing "antimatter" */
#endif
rtDeclareVariable(unsigned int, mask, , ) = 0u; /* Bitmask of materials to be clipped. */

/* Context variables */
rtDeclareVariable(rtObject, top_object, , );

/* OptiX variables */
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(PerRayData_ambient_record, prd_ambient, rtPayload, );
rtDeclareVariable(PerRayData_point_cloud, prd_point_cloud, rtPayload, );

/* Attributes */
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );


RT_PROGRAM void closest_hit_radiance()
{
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));

	if (dot(world_geometric_normal, ray.direction) < 0.0f) {
		/* Entering a volume */
		prd.mask |= mask;
	}
	//else if (prd_point_cloud.mask && prd_point_cloud.inside > 0 && alt_mat > -1) {
	//	/* Leaving a volume and rendering the alternate material */
	// TODO implement this
	//	return;
	//}
	else {
		/* Just leave the volume */
		prd.mask &= ~mask;
	}

	/* Continue the ray */
	float3 hit_point = ray.origin + t_hit * ray.direction;
	float3 snormal = faceforward(world_geometric_normal, -ray.direction, world_geometric_normal);
	Ray new_ray = make_Ray(ray.origin, ray.direction, ray.ray_type, ray_start(hit_point, ray.direction, snormal, RAY_START) + t_hit, RAY_END);
	rtTrace(top_object, new_ray, prd);
}


RT_PROGRAM void closest_hit_shadow()
{
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));

	if (dot(world_geometric_normal, ray.direction) < 0.0f) {
		/* Entering a volume */
		prd_shadow.mask |= mask;
	}
	//else if (prd_point_cloud.mask && prd_point_cloud.inside > 0 && alt_mat > -1) {
	//	/* Leaving a volume and rendering the alternate material */
	// TODO implement this
	//	return;
	//}
	else {
		/* Just leave the volume */
		prd_shadow.mask &= ~mask;
	}

	/* Continue the ray */
	float3 hit_point = ray.origin + t_hit * ray.direction;
	float3 snormal = faceforward(world_geometric_normal, -ray.direction, world_geometric_normal);
	Ray new_ray = make_Ray(ray.origin, ray.direction, ray.ray_type, ray_start(hit_point, ray.direction, snormal, RAY_START) + t_hit, RAY_END);
	rtTrace(top_object, new_ray, prd_shadow);
}


RT_PROGRAM void closest_hit_point_cloud()
{
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));

	if (dot(world_geometric_normal, ray.direction) < 0.0f) {
		/* Entering a volume */
		prd_point_cloud.mask |= mask;
	}
	//else if (prd_point_cloud.mask && prd_point_cloud.inside > 0 && alt_mat > -1) {
	//	/* Leaving a volume and rendering the alternate material */
	// TODO implement this
	//	return;
	//}
	else {
		/* Just leave the volume */
		prd_point_cloud.mask &= ~mask;
	}

	/* Continue the ray */
	float3 hit_point = ray.origin + t_hit * ray.direction;
	float3 snormal = faceforward(world_geometric_normal, -ray.direction, world_geometric_normal);
	Ray new_ray = make_Ray(ray.origin, ray.direction, ray.ray_type, ray_start(hit_point, ray.direction, snormal, RAY_START) + t_hit, RAY_END);
	rtTrace(top_object, new_ray, prd_point_cloud);
}

#endif /* ANTIMATTER */
