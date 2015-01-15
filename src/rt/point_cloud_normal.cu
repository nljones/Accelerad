/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_common.h"

using namespace optix;

/* OptiX variables */
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_point_cloud, prd, rtPayload, );

/* Attributes */
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );


RT_PROGRAM void any_hit_point_cloud_glass()
{
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

	prd.backup.pos = ray.origin + t_hit * ray.direction;
	prd.backup.dir = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	//TODO should probably use first intersection only and send transmitted ray
	rtIgnoreIntersection();
}

RT_PROGRAM void closest_hit_point_cloud()
{
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

	prd.result.pos = ray.origin + t_hit * ray.direction;
	prd.result.dir = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
}

RT_PROGRAM void point_cloud_miss()
{
	prd.result = prd.backup;
}
