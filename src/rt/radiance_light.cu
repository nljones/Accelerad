/*
 * Copyright (c) 2013-2016 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_common.h"
#ifdef CONTRIB
#include "optix_shader_contrib.h"
#endif

using namespace optix;

/* Material variables */
#ifdef HIT_TYPE
rtDeclareVariable(unsigned int, type, , ); /* The material type representing "light", "illum", "glow", or "spot" */
#endif
rtDeclareVariable(float3,       color, , );
rtDeclareVariable(float,        maxrad, , ) = RAY_END;
rtDeclareVariable(float,        siz, , ) = -1.0f;		/* output solid angle or area */
rtDeclareVariable(float,        flen, , );				/* focal length (negative if distant source) */
rtDeclareVariable(float3,       aim, , );				/* aim direction or center */
rtDeclareVariable(rtCallableProgramId<float3(const float3, const float3)>, function, , );		/* function or texture modifier */

/* Context variables */
rtDeclareVariable(int,          directvis, , );		/* Boolean switch for light source visibility (dv) */
#ifdef ANTIMATTER
rtDeclareVariable(rtObject, top_object, , );
#endif

/* OptiX variables */
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

/* Attributes */
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(int, surface_id, attribute surface_id, );
#ifdef ANTIMATTER
rtDeclareVariable(int, mat_id, attribute mat_id, );
#endif


RT_METHOD int spotout();


RT_PROGRAM void closest_hit_shadow()
{
#ifdef ANTIMATTER
	if (prd_shadow.mask & (1 << mat_id)) {
		prd_shadow.result = make_float3(0.0f);
		return;
	}
#endif /* ANTIMATTER */

	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));

	if ( t_hit > maxrad || spotout() || dot( world_shading_normal, ray.direction ) > 0.0f || surface_id != -prd_shadow.target - 1 )
		prd_shadow.result = make_float3( 0.0f );
	else if ( function > RT_PROGRAM_ID_NULL )
		prd_shadow.result = color * function( ray.direction, world_shading_normal );
	else
		prd_shadow.result = color;
}

RT_PROGRAM void closest_hit_radiance()
{
#ifdef ANTIMATTER
	if (prd.mask & (1 << mat_id)) {
		float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
		prd.inside += dot(world_geometric_normal, ray.direction) < 0.0f ? 1 : -1;

		/* Continue the ray */
		float3 snormal = faceforward(world_geometric_normal, -ray.direction, world_geometric_normal);
		Ray new_ray = make_Ray(ray.origin, ray.direction, ray.ray_type, ray_start(ray.origin + t_hit * ray.direction, ray.direction, snormal, RAY_START) + t_hit, RAY_END);
		rtTrace(top_object, new_ray, prd);
		return;
	}
#endif /* ANTIMATTER */

	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));

	// no contribution to ambient calculation
	if ( !directvis || 0.0f > maxrad && prd.depth > 0 || prd.ambient_depth > 0 || spotout() || dot( world_shading_normal, ray.direction ) > 0.0f ) //TODO need a better ambient test
		prd.result = make_float3( 0.0f );
	else if ( function > RT_PROGRAM_ID_NULL )
		prd.result = color * function( ray.direction, world_shading_normal );
	else
		prd.result = color;
	prd.distance = t_hit;

#ifdef HIT_TYPE
	prd.hit_type = type;
#endif
#ifdef CONTRIB
	contribution(prd.rcoef, prd.result, ray.direction);
#endif
}

RT_METHOD int spotout()
{
	if ( siz < -FTINY )
		return(0); /* Not a spotlight */
	if ( flen < -FTINY ) {		/* distant source */
		const float3 vd = aim - ray.origin;
		float d = dot( ray.direction, vd );
		/*			wrong side?
		if (d <= FTINY)
			return(1);	*/
		d = dot( vd, vd ) - d * d;
		return (M_PIf * d > siz); /* If true then out */
	}
					/* local source */
	return (siz < 2.0f * M_PIf * (1.0f + dot(aim, ray.direction)));	/* If true then out */
}
