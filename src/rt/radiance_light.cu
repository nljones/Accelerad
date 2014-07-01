/*
 * Copyright (c) 2013-2014 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_common.h"

using namespace optix;

/* Material variables */
rtDeclareVariable(float3,       color, , );

/* OptiX variables */
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );

/* Attributes */
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

RT_PROGRAM void closest_hit_radiance()
{
	// no contribution to ambient calculation
	if ( prd.ambient_depth > 0 ) //TODO need a better test
		prd.result = make_float3( 0.0f );
	else
		prd.result = color;
	prd.distance = t_hit;

#ifdef HIT_TYPE
	prd.hit_type = MAT_LIGHT;
#endif
}
