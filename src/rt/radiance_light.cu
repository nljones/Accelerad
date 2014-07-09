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
rtDeclareVariable(float,        maxrad, , ) = RAY_END;
rtDeclareVariable(float,        siz, , ) = -1.0f;		/* output solid angle or area */
rtDeclareVariable(float,        flen, , );				/* focal length (negative if distant source) */
rtDeclareVariable(float3,       aim, , );				/* aim direction or center */

/* OptiX variables */
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

/* Attributes */
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 


static __device__ __inline__ int spotout();


RT_PROGRAM void closest_hit_shadow()
{
	if ( t_hit > maxrad || spotout() )
		prd_shadow.result = make_float3( 0.0f );
	else
		prd_shadow.result = color;
}

RT_PROGRAM void closest_hit_radiance()
{
	// no contribution to ambient calculation
	if ( t_hit > maxrad || prd.ambient_depth > 0 || spotout() ) //TODO need a better ambient test and handle maxrad < 0
		prd.result = make_float3( 0.0f );
	else
		prd.result = color;
	prd.distance = t_hit;

#ifdef HIT_TYPE
	prd.hit_type = MAT_LIGHT;
#endif
}

static __device__ __inline__ int spotout()
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
		if ( M_PIf * d > siz )
			return(1);	/* out */
		return(0);	/* OK */
	}
					/* local source */
	if ( siz < 2.0f * M_PIf * ( 1.0f + dot( aim, ray.direction ) ) )
		return(1);	/* out */
	return(0);	/* OK */
}
