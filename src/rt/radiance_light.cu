/*
 * Copyright (c) 2013-2014 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_common.h"

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
#ifdef CALLABLE
rtDeclareVariable(rtCallableProgramId<float(float3)>, function, , );		/* function or texture modifier */
#else
rtDeclareVariable(int,          lindex, , ) = -1;		/* function or texture modifier */

/* Geometry instance variables */
rtBuffer<Light> light_sources;
#endif

/* Context variables */
rtDeclareVariable(int,          directvis, , );		/* Boolean switch for light source visibility (dv) */

/* OptiX variables */
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

/* Attributes */
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 


RT_METHOD int spotout();
#ifndef CALLABLE
RT_METHOD float texture_function( const float3& normal );
#endif


RT_PROGRAM void closest_hit_shadow()
{
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	//float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

	//float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	if ( t_hit > maxrad || spotout() || dot( world_shading_normal, ray.direction ) > 0.0f )
		prd_shadow.result = make_float3( 0.0f );
#ifdef CALLABLE
	else if ( function > RT_PROGRAM_ID_NULL )
		prd_shadow.result = color * function( ray.direction );
#else
	else if ( lindex > -1 )
		prd_shadow.result = color * texture_function( world_shading_normal );
#endif
	else
		prd_shadow.result = color;
}

RT_PROGRAM void closest_hit_radiance()
{
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	//float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

	//float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	// no contribution to ambient calculation
	if ( !directvis || t_hit > maxrad || prd.ambient_depth > 0 || spotout() || dot( world_shading_normal, ray.direction ) > 0.0f ) //TODO need a better ambient test and handle maxrad < 0
		prd.result = make_float3( 0.0f );
#ifdef CALLABLE
	else if ( function > RT_PROGRAM_ID_NULL )
		prd.result = color * function( ray.direction );
#else
	else if ( lindex > -1 )
		prd.result = color * texture_function( world_shading_normal );
#endif
	else
		prd.result = color;
	prd.distance = t_hit;

#ifdef HIT_TYPE
	prd.hit_type = type;
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
		if ( M_PIf * d > siz )
			return(1);	/* out */
		return(0);	/* OK */
	}
					/* local source */
	if ( siz < 2.0f * M_PIf * ( 1.0f + dot( aim, ray.direction ) ) )
		return(1);	/* out */
	return(0);	/* OK */
}

#ifndef CALLABLE
RT_METHOD float texture_function( const float3& normal )
{
	const Light light_source = light_sources[lindex];

	float phi = acosf( dot( ray.direction, normalize( light_source.w ) ) );
	float theta = atan2f( -dot( ray.direction, normalize( light_source.v ) ), -dot( ray.direction, normalize( light_source.u ) ) );
	theta += 2.0f * M_PIf * ( theta < 0.0f );

	/* Normalize to [0, 1] within range */
	phi = ( 180.0f * M_1_PIf * phi - light_source.min.x ) / ( light_source.max.x - light_source.min.x );
	theta = ( 180.0f * M_1_PIf * theta - light_source.min.y ) / ( light_source.max.y - light_source.min.y );

	float rdot = dot( ray.direction, normal );
	return light_source.multiplier * rtTex2D<float>( light_source.texture, phi, theta ) / fabsf( rdot ); // this is flatcorr from source.cal
}
#endif /* CALLLABLE */
