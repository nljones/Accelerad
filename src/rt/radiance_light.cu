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
rtDeclareVariable(unsigned int, type, , ); /* The material type representing "light", "glow", or "spot" */
#endif
rtDeclareVariable(float3,       color, , );
rtDeclareVariable(float,        maxrad, , ) = RAY_END;
rtDeclareVariable(float,        siz, , ) = -1.0f;		/* output solid angle or area */
rtDeclareVariable(float,        flen, , );				/* focal length (negative if distant source) */
rtDeclareVariable(float3,       aim, , );				/* aim direction or center */
rtDeclareVariable(unsigned int, function, , ) = RT_PROGRAM_ID_NULL;		/* function or texture modifier */

/* OptiX variables */
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

/* Attributes */
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 


static __device__ __inline__ int spotout();
static __device__ __inline__ float texture_function( const float3& normal );


RT_PROGRAM void closest_hit_shadow()
{
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	//float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

	//float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	if ( t_hit > maxrad || spotout() )
		prd_shadow.result = make_float3( 0.0f );
	else if ( function > RT_PROGRAM_ID_NULL )
		prd.result = color * texture_function( world_shading_normal );
	else
		prd_shadow.result = color;
}

RT_PROGRAM void closest_hit_radiance()
{
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	//float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

	//float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	// no contribution to ambient calculation
	if ( t_hit > maxrad || prd.ambient_depth > 0 || spotout() ) //TODO need a better ambient test and handle maxrad < 0
		prd.result = make_float3( 0.0f );
	else if ( function > RT_PROGRAM_ID_NULL )
		prd.result = color * texture_function( world_shading_normal );
	else
		prd.result = color;
	prd.distance = t_hit;

#ifdef HIT_TYPE
	prd.hit_type = type;
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

static __device__ __inline__ float texture_function( const float3& normal )
{
#ifdef CALLABLE
	return function( ray.direction );
#else /* CALLLABLE */
	const float3 u = make_float3( 1.0f, 0.0f, 0.0f ); //TODO use orientation of fixture
	const float3 v = make_float3( 0.0f, 1.0f, 0.0f );
	const float3 w = make_float3( 0.0f, 0.0f, 1.0f );

	float phi = acosf( dot( ray.direction, w ) );
	float theta = atan2f( dot( ray.direction, v ), dot( ray.direction, u ) );
	theta += 2.0f * M_PIf * ( theta < 0.0f );

	/* Normalize to [0, 1] */
	phi *= M_1_PIf * 2.0f; //TODO some IES files are 0-180 degrees rather than 0-90 degrees
	theta *= M_1_PIf * 0.5f;

	float rdot = dot( ray.direction, normal );
	return rtTex2D<float>( function, phi, theta ) / fabsf( rdot ); // this is flatcorr from source.cal
#endif /* CALLLABLE */
}
