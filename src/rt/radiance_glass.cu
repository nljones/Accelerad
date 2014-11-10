/*
 * Copyright (c) 2013-2014 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_common.h"

using namespace optix;

/* Context variables */
rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type, , );
rtDeclareVariable(rtObject,     top_object, , );
//rtDeclareVariable(rtObject,     top_shadower, , );

rtDeclareVariable(float3,       CIE_rgbf, , ); /* This is the value [ CIE_rf, CIE_gf, CIE_bf ] from color.h */

rtDeclareVariable(float,        minweight, , ); /* minimum ray weight */
rtDeclareVariable(int,          maxdepth, , ); /* maximum recursion depth */

/* Material variables */
#ifdef HIT_TYPE
rtDeclareVariable(unsigned int, type, , ); /* The material type representing "glass" or "dielectric" */
#endif
rtDeclareVariable(float,        rindex, , ) = 1.52f; /* Refractive index, usually 1.52 */
rtDeclareVariable(float3,       color, , ); /* The material color given by the rad file "glass" object */

/* OptiX variables */
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow, rtPayload, );

/* Attributes */
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );


RT_METHOD float bright( const float3 &rgb );


RT_PROGRAM void closest_hit_shadow()
{
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

	float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	PerRayData_shadow new_prd;             
	float3 result = make_float3( 0.0f );
	float3 hit_point = ray.origin + t_hit * ray.direction;
	float3 mcolor = color;

	/* check transmission */
	bool hastrans = optix::fmaxf( mcolor ) > 1e-15f;
	if (!hastrans) {
		return;
	}
	mcolor = optix::fmaxf( mcolor, make_float3( 1e-15f ) ); // no color channel should be smaller than 1e-15

	/* get modifiers */
	// we'll skip this for now

	/* perturb normal */
	// if there's a bump map, we use that, else
	float pdot = -optix::dot( ray.direction, ffnormal );

	/* angular transmission */
	float cos2 = sqrtf( 1.0f + ( pdot * pdot - 1.0f ) / ( rindex * rindex ) );
	//if (hastrans) {
	mcolor = make_float3( powf( mcolor.x, 1.0f / cos2 ), powf( mcolor.y, 1.0f / cos2 ), powf( mcolor.z, 1.0f / cos2 ) );
	//}

	/* compute reflection */
	float r1e = (pdot - rindex*cos2) / (pdot + rindex*cos2);
	r1e *= r1e;
	float r1m = (1.0f/pdot - rindex/cos2) / (1.0f/pdot + rindex/cos2);
	r1m *= r1m;

	/* compute transmission */
	//if (hastrans) {
		float3 trans = 0.5f * (1.0f-r1e) * (1.0f-r1e) * mcolor / (1.0f - r1e * r1e * mcolor * mcolor);
		trans       += 0.5f * (1.0f-r1m) * (1.0f-r1m) * mcolor / (1.0f - r1m * r1m * mcolor * mcolor);

		/* modify by pattern */
		//trans *= pcol;

		/* transmitted ray */
		//new_prd.depth = prd.depth + 1;
		new_prd.target = prd_shadow.target;
		new_prd.result = make_float3( 0.0f );
		optix::Ray trans_ray = optix::make_Ray( hit_point, ray.direction, shadow_ray_type, ray_start( hit_point, ray.direction, ffnormal, RAY_START ), RAY_END );
		rtTrace(top_object, trans_ray, new_prd);
		result += new_prd.result * trans;
	//}

	// pass the color back up the tree
	prd_shadow.result = result;
}


RT_PROGRAM void closest_hit_radiance()
{
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

	float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	PerRayData_radiance new_prd;             
	float3 result = make_float3( 0.0f );
	float3 hit_point = ray.origin + t_hit * ray.direction;
	float3 mcolor = color;

	/* check transmission */
	bool hastrans = optix::fmaxf( mcolor ) > 1e-15f;
	if (hastrans) {
		mcolor = optix::fmaxf( mcolor, make_float3( 1e-15f ) ); // no color channel should be smaller than 1e-15
	} // else we return if it's a shadow ray, which it isn't

	/* get modifiers */
	// we'll skip this for now

	float transtest = 0.0f, transdist = 0.0f;
	float mirtest = 0.0f, mirdist = 0.0f;

	/* perturb normal */
	// if there's a bump map, we use that, else
	float pdot = -optix::dot( ray.direction, ffnormal );

	/* angular transmission */
	float cos2 = sqrtf( 1.0f + ( pdot * pdot - 1.0f ) / ( rindex * rindex ) );
	if (hastrans) {
		mcolor = make_float3( powf( mcolor.x, 1.0f / cos2 ), powf( mcolor.y, 1.0f / cos2 ), powf( mcolor.z, 1.0f / cos2 ) );
	}

	/* compute reflection */
	float r1e = (pdot - rindex*cos2) / (pdot + rindex*cos2);
	r1e *= r1e;
	float r1m = (1.0f/pdot - rindex/cos2) / (1.0f/pdot + rindex/cos2);
	r1m *= r1m;

	/* compute transmission */
	if (hastrans) {
		float3 trans = 0.5f * (1.0f-r1e) * (1.0f-r1e) * mcolor / (1.0f - r1e * r1e * mcolor * mcolor);
		trans       += 0.5f * (1.0f-r1m) * (1.0f-r1m) * mcolor / (1.0f - r1m * r1m * mcolor * mcolor);

		/* modify by pattern */
		//trans *= pcol;

		/* transmitted ray */
		new_prd.weight = prd.weight * fmaxf(trans);
		if (new_prd.weight >= minweight) {
			new_prd.depth = prd.depth;
			new_prd.ambient_depth = prd.ambient_depth;
			//new_prd.seed = prd.seed;//lcg( prd.seed );
			new_prd.state = prd.state;
#ifdef FILL_GAPS
			new_prd.primary = 0;
#endif
#ifdef RAY_COUNT
			new_prd.ray_count = 1;
#endif
#ifdef HIT_COUNT
			new_prd.hit_count = 0;
#endif
			Ray trans_ray = make_Ray( hit_point, ray.direction, radiance_ray_type, ray_start( hit_point, ray.direction, ffnormal, RAY_START ), RAY_END );
			rtTrace(top_object, trans_ray, new_prd);
			float3 rcol = new_prd.result * trans;
			result += rcol;
			transtest = 2.0f * bright( rcol );
			transdist = t_hit + new_prd.distance;
#ifdef RAY_COUNT
			prd.ray_count += new_prd.ray_count;
#endif
#ifdef HIT_COUNT
			prd.hit_count += new_prd.hit_count;
#endif
		}
	}
	// stop if it's a shadow ray, which it isn't

	/* compute reflectance */
	float3 refl = 0.5f * r1e * ( 1.0f + (1.0f-2.0f*r1e) * mcolor * mcolor ) / (1.0f - r1e * r1e * mcolor * mcolor );
	refl       += 0.5f * r1m * ( 1.0f + (1.0f-2.0f*r1m) * mcolor * mcolor ) / (1.0f - r1m * r1m * mcolor * mcolor );

	/* reflected ray */
	new_prd.weight = prd.weight * fmaxf(refl);
	new_prd.depth = prd.depth + 1;
	if (new_prd.weight >= minweight && new_prd.depth <= abs(maxdepth)) {
		new_prd.ambient_depth = prd.ambient_depth;
		//new_prd.seed = prd.seed;//lcg( prd.seed );
		new_prd.state = prd.state;
#ifdef FILL_GAPS
		new_prd.primary = 0;
#endif
#ifdef RAY_COUNT
		new_prd.ray_count = 1;
#endif
#ifdef HIT_COUNT
		new_prd.hit_count = 0;
#endif
		float3 R = reflect( ray.direction, ffnormal );
		Ray refl_ray = make_Ray( hit_point, R, radiance_ray_type, ray_start( hit_point, R, ffnormal, RAY_START ), RAY_END );
		rtTrace(top_object, refl_ray, new_prd);
		float3 rcol = new_prd.result * refl;
		result += rcol;
		mirtest = 2.0f * bright( rcol );
		mirdist = t_hit + new_prd.distance;
#ifdef RAY_COUNT
		prd.ray_count += new_prd.ray_count;
#endif
#ifdef HIT_COUNT
		prd.hit_count += new_prd.hit_count;
#endif
	}
  
	/* check distance */
	float d = bright( result );
	if (transtest > d)
		prd.distance = transdist;
	else if (mirtest > d)
		prd.distance = mirdist;
	else
		prd.distance = t_hit;

	// pass the color back up the tree
	prd.result = result;
	
#ifdef HIT_TYPE
	prd.hit_type = type;
#endif
}

RT_METHOD float bright( const float3 &rgb )
{
	return dot( rgb, CIE_rgbf );
}
