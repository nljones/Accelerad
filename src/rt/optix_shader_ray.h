/*
 *  optix_shader_ray.h - structures and routines for ray types on GPUs.
 */

#pragma once

#include "accelerad_copyright.h"

#include "optix_shader_common.h"
#include "optix_shader_daysim.h"
#include "optix_shader_random.h"

/* Ray payloads */
struct PerRayData_radiance
{
	float3 result;
	float distance;
	float3 mirror;
	float mirror_distance;
	float tmax;			/* maximum distance (aft clipping plane) */
	float weight;
	int depth;
	int ambient_depth;
	rand_state* state;
#ifdef CONTRIB
	contrib3 rcoef;
#endif
#ifdef ANTIMATTER
	int inside;			/* counter for number of volumes traversed */
	unsigned int mask;	/* mask for materials to skip */
#endif
#ifdef DAYSIM_COMPATIBLE
	DaysimCoef dc;	/* daylight coefficients */
#endif
#ifdef RAY_COUNT
	int ray_count;
#endif
#ifdef HIT_COUNT
	int hit_count;
#endif
#ifdef HIT_TYPE
	int hit_type;
#endif
};

struct PerRayData_shadow
{
	float3 result;
	int target;
#ifdef CONTRIB
	contrib3 rcoef;
#endif
#ifdef ANTIMATTER
	int inside;			/* counter for number of volumes traversed */
	unsigned int mask;	/* mask for materials to skip */
#endif
#ifdef DAYSIM_COMPATIBLE
	DaysimCoef dc;	/* daylight coefficients */
#endif
};

struct PerRayData_ambient
{
	float3 result;
	int ambient_depth;
	float3 surface_point;
	float wsum;
	float3 surface_normal;
	float weight;
#ifdef DAYSIM_COMPATIBLE
	DaysimCoef dc;	/* daylight coefficients */
#endif
#ifdef HIT_COUNT
	int hit_count;
#endif
};

struct PerRayData_ambient_record
{
	AmbientRecord result;
	rand_state* state;
#ifdef DAYSIM_COMPATIBLE
	DaysimCoef dc;	/* daylight coefficients */
#endif
};

struct IntersectData
{
	float3 ray_direction;	/* Ray direction */
	int ray_type;		/* Ray type */
	float3 hit;			/* Ray hit point */
	float t;			/* Ray hit parameter */
	float3 world_geometric_normal;	/* World geometric normal */
	float3 world_shading_normal;	/* World shading normal */
	int surface_id;		/* Unique ID per surface (not necessarily unique per triangle) */
	MaterialData mat;	/* Material properties of intersected triangle */
};

rtDeclareVariable(float, minweight, , ) = 0.0f;	/* minimum ray weight (lw) */
rtDeclareVariable(int, maxdepth, , ) = 0;	/* maximum recursion depth (lr) */

RT_METHOD int rayorigin(PerRayData_radiance& new_prd, const PerRayData_radiance& prd, const float3& rcoef, const int& d, const int& ad);
RT_METHOD void setupPayload(PerRayData_radiance& prd);
RT_METHOD void resolvePayload(PerRayData_radiance& parent, PerRayData_radiance& prd);
RT_METHOD float rayDistance(PerRayData_radiance& prd);

/* Based on rayoringin() from raytrace.c */
RT_METHOD int rayorigin(PerRayData_radiance& new_prd, const PerRayData_radiance& prd, const float3& rcoef, const int& d, const int& ad)
{
	new_prd.weight = prd.weight * fminf(fmaxf(rcoef), 1.0f);
	if (new_prd.weight <= 0.0f)			/* check for expiration */
		return 0;

	//new_prd.seed = prd.seed;//lcg( prd.seed );
	new_prd.state = prd.state;
	new_prd.depth = prd.depth + d;
	new_prd.ambient_depth = prd.ambient_depth + ad;
	new_prd.tmax = d ? RAY_END : prd.tmax;
#ifdef CONTRIB
	new_prd.rcoef = prd.rcoef * rcoef;
#endif
#ifdef ANTIMATTER
	new_prd.mask = prd.mask;
	new_prd.inside = prd.inside;
#endif

	if (maxdepth <= 0) {	/* Russian roulette */
		//if (minweight <= 0.0f)
		//	error(USER, "zero ray weight in Russian roulette");
		if (maxdepth < 0 && new_prd.depth > -maxdepth)
			return 0;		/* upper reflection limit */
		if (new_prd.weight >= minweight)
			return 1;
		if (curand_uniform(prd.state) > new_prd.weight / minweight)
			return 0;
#ifdef CONTRIB
		new_prd.rcoef *= minweight / new_prd.weight;	/* promote survivor */
#endif
		new_prd.weight = minweight;
		return 1;
	}

	return (new_prd.weight >= minweight && new_prd.depth <= maxdepth);
}

RT_METHOD void setupPayload(PerRayData_radiance& prd)
{
#ifdef DAYSIM_COMPATIBLE
	daysimSet(prd.dc, 0.0f);
#endif
#ifdef RAY_COUNT
	prd.ray_count = 1;
#endif
#ifdef HIT_COUNT
	prd.hit_count = 0;
#endif
}

RT_METHOD void resolvePayload(PerRayData_radiance& parent, PerRayData_radiance& prd)
{
#ifdef RAY_COUNT
	parent.ray_count += prd.ray_count;
#endif
#ifdef HIT_COUNT
	parent.hit_count += prd.hit_count;
#endif
}

RT_METHOD float rayDistance(PerRayData_radiance& prd)
{
	return bright(prd.mirror) > 0.5f * bright(prd.result) ? prd.mirror_distance : prd.distance;
}
