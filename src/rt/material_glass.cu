/*
 *  material_glass.cu - hit programs for glass materials on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_ray.h"
#ifdef CONTRIB_DOUBLE
#include "optix_double.h"
#endif

using namespace optix;

/* Context variables */
rtDeclareVariable(rtObject,     top_object, , );


RT_CALLABLE_PROGRAM PerRayData_shadow closest_hit_glass_shadow(IntersectData const&data, PerRayData_shadow prd_shadow)
{
	float3 ffnormal = faceforward(data.world_shading_normal, -data.ray_direction, data.world_geometric_normal);
	const float3 snormal = faceforward(data.world_geometric_normal, -data.ray_direction, data.world_geometric_normal);

	float3 mcolor = data.mat.color;

	/* check transmission */
	const bool hastrans = fmaxf( mcolor ) > 1e-15f;
	if (!hastrans) {
		return prd_shadow;
	}
	mcolor = fmaxf(mcolor, make_float3(1e-15f)); // no color channel should be smaller than 1e-15

	/* get modifiers */
	// we'll skip this for now

	/* perturb normal */
	// if there's a bump map, we use that, else
	float pdot = -dot( data.ray_direction, ffnormal );
	if (pdot < 0.0f) {		/* fix orientation from raynormal in raytrace.c */
		ffnormal += 2.0f * pdot * data.ray_direction;
		pdot = -pdot;
	}

	/* angular transmission */
	float cos2 = sqrtf(1.0f + (pdot * pdot - 1.0f) / (data.mat.params.r_index * data.mat.params.r_index));
	//if (hastrans) {
	mcolor = make_float3( powf( mcolor.x, 1.0f / cos2 ), powf( mcolor.y, 1.0f / cos2 ), powf( mcolor.z, 1.0f / cos2 ) );
	//}

	/* compute reflection */
	float r1e = (pdot - data.mat.params.r_index * cos2) / (pdot + data.mat.params.r_index * cos2);
	r1e *= r1e;
	float r1m = (1.0f / pdot - data.mat.params.r_index / cos2) / (1.0f / pdot + data.mat.params.r_index / cos2);
	r1m *= r1m;

	/* compute transmission */
	//if (hastrans) {
		float3 trans = 0.5f * (1.0f-r1e) * (1.0f-r1e) * mcolor / (1.0f - r1e * r1e * mcolor * mcolor);
		trans       += 0.5f * (1.0f-r1m) * (1.0f-r1m) * mcolor / (1.0f - r1m * r1m * mcolor * mcolor);

		/* modify by pattern */
		//trans *= pcol;

		/* transmitted ray */
#ifdef CONTRIB
		prd_shadow.rcoef *= trans;
#endif
		Ray trans_ray = make_Ray(data.hit, data.ray_direction, SHADOW_RAY, ray_start(data.hit, data.ray_direction, snormal, RAY_START), RAY_END);
		rtTrace(top_object, trans_ray, prd_shadow);
		prd_shadow.result *= trans;
#ifdef DAYSIM_COMPATIBLE
		daysimScale(prd_shadow.dc, trans.x);
#endif
	//}
	return prd_shadow;
}


RT_CALLABLE_PROGRAM PerRayData_radiance closest_hit_glass_radiance(IntersectData const&data, PerRayData_radiance prd)
{
	float3 ffnormal = faceforward(data.world_shading_normal, -data.ray_direction, data.world_geometric_normal);
	const float3 snormal = faceforward(data.world_geometric_normal, -data.ray_direction, data.world_geometric_normal);

	PerRayData_radiance new_prd;
	float3 result = prd.mirror = make_float3(0.0f);
	float3 mcolor = data.mat.color;

	/* check transmission */
	const bool hastrans = fmaxf( mcolor ) > 1e-15f;
	if (hastrans) {
		mcolor = fmaxf( mcolor, make_float3( 1e-15f ) ); // no color channel should be smaller than 1e-15
	} // else we return if it's a shadow ray, which it isn't

	/* get modifiers */
	// we'll skip this for now

	/* perturb normal */
	float3 pert = snormal - ffnormal;
	int hastexture = dot(pert, pert) > FTINY * FTINY;
	float pdot = -dot(data.ray_direction, ffnormal);
	if (pdot < 0.0f) {		/* fix orientation from raynormal in raytrace.c */
		ffnormal += 2.0f * pdot * data.ray_direction;
		pdot = -pdot;
	}

	/* angular transmission */
	float cos2 = sqrtf(1.0f + (pdot * pdot - 1.0f) / (data.mat.params.r_index * data.mat.params.r_index));
	if (hastrans) {
		mcolor = make_float3( powf( mcolor.x, 1.0f / cos2 ), powf( mcolor.y, 1.0f / cos2 ), powf( mcolor.z, 1.0f / cos2 ) );
	}

	/* compute reflection */
	float r1e = (pdot - data.mat.params.r_index * cos2) / (pdot + data.mat.params.r_index * cos2);
	r1e *= r1e;
	float r1m = (1.0f / pdot - data.mat.params.r_index / cos2) / (1.0f / pdot + data.mat.params.r_index / cos2);
	r1m *= r1m;

	/* compute transmission */
	if (hastrans) {
		float3 trans = 0.5f * (1.0f - r1e) * (1.0f - r1e) * mcolor / (1.0f - r1e * r1e * mcolor * mcolor);
		trans       += 0.5f * (1.0f-r1m) * (1.0f-r1m) * mcolor / (1.0f - r1m * r1m * mcolor * mcolor);

		/* modify by pattern */
		//trans *= pcol;

		/* transmitted ray */
		if (rayorigin(new_prd, prd, trans, 0, 0)) {
			new_prd.result = make_float3(0.0f);
#ifdef DAYSIM_COMPATIBLE
			new_prd.dc = daysimNext(prd.dc);
#endif
			float3 R = data.ray_direction;

			if (!new_prd.ambient_depth && hastexture) {
				R = normalize(data.ray_direction + pert * (2.0f * (1.0f - data.mat.params.r_index)));
				if (isnan(R))
					R = data.ray_direction;
			}

			setupPayload(new_prd);
			Ray trans_ray = make_Ray(data.hit, R, RADIANCE_RAY, ray_start(data.hit, R, snormal, RAY_START), new_prd.tmax);
			rtTrace(top_object, trans_ray, new_prd);
			new_prd.result *= trans;
			result += new_prd.result;
			if (prd.ambient_depth || !hastexture)
				prd.distance = data.t + rayDistance(new_prd);
#ifdef DAYSIM_COMPATIBLE
			daysimAddScaled(prd.dc, new_prd.dc, trans.x);
#endif
			resolvePayload(prd, new_prd);
		}
	}
	// stop if it's a shadow ray, which it isn't

	/* compute reflectance */
	float3 refl = 0.5f * r1e * ( 1.0f + (1.0f-2.0f*r1e) * mcolor * mcolor ) / (1.0f - r1e * r1e * mcolor * mcolor );
	refl       += 0.5f * r1m * ( 1.0f + (1.0f-2.0f*r1m) * mcolor * mcolor ) / (1.0f - r1m * r1m * mcolor * mcolor );

	/* reflected ray */
	if (rayorigin(new_prd, prd, refl, 1, 0)) {
		new_prd.result = make_float3(0.0f);
#ifdef DAYSIM_COMPATIBLE
		new_prd.dc = daysimNext(prd.dc);
#endif
		setupPayload(new_prd);
		float3 R = reflect(data.ray_direction, ffnormal);
		Ray refl_ray = make_Ray(data.hit, R, RADIANCE_RAY, ray_start(data.hit, R, snormal, RAY_START), new_prd.tmax);
		rtTrace(top_object, refl_ray, new_prd);
		new_prd.result *= refl;
		prd.mirror = new_prd.result;
		result += new_prd.result;
		prd.mirror_distance = data.t;
		if (prd.ambient_depth || !hastexture)
			prd.mirror_distance += rayDistance(new_prd);
#ifdef DAYSIM_COMPATIBLE
		daysimAddScaled(prd.dc, new_prd.dc, refl.x);
#endif
		resolvePayload(prd, new_prd);
	}
  
	// pass the color back up the tree
	prd.result = result;
	return prd;
}
