/*
 *  material_intersect.cu - hit programs for the material on GPUs.
 */

#include "accelerad_copyright.h"

#include "otypes.h"

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_ray.h"
#include "optix_point_common.h"
#ifdef CONTRIB
#include "optix_shader_contrib.h"
#endif

using namespace optix;


/* Program variables */
rtDeclareVariable(unsigned int, backvis, , ) = 1u; /* backface visibility (bv) */

/* Context variables */
rtDeclareVariable(unsigned int, do_irrad, , ) = 0u;	/* Calculate irradiance (-i) */
rtDeclareVariable(unsigned int, frame, , ) = 0u;	/* Current frame number, starting from zero, for rvu only */

rtBuffer<MaterialData> material_data;	/* One entry per Radiance material. */

/* OptiX variables */
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(PerRayData_point_cloud, prd_point_cloud, rtPayload, );

/* Attributes */
//rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(int, surface_id, attribute surface_id, );
rtDeclareVariable(int, mat_id, attribute mat_id, );

#ifdef ANTIMATTER
/* Context variables */
rtDeclareVariable(rtObject, top_object, , );
#endif


RT_PROGRAM void any_hit()
{
	if (mat_id < 0 || mat_id >= material_data.size()) {
		rtIgnoreIntersection();
	}
	else {
		const MaterialData mat = material_data[mat_id];

		// Backface visibility
		if (mat.type != MAT_CLIP && !backvis && dot(geometric_normal, ray.direction) > 0) {
			rtIgnoreIntersection();
		}
	}
}

RT_PROGRAM void closest_hit_radiance()
{
	IntersectData data;
	data.mat = material_data[mat_id];
	data.ray_type = ray.ray_type;
	data.t = t_hit;
	data.ray_direction = ray.direction;
	data.hit = ray.origin + t_hit * ray.direction;

	data.world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	data.world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));

	data.surface_id = surface_id;

#ifdef ANTIMATTER
	bool continue_ray = false;
	if (data.mat.type == MAT_CLIP) {
		if (dot(data.world_geometric_normal, ray.direction) < 0.0f) {
			/* Entering a volume */
			prd.mask |= data.mat.params.mask;
			continue_ray = true;
		}
		else if ((prd.mask & data.mat.params.mask) && prd.inside > 0 && data.mat.proxy > -1) {
			/* Leaving a volume and rendering the alternate material */
			data.mat = material_data[data.mat.proxy]; // TODO this will produce odd results if the proxy material is transparent
		}
		else {
			/* Just leave the volume */
			prd.mask &= ~data.mat.params.mask;
			continue_ray = true;
		}
	}
	else if (prd.mask & (1 << mat_id)) {
		/* Entering or leaving the material while in antimatter. */
		prd.inside += dot(data.world_geometric_normal, ray.direction) < 0.0f ? 1 : -1;
		continue_ray = true;
	}
	if (continue_ray) {
		/* Continue the ray */
		const float3 normal = faceforward(data.world_geometric_normal, -ray.direction, data.world_geometric_normal);
		Ray new_ray = make_Ray(ray.origin, ray.direction, RADIANCE_RAY, ray_start(data.hit, ray.direction, normal, RAY_START) + t_hit, ray.tmax);
		rtTrace(top_object, new_ray, prd);
		return;
	}
#endif /* ANTIMATTER */

	if (data.mat.type == MAT_ILLUM) {
		if (data.mat.proxy < 0) return;
		data.mat = material_data[data.mat.proxy];
	}
	if (prd.depth == 0 && do_irrad)
		if (data.mat.type == MAT_PLASTIC || data.mat.type == MAT_METAL || data.mat.type == MAT_TRANS) {
			data.mat = material_data[0];
		}
	int radiance_program_id = data.mat.radiance_program_id;
	if (prd.depth == 0 && frame)
		radiance_program_id = data.mat.diffuse_program_id;

	/* Call the material's callable program. */
	if (radiance_program_id != RT_PROGRAM_ID_NULL)
		prd = rtMarkedCallableProgramId<PerRayData_radiance(IntersectData const&, PerRayData_radiance)>(radiance_program_id, "closest_hit_radiance_call_site")(data, prd);

#ifdef HIT_TYPE
	prd.hit_type = data.mat.type;
#endif
#ifdef CONTRIB
	contribution(prd.rcoef, prd.result, ray.direction, data.mat.contrib_index, data.mat.contrib_function);
#endif
}

RT_PROGRAM void closest_hit_shadow()
{
	IntersectData data;
	data.mat = material_data[mat_id];
	data.ray_type = ray.ray_type;
	data.t = t_hit;
	data.ray_direction = ray.direction;
	data.hit = ray.origin + t_hit * ray.direction;

	data.world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	data.world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));

	data.surface_id = surface_id;

#ifdef ANTIMATTER
	bool continue_ray = false;
	if (data.mat.type == MAT_CLIP) {
		if (dot(data.world_geometric_normal, ray.direction) < 0.0f) {
			/* Entering a volume */
			prd_shadow.mask |= data.mat.params.mask;
			continue_ray = true;
		}
		else if ((prd_shadow.mask & data.mat.params.mask) && prd_shadow.inside > 0 && data.mat.proxy > -1) {
			/* Leaving a volume and rendering the alternate material */
			data.mat = material_data[data.mat.proxy]; // TODO this will produce odd results if the proxy material is transparent
		}
		else {
			/* Just leave the volume */
			prd_shadow.mask &= ~data.mat.params.mask;
			continue_ray = true;
		}
	}
	else if (prd_shadow.mask & (1 << mat_id)) {
		/* Entering or leaving the material while in antimatter. */
		prd_shadow.inside += dot(data.world_geometric_normal, ray.direction) < 0.0f ? 1 : -1;
		continue_ray = true;
	}
	if (continue_ray) {
		/* Continue the ray */
		const float3 normal = faceforward(data.world_geometric_normal, -ray.direction, data.world_geometric_normal);
		Ray new_ray = make_Ray(ray.origin, ray.direction, SHADOW_RAY, ray_start(data.hit, ray.direction, normal, RAY_START) + t_hit, ray.tmax);
		rtTrace(top_object, new_ray, prd_shadow);
		return;
	}
#endif /* ANTIMATTER */

	if (data.mat.shadow_program_id != RT_PROGRAM_ID_NULL)
		prd_shadow = rtMarkedCallableProgramId<PerRayData_shadow(IntersectData const&, PerRayData_shadow)>(data.mat.shadow_program_id, "closest_hit_shadow_call_site")(data, prd_shadow);

	//#ifdef CONTRIB
	//	contribution(prd_shadow.rcoef, prd_shadow.result, ray.direction, data.mat.contrib_index, data.mat.contrib_function); //TODO calculate contribution of shadow?
	//#endif
}

RT_PROGRAM void closest_hit_point_cloud()
{
	IntersectData data;
	data.mat = material_data[mat_id];
	data.ray_type = ray.ray_type;
	data.t = t_hit;
	data.ray_direction = ray.direction;
	data.hit = ray.origin + t_hit * ray.direction;

	data.world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	data.world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));

	data.surface_id = surface_id;

#ifdef ANTIMATTER
	bool continue_ray = false;
	if (data.mat.type == MAT_CLIP) {
		if (dot(data.world_geometric_normal, ray.direction) < 0.0f) {
			/* Entering a volume */
			prd_shadow.mask |= data.mat.params.mask;
			continue_ray = true;
		}
		else if ((prd_shadow.mask & data.mat.params.mask) && prd_shadow.inside > 0 && data.mat.proxy > -1) {
			/* Leaving a volume and rendering the alternate material */
			data.mat = material_data[data.mat.proxy]; // TODO this will produce odd results if the proxy material is transparent
		}
		else {
			/* Just leave the volume */
			prd_shadow.mask &= ~data.mat.params.mask;
			continue_ray = true;
		}
	}
	else if (prd_shadow.mask & (1 << mat_id)) {
		/* Entering or leaving the material while in antimatter. */
		prd_shadow.inside += dot(data.world_geometric_normal, ray.direction) < 0.0f ? 1 : -1;
		continue_ray = true;
	}
	if (continue_ray) {
		/* Continue the ray */
		const float3 normal = faceforward(data.world_geometric_normal, -ray.direction, data.world_geometric_normal);
		Ray new_ray = make_Ray(ray.origin, ray.direction, POINT_CLOUD_RAY, ray_start(data.hit, ray.direction, normal, RAY_START) + t_hit, ray.tmax);
		rtTrace(top_object, new_ray, prd_point_cloud);
		return;
	}
#endif /* ANTIMATTER */

	if (data.mat.point_cloud_program_id != RT_PROGRAM_ID_NULL)
		prd_point_cloud = rtMarkedCallableProgramId<PerRayData_point_cloud(IntersectData const&, PerRayData_point_cloud)>(data.mat.point_cloud_program_id, "closest_hit_point_cloud_call_site")(data, prd_point_cloud);
}
