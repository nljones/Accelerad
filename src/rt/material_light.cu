/*
 *  material_light.cu - hit programs for light materials on GPUs.
 */

#include "accelerad_copyright.h"

#include "otypes.h"

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_ray.h"


using namespace optix;

/* Context variables */
rtDeclareVariable(int,          directvis, , );		/* Boolean switch for light source visibility (dv) */


RT_METHOD int spotout(const IntersectData &data);

/* wrongsource *
 *
 * This source is the wrong source (ie. overcounted) if we are
 * aimed to a different source than the one we hit and the one
 * we hit is not an illum that should be passed.
 */

#define  wrongsource(prd_shadow, data)	(data.surface_id != -prd_shadow.target - 1)// && \
//				(data.mat.type != MAT_ILLUM || illumblock(m,r)))

/* distglow *
 *
 * A distant glow is an object that sometimes acts as a light source,
 * but is too far away from the test point to be one in this case.
 * (Glows with negative radii should NEVER participate in illumination.)
 */
#define  distglow(data)	(data.mat.type == MAT_GLOW && \
				data.mat.params.l.maxrad >= -FTINY && \
				data.t > data.mat.params.l.maxrad)

/* badcomponent *
 *
 * We must avoid counting light sources in the ambient calculation,
 * since the direct component is handled separately.  Therefore, any
 * ambient ray which hits an active light source must be discarded.
 * The same is true for stray specular samples, since the specular
 * contribution from light sources is calculated separately.
 */
#define  badcomponent(prd, data)   ((prd.ambient_depth > 0 && \
				!(dot(data.world_shading_normal, data.ray_direction) > 0.0f || \
		/* not 100% correct */	distglow(data))))

/* srcignore *
 *
 * The -dv flag is normally on for sources to be visible. Not for shadow rays.
 */
#define  srcignore(prd, data)	!(directvis || (distglow(data) && !prd.depth))

RT_CALLABLE_PROGRAM PerRayData_shadow closest_hit_light_shadow(IntersectData const&data, PerRayData_shadow prd_shadow)
{
	if (wrongsource(prd_shadow, data) || dot(data.world_shading_normal, data.ray_direction) > 0.0f || spotout(data)) {
		prd_shadow.result = make_float3(0.0f);
#ifdef CONTRIB
		prd_shadow.rcoef = make_contrib3(0.0f);
#endif
	}
	else if (data.mat.params.l.function > RT_PROGRAM_ID_NULL)
		prd_shadow.result = data.mat.color * ((rtCallableProgramId<float3(const float3, const float3)>)data.mat.params.l.function)(data.ray_direction, data.world_shading_normal);
	else
		prd_shadow.result = data.mat.color;
	return prd_shadow;
}

RT_CALLABLE_PROGRAM PerRayData_radiance closest_hit_light_radiance(IntersectData const&data, PerRayData_radiance prd)
{
	// no contribution to ambient calculation
	if (badcomponent(prd, data) || srcignore(prd, data)) {
		prd.result = make_float3(0.0f);
#ifdef CONTRIB
		prd.rcoef = make_contrib3(0.0f);
#endif
	}
	else if (dot(data.world_shading_normal, data.ray_direction) > 0.0f || spotout(data))
		prd.result = make_float3(0.0f);
	else if (data.mat.params.l.function > RT_PROGRAM_ID_NULL)
		prd.result = data.mat.color * ((rtCallableProgramId<float3(const float3, const float3)>)data.mat.params.l.function)(data.ray_direction, data.world_shading_normal);
	else
		prd.result = data.mat.color;
	prd.mirror = make_float3(0.0f);
	return prd;
}

RT_METHOD int spotout(const IntersectData &data)
{
	if (data.mat.type != MAT_SPOT)
		return(0); /* Not a spotlight */
	if (data.mat.params.l.flen < -FTINY) {		/* distant source */
		const float3 ray_origin = data.hit - data.t * data.ray_direction;
		const float3 vd = data.mat.params.l.aim - ray_origin;
		float d = dot(data.ray_direction, vd);
		/*			wrong side?
		if (d <= FTINY)
			return(1);	*/
		d = dot( vd, vd ) - d * d;
		return (M_PIf * d > data.mat.params.l.siz); /* If true then out */
	}
					/* local source */
	return (data.mat.params.l.siz < 2.0f * M_PIf * (1.0f + dot(data.mat.params.l.aim, data.ray_direction)));	/* If true then out */
}
