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


RT_CALLABLE_PROGRAM PerRayData_shadow closest_hit_light_shadow(IntersectData const&data, PerRayData_shadow prd_shadow)
{
	if (data.t > data.mat.params.l.maxrad || spotout(data) || dot(data.world_shading_normal, data.ray_direction) > 0.0f || data.surface_id != -prd_shadow.target - 1)
		prd_shadow.result = make_float3( 0.0f );
	else if (data.mat.params.l.function > RT_PROGRAM_ID_NULL)
		prd_shadow.result = data.mat.color * ((rtCallableProgramId<float3(const float3, const float3)>)data.mat.params.l.function)(data.ray_direction, data.world_shading_normal);
	else
		prd_shadow.result = data.mat.color;
	return prd_shadow;
}

RT_CALLABLE_PROGRAM PerRayData_radiance closest_hit_light_radiance(IntersectData const&data, PerRayData_radiance prd)
{
	// no contribution to ambient calculation
	if (!directvis || data.t > data.mat.params.l.maxrad && prd.depth > 0 || prd.ambient_depth > 0 || spotout(data) || dot(data.world_shading_normal, data.ray_direction) > 0.0f) //TODO need a better ambient test
		prd.result = make_float3( 0.0f );
	else if (data.mat.params.l.function > RT_PROGRAM_ID_NULL)
		prd.result = data.mat.color * ((rtCallableProgramId<float3(const float3, const float3)>)data.mat.params.l.function)(data.ray_direction, data.world_shading_normal);
	else
		prd.result = data.mat.color;
	prd.mirror = make_float3(0.0f);
	prd.distance = prd.mirror_distance = data.t;
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
