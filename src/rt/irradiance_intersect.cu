/*
 *  irradiance_intersect.cu - intersection program for virtual Lambertian surface on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "optix_shader_common.h"

using namespace optix;

/* OptiX variables */
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

/* Attributes */
rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(int, surface_id, attribute surface_id, );
rtDeclareVariable(int, mat_id, attribute mat_id, );

RT_PROGRAM void irradiance_intersect( int primIdx )
{
	/* There should always be an intersection at t=0. */
	if ( rtPotentialIntersection( 0.0f ) ) {
		geometric_normal = shading_normal = normalize( -ray.direction );
		texcoord = make_float3( 0.0f, 0.0f, 0.0f );
		surface_id = -1; // Not a real surface

		/* Lambert material is material 0 */
		mat_id = 0;
		rtReportIntersection(0);
	}
}

RT_PROGRAM void irradiance_bounds(int primIdx, float result[6])
{  
	optix::Aabb* aabb = (optix::Aabb*)result;

	/* The single instance covers the entire scene. */
	aabb->m_min = make_float3( -RT_DEFAULT_MAX );
	aabb->m_max = make_float3( RT_DEFAULT_MAX );
}

