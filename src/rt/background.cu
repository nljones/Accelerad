/*
 *  background.cu - miss program for ray tracing on GPUs.
 */

#include "accelerad_copyright.h"

//#define RT_USE_TEMPLATED_RTCALLABLEPROGRAM
#include <optix_world.h>
#include "optix_shader_common.h"
#include "optix_shader_ray.h"
#ifdef CONTRIB_DOUBLE
#include "optix_double.h"
#endif

using namespace optix;

/* Program variables */
#ifdef HIT_TYPE
rtDeclareVariable(unsigned int, type, , ); /* The material type representing "source" */
#endif

/* Context variables */
rtBuffer<DistantLight> lights;
#ifdef CONTRIB
rtBuffer<contrib4, 3> contrib_buffer; /* accumulate contributions */
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(unsigned int, contrib, , ) = 0u;		/* Boolean switch for computing contributions (V) */
#endif
//rtBuffer<rtCallableProgramId<float(const float3)> > functions;
//rtDeclareVariable(rtCallableProgramId<float(float3)>, func, , );
//rtDeclareVariable(rtCallableProgramX<float(float3)>, func, , );
rtDeclareVariable(int, directvis, , );		/* Boolean switch for light source visibility (dv) */

/* OptiX variables */
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );


#ifdef DAYSIM_COMPATIBLE
RT_METHOD unsigned int daysimComputePatch(const float3 dir);
#endif

RT_PROGRAM void miss()
{
	prd_radiance.result = make_float3( 0.0f );
	prd_radiance.distance = ray.tmax;
	if ( ray.tmax < RAY_END ) // ray length was truncated
		return;

	const float3 H = optix::normalize(ray.direction);

	// compute direct lighting
	unsigned int num_lights = lights.size();
	for (int i = 0; i < num_lights; ++i) {
		DistantLight light = lights[i];

		// no contribution to ambient calculation
		if (prd_radiance.ambient_depth && light.casts_shadow) // badcomponent() in source.c
			continue; // TODO also no contribution from specular

		// get the angle bwetween the light direction and the view
		float3 L = optix::normalize(light.pos);
		float lDh = optix::dot( L, H );
		float solid_angle = 2.0f * M_PIf * (1.0f - lDh);

		if (solid_angle <= light.solid_angle) {
			if (!directvis && light.casts_shadow) { // srcignore() in source.c
				prd_radiance.result = make_float3(0.0f);
				break;
			}
			float3 color = light.color;
			if (light.function != RT_PROGRAM_ID_NULL)
				color *= ((rtCallableProgramId<float3(const float3, const float3)>)light.function)(H, -H);
			prd_radiance.result += color;
#ifdef DAYSIM_COMPATIBLE
			if (daylightCoefficients >= 2) {
				daysimAddCoef(prd_radiance.dc, daysimComputePatch(ray.direction), color.x);
			}
#endif /* DAYSIM_COMPATIBLE */
#ifdef CONTRIB
			if (light.contrib_index >= 0) {
				contrib3 contr = prd_radiance.rcoef;
				if (contrib)
					contr *= color;
				int contr_index = light.contrib_index;
				if (light.contrib_function != RT_PROGRAM_ID_NULL)
					contr_index += ((rtCallableProgramId<int(const float3)>)light.contrib_function)(H);
				if (contr_index >= light.contrib_index)
					contrib_buffer[make_uint3(contr_index, launch_index.x, launch_index.y)] += make_contrib4(contr);
			}
#endif /* CONTRIB */
		}
	}

#ifdef HIT_TYPE
	prd_radiance.hit_type = type;
#endif
}

RT_PROGRAM void miss_shadow()
{
	prd_shadow.result = make_float3(0.0f);
	if (ray.tmax < RAY_END) // ray length was truncated
		return;

	const float3 H = optix::normalize(ray.direction);

	// compute direct lighting
	if ( prd_shadow.target >= 0 && prd_shadow.target < lights.size() ) {
		DistantLight light = lights[prd_shadow.target];
		if (light.casts_shadow) {

			// get the angle bwetween the light direction and the view
			float3 L = optix::normalize(light.pos);
			float lDh = optix::dot( L, H );
			float solid_angle = 2.0f * M_PIf * (1.0f - lDh);

			if (solid_angle <= light.solid_angle) {
				float3 color = light.color;
				if (light.function != RT_PROGRAM_ID_NULL)
					color *= ((rtCallableProgramId<float3(const float3, const float3)>)light.function)(H, -H);
				prd_shadow.result = color;
#ifdef DAYSIM_COMPATIBLE
				if (daylightCoefficients >= 2) {
					// TODO This assumes that all sources are sun positions in numerical order
					// TODO If files are merged, add 148 to prd_shadow.target
					daysimAddCoef(prd_shadow.dc, prd_shadow.target, color.x);
				}
#endif /* DAYSIM_COMPATIBLE */
#ifdef CONTRIB
				if (light.contrib_index >= 0) {
					contrib3 contr = prd_shadow.rcoef;
					if (contrib)
						contr *= color;
					int contr_index = light.contrib_index;
					if (light.contrib_function != RT_PROGRAM_ID_NULL)
						contr_index += ((rtCallableProgramId<int(const float3)>)light.contrib_function)(H);
					if (contr_index >= light.contrib_index)
						contrib_buffer[make_uint3(contr_index, launch_index.x, launch_index.y)] += make_contrib4(contr);
				}
#endif /* CONTRIB */
			}
		}
	}
}

#ifdef DAYSIM_COMPATIBLE
/*
* Computes the sky/ground patch hit by a ray in direction (dx,dy,dz)
* according to the Tregenza sky division.
*/
RT_METHOD unsigned int daysimComputePatch(const float3 dir)
{
	if (dir.z > 0.0f) { // sky
		const unsigned int number[8] = { 0, 30, 60, 84, 108, 126, 138, 144 };
		const float ring_division[8] = { 30.0f, 30.0f, 24.0f, 24.0f, 18.0f, 12.0f, 6.0f, 0.0f };
		int ringnumber = (int)(asinf(dir.z) * 15.0f * M_1_PIf);
		// origin of the number "15":
		// according to Tregenza, the celestial hemisphere is divided into 7 bands and
		// the zenith patch. The bands range from:
		//												altitude center
		// Band 1		0 to 12 Deg			30 patches	6
		// Band 2		12 to 24 Deg		30 patches	18
		// Band 3		24 to 36 Deg		24 patches	30
		// Band 4		36 to 48 Deg		24 patches	42
		// Band 5		48 to 60 Deg		18 patches	54
		// Band 6		60 to 72 Deg		12 patches	66
		// Band 7		72 to 84 Deg		 6 patches	78
		// Band 8		84 to 90 Deg		 1 patch 	90
		// since the zenith patch is only takes 6Deg instead of 12, the arc length
		// between 0 and 90 Deg (equlas o and Pi/2) is divided into 7.5 units:
		// Therefore, 7.5 units = (int) asin(z=1)/(Pi/2)
		//				1 unit = asin(z)*(2*7.5)/Pi)
		//				1 unit = asin(z)*(15)/Pi)
		// Note that (int) always rounds to the next lowest integer
		return number[ringnumber] + ring_division[ringnumber] * (atan2f(dir.y, dir.x) * 0.5f * M_1_PIf + (dir.y >= 0.0f ? 0.0f : 1.0f));
	}
	// ground
	if (dir.z >= -0.17365f)
		return 145;
	if (dir.z >= -0.5f)
		return 146;
	return 147;
}
#endif /* DAYSIM_COMPATIBLE */
