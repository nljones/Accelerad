/*
 *  background.cu - miss program for ray tracing on GPUs.
 */

#include "accelerad_copyright.h"

#ifdef HIT_TYPE
#include "otypes.h"	/* For definition of OBJ_SOURCE */
#endif

#include <optix_world.h>
#include "optix_shader_ray.h"
#ifdef CONTRIB
#include "optix_shader_contrib.h"
#endif

using namespace optix;

/* Context variables */
rtBuffer<DistantLight> lights;
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
	prd_radiance.result = prd_radiance.mirror = make_float3(0.0f);
	prd_radiance.distance = prd_radiance.mirror_distance = prd_radiance.tmax;
	if (prd_radiance.tmax < RAY_END) // ray length was truncated
		return;

	const float3 H = optix::normalize(ray.direction);

	// compute direct lighting
	int foundsrc = -1, glowsrc = -1;
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

		// Check to see if ray is within solid angle of source
		if (solid_angle <= light.solid_angle) {
			// Use first hit
			if (light.casts_shadow) {
				foundsrc = i;
				break;
			}
			// If it's a glow or transparent illum, just remember it
			if (glowsrc == -1) glowsrc = i;
		}
	}

	// Do we need fallback?
	if (foundsrc == -1) {
		if (glowsrc == -1) return;
		foundsrc = glowsrc;
	}

	DistantLight light = lights[foundsrc];
	if (!directvis && light.casts_shadow) { // srcignore() in source.c
		prd_radiance.result = make_float3(0.0f);
	}
	else {
		float3 color = light.color;
		if (light.function != RT_PROGRAM_ID_NULL)
			color *= ((rtCallableProgramId<float3(const float3, const float3)>)light.function)(H, -H);
		prd_radiance.result = color;
#ifdef DAYSIM_COMPATIBLE
		if (daylightCoefficients >= 2) {
			daysimAddCoef(prd_radiance.dc, daysimComputePatch(ray.direction), color.x);
		}
#endif /* DAYSIM_COMPATIBLE */
	}
#ifdef CONTRIB
	contribution(prd_radiance.rcoef, prd_radiance.result, H, light.contrib_index, light.contrib_function);
#endif /* CONTRIB */

#ifdef HIT_TYPE
	prd_radiance.hit_type = OBJ_SOURCE;
#endif
}

RT_PROGRAM void miss_shadow()
{
	prd_shadow.result = make_float3(0.0f);

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
				contribution(prd_shadow.rcoef, color, H, light.contrib_index, light.contrib_function);
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
