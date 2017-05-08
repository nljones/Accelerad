/*
 *  isotropsky.cu - program for isotropic sky distribution on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>

struct Transform
{
	optix::Matrix<3,3> m;
};

/* Program variables */
rtDeclareVariable(float,      radiance, , ); /* isotropic sky radiance */
rtDeclareVariable(Transform,  transform, , ); /* transformation matrix, ignored */

// Calculate the isotropic sky value for the current ray direction.
// This function replicates the algorithm in isotrop_sky.cal distributed with Daysim.
RT_CALLABLE_PROGRAM float3 skybright(const float3 ignore0, const float3 ignore1)
{
	return make_float3(radiance);
}
