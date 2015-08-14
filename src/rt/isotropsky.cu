/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix_world.h>
//#include "optix_shader_common.h"

struct Transform
{
	optix::Matrix<3,3> m;
};

/* Program variables */
rtDeclareVariable(float,      skybright, , ); /* isotropic sky radiance */
rtDeclareVariable(Transform,  transform, , ); /* transformation matrix, ignored */

// Calculate the isotropic sky value for the current ray direction.
// This function replicates the algorithm in isotrop_sky.cal distributed with Daysim.
RT_CALLABLE_PROGRAM float3 isotrop_sky(const float3 ignore0, const float3 ignore1)
{
	return make_float3(skybright);
}
