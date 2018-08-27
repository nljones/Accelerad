/*
 *  uniform.cu - program for uniform sampling of sky and other surfaces on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix_world.h>

rtDeclareVariable(float3, normal, , );	/* Normal direction */

// Return 0 if the surface is hit from the front, -1 otherwise.
RT_CALLABLE_PROGRAM int front(const float3 direction)
{
	return optix::dot(direction, normal) < 0 ? 0 : -1;
}