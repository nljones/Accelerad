/*
 *  optix_shader_contrib.h - shader routines for generating contribution coefficients on GPUs.
 */

#pragma once

#include "accelerad_copyright.h"

#ifdef CONTRIB_DOUBLE
#include "optix_double.h"
#endif

#ifdef CONTRIB

rtBuffer<contrib4, 3> contrib_buffer;					/* Accumulate contributions */
rtDeclareVariable(unsigned int, contrib, , ) = 0u;	/* Boolean switch for computing contributions (V) */
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );


/* Compute and accumulate contributions */
RT_METHOD void contribution(const contrib3& rcoef, const float3& color, const float3& direction, const int& contrib_index, const int& contrib_function)
{
	if (contrib_index >= 0) {
		contrib3 contr = rcoef;
		if (contrib)
			contr *= color;
		int contr_index = contrib_index;
		if (contrib_function != RT_PROGRAM_ID_NULL)
			contr_index += ((rtCallableProgramId<int(const float3)>)contrib_function)(direction);
		if (contr_index >= contrib_index)
			contrib_buffer[make_uint3(contr_index, launch_index.x, launch_index.y)] += make_contrib4(contr);
	}
}

#endif /* CONTRIB */