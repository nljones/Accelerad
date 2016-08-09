/*
 * Copyright (c) 2013-2016 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#pragma once

#ifdef CONTRIB

rtBuffer<float4, 3> contrib_buffer;					/* Accumulate contributions */
rtDeclareVariable(unsigned int, contrib, , ) = 0u;	/* Boolean switch for computing contributions (V) */
rtDeclareVariable(int, contrib_index, , ) = -1;		/* Index of first bin for contribution accumulation */
rtDeclareVariable(rtCallableProgramId<int(const float3)>, contrib_function, , );	/* Function to choose bin for contribution accumulation */
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );


/* Compute and accumulate contributions */
RT_METHOD void contribution(const float weight, const float3& color, const float3& direction)
{
	if (contrib_index >= 0) {
		float3 contr = make_float3(weight);
		if (contrib)
			contr *= color;
		int contr_index = contrib_index;
		if (contrib_function != RT_PROGRAM_ID_NULL)
			contr_index += contrib_function(direction);
		if (contr_index >= contrib_index)
			contrib_buffer[make_uint3(contr_index, launch_index.x, launch_index.y)] += make_float4(contr);
	}
}

#endif /* CONTRIB */