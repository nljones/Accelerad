/*
 *  optix_point_common.h - declarations and structures for geometry sampling on GPUs.
 */

#pragma once

#include "accelerad_copyright.h"

#include <optixu_vector_types.h>

//#define AMBIENT_CELL

/* Structure to hold a point and direction */
typedef struct struct_point_direction
{
#if defined(__cplusplus)
	typedef optix::float3 float3;
#endif
	float3 pos;		/* position in space */
	float3 dir;		/* normal direction */
#ifdef AMBIENT_CELL
	uint2 cell;		/* cell index represented as 4 16-bit shorts */
#endif
} PointDirection;

#ifdef __CUDACC__
#include "optix_shader_random.h"

typedef rtBufferId<PointDirection, 1> PointDirectionBuffer;

/* Ray payload structure */
struct PerRayData_point_cloud
{
	float3 point;		/* hit point */
	float3 forward;		/* transmitted ray normal direction */
	float3 reverse;		/* reflected ray normal direction */
	uint3 index;		/* thread index */
	unsigned int seeds;	/* maximum seed index for thread */
	rand_state* state;
#ifdef ANTIMATTER
	int inside;			/* counter for number of volumes traversed */
	unsigned int mask;	/* mask for materials to skip */
#endif
};

/* OptiX method declaration in the style of RT_PROGRAM */
#ifndef RT_METHOD
#define RT_METHOD	static __forceinline__ __device__
#endif

RT_METHOD void clear(PointDirection& pd);

/* Clear the contents of object. */
RT_METHOD void clear(PointDirection& pd)
{
	pd.pos = pd.dir = make_float3(0.0f);
#ifdef AMBIENT_CELL
	pd.cell = make_uint2(0);
#endif
}

#endif /* __CUDACC__ */