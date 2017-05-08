/*
 *  optix_ambient_common.h - definitinos and structures for ambient sampling on GPUs.
 */

#pragma once

#include "accelerad_copyright.h"

#ifndef OLDAMB

#define AMB_PARALLEL	/* Trace reflected rays within each ambient calculation in parallel */
#ifndef AMB_PARALLEL
#define AMB_SAVE_MEM	/* Reduce global memory usage in ambient calculation by saving one row of samples at a time */
#endif
#ifndef AMB_SAVE_MEM
#define AMB_SUPER_SAMPLE	/* Perform ambient super sampling */
#endif

typedef struct {
#if defined(__cplusplus)
	typedef optix::float3 float3;
#endif
	float3	v;		/* hemisphere sample value */
	float	d;		/* reciprocal distance (1/rt) */
	float3	p;		/* intersection point */
#ifdef AMB_PARALLEL
#ifdef RAY_COUNT
	int ray_count;
#endif
#ifdef HIT_COUNT
	int hit_count;
#endif
#endif
} AmbientSample;		/* sample value */

#endif /* OLDAMB */
