/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#pragma once

#include <optixu_vector_types.h>

//#define RAY_COUNT
//#define HIT_COUNT
//#define HIT_TYPE
//#define OLDAMB
#define LIGHTS

#ifndef DAYSIM
//#define DAYSIM
#ifdef DAYSIM
#define DAYSIM_MAX_COEFS		148
//#define DAYSIM_MAX_COEFS		2306
typedef float DaysimCoef[DAYSIM_MAX_COEFS];
#endif
#endif

#define AMB_ROW_SIZE	32	/* Number of entries allowed per row of the ambient hemisphere. */

typedef struct struct_DistantLight
{
#if defined(__cplusplus)
	typedef optix::float3 float3;
#endif
	float3 pos;
	float3 color;
	float  solid_angle;
	int    function;     /* Reference to the OptiX buffer index of the brightness function for this light. */
	int    casts_shadow;
} DistantLight;

/* Structure to hold an ambient record */
typedef struct struct_ambient_record
{
#if defined(__cplusplus)
	typedef optix::float2 float2;
	typedef optix::float3 float3;
#endif
	float3 pos;		/* position in space */
	float3 val;		/* computed ambient value */
#ifndef OLDAMB
	float2 gpos;	/* (u,v) gradient wrt. position */
	float2 gdir;	/* (u,v) gradient wrt. direction */
	float2 rad;		/* anisotropic radii (rad.x <= rad.y) */
	int ndir;		/* encoded surface normal */
	int udir;		/* u-vector direction */
	unsigned int corral;	/* potential light leak direction flags */
#else
	float3 dir;		/* normal direction */
	float3 gpos;	/* gradient wrt. position */
	float3 gdir;	/* gradient wrt. direction */
	float  rad;		/* validity radius */
#endif
	float  weight;	/* weight of parent ray */
	unsigned int lvl;	/* recursion level of parent ray */
#ifdef DAYSIM
	DaysimCoef dc;	/* daylight coefficients */
#endif
#ifdef RAY_COUNT
	int ray_count;
#endif
#ifdef HIT_COUNT
	int hit_count;
#endif
} AmbientRecord;

/* Structure to hold ray data */
typedef struct struct_ray_data
{
#if defined(__cplusplus)
	typedef optix::float3 float3;
#endif
	float3 origin;	/* origin of ray */
	float3 dir;		/* normalized direction of ray */
	float3 val;		/* computed radiance value */
	//float3 contrib;	/* contribution coefficient w.r.t. parent */
	//float3 extinction;	/* medium extinction coefficient */
	//float3 hit;	/* point of intersection */
	//float3 pnorm;	/* normal at intersection (perturbed) */
	//float3 normal;	/* normal at intersection (unperturbed) */
	//float2 tex;	/* local (u,v) coordinates */
	float  max;		/* maximum distance (aft clipping plane) */
	float  weight;	/* cumulative weight (for termination) */
	float  length;	/* effective ray length */
	//float  t;		/* first intersection distance */
	//char*  surface;
	//char*  modifier;
	//char*  material;
#ifdef DAYSIM
	DaysimCoef dc;	/* daylight coefficients */
#endif
} RayData;

/* Structure to hold a point and direction */
typedef struct struct_point_direction
{
#if defined(__cplusplus)
	typedef optix::float3 float3;
#endif
	float3 pos;		/* position in space */
	float3 dir;		/* normal direction */
} PointDirection;
