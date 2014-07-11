/*
 * Copyright (c) 2013-2014 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#pragma once

#include <optixu_vector_types.h>

//#define RAY_COUNT
//#define HIT_COUNT
//#define HIT_TYPE
//#define CALLABLE
//#define OLDAMB
#define LIGHTS

/* Types of sky that may be displayed with an OptiX miss program. */
#ifndef CALLABLE
#define  SKY_NONE	0u		/* sun or other distant uniform source */
#define  SKY_CIE	1u		/* cie sky model implemented in skybright.cal */
#define  SKY_PEREZ	2u		/* perez sky model implemented in perezlum.cal */
#endif

typedef struct struct_DistantLight
{
#if defined(__cplusplus)
	typedef optix::float3 float3;
#endif
	float3 pos;
	float3 color;
	float  solid_angle;
#ifndef CALLABLE
	unsigned int type;   /* Reference to the sky type defined above. */
#endif
	int    function;     /* Reference to the OptiX buffer index of the brightness function for this light. */
	int    casts_shadow;
} DistantLight;

#ifndef CALLABLE
/* Structure for the parameters necessary to run skybright.cal. */
typedef struct struct_SkyBright
{
#if defined(__cplusplus)
	typedef optix::float3 float3;
#endif
	int type;      /* 1 for CIE clear, 2 for CIE overcast, 3 for uniform, 4 for CIE intermediate */
	float zenith;  /* zenith brightness */
	float ground;  /* ground plane brightness */
	float factor;  /* normalization factor based on sun direction */
	float3 sun;    /* sun direction */
} SkyBright;

/* Structure for the parameters necessary to run perezlum.cal. */
typedef struct struct_PerezLum
{
#if defined(__cplusplus)
	typedef optix::float3 float3;
#endif
	float diffuse; /* diffuse normalization */
	float ground;  /* ground brightness */
	float coef[5]; /* coefficients for the Perez model */
	float3 sun;    /* sun direction */
} PerezLum;
#endif

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
