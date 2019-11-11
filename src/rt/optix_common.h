/*
 *  optix_common.h - structures for transferring data between the CPU and GPU.
 */

#pragma once

#include "accelerad_copyright.h"

#include <optixu_vector_types.h>

#define RAY_COUNT
//#define HIT_COUNT
//#define HIT_TYPE
#define LIGHTS
#define ANTIMATTER
#define METRICS_DOUBLE
#define CONTRIB /* Calculate contribution coefficients (for rcontrib) */
#ifdef CONTRIB
#define CONTRIB_DOUBLE /* Use double-precision contribution coefficients */
#endif
//#define DAYSIM_COMPATIBLE
//#define PRINT_OPTIX /* Enable OptiX rtPrintf statements to standard out */


#ifdef METRICS_DOUBLE
typedef double metric;
#else
typedef float metric;
#endif

/*! Additional Exceptions */
typedef enum
{
	RT_EXCEPTION_INF = RT_EXCEPTION_USER,	/*!< Inf error */
	RT_EXCEPTION_NAN,		/*!< NaN error */
	RT_EXCEPTION_CUSTOM,	/*!< Custom user exception */
	RT_RETHROWN_EXCEPTION = 0x8000	/* Flag for rethrown exception */
} RTexceptionUser;

/* Ray types */
typedef enum
{
	RADIANCE_RAY = 0,	/* Radiance ray type */
	SHADOW_RAY,			/* Shadow ray type */
	AMBIENT_RAY,		/* Ray into ambient cache */
	AMBIENT_RECORD_RAY,	/* Ray to create ambient record */
	POINT_CLOUD_RAY,	/* Ray to create point cloud */

	RAY_TYPE_COUNT		/* Entry point count for ambient calculation */
} RTraytype;

typedef struct struct_DistantLight
{
#if defined(__cplusplus)
	typedef optix::float3 float3;
#endif
	float3 pos;				/* direction to source */
	float3 color;			/* soruce color */
	float  solid_angle;		/* solid angle covered by soruce */
	int    function;		/* OptiX callable program ID */
	int    casts_shadow;	/* true if source casts shadows (i.e., not a glow material) */
#ifdef CONTRIB
	int    contrib_index;		/* index of first bin for contribution accumulation */
	int    contrib_function;	/* function to choose bin for contribution accumulation */
#endif
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
	float2 gpos;	/* (u,v) gradient wrt. position */
	float2 gdir;	/* (u,v) gradient wrt. direction */
	float2 rad;		/* anisotropic radii (rad.x <= rad.y) */
	int ndir;		/* encoded surface normal */
	int udir;		/* u-vector direction */
	unsigned int corral;	/* potential light leak direction flags */
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
	float  weight;	/* cumulative weight (for termination) */
	float3 dir;		/* normalized direction of ray */
	float  max;		/* maximum distance (aft clipping plane) */
	float3 val;		/* computed radiance value */
	float  length;	/* effective ray length */
	float3 mirror;		/* radiance value of mirrored contribution */
	float  mirrored_length;	/* effective length of mirrored ray */
	//float3 contrib;	/* contribution coefficient w.r.t. parent */
	//float3 extinction;	/* medium extinction coefficient */
	//float3 hit;	/* point of intersection */
	//float3 pnorm;	/* normal at intersection (perturbed) */
	//float3 normal;	/* normal at intersection (unperturbed) */
	//float2 tex;	/* local (u,v) coordinates */
	//float  t;		/* first intersection distance */
	//char*  surface;
	//char*  modifier;
	//char*  material;
#ifdef RAY_COUNT
	int ray_count;
#endif
} RayData;

typedef struct struct_normal_data
{
	float spec;			/* The material specularity given by the rad file "plastic", "metal", or "trans" object */
	float rough;		/* The material roughness given by the rad file "plastic", "metal", or "trans" object */
	float trans;		/* The material transmissivity given by the rad file "trans" object */
	float tspec;		/* The material transmitted specular component given by the rad file "trans" object */
	unsigned int ambincl;	/* Flag to skip ambient calculation and use default (ae, aE, ai, aI) */
} struct_normal_data;

typedef struct struct_light_data
{
	float maxrad;		/* maximum radius for "glow" object */
	float siz;			/* output solid angle or area for "spotlight" object */
	float flen;			/* focal length for "spotlight" object (negative if distant source) */
	float3 aim;			/* aim direction or center for "spotlight" object */
	int function;		/* function or texture modifier */
} struct_light_data;

/* Structure to material parameters */
typedef struct struct_material_data
{
#if defined(__cplusplus)
	typedef optix::float3 float3;
#endif
	unsigned int type;	/* The material type */
	int proxy;			/* The index of the material to use for direct views (non-shadow rays) of this material */
	float3 color;		/* The material color */
	unsigned int mask;	/* Bitmask of antimatter materials affecting this material, or of this antimatter material. */

	union {
		struct_normal_data n;
		float r_index;		/* Refractive index of the "glass" object, usually 1.52 */
		struct_light_data l;
	} params;

	int radiance_program_id;	/* Program ID for radiance rays */
	int diffuse_program_id;		/* Program ID for diffuse rays */
	int shadow_program_id;		/* Program ID for shadow rays */
	int point_cloud_program_id;	/* Program ID for point cloud rays */

#ifdef CONTRIB
	int contrib_index;		/* index of first bin for contribution accumulation */
	int contrib_function;	/* function to choose bin for contribution accumulation */
#endif
} MaterialData;

/* Structure to hold ray parameters */
typedef struct struct_ray_parameters
{
#if defined(__cplusplus)
	typedef optix::float3 float3;
#endif
	float aft;			/* distance to aft clipping plane */
	float3 origin;		/* position in space */
	float3 direction;	/* normal direction */
	float distance;		/* focal length */
} RayParams;

/* Structure to hold evaluation metrics */
typedef struct struct_eval_metrics
{
	metric omega;	/* solid angle */
	metric ev;		/* contribution to vertical eye illuminance */
	metric avlum;	/* contribution to average luminance */
	metric dgp;		/* contribution to daylight glare probability */
	int flags;		/* flags for task regions */
} Metrics;
