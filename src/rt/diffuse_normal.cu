/*
 * Copyright (c) 2013-2016 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_common.h"

using namespace optix;

#define  TRANSMISSION

#ifndef  MAXITER
#define  MAXITER	10		/* maximum # specular ray attempts */
#endif
#define  MAXSPART	64		/* maximum partitions per source */
//#define frandom()	(rnd( prd.seed )/float(RAND_MAX))
//#define frandom()	(rnd( prd.seed ))

/* specularity flags */
#define  SP_REFL	01		/* has reflected specular component */
#define  SP_TRAN	02		/* has transmitted specular */
#define  SP_PURE	04		/* purely specular (zero roughness) */
#define  SP_FLAT	010		/* flat reflecting surface */
#define  SP_RBLT	020		/* reflection below sample threshold */
#define  SP_TBLT	040		/* transmission below threshold */

typedef struct {
	unsigned int specfl;		/* specularity flags, defined above */
	float3 mcolor;		/* color of this material */
	float3 scolor;		/* color of specular component */
	//float3 vrefl;		/* vector in direction of reflected ray */
	float3 prdir;		/* vector in transmitted direction */
	float3 normal;
	float3 hit;
	float  alpha2;		/* roughness squared */
	float  rdiff, rspec;	/* reflected specular, diffuse */
	float  trans;		/* transmissivity */
	float  tdiff, tspec;	/* transmitted specular, diffuse */
	float3 pnorm;		/* perturbed surface normal */
	float  pdot;		/* perturbed dot product */
}  NORMDAT;		/* normal material data */

/* Context variables */
rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, ambient_ray_type, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(rtObject, top_ambient, , );

rtDeclareVariable(float3, ambval, , );	/* This is the final value used in place of an indirect light calculation */
rtDeclareVariable(int, ambvwt, , );	/* As new indirect irradiances are computed, they will modify the default ambient value in a moving average, with the specified weight assigned to the initial value given on the command and all other weights set to 1 */
rtDeclareVariable(int, ambounce, , );	/* Ambient bounces (ab) */
//rtDeclareVariable(int,          ambres, , );	/* Ambient resolution (ar) */
rtDeclareVariable(float, ambacc, , );	/* Ambient accuracy (aa). This value will approximately equal the error from indirect illuminance interpolation */
rtDeclareVariable(int, ambdiv, , );	/* Ambient divisions (ad) */
rtDeclareVariable(int, ambdiv_final, , ); /* Number of ambient divisions for final-pass fill (ag) */
rtDeclareVariable(int, ambssamp, , );	/* Ambient super-samples (as) */
#ifdef OLDAMB
rtDeclareVariable(float, maxarad, , );	/* maximum ambient radius */
rtDeclareVariable(float, minarad, , );	/* minimum ambient radius */
#endif /* OLDAMB */
rtDeclareVariable(float, avsum, , );		/* computed ambient value sum (log) */
rtDeclareVariable(unsigned int, navsum, , );	/* number of values in avsum */

rtDeclareVariable(float, minweight, , );	/* minimum ray weight (lw) */
rtDeclareVariable(int, maxdepth, , );	/* maximum recursion depth (lr) */

rtBuffer<DistantLight> lights;

/* Material variables */
rtDeclareVariable(unsigned int, type, , );	/* The material type representing "plastic", "metal", or "trans" */
rtDeclareVariable(float3, color, , );	/* The material color given by the rad file "plastic", "metal", or "trans" object */
rtDeclareVariable(float, spec, , );	/* The material specularity given by the rad file "plastic", "metal", or "trans" object */
rtDeclareVariable(float, rough, , );	/* The material roughness given by the rad file "plastic", "metal", or "trans" object */
rtDeclareVariable(unsigned int, ambincl, , ) = 1u;	/* Flag to skip ambient calculation and use default (ae, aE, ai, aI) */

/* OptiX variables */
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );

/* Attributes */
//rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
#ifdef ANTIMATTER
rtDeclareVariable(int, mat_id, attribute mat_id, );
#endif


RT_METHOD float3 multambient(float3 aval, const float3& normal, const float3& pnormal, const float3& hit);
#ifdef DAYSIM_COMPATIBLE
RT_METHOD int doambient(float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit, DaysimCoef dc);
#else
RT_METHOD int doambient(float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit);
#endif


RT_PROGRAM void closest_hit_radiance()
{
	NORMDAT nd;

	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));

	/* check for back side */
	nd.pnorm = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);
	nd.normal = faceforward(world_geometric_normal, -ray.direction, world_geometric_normal);

	float3 result = make_float3(0.0f);
	nd.hit = ray.origin + t_hit * ray.direction;
	nd.mcolor = color;
	nd.scolor = make_float3(0.0f);
	nd.rspec = spec;
	nd.alpha2 = rough * rough;
	nd.specfl = 0u; /* specularity flags */

#ifdef ANTIMATTER
	if (prd.mask & (1 << mat_id)) {
		prd.inside += dot(world_geometric_normal, ray.direction) < 0.0f ? 1 : -1;

		/* Continue the ray */
		Ray new_ray = make_Ray(ray.origin, ray.direction, ray.ray_type, ray_start(nd.hit, ray.direction, nd.normal, RAY_START) + t_hit, RAY_END);
		rtTrace(top_object, new_ray, prd);
		return;
	}
#endif /* ANTIMATTER */

	/* get roughness */
	if (nd.alpha2 <= FTINY) {
		nd.specfl |= SP_PURE; // label this as a purely specular reflection
	}

	/* perturb normal */
	float3 pert = nd.normal - nd.pnorm;
	int hastexture = dot(pert, pert) > FTINY * FTINY;
	nd.pdot = -dot(ray.direction, nd.pnorm);
	if (nd.pdot < 0.0f) {		/* fix orientation from raynormal in raytrace.c */
		nd.pnorm += 2.0f * nd.pdot * ray.direction;
		nd.pdot = -nd.pdot;
	}
	if (nd.pdot < 0.001f)
		nd.pdot = 0.001f;			/* non-zero for dirnorm() */

	// if it's a face or a ring label as flat (currently we only support triangles, so everything is flat)
	nd.specfl |= SP_FLAT;

	/* modify material color */
	//nd.mcolor *= rtTex3D(rtTextureId id, texcoord.x, texcoord.y, texcoord.z).xyz;

	/* compute Fresnel approx. */
	float fest = 0.0f;
	if (nd.specfl & SP_PURE && nd.rspec >= FRESTHRESH) {
		fest = FRESNE(nd.pdot);
		nd.rspec += fest * (1.0f - nd.rspec);
	}

	/* compute transmission */
	nd.tdiff = nd.tspec = nd.trans = 0.0f; // because it's opaque

	/* diffuse reflection */
	nd.rdiff = 1.0f - nd.trans - nd.rspec;

	if (!(nd.specfl & SP_PURE && nd.rdiff <= FTINY && nd.tdiff <= FTINY)) { /* not 100% pure specular */
		/* ambient from this side */
		if (nd.rdiff > FTINY) {
			float3 aval = nd.mcolor * nd.rdiff;	/* modified by material color */
			if (nd.specfl & SP_RBLT)	/* add in specular as well? */
				aval += nd.scolor;
			result += multambient(aval, nd.normal, nd.pnorm, nd.hit);	/* add to returned color */
		}

#ifdef TRANSMISSION
		/* ambient from other side */
		if (nd.tdiff > FTINY) {
			float3 aval = nd.mcolor;	/* modified by material color */
			if (nd.specfl & SP_TBLT)
				aval *= nd.trans;
			else
				aval *= nd.tdiff;
			result += multambient(aval, -nd.normal, -nd.pnorm, nd.hit);	/* add to returned color */
		}
#endif /* TRANSMISSION */
	}

	prd.distance = t_hit;

	// pass the color back up the tree
	prd.result = result;

#ifdef HIT_TYPE
	prd.hit_type = type;
#endif
}


// Compute the ambient component and multiply by the coefficient.
RT_METHOD float3 multambient(float3 aval, const float3& normal, const float3& pnormal, const float3& hit)
{
	float 	d;

	/* ambient calculation */
	if (ambdiv > 0 && prd.ambient_depth < ambounce && ambincl) {
		float3 acol = aval;
	#ifdef DAYSIM_COMPATIBLE
		DaysimCoef dc = daysimNext(prd.dc);
		daysimSet(dc, 0.0f);
		d = doambient(&acol, normal, pnormal, hit, dc);
		if (d > FTINY)
			daysimAdd(prd.dc, dc);
	#else
		d = doambient(&acol, normal, pnormal, hit);
	#endif
		if (d > FTINY)
			return acol;
	}
					/* return global value */
	if ((ambvwt <= 0) || (navsum == 0)) {
#ifdef DAYSIM_COMPATIBLE
		daysimAdd(prd.dc, aval.x * ambval.x);
#endif
		return aval * ambval;
	}
	float l = bright(ambval);			/* average in computations */
	if (l > FTINY) {
		d = (logf(l)*(float)ambvwt + avsum) / (float)(ambvwt + navsum);
		d = expf(d) / l;
		aval *= ambval;	/* apply color of ambval */
#ifdef DAYSIM_COMPATIBLE
		daysimAdd(prd.dc, aval.x * ambval.x * d);
#endif
	}
	else {
		d = expf(avsum / (float)navsum);
#ifdef DAYSIM_COMPATIBLE
		daysimAdd(prd.dc, aval.x * d);
#endif
	}
	return aval * d;
}


/* sample indirect hemisphere, based on samp_hemi in ambcomp.c */
#ifdef DAYSIM_COMPATIBLE
RT_METHOD int doambient(float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit, DaysimCoef dc)
#else
RT_METHOD int doambient(float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit)
#endif
{
	float	d;
	float wt = prd.weight;

	/* set number of divisions */
	if (ambacc <= FTINY && wt > (d = 0.8f * fmaxf(*rcol) * wt / (ambdiv_final * minweight)))
		wt = d;			/* avoid ray termination */
	float3 acol = make_float3(0.0f);
	float3 acoef = *rcol;

	/* Setup from ambsample in ambcomp.c */
	PerRayData_radiance new_prd;
	/* generate hemispherical sample */
	/* ambient coefficient for weight */
	if (ambacc > FTINY)
		d = AVGREFL; // Reusing this variable
	else
		d = fmaxf(acoef);
	new_prd.weight = prd.weight * d;
	if (new_prd.weight < minweight) //if (rayorigin(&ar, AMBIENT, r, ar.rcoef) < 0)
		return(0);

	new_prd.depth = prd.depth + 1;
	new_prd.ambient_depth = prd.ambient_depth + 1;
	//new_prd.seed = prd.seed;//lcg( prd.seed );
	new_prd.state = prd.state;
#ifdef ANTIMATTER
	new_prd.mask = prd.mask;
	new_prd.inside = prd.inside;
#endif
#ifdef DAYSIM_COMPATIBLE
	new_prd.dc = daysimNext(dc);
#endif

	Ray amb_ray = make_Ray(hit, pnormal, radiance_ray_type, RAY_START, RAY_END); // Use normal point as temporary direction
	/* End ambsample setup */

	/* make tangent plane axes */
	float3 ux = getperpendicular(pnormal, prd.state);
	float3 uy = cross(pnormal, ux);

	/* ambsample in ambcomp.c */
	float2 spt = 0.01f + 0.98f * make_float2(curand_uniform(prd.state), curand_uniform(prd.state));
	SDsquare2disk(spt, spt.y, spt.x);
	float zd = sqrtf(1.0f - dot(spt, spt));
	amb_ray.direction = normalize(spt.x*ux + spt.y*uy + zd*pnormal);
	if (dot(amb_ray.direction, normal) <= 0) /* Prevent light leaks */
		return(0);
	amb_ray.tmin = ray_start(hit, amb_ray.direction, normal, RAY_START);

	setupPayload(new_prd);
	//Ray amb_ray = make_Ray( hit, rdir, radiance_ray_type, RAY_START, RAY_END );
	rtTrace(top_object, amb_ray, new_prd);
	resolvePayload(prd, new_prd);

	if (isnan(new_prd.result)) // TODO How does this happen?
		return(0);
	if (new_prd.distance <= FTINY)
		return(0);		/* should never happen */
	acol += new_prd.result * acoef;	/* add to our sum */
#ifdef DAYSIM_COMPATIBLE
	daysimAddScaled(dc, new_prd.dc, acoef.x);
#endif
	*rcol = acol;
	return(1);			/* all is well */
}
