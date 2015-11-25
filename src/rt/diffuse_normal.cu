/*
* Copyright (c) 2013-2015 Nathaniel Jones
* Massachusetts Institute of Technology
*/

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_common.h"

using namespace optix;

#define  AMBIENT
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

#ifdef AMBIENT
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
#endif /* AMBIENT */

rtDeclareVariable(float, minweight, , );	/* minimum ray weight (lw) */
rtDeclareVariable(int, maxdepth, , );	/* maximum recursion depth (lr) */

rtBuffer<DistantLight> lights;

/* Material variables */
rtDeclareVariable(unsigned int, type, , );	/* The material type representing "plastic", "metal", or "trans" */
rtDeclareVariable(float3, color, , );	/* The material color given by the rad file "plastic", "metal", or "trans" object */
rtDeclareVariable(float, spec, , );	/* The material specularity given by the rad file "plastic", "metal", or "trans" object */
rtDeclareVariable(float, rough, , );	/* The material roughness given by the rad file "plastic", "metal", or "trans" object */

/* OptiX variables */
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

/* Attributes */
//rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );


//RT_METHOD float3 dirnorm(Ray *shadow_ray, PerRayData_shadow *shadow_prd, const NORMDAT *nd, const float& omega);
//RT_METHOD float3 gaussamp(const NORMDAT *nd);
#ifdef AMBIENT
RT_METHOD float3 multambient(float3 aval, const float3& normal, const float3& pnormal, const float3& hit);
#ifdef DAYSIM_COMPATIBLE
RT_METHOD int doambient(float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit, DaysimCoef dc);
#else
RT_METHOD int doambient(float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit);
#endif
//RT_METHOD int ambsample( AMBHEMI *hp, const int& i, const int& j, const float3 normal, const float3 hit );
#endif /* AMBIENT */


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


#ifdef AMBIENT
// Compute the ambient component and multiply by the coefficient.
RT_METHOD float3 multambient(float3 aval, const float3& normal, const float3& pnormal, const float3& hit)
{
	int do_ambient = 1;
	float 	d;

	if (ambdiv <= 0)			/* no ambient calculation */
		goto dumbamb;
	/* check number of bounces */
	if (prd.ambient_depth >= ambounce)
		goto dumbamb;
	/* check ambient list */
	//if (ambincl != -1 && r->ro != NULL && ambincl != inset(ambset, r->ro->omod))
	//	goto dumbamb;

	if (ambacc > FTINY && navsum != 0) {			/* ambient storage */
		//if (tracktime)				/* sort to minimize thrashing */
		//	sortambvals(0);

		/* interpolate ambient value */
		//acol = make_float3( 0.0f );
		//d = sumambient(acol, r, normal, rdepth, &atrunk, thescene.cuorg, thescene.cusize);
		PerRayData_ambient ambient_prd;
		ambient_prd.result = make_float3(0.0f);
		ambient_prd.surface_normal = pnormal;
		ambient_prd.ambient_depth = prd.ambient_depth;
		ambient_prd.wsum = 0.0f;
		ambient_prd.weight = prd.weight;
#ifdef OLDAMB
		ambient_prd.state = prd.state;
#endif
#ifdef DAYSIM_COMPATIBLE
		ambient_prd.dc = daysimNext(prd.dc);
		daysimSet(ambient_prd.dc, 0.0f);
#endif
#ifdef HIT_COUNT
		ambient_prd.hit_count = 0;
#endif
		const float tmin = ray_start(hit, AMBIENT_RAY_LENGTH);
		Ray ambient_ray = make_Ray(hit, normal, ambient_ray_type, -tmin, tmin);
		rtTrace(top_ambient, ambient_ray, ambient_prd);
#ifdef HIT_COUNT
		prd.hit_count += ambient_prd.hit_count;
#endif
		if (ambient_prd.wsum > FTINY) { // TODO if miss program is called, set wsum = 1.0f or place this before ambacc == 0.0f
			ambient_prd.result *= 1.0f / ambient_prd.wsum;
#ifdef DAYSIM_COMPATIBLE
			daysimAddScaled(prd.dc, ambient_prd.dc, aval.x / ambient_prd.wsum);
#endif
			return aval * ambient_prd.result;
		}
		//rdepth++;				/* need to cache new value */
		//d = makeambient(acol, r, normal, rdepth-1); //TODO implement as miss program for ambient ray
		//rdepth--;
		//if ( dot( ambient_prd.result, ambient_prd.result) > FTINY) { // quick check to see if a value was returned by miss program
		//	return aval * ambient_prd.result;		/* got new value */
		//}

#ifdef FILL_GAPS
		do_ambient = prd.primary && ambdiv_final;
#else
		do_ambient = !prd.ambient_depth && ambdiv_final;
#endif
	}
	if (do_ambient) {			/* no ambient storage */
		/* Option to show error if nothing found */
		if (ambdiv_final < 0)
			rtThrow(RT_EXCEPTION_USER - ambdiv_final);

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
dumbamb:					/* return global value */
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
	int n = sqrtf(ambdiv_final * wt) + 0.5f;
	int i = 1 + 8 * (ambacc > FTINY);	/* minimum number of samples */
	if (n < i)
		n = i;
	const int nn = n * n;
	float3 acol = make_float3(0.0f);
	unsigned int sampOK = 0u;
	/* assign coefficient */
	float3 acoef = *rcol / nn;

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
#ifdef DAYSIM_COMPATIBLE
	new_prd.dc = daysimNext(dc);
#endif

	Ray amb_ray = make_Ray(hit, pnormal, radiance_ray_type, RAY_START, RAY_END); // Use normal point as temporary direction
	/* End ambsample setup */

	/* make tangent plane axes */
	float3 ux = getperpendicular(pnormal, prd.state);
	float3 uy = cross(pnormal, ux);
	/* sample divisions */
	for (i = n; i--;)
		for (int j = n; j--;) {
			//hp.sampOK += ambsample( &hp, i, j, normal, hit );
			/* ambsample in ambcomp.c */
			float2 spt = 0.1f + 0.8f * make_float2(curand_uniform(prd.state), curand_uniform(prd.state));
			SDsquare2disk(spt, (j + spt.y) / n, (i + spt.x) / n);
			float zd = sqrtf(1.0f - dot(spt, spt));
			amb_ray.direction = normalize(spt.x*ux + spt.y*uy + zd*pnormal);
			if (dot(amb_ray.direction, normal) <= 0) /* Prevent light leaks */
				continue;
			amb_ray.tmin = ray_start(hit, amb_ray.direction, normal, RAY_START);
			//dimlist[ndims++] = AI(hp,i,j) + 90171;

			setupPayload(new_prd, 0);
			//Ray amb_ray = make_Ray( hit, rdir, radiance_ray_type, RAY_START, RAY_END );
			rtTrace(top_object, amb_ray, new_prd);
			resolvePayload(prd, new_prd);

			//ndims--;
			if (isnan(new_prd.result)) // TODO How does this happen?
				continue;
			if (new_prd.distance <= FTINY)
				continue;		/* should never happen */
			acol += new_prd.result * acoef;	/* add to our sum */
#ifdef DAYSIM_COMPATIBLE
			daysimAddScaled(dc, new_prd.dc, acoef.x);
#endif
			sampOK++;
		}
	*rcol = acol;
	if (!sampOK) {		/* utter failure? */
		return(0);
	}
	if (sampOK < nn) {
		//hp.sampOK *= -1;	/* soft failure */
		return(1);
	}
	//n = ambssamp * wt + 0.5f;
	//if (n > 8) {			/* perform super-sampling? */
	//	ambsupersamp(hp, n);
	//	*rcol = hp.acol;
	//}
	return(1);			/* all is well */
}
#endif /* AMBIENT */
