/*
 *  material_diffuse.cu - hit program for diffuse-only reflection on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_ray.h"

using namespace optix;

#define  TRANSMISSION

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
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(rtObject, top_ambient, , );

rtDeclareVariable(float3, ambval, , );	/* This is the final value used in place of an indirect light calculation */
rtDeclareVariable(int, ambvwt, , );	/* As new indirect irradiances are computed, they will modify the default ambient value in a moving average, with the specified weight assigned to the initial value given on the command and all other weights set to 1 */
rtDeclareVariable(int, ambounce, , );	/* Ambient bounces (ab) */
//rtDeclareVariable(int,          ambres, , );	/* Ambient resolution (ar) */
rtDeclareVariable(float, ambacc, , );	/* Ambient accuracy (aa). This value will approximately equal the error from indirect illuminance interpolation */
rtDeclareVariable(int, ambdiv, , );	/* Ambient divisions (ad) */
rtDeclareVariable(int, ambdiv_final, , ); /* Number of ambient divisions for final-pass fill (ag) */
//rtDeclareVariable(int, ambssamp, , );	/* Ambient super-samples (as) */
rtDeclareVariable(float, avsum, , );		/* computed ambient value sum (log) */
rtDeclareVariable(unsigned int, navsum, , );	/* number of values in avsum */


RT_METHOD float3 multambient(float3 aval, const float3& normal, const float3& pnormal, const float3& hit, const unsigned int& ambincl, PerRayData_radiance &prd);
#ifdef DAYSIM_COMPATIBLE
RT_METHOD int doambient(float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit, PerRayData_radiance &prd, DaysimCoef dc);
#else
RT_METHOD int doambient(float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit, PerRayData_radiance &prd);
#endif


RT_CALLABLE_PROGRAM PerRayData_radiance closest_hit_diffuse_radiance(IntersectData const&data, PerRayData_radiance prd)
{
	NORMDAT nd;

	/* check for back side */
	nd.pnorm = faceforward(data.world_shading_normal, -data.ray_direction, data.world_geometric_normal);
	nd.normal = faceforward(data.world_geometric_normal, -data.ray_direction, data.world_geometric_normal);

	float3 result = make_float3(0.0f);
	nd.hit = data.hit;
	nd.mcolor = data.mat.color;
	nd.scolor = make_float3(0.0f);
	nd.rspec = data.mat.params.n.spec;
	nd.alpha2 = data.mat.params.n.rough * data.mat.params.n.rough;
	nd.specfl = 0u; /* specularity flags */

	/* get roughness */
	if (nd.alpha2 <= FTINY) {
		nd.specfl |= SP_PURE; // label this as a purely specular reflection
	}

	/* perturb normal */
	float3 pert = nd.normal - nd.pnorm;
	int hastexture = dot(pert, pert) > FTINY * FTINY;
	nd.pdot = -dot(data.ray_direction, nd.pnorm);
	if (nd.pdot < 0.0f) {		/* fix orientation from raynormal in raytrace.c */
		nd.pnorm += 2.0f * nd.pdot * data.ray_direction;
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
			result += multambient(aval, nd.normal, nd.pnorm, nd.hit, data.mat.params.n.ambincl, prd);	/* add to returned color */
		}

#ifdef TRANSMISSION
		/* ambient from other side */
		if (nd.tdiff > FTINY) {
			float3 aval = nd.mcolor;	/* modified by material color */
			if (nd.specfl & SP_TBLT)
				aval *= nd.trans;
			else
				aval *= nd.tdiff;
			result += multambient(aval, -nd.normal, -nd.pnorm, nd.hit, data.mat.params.n.ambincl, prd);	/* add to returned color */
		}
#endif /* TRANSMISSION */
	}

	prd.distance = data.t;

	// pass the color back up the tree
	prd.result = result;

	return prd;
}


// Compute the ambient component and multiply by the coefficient.
RT_METHOD float3 multambient(float3 aval, const float3& normal, const float3& pnormal, const float3& hit, const unsigned int& ambincl, PerRayData_radiance &prd)
{
	float 	d;

	/* ambient calculation */
	if (ambdiv > 0 && prd.ambient_depth < ambounce && ambincl) {
		float3 acol = aval;
	#ifdef DAYSIM_COMPATIBLE
		DaysimCoef dc = daysimNext(prd.dc);
		daysimSet(dc, 0.0f);
		d = doambient(&acol, normal, pnormal, hit, prd, dc);
		if (d > FTINY)
			daysimAdd(prd.dc, dc);
	#else
		d = doambient(&acol, normal, pnormal, hit, prd);
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
RT_METHOD int doambient(float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit, PerRayData_radiance &prd, DaysimCoef dc)
#else
RT_METHOD int doambient(float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit, PerRayData_radiance &prd)
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
	if (!rayorigin(new_prd, prd, acoef, 1, 1))
		return(0);

#ifdef DAYSIM_COMPATIBLE
	new_prd.dc = daysimNext(dc);
#endif

	/* End ambsample setup */

	/* make tangent plane axes */
	float3 ux = getperpendicular(pnormal, prd.state);
	float3 uy = cross(pnormal, ux);

	/* ambsample in ambcomp.c */
	float2 spt = make_float2(curand_uniform(prd.state), curand_uniform(prd.state));
	SDsquare2disk(spt, spt.y, spt.x);
	float zd = sqrtf(1.0f - dot(spt, spt));
	float3 direction = normalize(spt.x*ux + spt.y*uy + zd*pnormal);
	if (dot(direction, normal) <= 0) /* Prevent light leaks */
		return(0);

	setupPayload(new_prd);
	Ray amb_ray = make_Ray(hit, direction, RADIANCE_RAY, ray_start(hit, direction, normal, RAY_START), new_prd.tmax);
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
