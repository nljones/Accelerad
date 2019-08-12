/*
 *  material_normal.cu - hit programs for normal materials on GPUs.
 */

#include "accelerad_copyright.h"

#include "otypes.h"	/* For definition of MAT_METAL */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_ray.h"
#ifdef CONTRIB_DOUBLE
#include "optix_double.h"
#endif

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
rtDeclareVariable(rtObject,     top_object, , );
rtDeclareVariable(rtObject,     top_ambient, , );

rtDeclareVariable(float,        specthresh, , );	/* This is the minimum fraction of reflection or transmission, under which no specular sampling is performed */
rtDeclareVariable(float,        specjitter, , );	/* specular sampling (ss) */

#ifdef LIGHTS
rtDeclareVariable(float,        dstrsrc, , ); /* direct jitter (dj) */
rtDeclareVariable(float,        srcsizerat, , );	/* direct sampling ratio (ds) */
//rtDeclareVariable(float,        shadthresh, , );	/* direct threshold (dt) */
//rtDeclareVariable(float,        shadcert, , );	/* direct certainty (dc) */
//rtDeclareVariable(int,          directrelay, , );	/* direct relays for secondary sources (dr) */
//rtDeclareVariable(int,          vspretest, , );	/* direct presampling density for secondary sources (dp) */
#endif /* LIGHTS */

#ifdef AMBIENT
rtDeclareVariable(float3,       ambval, , );	/* This is the final value used in place of an indirect light calculation */
rtDeclareVariable(int,          ambvwt, , );	/* As new indirect irradiances are computed, they will modify the default ambient value in a moving average, with the specified weight assigned to the initial value given on the command and all other weights set to 1 */
rtDeclareVariable(int,          ambounce, , );	/* Ambient bounces (ab) */
//rtDeclareVariable(int,          ambres, , );	/* Ambient resolution (ar) */
rtDeclareVariable(float,        ambacc, , );	/* Ambient accuracy (aa). This value will approximately equal the error from indirect illuminance interpolation */
rtDeclareVariable(int,          ambdiv, , );	/* Ambient divisions (ad) */
rtDeclareVariable(int,          ambdiv_final, , ); /* Number of ambient divisions for final-pass fill (ag) */
//rtDeclareVariable(int,          ambssamp, , );	/* Ambient super-samples (as) */
rtDeclareVariable(float,        avsum, , );		/* computed ambient value sum (log) */
rtDeclareVariable(unsigned int, navsum, , );	/* number of values in avsum */
#endif /* AMBIENT */

rtDeclareVariable(float,        exposure, , ) = 0.0f; /* Current exposure (-pe), zero unless called from rvu */

rtBuffer<DistantLight> lights;

/* Geometry instance variables */
#ifdef LIGHTS
rtBuffer<float3> vertex_buffer;
rtBuffer<uint3>  lindex_buffer;    // position indices
#endif


RT_METHOD float3 dirnorm(Ray *shadow_ray, PerRayData_shadow *shadow_prd, const NORMDAT *nd, const float& omega, const float3& ray_dir, PerRayData_radiance &prd);
RT_METHOD float3 gaussamp(const NORMDAT *nd, const float3& ray_dir, PerRayData_radiance &prd);
#ifdef AMBIENT
RT_METHOD float3 multambient(float3 aval, const float3& normal, const float3& pnormal, const float3& hit, const unsigned int& ambincl, PerRayData_radiance &prd);
#ifdef DAYSIM_COMPATIBLE
RT_METHOD int doambient(float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit, PerRayData_radiance &prd, DaysimCoef dc);
#else
RT_METHOD int doambient(float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit, PerRayData_radiance &prd);
#endif
//RT_METHOD int ambsample( AMBHEMI *hp, const int& i, const int& j, const float3 normal, const float3 hit );
#endif /* AMBIENT */
#ifdef LIGHTS
RT_METHOD unsigned int flatpart( const float3& v, const float3& r0, const float3& r1, const float3& r2, const float& weight );
RT_METHOD float solid_angle( const float3& r0, const float3& r1, const float3& r2 );
RT_METHOD float3 barycentric( float2& lambda, const float3& r0, const float3& r1, const float3& r2, const int flip );
#endif /* LIGHTS */
//RT_METHOD float2 multisamp2(float r);
//RT_METHOD int ilhash(int3 d);


RT_CALLABLE_PROGRAM PerRayData_shadow closest_hit_normal_shadow(IntersectData const&data, PerRayData_shadow prd_shadow)
{
	NORMDAT nd;

	/* check for back side */
	nd.pnorm = faceforward(data.world_shading_normal, -data.ray_direction, data.world_geometric_normal);
	nd.normal = faceforward(data.world_geometric_normal, -data.ray_direction, data.world_geometric_normal);

	nd.hit = data.hit;
	nd.mcolor = data.mat.color;
	nd.rspec = data.mat.params.n.spec;
	nd.alpha2 = data.mat.params.n.rough * data.mat.params.n.rough;
	nd.specfl = 0u; /* specularity flags */

#ifdef TRANSMISSION
	if (data.mat.params.n.trans > 0.0f) { // type == MAT_TRANS
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
		nd.prdir = data.ray_direction;
		nd.trans = data.mat.params.n.trans * (1.0f - nd.rspec);
		nd.tspec = nd.trans * data.mat.params.n.tspec;
		if (nd.tspec > FTINY) {
			nd.specfl |= SP_TRAN;

			/* check threshold */
			if (!(nd.specfl & SP_PURE) && specthresh >= nd.tspec - FTINY)
				nd.specfl |= SP_TBLT;
			if (hastexture) { //TODO only if ambient depth == 0
				if (dot(nd.prdir - pert, nd.normal) < -FTINY)
					nd.prdir = normalize(nd.prdir - pert);	/* OK */
			}
		}
	}

	/* transmitted ray */
	if ((nd.specfl&(SP_TRAN | SP_PURE | SP_TBLT)) == (SP_TRAN | SP_PURE)) {
#ifdef CONTRIB
		prd_shadow.rcoef *= nd.mcolor * nd.tspec;
#endif
		Ray trans_ray = make_Ray(nd.hit, nd.prdir, SHADOW_RAY, ray_start(nd.hit, nd.prdir, nd.normal, RAY_START), RAY_END);
		rtTrace(top_object, trans_ray, prd_shadow);
		prd_shadow.result *= nd.mcolor * nd.tspec;
#ifdef DAYSIM_COMPATIBLE
		daysimScale(prd_shadow.dc, nd.mcolor.x * nd.tspec);
#endif
	}
#endif /* TRANSMISSION */
	return prd_shadow;
}


RT_CALLABLE_PROGRAM PerRayData_radiance closest_hit_normal_radiance(IntersectData const&data, PerRayData_radiance prd)
{
	NORMDAT nd;

	/* check for back side */
	nd.pnorm = faceforward(data.world_shading_normal, -data.ray_direction, data.world_geometric_normal);
	nd.normal = faceforward(data.world_geometric_normal, -data.ray_direction, data.world_geometric_normal);
	nd.hit = data.hit;

	PerRayData_radiance new_prd;
	float3 result = prd.mirror = make_float3(0.0f);
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
#ifdef TRANSMISSION
	nd.prdir = data.ray_direction;
	if (data.mat.params.n.trans > 0.0f) { // data.mat.type == MAT_TRANS
		nd.trans = data.mat.params.n.trans * (1.0f - nd.rspec);
		nd.tspec = nd.trans * data.mat.params.n.tspec;
		nd.tdiff = nd.trans - nd.tspec;
		if (nd.tspec > FTINY) {
			nd.specfl |= SP_TRAN;

							/* check threshold */
			if (!(nd.specfl & SP_PURE) && specthresh >= nd.tspec - FTINY)
				nd.specfl |= SP_TBLT;
			if (!prd.ambient_depth && hastexture) {
				if (dot(nd.prdir - pert, nd.normal) < -FTINY)
					nd.prdir = normalize(nd.prdir - pert);	/* OK */
			}
		}
	}

	/* diffuse reflection */
	nd.rdiff = 1.0f - nd.trans - nd.rspec;

	/* transmitted ray */
	if ((nd.specfl&(SP_TRAN | SP_PURE | SP_TBLT)) == (SP_TRAN | SP_PURE) && rayorigin(new_prd, prd, nd.mcolor * nd.tspec, 0, 0)) {
#ifdef DAYSIM_COMPATIBLE
		new_prd.dc = daysimNext(prd.dc);
#endif
		setupPayload(new_prd);
		Ray trans_ray = make_Ray(nd.hit, nd.prdir, RADIANCE_RAY, ray_start(nd.hit, nd.prdir, nd.normal, RAY_START), new_prd.tmax);
		rtTrace(top_object, trans_ray, new_prd);
		new_prd.result *= nd.mcolor * nd.tspec;
		result += new_prd.result;
#ifdef DAYSIM_COMPATIBLE
		daysimAddScaled(prd.dc, new_prd.dc, nd.mcolor.x * nd.tspec);
#endif
		if (nd.tspec >= 1.0f - FTINY) {
			/* completely transparent */
			prd.mirror = new_prd.mirror * nd.mcolor * nd.tspec;
			prd.mirror_distance = data.t + new_prd.mirror_distance;
			prd.distance = data.t + new_prd.distance;
		}
		else if (nd.tspec > nd.tdiff + nd.rdiff)
			prd.distance = data.t + rayDistance(new_prd);
		resolvePayload(prd, new_prd);
	}
#endif

	// return if it's a shadow ray, which it isn't

	/* get specular reflection */
	if (nd.rspec > FTINY) {
		nd.specfl |= SP_REFL;

		/* compute specular color */
		if (data.mat.type != MAT_METAL) {
			nd.scolor = make_float3(nd.rspec);
		} else {
			if (fest > FTINY) {
				float d = data.mat.params.n.spec * (1.0f - fest);
				nd.scolor = fest + nd.mcolor * d;
			} else {
				nd.scolor = nd.mcolor * nd.rspec;
			}
		}

		/* check threshold */
		if (!(nd.specfl & SP_PURE) && specthresh >= nd.rspec - FTINY) {
			nd.specfl |= SP_RBLT;
		}
	}

	/* reflected ray */
	if ((nd.specfl&(SP_REFL | SP_PURE | SP_RBLT)) == (SP_REFL | SP_PURE) && rayorigin(new_prd, prd, nd.scolor, 1, 0)) {
#ifdef DAYSIM_COMPATIBLE
		new_prd.dc = daysimNext(prd.dc);
#endif
		setupPayload(new_prd);
		float3 vrefl = reflect(data.ray_direction, nd.pnorm);
		Ray refl_ray = make_Ray(nd.hit, vrefl, RADIANCE_RAY, ray_start(nd.hit, vrefl, nd.normal, RAY_START), new_prd.tmax);
		rtTrace(top_object, refl_ray, new_prd);
		new_prd.result *= nd.scolor;
		prd.mirror = new_prd.result;
		result += new_prd.result;
		prd.mirror_distance = data.t;
#ifdef DAYSIM_COMPATIBLE
		daysimAddScaled(prd.dc, new_prd.dc, nd.scolor.x);
#endif
		if (nd.specfl & SP_FLAT && (prd.ambient_depth || !hastexture))
			prd.mirror_distance += rayDistance(new_prd);
		resolvePayload(prd, new_prd);
	}

	if (!(nd.specfl & SP_PURE && nd.rdiff <= FTINY && nd.tdiff <= FTINY)) { /* not 100% pure specular */
		/* checks *BLT flags */
		if (!(nd.specfl & SP_PURE))
			result += gaussamp(&nd, data.ray_direction, prd);

#ifdef AMBIENT
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
#endif /* AMBIENT */

		/* add direct component */
		// This is the call to direct() in source.c
		// Let's start at line 447, and not bother with sorting for now

		// compute direct lighting
		PerRayData_shadow shadow_prd;
#ifdef DAYSIM_COMPATIBLE
		shadow_prd.dc = daysimNext(prd.dc);
#endif
		Ray shadow_ray = make_Ray(nd.hit, nd.pnorm, SHADOW_RAY, RAY_START, RAY_END);

		/* contributions from distant lights (mainly the sun) */
		unsigned int num_lights = lights.size();
		for (unsigned int i = 0u; i < num_lights; i++) {
			const DistantLight light = lights[i];
			if ( light.casts_shadow ) {
				shadow_prd.target = i;
				shadow_ray.direction = normalize( light.pos ); //TODO implement direct jitter for distant light sources
				shadow_ray.tmin = ray_start(nd.hit, shadow_ray.direction, nd.normal, RAY_START);
				shadow_ray.tmax = RAY_END;
				result += dirnorm(&shadow_ray, &shadow_prd, &nd, light.solid_angle, data.ray_direction, prd);
			}
		}

#ifdef LIGHTS
		/* contributions from nearby lights */
		num_lights = lindex_buffer.size();
		for (unsigned int i = 0u; i < num_lights; i++) {
			const uint3 v_idx = lindex_buffer[i];

			const float3 r0 = vertex_buffer[v_idx.x] - nd.hit;
			const float3 r1 = vertex_buffer[v_idx.y] - nd.hit;
			const float3 r2 = vertex_buffer[v_idx.z] - nd.hit;
			float3 rdir = ( r0 + r1 + r2 ) / 3.0f;

			const unsigned int divs = flatpart( rdir, r0, r1, r2, prd.weight ); //TODO divisions should be smaller closer to the light source
			const float step = 1.0f / divs;

			for ( int j = 0; j < divs; j++ )
				for ( int k = 0; k < divs; k++ ) {
					float2 lambda = make_float2( step * j, step * k );
					const float3 p0 = barycentric( lambda, r0, r1, r2, k + j >= divs );

					lambda = make_float2( step * ( j + 1 ), step * k );
					const float3 p1 = barycentric( lambda, r0, r1, r2, k + j >= divs );

					lambda = make_float2( step * j, step * ( k + 1 ) );
					const float3 p2 = barycentric( lambda, r0, r1, r2, k + j >= divs );

					const float omega = solid_angle( p0, p1, p2 );

					if ( omega > FTINY ) {
						/* from nextssamp in srcsamp.c */
						rdir = ( p0 + p1 + p2 ) / 3.0f;
						if ( dstrsrc > FTINY ) {
							/* jitter sample using random barycentric coordinates */
							lambda = make_float2( curand_uniform( prd.state ), curand_uniform( prd.state ) );
							float3 vpos = barycentric( lambda, p0, p1, p2, lambda.x + lambda.y >= 1.0f );
							rdir += dstrsrc * ( vpos - rdir );
						}

						shadow_prd.target = -v_idx.x - 1; //TODO find a better way to identify surface
						shadow_ray.direction = normalize( rdir );
						shadow_ray.tmin = ray_start(nd.hit, shadow_ray.direction, nd.normal, RAY_START);
						shadow_ray.tmax = length(rdir) * 1.0001f;
						result += dirnorm(&shadow_ray, &shadow_prd, &nd, omega, data.ray_direction, prd);
					}
				}
		}
#endif /* LIGHTS */
	}

	// pass the color back up the tree
	prd.result = result;
	return prd;
}

/* compute source contribution */
RT_METHOD float3 dirnorm(Ray *shadow_ray, PerRayData_shadow *shadow_prd, const NORMDAT *nd, const float& omega, const float3& ray_dir, PerRayData_radiance &prd)
{
	float3 cval = make_float3( 0.0f );
	float ldot = dot(nd->pnorm, shadow_ray->direction);

#ifdef TRANSMISSION
	if (ldot < 0.0f ? nd->trans <= FTINY : nd->trans >= 1.0f - FTINY)
#else
	if ( ldot <= FTINY )
#endif
		return cval;
	
	/* Fresnel estimate */
	float lrdiff = nd->rdiff;
	float ltdiff = nd->tdiff;
	if (nd->specfl & SP_PURE && nd->rspec >= FRESTHRESH && (lrdiff > FTINY) | (ltdiff > FTINY)) {
		float dtmp = 1.0f - FRESNE(fabs(ldot));
		lrdiff *= dtmp;
		ltdiff *= dtmp;
	}

	if (ldot > FTINY && lrdiff > FTINY) {
		/*
		 *  Compute and add diffuse reflected component to returned
		 *  color.  The diffuse reflected component will always be
		 *  modified by the color of the material.
		 */
		float dtmp = ldot * omega * lrdiff * M_1_PIf;
		cval += nd->mcolor * dtmp;
	}
#ifdef TRANSMISSION
	if (ldot < -FTINY && ltdiff > FTINY) {
		/*
		 *  Compute diffuse transmission.
		 */
		float dtmp = -ldot * omega * ltdiff * M_1_PIf;
		cval += nd->mcolor * dtmp;
	}
#endif
	if (ldot > FTINY && (nd->specfl&(SP_REFL | SP_PURE)) == SP_REFL) {
		/*
		 *  Compute specular reflection coefficient using
		 *  Gaussian distribution model.
		 */
		/* roughness */
		float dtmp = nd->alpha2;
		/* + source if flat */
		if (nd->specfl & SP_FLAT)
			dtmp += omega * 0.25f * M_1_PIf;
		/* half vector */
		float3 vtmp = shadow_ray->direction - ray_dir;
		float d2 = dot(vtmp, nd->pnorm);
		d2 *= d2;
		float d3 = dot( vtmp, vtmp );
		float d4 = (d3 - d2) / d2;
		/* new W-G-M-D model */
		dtmp = expf(-d4/dtmp) * d3 / (M_PIf * d2*d2 * dtmp);
		/* worth using? */
		if (dtmp > FTINY) {
			dtmp *= ldot * omega;
			cval += nd->scolor * dtmp;
		}
	}
#ifdef TRANSMISSION
	if (ldot < -FTINY && (nd->specfl&(SP_TRAN | SP_PURE)) == SP_TRAN) {
		/*
		 *  Compute specular transmission.  Specular transmission
		 *  is always modified by material color.
		 */
						/* roughness + source */
		float dtmp = nd->alpha2 + omega * M_1_PIf;
						/* Gaussian */
		dtmp = expf((2.0f * dot(nd->prdir, shadow_ray->direction) - 2.0f) / dtmp) / (M_PIf * dtmp); // may need to perturb direction
						/* worth using? */
		if (dtmp > FTINY) {
			dtmp *= nd->tspec * omega * sqrtf(-ldot / nd->pdot);
			cval += nd->mcolor * dtmp;
		}
	}
#endif

	/* from direct() in source.c */
	if (fmaxf(cval) <= 0.0f)
		return cval;

	// cast shadow ray
	shadow_prd->result = make_float3(0.0f);
#ifdef CONTRIB
	shadow_prd->rcoef = prd.rcoef * cval;
#endif
#ifdef ANTIMATTER
	shadow_prd->mask = prd.mask;
	shadow_prd->inside = prd.inside;
#endif
#ifdef DAYSIM_COMPATIBLE
	daysimSet(shadow_prd->dc, 0.0f);
#endif
	rtTrace(top_object, *shadow_ray, *shadow_prd);

#ifdef DAYSIM_COMPATIBLE
	daysimAddScaled(prd.dc, shadow_prd->dc, cval.x);
#endif
	return cval * shadow_prd->result;
}

// sample Gaussian specular
RT_METHOD float3 gaussamp(const NORMDAT *nd, const float3& ray_dir, PerRayData_radiance &prd)
{
	float3 rcol = make_float3( 0.0f );

	/* This section is based on the gaussamp method in normal.c */
	if ((nd->specfl & (SP_REFL | SP_RBLT)) != SP_REFL && (nd->specfl & (SP_TRAN | SP_TBLT)) != SP_TRAN)
		return rcol;

	PerRayData_radiance gaus_prd;
	Ray gaus_ray = make_Ray(nd->hit, nd->pnorm, RADIANCE_RAY, RAY_START, RAY_END);

	float d;

	/* set up sample coordinates */
	float3 u = getperpendicular(nd->pnorm); // prd.state?
	float3 v = cross(nd->pnorm, u);

	unsigned int nstarget, nstaken, ntrials;

	/* compute reflection */
	if ((nd->specfl & (SP_REFL | SP_RBLT)) == SP_REFL && rayorigin(gaus_prd, prd, nd->scolor, 1, 1)) { //TODO the ambient depth increment is a hack to prevent the sun from affecting specular values
		float3 scolor = nd->scolor;
		nstarget = 1;
		if (specjitter > 1.5f) {	/* multiple samples? */ // By default it's 1.0
			nstarget = specjitter * prd.weight + 0.5f;
			if ( gaus_prd.weight <= minweight * nstarget )
				nstarget = gaus_prd.weight / minweight;
			if ( nstarget > 1 ) {
				d = 1.0f / nstarget;
				scolor *= d; //scolor, stored as ray rcoef
#ifdef CONTRIB
				gaus_prd.rcoef *= d;
#endif
				gaus_prd.weight *= d; // TODO make sure weight isn't changed by hit programs
			} else
				nstarget = 1;
		}
		float3 scol = make_float3( 0.0f );
#ifdef DAYSIM_COMPATIBLE
		DaysimCoef dc = daysimNext(prd.dc);
		if (nstarget > 1) {
			daysimSet(dc, 0.0f);
			gaus_prd.dc = daysimNext(dc);
		} else
			gaus_prd.dc = dc;
#endif
		//dimlist[ndims++] = (int)(size_t)np->mp;
		unsigned int maxiter = MAXITER * nstarget;
		for (nstaken = ntrials = 0; nstaken < nstarget && ntrials < maxiter; ntrials++) {
			float2 rv = make_float2( curand_uniform( prd.state ), curand_uniform( prd.state ) ); // should be evenly distributed in both dimensions
			d = 2.0f * M_PIf * rv.x;
			float cosp = cosf( d );
			float sinp = sinf( d );
			if ( ( 0.0f <= specjitter ) && ( specjitter < 1.0f ) )
				rv.y = 1.0f - specjitter * rv.y;
			if ( rv.y <= FTINY )
				d = 1.0f;
			else
				d = sqrtf(nd->alpha2 * -logf(rv.y));
			float3 h = nd->pnorm + d * (cosp * u + sinp * v);
			d = -2.0f * dot( h, ray_dir ) / ( 1.0f + d*d );
			gaus_ray.direction = ray_dir + h * d;

			/* sample rejection test */
			if ((d = dot(gaus_ray.direction, nd->normal)) <= FTINY)
				continue;

			gaus_ray.direction = normalize( gaus_ray.direction );
			gaus_ray.tmin = ray_start(nd->hit, gaus_ray.direction, nd->normal, RAY_START);
			gaus_ray.tmax = gaus_prd.tmax;

			setupPayload(gaus_prd);
			//if (nstaken) // check for prd data that needs to be cleared
			rtTrace(top_object, gaus_ray, gaus_prd);
			resolvePayload(prd, gaus_prd);

			/* W-G-M-D adjustment */
			if (nstarget > 1) {	
				d = 2.0f / (1.0f - dot(ray_dir, nd->normal) / d);
				scol += gaus_prd.result * d;
#ifdef DAYSIM_COMPATIBLE
				daysimAddScaled(dc, gaus_prd.dc, d);
#endif
			} else {
				rcol += gaus_prd.result * scolor;
#ifdef DAYSIM_COMPATIBLE
				daysimAddScaled(prd.dc, gaus_prd.dc, scolor.x);
#endif
			}

			++nstaken;
		}
		/* final W-G-M-D weighting */
		if (nstarget > 1) {
			scol *= scolor;
			d = (float)nstarget / ntrials;
			rcol += scol * d;
#ifdef DAYSIM_COMPATIBLE
			daysimAddScaled(prd.dc, dc, scolor.x * d);
#endif
		}
		//ndims--;
	}

#ifdef TRANSMISSION
	/* compute transmission */
	float3 mcolor = nd->mcolor * nd->tspec;	/* modified by color */
	if ((nd->specfl & (SP_TRAN | SP_TBLT)) == SP_TRAN && rayorigin(gaus_prd, prd, mcolor, 0, 0)) {
		nstarget = 1;
		if (specjitter > 1.5f) {	/* multiple samples? */ // By default it's 1.0
			nstarget = specjitter * prd.weight + 0.5f;
			if ( gaus_prd.weight <= minweight * nstarget )
				nstarget = gaus_prd.weight / minweight;
			if ( nstarget > 1 ) {
				d = 1.0f / nstarget;
				mcolor *= d; //mcolor, stored as ray rcoef
#ifdef CONTRIB
				gaus_prd.rcoef *= d;
#endif
				gaus_prd.weight *= d; // TODO make sure weight isn't changed by hit programs
			} else
				nstarget = 1;
		}
		//dimlist[ndims++] = (int)(size_t)np->mp;
		unsigned int maxiter = MAXITER * nstarget;
		for (nstaken = ntrials = 0; nstaken < nstarget && ntrials < maxiter; ntrials++) {
			float2 rv = make_float2( curand_uniform( prd.state ), curand_uniform( prd.state ) ); // should be evenly distributed in both dimensions
			d = 2.0f * M_PIf * rv.x;
			float cosp = cosf( d );
			float sinp = sinf( d );
			if ( ( 0.0f <= specjitter ) && ( specjitter < 1.0f ) )
				rv.y = 1.0f - specjitter * rv.y;
			if ( rv.y <= FTINY )
				d = 1.0f;
			else
				d = sqrtf(nd->alpha2 * -logf(rv.y));
			gaus_ray.direction = nd->prdir + d * (cosp * u + sinp * v); // ray direction is perturbed

			/* sample rejection test */
			if (dot(gaus_ray.direction, nd->normal) >= -FTINY)
				continue;

			gaus_ray.direction = normalize( gaus_ray.direction );
			gaus_ray.tmin = ray_start(nd->hit, gaus_ray.direction, nd->normal, RAY_START);
			gaus_ray.tmax = gaus_prd.tmax;

#ifdef DAYSIM_COMPATIBLE
			gaus_prd.dc = daysimNext(prd.dc);
#endif
			setupPayload(gaus_prd);
			//if (nstaken) // check for prd data that needs to be cleared
			rtTrace(top_object, gaus_ray, gaus_prd);
			resolvePayload(prd, gaus_prd);
			rcol += gaus_prd.result * mcolor;
			++nstaken;
#ifdef DAYSIM_COMPATIBLE
			daysimAddScaled(prd.dc, gaus_prd.dc, mcolor.x);
#endif
		}
		//ndims--;
	}
#endif
	//return make_float3(0.0f);
	return rcol;
}

#ifdef AMBIENT
// Compute the ambient component and multiply by the coefficient.
RT_METHOD float3 multambient(float3 aval, const float3& normal, const float3& pnormal, const float3& hit, const unsigned int& ambincl, PerRayData_radiance &prd)
{
	if (exposure && !prd.ambient_depth) // TODO exposure is hack to check if we are running rvu
		return make_float3(0.0f);

	int do_ambient = 1;
	float 	d;

	if (ambdiv <= 0)			/* no ambient calculation */
		goto dumbamb;
						/* check number of bounces */
	if (prd.ambient_depth >= ambounce)
		goto dumbamb;
						/* check ambient list */
	if (!ambincl)
		goto dumbamb;

	if (ambacc > FTINY && navsum != 0) {			/* ambient storage */
		//if (tracktime)				/* sort to minimize thrashing */
		//	sortambvals(0);

		/* interpolate ambient value */
		//acol = make_float3( 0.0f );
		//d = sumambient(acol, r, normal, rdepth, &atrunk, thescene.cuorg, thescene.cusize);
		PerRayData_ambient ambient_prd;
		ambient_prd.result = make_float3( 0.0f );
		ambient_prd.surface_point = hit;
		ambient_prd.surface_normal = pnormal;
		ambient_prd.ambient_depth = prd.ambient_depth;
		ambient_prd.wsum = 0.0f;
		ambient_prd.weight = prd.weight;
#ifdef DAYSIM_COMPATIBLE
		ambient_prd.dc = daysimNext(prd.dc);
		daysimSet(ambient_prd.dc, 0.0f);
#endif
#ifdef HIT_COUNT
		ambient_prd.hit_count = 0;
#endif
		const float tmax = ray_start(hit, AMBIENT_RAY_LENGTH);
		Ray ambient_ray = make_Ray(hit - normal * tmax, normal, AMBIENT_RAY, 0.0f, 2.0f * tmax);
		rtTrace(top_ambient, ambient_ray, ambient_prd, RT_VISIBILITY_ALL, RT_RAY_FLAG_DISABLE_CLOSESTHIT);
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

		do_ambient = !prd.ambient_depth && ambdiv_final;
	}
	if (do_ambient) {			/* no ambient storage */
		/* Option to show error if nothing found */
		if (ambdiv_final < 0)
			rtThrow(RT_EXCEPTION_CUSTOM - ambdiv_final);

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
	} else {
		d = expf( avsum / (float)navsum );
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
	if (wt > (d = 0.8f * fmaxf(*rcol) * wt / (ambdiv_final * minweight))) // Ignore ambacc <= FTINY check because this is faking ambacc == 0 calc
		wt = d;			/* avoid ray termination */
	int n = sqrtf(ambdiv_final * wt) + 0.5f;
	int i = 1 + 5 * (ambacc > FTINY);	/* minimum number of samples */
	if (n < i)
		n = i;
	const int nn = n * n;
	float3 acol = make_float3( 0.0f );
	unsigned int sampOK = 0u;
					/* assign coefficient */
	float3 acoef = *rcol / nn;

	/* Setup from ambsample in ambcomp.c */
	PerRayData_radiance new_prd;
#ifdef DAYSIM_COMPATIBLE
	new_prd.dc = daysimNext(dc);
#endif

	Ray amb_ray = make_Ray( hit, pnormal, RADIANCE_RAY, RAY_START, RAY_END ); // Use normal point as temporary direction
	/* End ambsample setup */

					/* make tangent plane axes */
	float3 ux = getperpendicular( pnormal, prd.state );
	float3 uy = cross( pnormal, ux );
					/* sample divisions */
	for (i = n; i--; )
	    for (int j = n; j--; ) {
			if (!rayorigin(new_prd, prd, acoef, 1, 1))
				continue;

			//hp.sampOK += ambsample( &hp, i, j, normal, hit );
			/* ambsample in ambcomp.c */
			float2 spt = make_float2(curand_uniform(prd.state), curand_uniform(prd.state));
			if (i > 0 && i < n - 1 && j > 0 && j < n - 1) /* avoid coincident samples */
				spt = 0.1f + 0.8f * spt;
			SDsquare2disk( spt, (j+spt.y) / n, (i+spt.x) / n );
			float zd = sqrtf( 1.0f - dot( spt, spt ) );
			amb_ray.direction = normalize( spt.x*ux + spt.y*uy + zd*pnormal );
			if (dot(amb_ray.direction, normal) <= 0) /* Prevent light leaks */
				continue;
			amb_ray.tmin = ray_start( hit, amb_ray.direction, normal, RAY_START );
			amb_ray.tmax = new_prd.tmax;
			//dimlist[ndims++] = AI(hp,i,j) + 90171;

			setupPayload(new_prd);
			//Ray amb_ray = make_Ray( hit, rdir, RADIANCE_RAY, RAY_START, new_prd.tmax );
			rtTrace(top_object, amb_ray, new_prd);
			resolvePayload(prd, new_prd);

			//ndims--;
			if ( isnan( new_prd.result ) ) // TODO How does this happen?
				continue;
			if ( new_prd.distance <= FTINY )
				continue;		/* should never happen */
			acol += new_prd.result * acoef;	/* add to our sum */
#ifdef DAYSIM_COMPATIBLE
			daysimAddScaled(dc, new_prd.dc, acoef.x);
#endif
			sampOK++;
		}
	*rcol = acol;
	if ( !sampOK ) {		/* utter failure? */
		return( 0 );
	}
	if ( sampOK < nn ) {
		//hp.sampOK *= -1;	/* soft failure */
		return( 1 );
	}
	//n = ambssamp * wt + 0.5f;
	//if (n > 8) {			/* perform super-sampling? */
	//	ambsupersamp(hp, n);
	//	*rcol = hp.acol;
	//}
	return( 1 );			/* all is well */
}
#endif /* AMBIENT */

#ifdef LIGHTS
/* partition a flat source */
RT_METHOD unsigned int flatpart( const float3& v, const float3& r0, const float3& r1, const float3& r2, const float& weight )
{
	//float3 vp = source[si->sn].snorm;
	//if ( dot( v, vp ) <= 0.0f )		/* behind source */
	//	return 0u;

	if ( srcsizerat <= FTINY )
		return 1u;

	float d;

	/* Find longest edge */
	float3 vp = r1 - r0;
	float d2 = dot( vp, vp );
	vp = r2 - r1;
	if ( ( d = dot( vp, vp ) ) > d2 )
		d2 = d;
	vp = r2 - r0;
	if ( ( d = dot( vp, vp ) ) > d2 )
		d2 = d;

	/* Find minimum partition size */
	d = srcsizerat / weight;
	d *= d * dot( v, v );

	/* Find number of partions */
	d2 /= d;
	if ( d2 < 1.0f )
		return 1u;
	if ( d2 > ( d = MAXSPART >> 1 ) ) // Divide maximum partitions by two going from rectangle to triangle
		d2 = d;
	return (unsigned int)sqrtf( d2 );
}

/* Solid angle calculation from "The solid angle of a plane triangle", A van Oosterom and J Strackee */
RT_METHOD float solid_angle( const float3& r0, const float3& r1, const float3& r2 )
{
	const float l0 = length( r0 );
	const float l1 = length( r1 );
	const float l2 = length( r2 );

	const float numerator = dot( r0, cross( r1, r2 ) );
	const float denominator = l0 * l1 * l2 + dot( r0, r1 ) * l2 + dot( r0, r2 ) * l1 + dot( r1, r2 ) * l0;
	return 2.0f * fabsf( atan2( numerator, denominator ) );
}

/* Compute point from barycentric coordinates and flip if outside triangle */
RT_METHOD float3 barycentric( float2& lambda, const float3& r0, const float3& r1, const float3& r2, const int flip )
{
	if ( flip )
		lambda = 1.0f - lambda;
	return r0 * ( 1.0f - lambda.x - lambda.y ) + r1 * lambda.x + r2 * lambda.y;
}
#endif /* LIGHTS */

/* convert 1-dimensional sample to 2 dimensions, based on multisamp.c */
//RT_METHOD float2 multisamp2(float r)	/* 1-dimensional sample [0,1) */
//{
//	int	j;
//	register int	k;
//	int2	ti;
//	float	s;
//
//	ti = make_int2( 0 );
//	j = 8;
//	while (j--) {
//		k = s = r*(1<<2);
//		r = s - k;
//		ti += ti + make_int2( ((k>>2) & 1), ((k>>1) & 1) );
//	}
//	ti += make_int2( frandom() );
//	ti *= 1.0f/256.0f;
//}

/* hash a set of integer values */
//RT_METHOD int ilhash(int3 d)
//{
//	register int  hval;
//
//	hval = 0;
//	hval ^= d.x * 73771;
//	hval ^= d.y * 96289;
//	hval ^= d.z * 103699;
//	return(hval & 0x7fffffff);
//}
