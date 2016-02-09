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
//#define  ITERATIVE

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
rtDeclareVariable(unsigned int, shadow_ray_type, , );
rtDeclareVariable(unsigned int, ambient_ray_type, , );
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
rtDeclareVariable(int,          ambssamp, , );	/* Ambient super-samples (as) */
#ifdef OLDAMB
rtDeclareVariable(float,        maxarad, , );	/* maximum ambient radius */
rtDeclareVariable(float,        minarad, , );	/* minimum ambient radius */
#endif /* OLDAMB */
rtDeclareVariable(float,        avsum, , );		/* computed ambient value sum (log) */
rtDeclareVariable(unsigned int, navsum, , );	/* number of values in avsum */
#endif /* AMBIENT */

rtDeclareVariable(float,        minweight, , );	/* minimum ray weight (lw) */
rtDeclareVariable(int,          maxdepth, , );	/* maximum recursion depth (lr) */

rtBuffer<DistantLight> lights;

/* Program variables */
rtDeclareVariable(unsigned int, metal, , );	/* The material type representing "metal" */

/* Material variables */
rtDeclareVariable(unsigned int, type, , );	/* The material type representing "plastic", "metal", or "trans" */
rtDeclareVariable(float3,       color, , );	/* The material color given by the rad file "plastic", "metal", or "trans" object */
rtDeclareVariable(float,        spec, , );	/* The material specularity given by the rad file "plastic", "metal", or "trans" object */
rtDeclareVariable(float,        rough, , );	/* The material roughness given by the rad file "plastic", "metal", or "trans" object */
rtDeclareVariable(float,        transm, , ) = 0.0f;	/* The material transmissivity given by the rad file "trans" object */
rtDeclareVariable(float,        tspecu, , ) = 0.0f;	/* The material transmitted specular component given by the rad file "trans" object */

/* Geometry instance variables */
#ifdef LIGHTS
rtBuffer<float3> vertex_buffer;
rtBuffer<uint3>  lindex_buffer;    // position indices
#endif

/* OptiX variables */
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow, rtPayload, );

/* Attributes */
//rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );


RT_METHOD float3 dirnorm(Ray *shadow_ray, PerRayData_shadow *shadow_prd, const NORMDAT *nd, const float& omega);
RT_METHOD float3 gaussamp(const NORMDAT *nd);
#ifdef AMBIENT
RT_METHOD float3 multambient(float3 aval, const float3& normal, const float3& pnormal, const float3& hit);
#ifndef OLDAMB
#ifdef DAYSIM_COMPATIBLE
RT_METHOD int doambient(float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit, DaysimCoef dc);
#else
RT_METHOD int doambient(float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit);
#endif
//RT_METHOD int ambsample( AMBHEMI *hp, const int& i, const int& j, const float3 normal, const float3 hit );
#else /* OLDAMB */
#ifdef DAYSIM_COMPATIBLE
RT_METHOD float doambient( float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit, DaysimCoef dc );
RT_METHOD int divsample( AMBSAMP  *dp, AMBHEMI  *h, const float3& normal, const float3& hit, DaysimCoef dc );
#else
RT_METHOD float doambient( float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit );
RT_METHOD int divsample( AMBSAMP  *dp, AMBHEMI  *h, const float3& normal, const float3& hit );
#endif
RT_METHOD void inithemi( AMBHEMI  *hp, float3 ac, const float3& normal );
//RT_METHOD void comperrs( AMBSAMP *da, AMBHEMI *hp );
//RT_METHOD int ambcmp( const void *p1, const void *p2 );
#endif /* OLDAMB */
#endif /* AMBIENT */
#ifdef LIGHTS
RT_METHOD unsigned int flatpart( const float3& v, const float3& r0, const float3& r1, const float3& r2 );
RT_METHOD float solid_angle( const float3& r0, const float3& r1, const float3& r2 );
RT_METHOD float3 barycentric( float2& lambda, const float3& r0, const float3& r1, const float3& r2, const int flip );
#endif /* LIGHTS */
//RT_METHOD float2 multisamp2(float r);
//RT_METHOD int ilhash(int3 d);


#ifndef LIGHTS
RT_PROGRAM void any_hit_shadow()
{
	// this material is opaque, so it fully attenuates all shadow rays
	prd_shadow.result = make_float3(0.0f);
	rtTerminateRay();
}
#endif


RT_PROGRAM void closest_hit_radiance()
{
	NORMDAT nd;

	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

	/* check for back side */
	nd.pnorm = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
	nd.normal = faceforward( world_geometric_normal, -ray.direction, world_geometric_normal );

	PerRayData_radiance new_prd;
	float3 result = make_float3( 0.0f );
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
#ifdef TRANSMISSION
	float transtest = 0.0f, transdist = t_hit;
	nd.prdir = ray.direction;
	if (transm > 0.0f) { // type == MAT_TRANS
		nd.trans = transm * (1.0f - nd.rspec);
		nd.tspec = nd.trans * tspecu;
		nd.tdiff = nd.trans - nd.tspec;
		if (nd.tspec > FTINY) {
			nd.specfl |= SP_TRAN;

							/* check threshold */
			if (!(nd.specfl & SP_PURE) && specthresh >= nd.tspec - FTINY)
				nd.specfl |= SP_TBLT;
			if (prd.ambient_depth || !hastexture) {
				transtest = 2.0f;
			} else {
				if (dot(nd.prdir - pert, nd.normal) < -FTINY)
					nd.prdir = normalize(nd.prdir - pert);	/* OK */
			}
		}
	}

	/* transmitted ray */
	if ((nd.specfl&(SP_TRAN | SP_PURE | SP_TBLT)) == (SP_TRAN | SP_PURE)) {
		new_prd.weight = prd.weight * fmaxf(nd.mcolor) * nd.tspec;
		if (new_prd.weight >= minweight) {
			new_prd.depth = prd.depth;
			new_prd.ambient_depth = prd.ambient_depth;
			new_prd.state = prd.state;
#ifdef DAYSIM_COMPATIBLE
			new_prd.dc = daysimNext(prd.dc);
#endif
			setupPayload(new_prd, 0);
			Ray trans_ray = make_Ray(nd.hit, nd.prdir, radiance_ray_type, ray_start(nd.hit, nd.prdir, nd.normal, RAY_START), RAY_END);
			rtTrace(top_object, trans_ray, new_prd);
			float3 rcol = new_prd.result * nd.mcolor * nd.tspec;
			result += rcol;
#ifdef DAYSIM_COMPATIBLE
			daysimAddScaled(prd.dc, new_prd.dc, nd.mcolor.x * nd.tspec);
#endif
			transtest *= bright(rcol);
			transdist = t_hit + new_prd.distance;
			resolvePayload(prd, new_prd);
		}
	}
	else
		transtest = 0.0f;
#endif

	// return if it's a shadow ray, which it isn't

	/* get specular reflection */
	if (nd.rspec > FTINY) {
		nd.specfl |= SP_REFL;

		/* compute specular color */
		if (type != metal) {
			nd.scolor = make_float3(nd.rspec);
		} else {
			if (fest > FTINY) {
				float d = spec * (1.0f - fest);
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
	float mirtest = 0.0f, mirdist = t_hit;
	if ((nd.specfl&(SP_REFL | SP_PURE | SP_RBLT)) == (SP_REFL | SP_PURE)) {
		new_prd.weight = prd.weight * fmaxf(nd.scolor);
		new_prd.depth = prd.depth + 1;
		if (new_prd.weight >= minweight && new_prd.depth <= abs(maxdepth)) {
			new_prd.ambient_depth = prd.ambient_depth;
			new_prd.state = prd.state;
#ifdef DAYSIM_COMPATIBLE
			new_prd.dc = daysimNext(prd.dc);
#endif
			setupPayload(new_prd, 0);
			float3 vrefl = reflect(ray.direction, nd.pnorm);
			Ray refl_ray = make_Ray(nd.hit, vrefl, radiance_ray_type, ray_start(nd.hit, vrefl, nd.normal, RAY_START), RAY_END);
			rtTrace(top_object, refl_ray, new_prd);
			float3 rcol = new_prd.result * nd.scolor;
			result += rcol;
#ifdef DAYSIM_COMPATIBLE
			daysimAddScaled(prd.dc, new_prd.dc, nd.scolor.x);
#endif
			if (nd.specfl & SP_FLAT && (prd.ambient_depth || !hastexture)) {
				mirtest = 2.0f * bright(rcol);
				mirdist = t_hit + new_prd.distance;
			}
			resolvePayload(prd, new_prd);
		}
	}

	/* diffuse reflection */
	nd.rdiff = 1.0f - nd.trans - nd.rspec;

	if (!(nd.specfl & SP_PURE && nd.rdiff <= FTINY && nd.tdiff <= FTINY)) { /* not 100% pure specular */

		/* checks *BLT flags */
		if (!(nd.specfl & SP_PURE))
			result += gaussamp(&nd);

#ifdef AMBIENT
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
#endif /* AMBIENT */

		/* add direct component */
		// This is the call to direct() in source.c
		// Let's start at line 447, and not bother with sorting for now

		// compute direct lighting
		PerRayData_shadow shadow_prd;
#ifdef DAYSIM_COMPATIBLE
		shadow_prd.dc = daysimNext(prd.dc);
#endif
		Ray shadow_ray = make_Ray(nd.hit, nd.pnorm, shadow_ray_type, RAY_START, RAY_END);

		/* contributions from distant lights (mainly the sun) */
		unsigned int num_lights = lights.size();
		for (unsigned int i = 0u; i < num_lights; i++) {
			const DistantLight light = lights[i];
			if ( light.casts_shadow ) {
				shadow_prd.target = i;
				shadow_ray.direction = normalize( light.pos );
				shadow_ray.tmin = ray_start(nd.hit, shadow_ray.direction, nd.normal, RAY_START);
				shadow_ray.tmax = RAY_END;
				result += dirnorm(&shadow_ray, &shadow_prd, &nd, light.solid_angle);
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

			const unsigned int divs = flatpart( rdir, r0, r1, r2 ); //TODO divisions should be smaller closer to the light source
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
						result += dirnorm(&shadow_ray, &shadow_prd, &nd, omega);
					}
				}
		}
#endif /* LIGHTS */
	}

	/* check distance */
	float d = bright( result );
#ifdef TRANSMISSION
	if (transtest > d)
		prd.distance = transdist;
	else
#endif
	if (mirtest > d)
		prd.distance = mirdist;
	else
		prd.distance = t_hit;

	// pass the color back up the tree
	prd.result = result;

#ifdef HIT_TYPE
	prd.hit_type = type;
#endif
}

/* compute source contribution */
RT_METHOD float3 dirnorm(Ray *shadow_ray, PerRayData_shadow *shadow_prd, const NORMDAT *nd, const float& omega)
{
	float3 cval = make_float3( 0.0f );
	float ldot = dot(nd->pnorm, shadow_ray->direction);

#ifdef TRANSMISSION
	if (ldot < 0.0f ? nd->trans <= FTINY : nd->trans >= 1.0f - FTINY)
#else
	if ( ldot <= FTINY )
#endif
		return cval;

	// cast shadow ray
	shadow_prd->result = make_float3( 0.0f );
#ifdef DAYSIM_COMPATIBLE
	daysimSet(shadow_prd->dc, 0.0f);
#endif
	rtTrace( top_object, *shadow_ray, *shadow_prd );
	if( fmaxf( shadow_prd->result ) <= 0.0f )
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
		float3 vtmp = shadow_ray->direction - ray.direction;
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
#ifdef DAYSIM_COMPATIBLE
	daysimAddScaled(prd.dc, shadow_prd->dc, cval.x);
#endif
	return cval * shadow_prd->result;
}

// sample Gaussian specular
RT_METHOD float3 gaussamp(const NORMDAT *nd)
{
	float3 rcol = make_float3( 0.0f );

	/* This section is based on the gaussamp method in normal.c */
	if ((nd->specfl & (SP_REFL | SP_RBLT)) != SP_REFL && (nd->specfl & (SP_TRAN | SP_TBLT)) != SP_TRAN)
		return rcol;

	PerRayData_radiance gaus_prd;
	gaus_prd.depth = prd.depth + 1;
	if ( gaus_prd.depth > abs(maxdepth) )
		return rcol;
	gaus_prd.ambient_depth = prd.ambient_depth + 1; //TODO the increment is a hack to prevent the sun from affecting specular values
	//gaus_prd.seed = prd.seed;//lcg( prd.seed );
	gaus_prd.state = prd.state;
	Ray gaus_ray = make_Ray(nd->hit, nd->pnorm, radiance_ray_type, RAY_START, RAY_END);

	float d;

	/* set up sample coordinates */
	float3 u = getperpendicular(nd->pnorm); // prd.state?
	float3 v = cross(nd->pnorm, u);

	unsigned int nstarget, nstaken, ntrials;

	/* compute reflection */
	gaus_prd.weight = prd.weight * fmaxf(nd->scolor);
	if ((nd->specfl & (SP_REFL | SP_RBLT)) == SP_REFL && gaus_prd.weight >= minweight) {
		float3 scolor = nd->scolor;
		nstarget = 1;
		if (specjitter > 1.5f) {	/* multiple samples? */ // By default it's 1.0
			nstarget = specjitter * prd.weight + 0.5f;
			if ( gaus_prd.weight <= minweight * nstarget )
				nstarget = gaus_prd.weight / minweight;
			if ( nstarget > 1 ) {
				d = 1.0f / nstarget;
				scolor *= d; //scolor, stored as ray rcoef
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
			d = -2.0f * dot( h, ray.direction ) / ( 1.0f + d*d );
			gaus_ray.direction = ray.direction + h * d;

			/* sample rejection test */
			if ((d = dot(gaus_ray.direction, nd->normal)) <= FTINY)
				continue;

			gaus_ray.direction = normalize( gaus_ray.direction );
			gaus_ray.tmin = ray_start(nd->hit, gaus_ray.direction, nd->normal, RAY_START);
			setupPayload(gaus_prd, 0);
			//if (nstaken) // check for prd data that needs to be cleared
			rtTrace(top_object, gaus_ray, gaus_prd);
			resolvePayload(prd, gaus_prd);

			/* W-G-M-D adjustment */
			if (nstarget > 1) {	
				d = 2.0f / (1.0f - dot(ray.direction, nd->normal) / d);
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
	gaus_prd.weight = prd.weight * fmaxf(mcolor);
	gaus_prd.ambient_depth = prd.ambient_depth;
	if ((nd->specfl & (SP_TRAN | SP_TBLT)) == SP_TRAN && gaus_prd.weight >= minweight) {
		nstarget = 1;
		if (specjitter > 1.5f) {	/* multiple samples? */ // By default it's 1.0
			nstarget = specjitter * prd.weight + 0.5f;
			if ( gaus_prd.weight <= minweight * nstarget )
				nstarget = gaus_prd.weight / minweight;
			if ( nstarget > 1 ) {
				d = 1.0f / nstarget;
				mcolor *= d; //mcolor, stored as ray rcoef
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
#ifdef DAYSIM_COMPATIBLE
			gaus_prd.dc = daysimNext(prd.dc);
#endif
			setupPayload(gaus_prd, 0);
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

#ifdef ITERATIVE
	if (!prd.ambient_depth)
		return make_float3(0.0f);
#else /* ITERATIVE */
	if (ambacc > FTINY && navsum != 0) {			/* ambient storage */
		//if (tracktime)				/* sort to minimize thrashing */
		//	sortambvals(0);

		/* interpolate ambient value */
		//acol = make_float3( 0.0f );
		//d = sumambient(acol, r, normal, rdepth, &atrunk, thescene.cuorg, thescene.cusize);
		PerRayData_ambient ambient_prd;
		ambient_prd.result = make_float3( 0.0f );
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
		const float tmin = ray_start( hit, AMBIENT_RAY_LENGTH );
		Ray ambient_ray = make_Ray( hit, normal, ambient_ray_type, -tmin, tmin );
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
#endif /* ITERATIVE */
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
	} else {
		d = expf( avsum / (float)navsum );
#ifdef DAYSIM_COMPATIBLE
		daysimAdd(prd.dc, aval.x * d);
#endif
	}
	return aval * d;
}

#ifndef OLDAMB
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
#ifdef ITERATIVE
	int i, n = 1;
#else /* ITERATIVE */
	int n = sqrtf(ambdiv_final * wt) + 0.5f;
	int i = 1 + 8 * (ambacc > FTINY);	/* minimum number of samples */
	if (n < i)
		n = i;
#endif /* ITERATIVE */
	const int nn = n * n;
	float3 acol = make_float3( 0.0f );
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
		d = fmaxf( acoef );
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

	Ray amb_ray = make_Ray( hit, pnormal, radiance_ray_type, RAY_START, RAY_END ); // Use normal point as temporary direction
	/* End ambsample setup */

					/* make tangent plane axes */
	float3 ux = getperpendicular( pnormal, prd.state );
	float3 uy = cross( pnormal, ux );
					/* sample divisions */
	for (i = n; i--; )
	    for (int j = n; j--; ) {
			//hp.sampOK += ambsample( &hp, i, j, normal, hit );
			/* ambsample in ambcomp.c */
#ifdef ITERATIVE
			float2 spt = 0.01f + 0.98f * make_float2(curand_uniform(prd.state), curand_uniform(prd.state));
#else /* ITERATIVE */
			float2 spt = 0.1f + 0.8f * make_float2( curand_uniform( prd.state ), curand_uniform( prd.state ) );
#endif /* ITERATIVE */
			SDsquare2disk( spt, (j+spt.y) / n, (i+spt.x) / n );
			float zd = sqrtf( 1.0f - dot( spt, spt ) );
			amb_ray.direction = normalize( spt.x*ux + spt.y*uy + zd*pnormal );
			if (dot(amb_ray.direction, normal) <= 0) /* Prevent light leaks */
				continue;
			amb_ray.tmin = ray_start( hit, amb_ray.direction, normal, RAY_START );
			//dimlist[ndims++] = AI(hp,i,j) + 90171;

			setupPayload(new_prd, 0);
			//Ray amb_ray = make_Ray( hit, rdir, radiance_ray_type, RAY_START, RAY_END );
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
#else /* OLDAMB */
#ifdef DAYSIM_COMPATIBLE
RT_METHOD float doambient( float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit, DaysimCoef dc )
#else
RT_METHOD float doambient( float3 *rcol, const float3& normal, const float3& pnormal, const float3& hit )
#endif
{
	float  b;//, d;
	AMBHEMI  hemi;
	AMBSAMP  *div;
	AMBSAMP  dnew;
	float3  acol;
	AMBSAMP  *dp;
	float  arad;
	int  divcnt;
	int  i, j;
					/* initialize hemisphere */
	inithemi(&hemi, *rcol, pnormal);
	divcnt = hemi.nt * hemi.np;
					/* initialize */
	//if (pg != NULL)
	//	pg[0] = pg[1] = pg[2] = 0.0;
	//if (dg != NULL)
	//	dg[0] = dg[1] = dg[2] = 0.0;
	*rcol = make_float3( 0.0f );
	if (divcnt == 0)
		return(0.0f); //TODO does this change the value of rcol in the calling method?
					/* allocate super-samples */
	//if (hemi.ns > 0) {// || pg != NULL || dg != NULL) {
	//	div = (AMBSAMP *)malloc(divcnt*sizeof(AMBSAMP));
	//	//if (div == NULL) // This is 0
	//	//	error(SYSTEM, "out of memory in doambient");
	//} else
		div = NULL; // This is 0
					/* sample the divisions */
	arad = 0.0f;
	acol = make_float3( 0.0f );
	if ((dp = div) == NULL)
		dp = &dnew;
	divcnt = 0;
	for (i = 0; i < hemi.nt; i++)
		for (j = 0; j < hemi.np; j++) {
			dp->t = i; dp->p = j;
			dp->v = make_float3( 0.0f );
			dp->r = 0.0f;
			dp->n = 0;
#ifdef DAYSIM_COMPATIBLE
			if (divsample( dp, &hemi, normal, hit, dc ) < 0) {
#else
			if (divsample( dp, &hemi, normal, hit ) < 0) {
#endif
				if (div != NULL)
					dp++;
				continue;
			}
			arad += dp->r;
			divcnt++;
			if (div != NULL)
				dp++;
			else
				acol += dp->v;
		}
	if (!divcnt) {
		//if (div != NULL)
		//	free((void *)div);
		return(0.0f);		/* no samples taken */
	}
	if (divcnt < hemi.nt*hemi.np) {
		//pg = dg = NULL;		/* incomplete sampling */
		hemi.ns = 0;
	} else if (arad > FTINY && divcnt/arad < minarad) {
		hemi.ns = 0;		/* close enough */
	} else if (hemi.ns > 0) {	/* else perform super-sampling? */
		//comperrs(div, &hemi);			/* compute errors */
		//qsort(div, divcnt, sizeof(AMBSAMP), ambcmp);	/* sort divs */ TODO necessary?
						/* super-sample */
		for (i = hemi.ns; i > 0; i--) {
			dnew = *div;
#ifdef DAYSIM_COMPATIBLE
			if (divsample( &dnew, &hemi, normal, hit, dc ) < 0) {
#else
			if (divsample( &dnew, &hemi, normal, hit ) < 0) {
#endif
				dp++;
				continue;
			}
			dp = div;		/* reinsert */
			j = divcnt < i ? divcnt : i;
			while (--j > 0 && dnew.k < dp[1].k) {
				*dp = *(dp+1);
				dp++;
			}
			*dp = dnew;
		}
		//if (pg != NULL || dg != NULL)	/* restore order */
		//	qsort(div, divcnt, sizeof(AMBSAMP), ambnorm);
	}
					/* compute returned values */
	if (div != NULL) {
		arad = 0.0f;		/* note: divcnt may be < nt*np */
		for (i = hemi.nt*hemi.np, dp = div; i-- > 0; dp++) {
			arad += dp->r;
			if (dp->n > 1) {
				b = 1.0f/dp->n;
				dp->v *= b;
				dp->r *= b;
				dp->n = 1;
			}
			acol += dp->v;
		}
		//b = bright(acol);
		//if (b > FTINY) {
		//	b = 1.0f/b;	/* compute & normalize gradient(s) */
		//	//if (pg != NULL) {
		//	//	posgradient(pg, div, &hemi);
		//	//	for (i = 0; i < 3; i++)
		//	//		pg[i] *= b;
		//	//}
		//	//if (dg != NULL) {
		//	//	dirgradient(dg, div, &hemi);
		//	//	for (i = 0; i < 3; i++)
		//	//		dg[i] *= b;
		//	//}
		//}
		//free((void *)div);
	}
	*rcol = acol;
	if (arad <= FTINY)
		arad = maxarad;
	else
		arad = (divcnt+hemi.ns)/arad;
	//if (pg != NULL) {		/* reduce radius if gradient large */
	//	d = DOT(pg,pg);
	//	if (d*arad*arad > 1.0f)
	//		arad = 1.0f/sqrtf(d);
	//}
	if (arad < minarad) {
		arad = minarad;
		//if (pg != NULL && d*arad*arad > 1.0f) {	/* cap gradient */
		//	d = 1.0f/arad/sqrtf(d);
		//	for (i = 0; i < 3; i++)
		//		pg[i] *= d;
		//}
	}
	if ((arad /= sqrtf(prd.weight)) > maxarad)
		arad = maxarad;
	return(arad);
}

/* initialize sampling hemisphere */
RT_METHOD void inithemi( AMBHEMI  *hp, float3 ac, const float3& normal )
{
	float	d;
	int  i;
	float wt = prd.weight;
					/* set number of divisions */
	if (ambacc <= FTINY && wt > (d = 0.8f * fmaxf(ac) * wt / (ambdiv_final * minweight)))
		wt = d;			/* avoid ray termination */
	hp->nt = sqrtf(ambdiv_final * wt * M_1_PIf) + 0.5f;
	i = ambacc > FTINY ? 3 : 1;	/* minimum number of samples */
	if (hp->nt < i)
		hp->nt = i;
	hp->np = M_PIf * hp->nt + 0.5f;
					/* set number of super-samples */
	hp->ns = ambssamp * wt + 0.5f;
					/* assign coefficient */
	hp->acoef = ac;
	d = 1.0f/(hp->nt*hp->np);
	hp->acoef *= d;
					/* make axes */
	hp->uz = normal;
	hp->ux = getperpendicular(hp->uz);
	hp->uy = cross(hp->uz, hp->ux);
}

/* sample a division */
#ifdef DAYSIM_COMPATIBLE
RT_METHOD int divsample( AMBSAMP  *dp, AMBHEMI  *h, const float3& normal, const float3& hit, DaysimCoef dc )
#else
RT_METHOD int divsample( AMBSAMP  *dp, AMBHEMI  *h, const float3& normal, const float3& hit )
#endif
{
	PerRayData_radiance new_prd;
	//RAY  ar;
	//float3 rcoef; /* contribution coefficient w.r.t. parent */
	//int3  hlist;
	float2  spt;
	float  xd, yd, zd;
	float  b2;
	float  phi;
					/* ambient coefficient for weight */
	if (ambacc > FTINY)
		b2 = AVGREFL; // Reusing this variable
	else
		b2 = fmaxf(h->acoef);
	new_prd.weight = prd.weight * b2;
	if (new_prd.weight < minweight) //if (rayorigin(&ar, AMBIENT, r, ar.rcoef) < 0)
		return(-1);
	//if (ambacc > FTINY) {
	//	rcoef *= h->acoef;
	//	rcoef *= 1.0f / AVGREFL; // This all seems unnecessary
	//}
	//hlist = make_int3( prd.seed, dp->t, dp->p );
	//multisamp(spt, 2, urand(ilhash(hlist,3)+dp->n));//TODO implement
	//spt = multisamp2( frandom() );
	//int il = ilhash( hlist );
	//spt = make_float2( rnd( il ) );
	//spt = make_float2( rnd( prd.seed ) );
	spt = make_float2( curand_uniform( prd.state ), curand_uniform( prd.state ) );
	zd = sqrtf((dp->t + spt.x)/h->nt);
	phi = 2.0f*M_PIf * (dp->p + spt.y)/h->np;
	xd = cosf(phi) * zd;
	yd = sinf(phi) * zd;
	zd = sqrtf(1.0f - zd*zd);
	float3 rdir = normalize( xd*h->ux + yd*h->uy + zd*h->uz );
	//dimlist[ndims++] = dp->t*h->np + dp->p + 90171;

	new_prd.depth = prd.depth + 1;
	new_prd.ambient_depth = prd.ambient_depth + 1;
	//new_prd.seed = prd.seed;//lcg( prd.seed );
	new_prd.state = prd.state;
#ifdef DAYSIM_COMPATIBLE
	new_prd.dc = daysimNext(dc);
#endif
	setupPayload(new_prd, 0);
	Ray amb_ray = make_Ray( hit, rdir, radiance_ray_type, ray_start( hit, rdir, normal, RAY_START ), RAY_END );
	rtTrace(top_object, amb_ray, new_prd);
	resolvePayload(prd, new_prd);

	//ndims--;
	if ( isnan( new_prd.result ) ) // TODO How does this happen?
		return(-1);
	new_prd.result *= h->acoef;	/* apply coefficient */
#ifdef DAYSIM_COMPATIBLE
	daysimAddScaled(dc, new_prd.dc, h->acoef.x);
#endif
	dp->v += new_prd.result;
					/* use rt to improve gradient calc */
	if (new_prd.distance > FTINY && new_prd.distance < RAY_END)
		dp->r += 1.0f/new_prd.distance;

					/* (re)initialize error */
	if (dp->n++) {
		b2 = bright(dp->v)/dp->n - bright(new_prd.result);
		b2 = b2*b2 + dp->k*((dp->n-1)*(dp->n-1));
		dp->k = b2/(dp->n*dp->n);
	} else
		dp->k = 0.0f;
	return(0);
}

/* compute initial error estimates */
//RT_METHOD void comperrs( AMBSAMP *da, AMBHEMI *hp )
//{
//	float  b, b2;
//	int  i, j;
//	AMBSAMP  *dp;
//				/* sum differences from neighbors */
//	dp = da;
//	for (i = 0; i < hp->nt; i++)
//		for (j = 0; j < hp->np; j++) {
////#ifdef  DEBUG
////			if (dp->t != i || dp->p != j)
////				error(CONSISTENCY,
////					"division order in comperrs");
////#endif
//			b = bright(dp[0].v);
//			if (i > 0) {		/* from above */
//				b2 = bright(dp[-hp->np].v) - b;
//				b2 *= b2 * 0.25f;
//				dp[0].k += b2;
//				dp[-hp->np].k += b2;
//			}
//			if (j > 0) {		/* from behind */
//				b2 = bright(dp[-1].v) - b;
//				b2 *= b2 * 0.25f;
//				dp[0].k += b2;
//				dp[-1].k += b2;
//			} else {		/* around */
//				b2 = bright(dp[hp->np-1].v) - b;
//				b2 *= b2 * 0.25f;
//				dp[0].k += b2;
//				dp[hp->np-1].k += b2;
//			}
//			dp++;
//		}
//				/* divide by number of neighbors */
//	dp = da;
//	for (j = 0; j < hp->np; j++)		/* top row */
//		(dp++)->k *= 1.0f/3.0f;
//	if (hp->nt < 2)
//		return;
//	for (i = 1; i < hp->nt-1; i++)		/* central region */
//		for (j = 0; j < hp->np; j++)
//			(dp++)->k *= 0.25f;
//	for (j = 0; j < hp->np; j++)		/* bottom row */
//		(dp++)->k *= 1.0f/3.0f;
//}

/* decreasing order */
//RT_METHOD int ambcmp( const void *p1, const void *p2 )
//{
//	const AMBSAMP	*d1 = (const AMBSAMP *)p1;
//	const AMBSAMP	*d2 = (const AMBSAMP *)p2;
//
//	if (d1->k < d2->k)
//		return(1);
//	if (d1->k > d2->k)
//		return(-1);
//	return(0);
//}
#endif /* OLDAMB */
#endif /* AMBIENT */

#ifdef LIGHTS
/* partition a flat source */
RT_METHOD unsigned int flatpart( const float3& v, const float3& r0, const float3& r1, const float3& r2 )
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
	d = srcsizerat / prd.weight;
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
