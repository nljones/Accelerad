/*
 * Copyright (c) 2013-2014 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_common.h"

using namespace optix;

#define  AMBIENT
//#define  TRANSMISSION
//#define  FILL_GAPS_LAST_ONLY

#ifndef  MAXITER
#define  MAXITER	10		/* maximum # specular ray attempts */
#endif
#define  FRESNE(ci)	(expf(-5.85f*(ci)) - 0.00287989916f) /* estimate of Fresnel function */
#define  FRESTHRESH	0.017999f	/* minimum specularity for approx. */
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

/* Context variables */
rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type, , );
rtDeclareVariable(unsigned int, ambient_ray_type, , );
rtDeclareVariable(rtObject,     top_object, , );
rtDeclareVariable(rtObject,     top_shadower, , );
rtDeclareVariable(rtObject,     top_ambient, , );

rtDeclareVariable(float3,       CIE_rgbf, , ); /* This is the value [ CIE_rf, CIE_gf, CIE_bf ] from color.h */

rtDeclareVariable(float,        specthresh, , ); /* This is the minimum fraction of reflection or transmission, under which no specular sampling is performed */
rtDeclareVariable(float,        specjitter, , );

#ifdef AMBIENT
rtDeclareVariable(float3,       ambval, , ); /* This is the final value used in place of an indirect light calculation */
rtDeclareVariable(int,          ambvwt, , ); /* As new indirect irradiances are computed, they will modify the default ambient value in a moving average, with the specified weight assigned to the initial value given on the command and all other weights set to 1 */
rtDeclareVariable(int,          ambounce, , ); /* Ambient bounces (ab) */
//rtDeclareVariable(int,          ambres, , ); /* Ambient resolution (ar) */
rtDeclareVariable(float,        ambacc, , ); /* Ambient accuracy (aa). This value will approximately equal the error from indirect illuminance interpolation */
rtDeclareVariable(int,          ambdiv, , ); /* Ambient divisions (ad) */
rtDeclareVariable(int,          ambssamp, , ); /* Ambient super-samples (as) */
rtDeclareVariable(float,        maxarad, , ); /* maximum ambient radius */
rtDeclareVariable(float,        minarad, , ); /* minimum ambient radius */
rtDeclareVariable(float,        avsum, , ); /* computed ambient value sum (log) */
rtDeclareVariable(unsigned int, navsum, , ); /* number of values in avsum */
#endif /* AMBIENT */

rtDeclareVariable(float,        minweight, , ); /* minimum ray weight (lw) */
rtDeclareVariable(int,          maxdepth, , ); /* maximum recursion depth (lr) */

//#ifdef FILL_GAPS_LAST_ONLY
//rtDeclareVariable(unsigned int, level, , ) = 0u;
//#endif

rtBuffer<DistantLight> lights;

/* Program variables */
rtDeclareVariable(unsigned int, metal, , ); /* The material type representing "metal" */

/* Material variables */
rtDeclareVariable(unsigned int, type, , ); /* The material type representing "plastic", "metal", or "trans" */
rtDeclareVariable(float3,       color, , ); /* The material color given by the rad file "plastic", "metal", or "trans" object */
rtDeclareVariable(float,        spec, , ); /* The material specularity given by the rad file "plastic", "metal", or "trans" object */
rtDeclareVariable(float,        rough, , ); /* The material roughness given by the rad file "plastic", "metal", or "trans" object */
rtDeclareVariable(float,        transm, , ) = 0.0f; /* The material transmissivity given by the rad file "trans" object */
rtDeclareVariable(float,        tspecu, , ) = 0.0f; /* The material transmitted specular component given by the rad file "trans" object */

/* OptiX variables */
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow, rtPayload, );

/* Attributes */
//rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );


static __device__ float3 gaussamp( const unsigned int& specfl, float3 scolor, float3 mcolor, const float3& normal, const float3& hit, const float& alpha2, const float& tspec );
#ifdef AMBIENT
static __device__ float3 multambient( float3 aval, const float3& normal, const float3& hit );
#ifndef OLDAMB
static __device__ int doambient( float3 *rcol, const float3& normal, const float3& hit );
//static __device__ __inline__ int ambsample( AMBHEMI *hp, const int& i, const int& j, const float3 normal, const float3 hit );
#else /* OLDAMB */
static __device__ float doambient( float3 *rcol, const float3& normal, const float3& hit );
static __device__ __inline__ void inithemi( AMBHEMI  *hp, float3 ac, const float3& normal );
static __device__ int divsample( AMBSAMP  *dp, AMBHEMI  *h, const float3& hit );
//static __device__ void comperrs( AMBSAMP *da, AMBHEMI *hp );
//static __device__ int ambcmp( const void *p1, const void *p2 );
#endif /* OLDAMB */
#endif /* AMBIENT */
//static __device__ float2 multisamp2(float r);
//static __device__ __inline__ int ilhash(int3 d);
static __device__ __inline__ float bright( const float3 &rgb );


RT_PROGRAM void any_hit_shadow()
{
	// this material is opaque, so it fully attenuates all shadow rays
	prd_shadow.result = make_float3(0.0f);
	rtTerminateRay();
}


RT_PROGRAM void closest_hit_radiance()
{
	/* easy shadow test */
	// if this is a shadow ray and not a trans material, return

	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

	/* check for back side */
	// if backvis is false, create a new ray starting from the hit point (i.e., ignore this hit)
	float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	PerRayData_radiance new_prd;
	float3 result = make_float3( 0.0f );
	float3 hit_point = ray.origin + t_hit * ray.direction;
	float3 mcolor = color;
	float3 scolor = make_float3( 0.0f );
	float rspec = spec;
	float alpha2 = rough * rough;
	unsigned int specfl = 0u; /* specularity flags */

	/* get modifiers */
	// we'll skip this for now

	/* get roughness */
	if (alpha2 <= FTINY) {
		specfl |= SP_PURE; // label this as a purely specular reflection
	}

	/* perturb normal */
	// if there's a bump map, we use that, else
	float pdot = -dot( ray.direction, ffnormal );

	// if it's a face or a ring (which it is currently) label as flat
	specfl |= SP_FLAT;

	/* modify material color */
	//mcolor *= rtTex3D(rtTextureId id, texcoord.x, texcoord.y, texcoord.z).xyz;

	/* compute Fresnel approx. */
	float fest = 0.0f;
	if (specfl & SP_PURE && rspec >= FRESTHRESH) {
		fest = FRESNE( pdot );
		rspec += fest * ( 1.0f - rspec );
	}

	/* compute transmission */
	float tdiff = 0.0f, tspec = 0.0f, trans = 0.0f; // because it's opaque
#ifdef TRANSMISSION
	float transtest = 0.0f, transdist = 0.0f;
	if (transm > 0.0f) { // type == MAT_TRANS
		trans = transm * (1.0f - rspec);
		tspec = trans * tspecu;
		tdiff = trans - tspec;
		if (tspec > FTINY) {
			specfl |= SP_TRAN;
							/* check threshold */
			if (!(specfl & SP_PURE) && specthresh >= tspec - FTINY)
				specfl |= SP_TBLT;
			//if (!hastexture || r->crtype & SHADOW) {
			//	VCOPY(nd.prdir, r->rdir);
			//	transtest = 2;
			//} else {
			//	for (i = 0; i < 3; i++)		/* perturb */
			//		nd.prdir[i] = r->rdir[i] - r->pert[i];
			//	if (DOT(nd.prdir, r->ron) < -FTINY)
			//		nd.prdir = normalize(nd.prdir);	/* OK */
			//	else
			//		VCOPY(nd.prdir, r->rdir);
			//}
		}
	}

	/* transmitted ray */
	if ((specfl&(SP_TRAN|SP_PURE|SP_TBLT)) == (SP_TRAN|SP_PURE)) {
		new_prd.weight = prd.weight * fmaxf(mcolor) * tspec;
		if (new_prd.weight >= minweight) {
			new_prd.depth = prd.depth;
			new_prd.ambient_depth = prd.ambient_depth;
			new_prd.state = prd.state;
#ifdef FILL_GAPS
			new_prd.primary = 0;
#endif
#ifdef RAY_COUNT
			new_prd.ray_count = 1;
#endif
#ifdef HIT_COUNT
			new_prd.hit_count = 0;
#endif
			float3 R = ray.direction; //TODO may need to perturb
			Ray trans_ray = make_Ray( hit_point, R, radiance_ray_type, RAY_START, RAY_END );
			rtTrace(top_object, trans_ray, new_prd);
			float3 rcol = new_prd.result * mcolor * tspec;
			result += rcol;
			transtest = 2.0f * bright( rcol );
			transdist = t_hit + new_prd.distance;
#ifdef RAY_COUNT
			prd.ray_count += new_prd.ray_count;
#endif
#ifdef HIT_COUNT
			prd.hit_count += new_prd.hit_count;
#endif
		}
	}
#endif

	// return if it's a shadow ray, which it isn't

	/* get specular reflection */
	if (rspec > FTINY) {
		specfl |= SP_REFL;

		/* compute specular color */
		if (type != metal) {
			scolor = make_float3( rspec );
		} else {
			if (fest > FTINY) {
				float d = spec * (1.0f - fest);
				scolor = fest + mcolor * d;
			} else {
				scolor = mcolor * rspec;
			}
		}

		/* check threshold */
		if (!(specfl & SP_PURE) && specthresh >= rspec - FTINY) {
			specfl |= SP_RBLT;
		}
	}

	/* reflected ray */
	float mirtest = 0.0f, mirdist = 0.0f;
	if ((specfl&(SP_REFL|SP_PURE|SP_RBLT)) == (SP_REFL|SP_PURE)) {
		new_prd.weight = prd.weight * fmaxf(scolor);
		new_prd.depth = prd.depth + 1;
		if (new_prd.weight >= minweight && new_prd.depth <= abs(maxdepth)) {
			new_prd.ambient_depth = prd.ambient_depth;
			new_prd.state = prd.state;
#ifdef FILL_GAPS
			new_prd.primary = 0;
#endif
#ifdef RAY_COUNT
			new_prd.ray_count = 1;
#endif
#ifdef HIT_COUNT
			new_prd.hit_count = 0;
#endif
			float3 R = reflect( ray.direction, ffnormal );
			Ray refl_ray = make_Ray( hit_point, R, radiance_ray_type, RAY_START, RAY_END );
			rtTrace(top_object, refl_ray, new_prd);
			float3 rcol = new_prd.result * scolor;
			result += rcol;
			mirtest = 2.0f * bright( rcol );
			mirdist = t_hit + new_prd.distance;
#ifdef RAY_COUNT
			prd.ray_count += new_prd.ray_count;
#endif
#ifdef HIT_COUNT
			prd.hit_count += new_prd.hit_count;
#endif
		}
	}

	/* diffuse reflection */
	float rdiff = 1.0f - trans - rspec;

	if (!(specfl & SP_PURE && rdiff <= FTINY && tdiff <= FTINY)) { /* not 100% pure specular */

		/* checks *BLT flags */
		if ( !(specfl & SP_PURE) )
			result += gaussamp( specfl, scolor, mcolor, ffnormal, hit_point, alpha2, tspec );

#ifdef AMBIENT
		/* ambient from this side */
		if (rdiff > FTINY) {
			float3 aval = mcolor * rdiff;	/* modified by material color */
			if (specfl & SP_RBLT)	/* add in specular as well? */
				aval += scolor;
			result += multambient(aval, ffnormal, hit_point);	/* add to returned color */
		}

#ifdef TRANSMISSION
		/* ambient from other side */
		if (tdiff > FTINY) {
			float3 aval = mcolor;	/* modified by material color */
			if (specfl & SP_TBLT)
				aval *= trans;
			else
				aval *= tdiff;
			result += multambient(aval, -ffnormal, hit_point);	/* add to returned color */
		}
#endif
#endif

		/* add direct component */
		// This is the call to direct() in source.c
		// Let's start at line 447, and not bother with sorting for now

		// compute direct lighting
		unsigned int num_lights = lights.size();
		PerRayData_shadow shadow_prd;
		Ray shadow_ray = make_Ray( hit_point, ffnormal, shadow_ray_type, RAY_START, RAY_END );
		for(int i = 0; i < num_lights; ++i) {
			DistantLight light = lights[i];
			//float Ldist = optix::length(light.pos - hit_point);
			//float3 L = optix::normalize(light.pos - hit_point);
			shadow_ray.direction = normalize(light.pos);
			float ldot = dot( ffnormal, shadow_ray.direction );

			// cast shadow ray
#ifdef TRANSMISSION
			if ( light.casts_shadow ) {
#else
			if ( ldot > 0.0f && light.casts_shadow ) { // assuming it's not a TRANS material
#endif
				shadow_prd.result = make_float3(0.0f);
				rtTrace(top_shadower, shadow_ray, shadow_prd);
				if( fmaxf(shadow_prd.result) > 0.0f ) {

					/* This comes from direct() in normal.c */
					float3 cval = make_float3( 0.0f );

					/* Fresnel estimate */
					float lrdiff = rdiff;
					float ltdiff = tdiff;
					if (specfl & SP_PURE && rspec >= FRESTHRESH && (lrdiff > FTINY) | (ltdiff > FTINY)) {
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
						float dtmp = ldot * light.solid_angle * lrdiff * (1.0f/M_PIf);
						cval += mcolor * dtmp;
					}
					if (ldot > FTINY && (specfl&(SP_REFL|SP_PURE)) == SP_REFL) {
						/*
						 *  Compute specular reflection coefficient using
						 *  Gaussian distribution model.
						 */
						/* roughness */
						float dtmp = alpha2;
						/* + source if flat */
						if (specfl & SP_FLAT)
							dtmp += light.solid_angle * (0.25f/M_PIf);
						/* half vector */
						float3 vtmp = shadow_ray.direction - ray.direction;
						float d2 = dot( vtmp, ffnormal );
						d2 *= d2;
						float d3 = dot( vtmp, vtmp );
						float d4 = (d3 - d2) / d2;
						/* new W-G-M-D model */
						dtmp = expf(-d4/dtmp) * d3 / (M_PIf * d2*d2 * dtmp);
						/* worth using? */
						if (dtmp > FTINY) {
							dtmp *= ldot * light.solid_angle;
							cval += scolor * dtmp;
						}
					}
#ifdef TRANSMISSION
					if (ldot < -FTINY && ltdiff > FTINY) {
						/*
						 *  Compute diffuse transmission.
						 */
						float dtmp = -ldot * light.solid_angle * ltdiff * (1.0f/M_PIf);
						cval += mcolor * dtmp;
					}
					if (ldot < -FTINY && (specfl&(SP_TRAN|SP_PURE)) == SP_TRAN) {
						/*
						 *  Compute specular transmission.  Specular transmission
						 *  is always modified by material color.
						 */
										/* roughness + source */
						float dtmp = alpha2 + light.solid_angle * (1.0f/M_PIf);
										/* Gaussian */
						dtmp = expf( ( 2.0f * dot( ray.direction, shadow_ray.direction ) - 2.0f ) / dtmp ) / ( M_PIf * dtmp ); // may need to perturb direction
										/* worth using? */
						if (dtmp > FTINY) {
							dtmp *= tspec * light.solid_angle * sqrtf( -ldot / pdot );
							cval += mcolor * dtmp;
						}
					}
#endif
					result += cval * shadow_prd.result;
				} /* End direct() in normal.c */
			}
		}

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

// sample Gaussian specular
static __device__ float3 gaussamp( const unsigned int& specfl, float3 scolor, float3 mcolor, const float3& normal, const float3& hit, const float& alpha2, const float& tspec )
{
	float3 rcol = make_float3( 0.0f );

	/* This section is based on the gaussamp method in normal.c */
	if ((specfl & (SP_REFL|SP_RBLT)) != SP_REFL && (specfl & (SP_TRAN|SP_TBLT)) != SP_TRAN)
		return rcol;

	PerRayData_radiance gaus_prd;
	gaus_prd.depth = prd.depth + 1;
	if ( gaus_prd.depth > abs(maxdepth) )
		return rcol;
	gaus_prd.ambient_depth = prd.ambient_depth + 1; //TODO the increment is a hack to prevent the sun from affecting specular values
	//gaus_prd.seed = prd.seed;//lcg( prd.seed );
	gaus_prd.state = prd.state;
	Ray gaus_ray = make_Ray( hit, normal, radiance_ray_type, RAY_START, RAY_END );

	float d;

	/* set up sample coordinates */
	float3 v = cross_direction( normal ); // should be using perturned normal, but currently using ffnormal
	float3 u = normalize( cross( v, normal ) );
	v = normalize( cross( normal, u ) );

	unsigned int nstarget, nstaken, ntrials;

	/* compute reflection */
	gaus_prd.weight = prd.weight * fmaxf(scolor);
	if ( (specfl & (SP_REFL|SP_RBLT)) == SP_REFL && gaus_prd.weight >= minweight ) {
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
				d = sqrtf( alpha2 * -logf( rv.y ) ); // alpha2
			float3 h = normal + d * ( cosp * u + sinp * v ); // normal is perturbed
			d = -2.0f * dot( h, ray.direction ) / ( 1.0f + d*d );
			gaus_ray.direction = ray.direction + h * d;

			/* sample rejection test */
			if ( ( d = dot( gaus_ray.direction, normal ) ) <= FTINY) // this is ron, is this perturbed?
				continue;

			gaus_ray.direction = normalize( gaus_ray.direction );
#ifdef FILL_GAPS
			gaus_prd.primary = 0;
#endif
#ifdef RAY_COUNT
			gaus_prd.ray_count = 1;
#endif
#ifdef HIT_COUNT
			gaus_prd.hit_count = 0;
#endif
			//if (nstaken) // check for prd data that needs to be cleared
			rtTrace(top_object, gaus_ray, gaus_prd);
#ifdef RAY_COUNT
			prd.ray_count += gaus_prd.ray_count;
#endif
#ifdef HIT_COUNT
			prd.hit_count += gaus_prd.hit_count;
#endif

			/* W-G-M-D adjustment */
			if (nstarget > 1) {	
				d = 2.0f / ( 1.0f + dot( ray.direction, normal ) / d );
				scol += gaus_prd.result * d;
			} else {
				rcol += gaus_prd.result * scolor;
			}

			++nstaken;
		}
		/* final W-G-M-D weighting */
		if (nstarget > 1) {
			scol *= scolor;
			d = (float)nstarget / ntrials;
			rcol += scol * d;
		}
		//ndims--;
	}

#ifdef TRANSMISSION
	/* compute transmission */
	mcolor *= tspec;	/* modified by color */
	gaus_prd.weight = prd.weight * fmaxf(mcolor);
	if ( ( specfl & (SP_TRAN|SP_TBLT)) == SP_TRAN && gaus_prd.weight >= minweight ) {
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
				d = sqrtf( alpha2 * -logf( rv.y ) );
			float3 h = ray.direction + d * ( cosp * u + sinp * v ); // ray direction is perturbed
			d = -2.0f * dot( h, ray.direction ) / ( 1.0f + d*d );
			gaus_ray.direction = ray.direction + h * d;

			/* sample rejection test */
			if ( ( d = dot( gaus_ray.direction, normal ) ) >= -FTINY) // this is ron, is this perturbed?
				continue;

			gaus_ray.direction = normalize( gaus_ray.direction );
#ifdef RAY_COUNT
			gaus_prd.ray_count = 1;
#endif
			//if (nstaken) // check for prd data that needs to be cleared
			rtTrace(top_object, gaus_ray, gaus_prd);
#ifdef RAY_COUNT
			prd.ray_count += gaus_prd.ray_count;
#endif
			rcol += gaus_prd.result * mcolor;
			++nstaken;
		}
		//ndims--;
	}
#endif
	//return make_float3(0.0f);
	return rcol;
}

#ifdef AMBIENT
// Compute the ambient component and multiply by the coefficient.
static __device__ float3 multambient( float3 aval, const float3& normal, const float3& hit )
{
	//static int  rdepth = 0;			/* ambient recursion */ //This is part of the ray for parallelism
	float 	d;

	if (ambdiv <= 0)			/* no ambient calculation */
		goto dumbamb;
						/* check number of bounces */
	if (prd.ambient_depth >= ambounce)
		goto dumbamb;
						/* check ambient list */
	//if (ambincl != -1 && r->ro != NULL && ambincl != inset(ambset, r->ro->omod))
	//	goto dumbamb;

	if ( ambacc <= FTINY || navsum == 0 ) {			/* no ambient storage */
		float3 acol = aval;
		//rdepth++;
		d = doambient( &acol, normal, hit );
		//rdepth--;
		if (d > FTINY)
			return acol;
	} else {
		//if (tracktime)				/* sort to minimize thrashing */
		//	sortambvals(0);

		/* interpolate ambient value */
		//acol = make_float3( 0.0f );
		//d = sumambient(acol, r, normal, rdepth, &atrunk, thescene.cuorg, thescene.cusize);
		PerRayData_ambient ambient_prd;
		ambient_prd.result = make_float3( 0.0f );
		ambient_prd.surface_normal = normal;
		ambient_prd.weight = prd.weight;
		ambient_prd.wsum = 0.0f;
		ambient_prd.ambient_depth = prd.ambient_depth;
		ambient_prd.state = prd.state;
#ifdef HIT_COUNT
		ambient_prd.hit_count = 0;
#endif
		Ray ambient_ray = make_Ray( hit, normal, ambient_ray_type, -AMBIENT_RAY_LENGTH, AMBIENT_RAY_LENGTH );
		rtTrace(top_ambient, ambient_ray, ambient_prd);
#ifdef HIT_COUNT
		prd.hit_count += ambient_prd.hit_count;
#endif
		if (ambient_prd.wsum > FTINY) { // TODO if miss program is called, set wsum = 1.0f or place this before ambacc == 0.0f
			ambient_prd.result *= 1.0f / ambient_prd.wsum;
			return aval * ambient_prd.result;
		}
		//rdepth++;				/* need to cache new value */
		//d = makeambient(acol, r, normal, rdepth-1); //TODO implement as miss program for ambient ray
		//rdepth--;
		//if ( dot( ambient_prd.result, ambient_prd.result) > FTINY) { // quick check to see if a value was returned by miss program
		//	return aval * ambient_prd.result;		/* got new value */
		//}

#ifdef FILL_GAPS
		if ( prd.primary ) {
			float3 acol = aval;
			//rdepth++;
			d = doambient( &acol, normal, hit );
			//rdepth--;
			if (d > FTINY)
				return acol;
		}
#elif defined FILL_GAPS_LAST_ONLY
		//if ( prd.ambient_depth == level ) {
		if ( prd.ambient_depth == 0 ) {
			float3 acol = aval;
			//rdepth++;
			d = doambient( &acol, normal, hit );
			//rdepth--;
			if (d > FTINY)
				return acol;
		}
#endif
	}
dumbamb:					/* return global value */
	if ((ambvwt <= 0) || (navsum == 0)) {
		return aval * ambval;
	}
	float l = bright(ambval);			/* average in computations */
	if (l > FTINY) {
		d = (logf(l)*(float)ambvwt + avsum) / (float)(ambvwt + navsum);
		d = expf(d) / l;
		aval *= ambval;	/* apply color of ambval */
	} else {
		d = expf( avsum / (float)navsum );
	}
	return aval * d;
}

#ifndef OLDAMB
/* sample indirect hemisphere, based on samp_hemi in ambcomp.c */
static __device__ int doambient( float3 *rcol, const float3& normal, const float3& hit )
{
	float	d;
	int	j;
	float wt = prd.weight;

					/* set number of divisions */
	if (ambacc <= FTINY && wt > (d = 0.8f * fmaxf(*rcol) * wt / (ambdiv*minweight)))
		wt = d;			/* avoid ray termination */
	int n = sqrtf(ambdiv * wt) + 0.5f;
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

	Ray amb_ray = make_Ray( hit, hit, radiance_ray_type, RAY_START, RAY_END ); // Use hit point as temporary direction
	/* End ambsample setup */

					/* make tangent plane axes */
	float3 uy = make_float3( curand_uniform( prd.state ), curand_uniform( prd.state ), curand_uniform( prd.state ) ) - 0.5f;
	uy = fmaxf( cross_direction( normal ) * 2.0f - 1.0f, uy );
	float3 ux = cross( uy, normal );
	ux = normalize( ux );
	uy = cross( normal, ux );
					/* sample divisions */
	for (i = n; i--; )
	    for (j = n; j--; ) {
			//hp.sampOK += ambsample( &hp, i, j, normal, hit );
			/* ambsample in ambcomp.c */
			float2 spt = 0.1f + 0.8f * make_float2( curand_uniform( prd.state ), curand_uniform( prd.state ) );
			SDsquare2disk( spt, (j+spt.y) / n, (i+spt.x) / n );
			float zd = sqrtf( 1.0f - dot( spt, spt ) );
			amb_ray.direction = normalize( spt.x*ux + spt.y*uy + zd*normal );
			//dimlist[ndims++] = AI(hp,i,j) + 90171;

#ifdef FILL_GAPS
			new_prd.primary = 0;
#endif
#ifdef RAY_COUNT
			new_prd.ray_count = 1;
#endif
			//Ray amb_ray = make_Ray( hit, rdir, radiance_ray_type, RAY_START, RAY_END );
			rtTrace(top_object, amb_ray, new_prd);
#ifdef RAY_COUNT
			prd.ray_count += new_prd.ray_count;
#endif

			//ndims--;
			if ( isnan( new_prd.result ) ) // TODO How does this happen?
				continue;
			if ( new_prd.distance <= FTINY )
				continue;		/* should never happen */
			acol += new_prd.result * acoef;	/* add to our sum */
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
static __device__ float doambient( float3 *rcol, const float3& normal, const float3& hit )
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
	inithemi(&hemi, *rcol, normal);
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
			if (divsample( dp, &hemi, hit ) < 0) {
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
			if (divsample( &dnew, &hemi, hit ) < 0) {
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
	//		arad = 1.0f/sqrt(d);
	//}
	if (arad < minarad) {
		arad = minarad;
		//if (pg != NULL && d*arad*arad > 1.0f) {	/* cap gradient */
		//	d = 1.0f/arad/sqrt(d);
		//	for (i = 0; i < 3; i++)
		//		pg[i] *= d;
		//}
	}
	if ((arad /= sqrt(prd.weight)) > maxarad)
		arad = maxarad;
	return(arad);
}

/* initialize sampling hemisphere */
static __device__ __inline__ void inithemi( AMBHEMI  *hp, float3 ac, const float3& normal )
{
	float	d;
	int  i;
	float wt = prd.weight;
					/* set number of divisions */
	if (ambacc <= FTINY && wt > (d = 0.8f * fmaxf(ac) * wt / (ambdiv*minweight)))
		wt = d;			/* avoid ray termination */
	hp->nt = sqrtf(ambdiv * wt / M_PIf) + 0.5f;
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
	hp->uy = cross_direction( hp->uz );
	hp->ux = normalize( cross(hp->uy, hp->uz) );
	hp->uy = normalize( cross(hp->uz, hp->ux) );
}

/* sample a division */
static __device__ int divsample( AMBSAMP  *dp, AMBHEMI  *h, const float3& hit )
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
#ifdef FILL_GAPS
	new_prd.primary = 0;
#endif
#ifdef RAY_COUNT
	new_prd.ray_count = 1;
#endif
	Ray amb_ray = make_Ray( hit, rdir, radiance_ray_type, RAY_START, RAY_END );
	rtTrace(top_object, amb_ray, new_prd);
#ifdef RAY_COUNT
	prd.ray_count += new_prd.ray_count;
#endif

	//ndims--;
	if ( isnan( new_prd.result ) ) // TODO How does this happen?
		return(-1);
	new_prd.result *= h->acoef;	/* apply coefficient */
	dp->v += new_prd.result;
					/* use rt to improve gradient calc */
	if (new_prd.distance > FTINY && new_prd.distance < RAY_END)
		dp->r += 1.0f/new_prd.distance; //TODO should this be sum of distances?

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
//static __device__ void comperrs( AMBSAMP *da, AMBHEMI *hp )
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
//static __device__ int ambcmp( const void *p1, const void *p2 )
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

//static __device__ float nextssamp( float3 *rdir, const DistantLight* light, const float3 hit,			/* compute sample for source, rtn. distance */
//	RAY  *r,		/* origin is read, direction is set */
//	SRCINDEX  *si		/* source index (modified to current) */\
//)
//{
//	int3  cent, size;
//	int2  parr;
////	SRCREC  *srcp;
//	float3  vpos;
//	float  d;
//	//int  i;
//nextsample:
//	//while (++si->sp >= si->np) {	/* get next sample */
//	//	if (++si->sn >= nsources)
//	//		return(0.0);	/* no more */
//	//	if (srcskip(source+si->sn, r->rorg))
//	//		si->np = 0;
//	//	else if (srcsizerat <= FTINY)
//	//		nopart(si, r);
//	//	else {
//	//		for (i = si->sn; source[i].sflags & SVIRTUAL;
//	//				i = source[i].sa.sv.sn)
//	//			;		/* partition source */
//	//		(*sfun[source[i].so->otype].of->partit)(si, r);
//	//	}
//	//	si->sp = -1;
//	//}
//					/* get partition */
//	cent = make_int3( 0 );
//	size = make_int3( MAXSPART );
//	parr = make_int2( 0, si->sp );
//	//if (!skipparts(cent, size, parr, si->spt)) //TODO implement this
//	//	error(CONSISTENCY, "bad source partition in nextssamp");
//					/* compute sample */
//	//srcp = source + si->sn;
//	if (dstrsrc > FTINY) {			/* jitter sample */
//		//dimlist[ndims] = si->sn + 8831;
//		//dimlist[ndims+1] = si->sp + 3109;
//		//d = urand(ilhash(dimlist,ndims+2)+samplendx);
//		if (srcp->sflags & SFLAT) {
//			//multisamp(vpos, 2, d);
//			vpos = make_float3( curand_uniform( prd.state ), curand_uniform( prd.state ), 0.5f );
//		} else
//			//multisamp(vpos, 3, d);
//			vpos = make_float3( curand_uniform( prd.state ), curand_uniform( prd.state ), curand_uniform( prd.state ) );
//		vpos = dstrsrc * ( 1.0f - 2.0f * vpos ) * (float3)size * ( 1.0f / MAXSPART );
//	} else
//		vpos = make_float3( 0.0f );
//
//	vpos += cent * ( 1.0f / MAXSPART );
//					/* avoid circular aiming failures */
//	if ((srcp->sflags & SCIR) && (si->np > 1 || dstrsrc > 0.7f)) {
//		float3 trim = make_float3( 0.0f );
//		if (srcp->sflags & (SFLAT|SDISTANT)) {
//			d = 1.12837917f;		/* correct setflatss() */
//			trim.x = d * sqrtf( 1.0f - 0.5f * vpos.y*vpos.y );
//			trim.y = d * sqrtf( 1.0f - 0.5f * vpos.x*vpos.x );
//		} else {
//			trim.z = trim.x = vpos.x*vpos.x;
//			d = vpos.y*vpos.y;
//			if (d > trim.z) trim.z = d;
//			trim.x += d;
//			d = vpos.z*vpos.z;
//			if (d > trim.z) trim.z = d;
//			trim.x += d;
//			if (trim.x > FTINY*FTINY) {
//				d = 1.0f / 0.7236f;	/* correct sphsetsrc() */
//				trim = make_float3( d * sqrtf( trim.z / trim.x ) );
//			} else
//				trim = make_float3( 0.0f );
//		}
//		vpos *= trim;
//	}
//					/* compute direction */
//	*rdir = light.pos + //TODO
//	for (i = 0; i < 3; i++)
//		r->rdir[i] = srcp->sloc[i] +
//				vpos[SU]*srcp->ss[SU][i] +
//				vpos[SV]*srcp->ss[SV][i] +
//				vpos[SW]*srcp->ss[SW][i];
//
//	if (!(srcp->sflags & SDISTANT))
//		*rdir -= hit;
//					/* compute distance */
//	if ((d = normalize(r->rdir)) == 0.0)
//		goto nextsample;		/* at source! */
//
//					/* compute sample size */
//	if (srcp->sflags & SFLAT) {
//		si->dom = sflatform(si->sn, r->rdir);
//		si->dom *= size.x * size.y * ( 1.0f / ( MAXSPART * MAXSPART ) );
//	} else if (srcp->sflags & SCYL) {
//		si->dom = scylform(si->sn, r->rdir);
//		si->dom *= size.x * ( 1.0f / MAXSPART );
//	} else {
//		si->dom = size.x * size.y * size.z * ( 1.0f / ( MAXSPART * MAXSPART * MAXSPART ) );
//	}
//	if (srcp->sflags & SDISTANT) {
//		si->dom *= srcp->ss2;
//		return(FHUGE);
//	}
//	if (si->dom <= 1e-4)
//		goto nextsample;		/* behind source? */
//	si->dom *= srcp->ss2/(d*d);
//	return(d);		/* sample OK, return distance */
//}

/* convert 1-dimensional sample to 2 dimensions, based on multisamp.c */
//static __device__ __inline__ float2 multisamp2(float r)	/* 1-dimensional sample [0,1) */
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
//static __device__ __inline__ int ilhash(int3 d)
//{
//	register int  hval;
//
//	hval = 0;
//	hval ^= d.x * 73771;
//	hval ^= d.y * 96289;
//	hval ^= d.z * 103699;
//	return(hval & 0x7fffffff);
//}

static __device__ __inline__ float bright( const float3 &rgb )
{
	return dot( rgb, CIE_rgbf );
}

