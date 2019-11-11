/*
 *  ambient_normal.cu - hit programs for ambient sampling on GPUs.
 */

#include "accelerad_copyright.h"

#include "otypes.h"

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include "optix_shader_ray.h"
#include "optix_shader_ambient.h"
#include "optix_ambient_common.h"

using namespace optix;

#define threadIndex()	(launch_index.y / stride)
//#define threadIndex()	((launch_index.x + launch_dim.x * launch_index.y) / stride)
#define CORRAL
#define hessrow(i)	hess_row_buffer[make_uint2(i, threadIndex())]
#define gradrow(i)	grad_row_buffer[make_uint2(i, threadIndex())]
#ifdef AMB_SAVE_MEM
#define prevrow(i)	amb_samp_buffer[make_uint2(i, threadIndex())]
#define corral_u(i)	corral_u_buffer[make_uint2(i, threadIndex())]
#define corral_d(i)	corral_d_buffer[make_uint2(i, threadIndex())]
#else /* AMB_SAVE_MEM */
#ifdef DAYSIM_COMPATIBLE
#define ambsam(i,j)	amb_samp_buffer[make_uint3(i, j, threadIndex() + segment_offset)]
#else /* DAYSIM_COMPATIBLE */
#define ambsam(i,j)	amb_samp_buffer[make_uint3(i, j, threadIndex())]
#endif /* DAYSIM_COMPATIBLE */
#ifdef AMB_SUPER_SAMPLE
#define earr(i,j)	earr_buffer[make_uint3(i, j, threadIndex())]
#endif
#endif /* AMB_SAVE_MEM */

typedef struct {
	unsigned int	ns;		/* number of samples per axis */
	int	sampOK;		/* acquired full sample set? */
	float3	acoef;		/* division contribution coefficient */
	float3	acol;		/* accumulated color */
	float3	ux, uy;		/* tangent axis unit vectors */
} AMBHEMI;		/* ambient sample hemisphere */

typedef struct {
	float3 r_i, r_i1, e_i, rcp, rI2_eJ2;
	float I1, I2;
} FFTRI;		/* vectors and coefficients for Hessian calculation */

/* Context variables */
rtDeclareVariable(rtObject,     top_object, , );
rtDeclareVariable(unsigned int, stride, , ) = 1u; /* Spacing between used threads in warp. */
#ifdef DAYSIM_COMPATIBLE
rtDeclareVariable(unsigned int, segment_offset, , ) = 0u; /* Offset into data if computed with multiple segments */
#endif /* DAYSIM_COMPATIBLE */

//rtDeclareVariable(float,        specthresh, , ); /* This is the minimum fraction of reflection or transmission, under which no specular sampling is performed */
//rtDeclareVariable(float,        specjitter, , );

//rtDeclareVariable(float3,       ambval, , ); /* This is the final value used in place of an indirect light calculation */
//rtDeclareVariable(int,          ambvwt, , ); /* As new indirect irradiances are computed, they will modify the default ambient value in a moving average, with the specified weight assigned to the initial value given on the command and all other weights set to 1 */
//rtDeclareVariable(int,          ambounce, , ); /* Ambient bounces (ab) */
//rtDeclareVariable(int,          ambres, , ); /* Ambient resolution (ar) */
rtDeclareVariable(float,        ambacc, , ); /* Ambient accuracy (aa). This value will approximately equal the error from indirect illuminance interpolation */
rtDeclareVariable(int,          ambdiv, , ); /* Ambient divisions (ad) */
rtDeclareVariable(int,          ambssamp, , ); /* Ambient super-samples (as) */
rtDeclareVariable(float,        maxarad, , ); /* maximum ambient radius */
rtDeclareVariable(float,        minarad, , ); /* minimum ambient radius */
//rtDeclareVariable(float,        avsum, , ); /* computed ambient value sum (log) */
//rtDeclareVariable(unsigned int, navsum, , ); /* number of values in avsum */

rtBuffer<MaterialData> material_data;	/* One entry per Radiance material. */

/* Program variables */
rtBuffer<optix::Matrix<3, 3>, 2> hess_row_buffer;
rtBuffer<float3, 2>              grad_row_buffer;
#ifdef AMB_SAVE_MEM
rtBuffer<AmbientSample, 2>       amb_samp_buffer;
rtBuffer<float2, 2>              corral_u_buffer;
rtBuffer<float, 2>               corral_d_buffer;
#else /* AMB_SAVE_MEM */
rtBuffer<AmbientSample, 3>       amb_samp_buffer;
#ifdef AMB_SUPER_SAMPLE
rtBuffer<float, 3>               earr_buffer;
#endif
#endif /* AMB_SAVE_MEM */

/* OptiX variables */
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_ambient_record, prd, rtPayload, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );

/* Attributes */
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(int, mat_id, attribute mat_id, );


#ifdef CHECK_OVERLAP
RT_METHOD int check_overlap( const float3& normal, const float3& hit );
RT_METHOD int plugaleak( const AmbientRecord* record, const float3& anorm, const float3& normal, const float3& hit, float ang );
#endif
RT_METHOD int doambient( float3 *rcol, optix::Matrix<2,3> *uv, float2 *ra, float2 *pg, float2 *dg, unsigned int *crlp, const float3& normal, const float3& hit );
RT_METHOD int ambsample(AMBHEMI *hp, AmbientSample *ap, const unsigned int& i, const unsigned int& j, const unsigned int& n, const float3& normal, const float3& hit);
#ifdef AMB_SAVE_MEM
RT_METHOD int samp_hemi(AMBHEMI *hp, float3 *rcol, float wt, optix::Matrix<2, 3> *uv, float2 *ra, float2 *pg, float2 *dg, unsigned int *crlp, const float3& normal, const float3& hit);
#else /* AMB_SAVE_MEM */
#ifdef AMB_SUPER_SAMPLE
RT_METHOD void getambdiffs(AMBHEMI *hp);
RT_METHOD void ambsupersamp(AMBHEMI *hp, int cnt, const float3& normal, const float3& hit);
#endif /* AMB_SUPER_SAMPLE */
RT_METHOD int samp_hemi( AMBHEMI *hp, float3 *rcol, float wt, const float3& normal, const float3& hit );
RT_METHOD void ambHessian( AMBHEMI *hp, optix::Matrix<2,3> *uv, float2 *ra, float2 *pg, const float3& normal, const float3& hit );
RT_METHOD void ambdirgrad( AMBHEMI *hp, const float3& u, const float3& v, float2 *dg, const float3& normal, const float3& hit );
RT_METHOD unsigned int ambcorral( AMBHEMI *hp, optix::Matrix<2,3> *uv, const float2& r, const float3& hit );
#endif /* AMB_SAVE_MEM */
RT_METHOD float back_ambval( const AmbientSample *n1, const AmbientSample *n2, const AmbientSample *n3 );
RT_METHOD void comp_fftri( FFTRI *ftp, const AmbientSample *n0, const AmbientSample *n1, const float3& hit );
RT_METHOD optix::Matrix<3,3> compose_matrix( const float3& va, const float3& vb );
RT_METHOD optix::Matrix<3,3> comp_hessian( FFTRI *ftp, const float3& normal );
RT_METHOD float3 comp_gradient( FFTRI *ftp, const float3& normal );
RT_METHOD optix::Matrix<2,2> eigenvectors( optix::Matrix<2,3> *uv, float2 *ra, optix::Matrix<3,3> *hessian );
//RT_METHOD float2 multisamp2(float r);
//RT_METHOD int ilhash(int3 d);

RT_PROGRAM void any_hit_ambient()
{
	if (mat_id < 0 || mat_id >= material_data.size()) {
		rtIgnoreIntersection();
	}
	else {
		MaterialData mat = material_data[mat_id];

		// Check if an alternate material is used
		while ((mat.type == MAT_CLIP || mat.type == MAT_ILLUM) && mat.proxy > -1)
			mat = material_data[mat.proxy];

		// This only applies to normal materials
		if (mat.type != MAT_PLASTIC && mat.type != MAT_METAL && mat.type != MAT_TRANS)
			rtIgnoreIntersection();

		// Check that this material is included
		if (!mat.params.n.ambincl)
			rtIgnoreIntersection();
	}
}

RT_PROGRAM void closest_hit_ambient()
{
	float3 ffnormal = -ray.direction;
	float3 hit_point = ray.origin + t_hit * ray.direction;

	/* compute weight */
	//float weight = 1.0f;
	//for (int i = prd.ambient_depth; i-- > 0; ) //TODO start at i-1?
	//	weight *= AVGREFL;
	//if (prd.weight < 0.1f * weight)	/* heuristic override */
	//	weight = 1.25f * prd.weight;
	float3 acol = make_float3( AVGREFL );
#ifdef DAYSIM_COMPATIBLE
	daysimSet(prd.dc, 0.0f);
#endif
	optix::Matrix<2,3> uv;
	float2 pg = make_float2( 0.0f );
	float2 dg = make_float2( 0.0f );
	float2 rad = make_float2( 0.0f );
	unsigned int corral = 0u;

	/* compute ambient */
	int i = doambient( &acol, &uv, &rad, &pg, &dg, &corral, ffnormal, hit_point );
	if ( !i || rad.x <= FTINY )	/* no Hessian or zero radius */
		return;

	acol *= 1.0f / AVGREFL;		/* undo assumed reflectance */

	//if (rn != r->ron)
	//	extambient(acol, &amb, r->rop, rn);	/* texture */

	// pass the color back up the tree
	prd.result.pos = hit_point;
	prd.result.val = acol;
	prd.result.gpos = pg;
	prd.result.gdir = dg;
	prd.result.rad = rad;
	prd.result.ndir = encodedir( ffnormal );
	prd.result.udir = encodedir( uv.getRow(0) );
	prd.result.corral = corral;
	//prd.result.lvl = lvl;
	//prd.result.weight = weight;
#ifdef DAYSIM_COMPATIBLE
	daysimScale(prd.dc, 1.0f / AVGREFL); // TODO Scaling should be done before extambient if textured
#endif
}

#ifdef CHECK_OVERLAP /* We don't have need for this currently. */
// based on sumambient from ambient.c
RT_METHOD int check_overlap( const float3& normal, const float3& hit )
{
	// Check that there is a parent
	if ( !prd.parent )
		return( 0 );

	/* Direction test using unperturbed normal */
	float3 w = decodedir( prd.parent->ndir );
	float d = dot( w, normal );
	if ( d <= 0.0f )		/* >= 90 degrees */
		return( 0 );

	float delta_r2 = 2.0f - 2.0f * d;	/* approx. radians^2 */
	const float minangle = 10.0f * M_PIf / 180.0f;
	float maxangle = minangle + ambacc;
					/* adjust maximum angle */
	//if (at->alist != NULL && (at->alist->lvl <= al) & (r->rweight < 0.6))
	//	maxangle = (maxangle - PI/2.)*pow(r->rweight,0.13) + PI/2.;
	if ( delta_r2 >= maxangle * maxangle )
		return( 0 );

	/* Modified ray behind test */
	float3 ck0 = hit - prd.parent->pos;
	d = dot( ck0, w );
	if ( d < -minarad * ambacc - 0.001f )
		return( 0 );
	d /= prd.parent->rad.x;
	float delta_t2 = d * d;
	if ( delta_t2 >= ambacc * ambacc )
		return( 0 );
	
	/* Elliptical radii test based on Hessian */
	float3 u = decodedir( prd.parent->udir );
	float3 v = cross( w, u );
	float uu, vv;
	d = (uu = dot( ck0, u )) / prd.parent->rad.x;
	delta_t2 += d * d;
	d = (vv = dot( ck0, v )) / prd.parent->rad.y;
	delta_t2 += d * d;
	if ( delta_t2 >= ambacc * ambacc )
		return( 0 );
	
	/* Test for potential light leak */
	if ( prd.parent->corral && plugaleak( prd.parent, w, normal, hit, atan2f( vv, uu ) ) )
		return( 0 );
	return( 1 );
}

/* Plug a potential leak where ambient cache value is occluded */
RT_METHOD int plugaleak( const AmbientRecord* record, const float3& anorm, const float3& normal, const float3& hit, float ang )
{
	const float cost70sq = 0.1169778f;	/* cos(70deg)^2 */
	float2 t;

	ang += 2.0f * M_PIf * (ang < 0);			/* check direction flags */
	if ( !(record->corral>>(int)( ang * 16.0f * M_1_PIf ) & 1) )
		return(0);
	/*
	 * Generate test ray, targeting 20 degrees above sample point plane
	 * along surface normal from cache position.  This should be high
	 * enough to miss local geometry we don't really care about.
	 */
	float3 vdif = record->pos - hit;
	float normdot = dot( anorm, normal );
	float ndotd = dot( vdif, normal );
	float nadotd = dot( vdif, anorm );
	float a = normdot * normdot - cost70sq;
	float b = 2.0f * ( normdot * ndotd - nadotd * cost70sq );
	float c = ndotd * ndotd - dot( vdif, vdif ) * cost70sq;
	if ( quadratic( &t, a, b, c ) != 2 )
		return(1);			/* should rarely happen */
	if ( t.y <= FTINY )
		return(0);			/* should fail behind test */

	float3 rdir = vdif + anorm * t.y;	/* further dist. > plane */
	Ray shadow_ray = make_Ray( hit, normalize( rdir ), SHADOW_RAY, RAY_START, length( rdir ) );
	PerRayData_shadow shadow_prd;
	shadow_prd.target = 0;
	shadow_prd.result = make_float3( 1.0f );
#ifdef CONTRIB
	shadow_prd.rcoef = make_contrib3(0.0f);
#endif
#ifdef ANTIMATTER
	shadow_prd.mask = 0u; //TODO check if we are in an antimatter volume
	shadow_prd.inside = 0;
#endif
	rtTrace( top_object, shadow_ray, shadow_prd );
	return( dot( shadow_prd.result, shadow_prd.result ) < 1.0f );	/* check for occluder */
}
#endif /* CHECK_OVERLAP */

RT_METHOD int doambient( float3 *rcol, optix::Matrix<2,3> *uv, float2 *ra, float2 *pg, float2 *dg, unsigned int *crlp, const float3& normal, const float3& hit )
{
	const float wt = prd.result.weight;
	AMBHEMI hp;

	if (!samp_hemi(&hp, rcol, wt,
#ifdef AMB_SAVE_MEM
		uv, ra, pg, dg, crlp,
#endif
		normal, hit))
		return(0);

	/* clear return values */
	//if (u != NULL)
	//	*u = make_float3( 0.0f );
	//if (v != NULL)
	//	*v = make_float3( 0.0f );
	//if (ra != NULL)
	//	*ra = make_float2( 0.0f );
	//if (pg != NULL)
	//	*pg = make_float2( 0.0f );
	//if (dg != NULL)
	//	*dg = make_float2( 0.0f );
	//if (crlp != NULL)
	//	*crlp = 0u;
	//if (hp == NULL)			/* sampling falure? */
	//	return(0);

	if ((ra == NULL) & (pg == NULL) & (dg == NULL) || (hp.sampOK < 0) | (hp.ns < 6)) { /* Hessian not requested/possible */
		return(-1);		/* value-only return value */
	}
#ifndef AMB_SAVE_MEM
	float	d, K;
	if ((d = bright(*rcol)) > FTINY) {	/* normalize Y values */
		d = 0.99f * ( hp.ns * hp.ns ) / d;
		K = 0.01f;
	} else {			/* or fall back on geometric Hessian */
		K = 1.0f;
		pg = NULL;
		dg = NULL;
		crlp = NULL;
	}
				/* relative Y channel from here on... */
	for (unsigned int i = 0; i < hp.ns; i++)
		for (unsigned int j = 0; j < hp.ns; j++) {
			ambsam(i, j).v.y = bright(ambsam(i, j).v) * d + K;
		}

	//if (uv == NULL)			/* make sure we have axis pointers */
	//	uv = my_uv;
					/* compute radii & pos. gradient */
	ambHessian( &hp, uv, ra, pg, normal, hit );

	if (dg != NULL)			/* compute direction gradient */
		ambdirgrad( &hp, uv->getRow(0), uv->getRow(1), dg, normal, hit );

	if (ra != NULL) {		/* scale/clamp radii */
		if (pg != NULL) {
			if ( ra->x * (d = fabsf( pg->x ) ) > 1.0f )
				ra->x = 1.0f / d;
			if ( ra->y * (d = fabsf( pg->y ) ) > 1.0f )
				ra->y = 1.0f / d;
			if ( ra->x > ra->y )
				ra->x = ra->y;
		}
		if ( ra->x < minarad ) {
			ra->x = minarad;
			if ( ra->y < minarad )
				ra->y = minarad;
		}
		*ra *= 1.0f / sqrtf( wt );
		if ( ra->y > 2.0f * ra->x )
			ra->y = 2.0f * ra->x;
		if ( ra->y > maxarad ) {
			ra->y = maxarad;
			if ( ra->x > maxarad )
				ra->x = maxarad;
		}
#ifdef CORRAL
					/* flag encroached directions */
		if (crlp != NULL)
			*crlp = ambcorral( &hp, uv, *ra * ambacc, hit );
#endif /* CORRAL */
		if (pg != NULL) {	/* cap gradient if necessary */
			d = pg->x*pg->x * ra->x*ra->x + pg->y*pg->y * ra->y*ra->y;
			if ( d > 1.0f ) {
				*pg *= 1.0f / sqrtf(d);
			}
		}
	}
#endif /* AMB_SAVE_MEM */
	//free(hp);			/* clean up and return */
	return(1);
}

/* sample indirect hemisphere, based on samp_hemi in ambcomp.c */
RT_METHOD int samp_hemi(
	AMBHEMI *hp,
	float3 *rcol,
	float wt,
#ifdef AMB_SAVE_MEM
	optix::Matrix<2, 3> *uv,
	float2 *ra,
	float2 *pg,
	float2 *dg,
	unsigned int *crlp,
#endif /* AMB_SAVE_MEM */
	const float3& normal,
	const float3& hit
)
{
					/* insignificance check */
	if (bright(*rcol) <= FTINY)
		return 0;
					/* set number of divisions */
#ifndef AMB_PARALLEL
	float d;
	if (ambacc <= FTINY && wt > (d = 0.8f * fmaxf(*rcol) * wt / (ambdiv*minweight))) //TODO second wt should be radiance ray weight
		wt = d;			/* avoid ray termination */
#endif
	unsigned int n = sqrtf(ambdiv * wt) + 0.5f;
	unsigned int i = 1 + 5 * (ambacc > FTINY);	/* minimum number of samples */
	if (n < i)
		n = i;
					/* allocate sampling array */
	hp->ns = n;
	hp->acol = make_float3( 0.0f );
	hp->sampOK = 0;
					/* assign coefficient */
	hp->acoef = *rcol * (1.0f / (n*n));
					/* make tangent plane axes */
#ifdef AMB_PARALLEL
	hp->ux = getperpendicular(normal);
#else
	hp->ux = getperpendicular( normal, prd.state );
#endif
	hp->uy = cross( normal, hp->ux );

#ifdef AMB_SAVE_MEM
	AmbientSample current, prev;

	/* ambHessian from ambcomp.c */
	optix::Matrix<3,3> hessian;
	float3 gradient = make_float3( 0.0f );
	hessian.setRow( 0, gradient ); // Set zero matrix
	hessian.setRow( 1, gradient );
	hessian.setRow( 2, gradient );
	optix::Matrix<3,3> hessianY;
	float3 gradientY = make_float3( 0.0f );
	hessianY.setRow( 0, gradientY ); // Set zero matrix
	hessianY.setRow( 1, gradientY );
	hessianY.setRow( 2, gradientY );

	FFTRI fftr;
					/* be sure to assign unit vectors */
	uv->setRow( 0, hp->ux );
	uv->setRow( 1, hp->uy );

	/* ambdirgrad from ambcomp.c */
	float3 dgsum = make_float3( 0.0f );	/* sum values times -tan(theta) */

#ifdef CORRAL
	/* ambcorral from ambcomp.c */
	const float max_d = 1.0f / ( minarad * ambacc + 0.001f );
	const float ang_res = M_PI_2f / hp->ns;
	const float ang_step = ang_res / ( (int)( 16.0f * M_1_PIf * ang_res ) + 1.01f );
	float avg_d = 0.0f;
	unsigned int corral_count = 0u;
#endif /* CORRAL */

					/* sample divisions */
	for ( i = 0; i < hp->ns; i++ ) {
		optix::Matrix<3,3> hesscol;	/* compute first vertical edge */
		float3 gradcol;

	    for (unsigned int j = 0; j < hp->ns; j++ ) {
			hp->sampOK += ambsample(hp, &current, i, j, 0, normal, hit);
			current.v.y = bright( current.v ); /* relative Y channel from here on... */

			/* ambHessian from ambcomp.c */
			if ( i ) {
				if ( j ) {
					optix::Matrix<3,3> hessdia;	/* compute triangle contributions */
					float3 graddia;
					optix::Matrix<3,3> hesstmp;
					float3 gradtmp;

					float backg = back_ambval( &prevrow(j - 1), &prevrow(j), &prev );
								/* diagonal (inner) edge */
					comp_fftri(&fftr, &prevrow(j), &prev, hit);
					hessdia = comp_hessian( &fftr, normal );
					hessian += ( hesstmp = hessrow(j - 1) + hessdia - hesscol );
					hessianY += backg * hesstmp;
					graddia = comp_gradient( &fftr, normal );
					gradient += ( gradtmp = gradrow(j - 1) + graddia - gradcol );
					gradientY += backg * gradtmp;
								/* initialize edge in next row */
					comp_fftri( &fftr, &current, &prev, hit );
					hessrow(j - 1) = comp_hessian( &fftr, normal );
					gradrow(j - 1) = comp_gradient( &fftr, normal );
								/* new column edge & paired triangle */
					backg = back_ambval( &current, &prev, &prevrow(j) );
					comp_fftri( &fftr, &prevrow(j), &current, hit );
					hesscol = comp_hessian( &fftr, normal );
					hessian += ( hesstmp = hessrow(j - 1) - hessdia + hesscol );
					hessianY += backg * hesstmp;
					gradcol = comp_gradient( &fftr, normal );
					gradient += ( gradtmp = gradrow(j - 1) - graddia + gradcol );
					gradientY += backg * gradtmp;
					if ( i < hp->ns-1 ) {
						hessrow(j - 1) *= -1.0f;
						gradrow(j - 1) = -gradrow(j - 1);
					}

#ifdef CORRAL
					/* ambcorral from ambcomp.c */
					if ( ( i < hp->ns * 3 / 4 ) && ( i >= hp->ns>>2 ) )
						if ( ( j < hp->ns * 3 / 4 ) && ( j >= hp->ns>>2 ) )
							avg_d += current.d;
#endif /* CORRAL */
				} else {
					comp_fftri(&fftr, &prevrow(0), &current, hit);
					hesscol = comp_hessian( &fftr, normal );
					gradcol = comp_gradient( &fftr, normal );
				}
			} else if ( j ) {
					/* compute first row of edges */
				comp_fftri( &fftr, &prev, &current, hit );
				hessrow(j - 1) = comp_hessian(&fftr, normal);
				gradrow(j - 1) = comp_gradient(&fftr, normal);
			}

			/* ambdirgrad from ambcomp.c */
					/* use vector for azimuth + 90deg */
			const float3 vd = current.p - hit;
					/* brightness over cosine factor */
			const float gfact = current.v.y / dot( normal, vd );
					/* sine = proj_radius/vd_length */
			dgsum += vd * gfact;

			if (j)
				prevrow(j - 1) = prev;
			else
				prevrow(hp->ns - 1) = prev;
			prev = current;

#ifdef CORRAL
			/* ambcorral from ambcomp.c */
			if ( !i || !j || i == hp->ns - 1 || j == hp->ns - 1 ) {
				if ( ( current.d <= FTINY ) | ( current.d >= max_d ) )
					continue;	/* too far or too near */
				corral_u(corral_count) = *uv * vd;
				corral_d(corral_count++) = current.d * current.d;
			}
#endif /* CORRAL */
		}
	}
#else /* AMB_SAVE_MEM */
					/* sample divisions */
	for (i = hp->ns; i--; )
		for (unsigned int j = hp->ns; j--;)
			hp->sampOK += ambsample(hp, &ambsam(i, j), i, j, 0, normal, hit);
#endif /* AMB_SAVE_MEM */
	*rcol = hp->acol;

	if (!hp->sampOK) {		/* utter failure? */
		return( 0 );
	}
	if (hp->sampOK < hp->ns * hp->ns) {
		hp->sampOK *= -1;	/* soft failure */
		return( 1 );
	}

#ifdef AMB_SAVE_MEM
	/* doambient from ambcomp.c */
	if ((d = bright(*rcol)) > FTINY) {	/* normalize Y values */
		d = 0.99f * ( hp->ns * hp->ns ) / d;
		hessian = d * hessianY + 0.01f * hessian;
		gradient = d * gradientY + 0.01f * gradient;
	} else {
		pg = dg = NULL;
		crlp = NULL;
	}

	/* ambHessian from ambcomp.c */
	optix::Matrix<2,2> ab;
	if ( ra )			/* extract eigenvectors & radii */
		ab = eigenvectors( uv, ra, &hessian );

	/* ambHessian from ambcomp.c */
	if ( pg )
		*pg = *uv * gradient;

	/* ambdirgrad from ambcomp.c */
	if ( dg ) {
		optix::Matrix<2,2> rotate;
		rotate[0] = rotate[3] = 0.0f;
		rotate[1] = -1.0f;
		rotate[2] = 1.0f;
		*dg = rotate * *uv * dgsum / (hp->ns*hp->ns);
	}

	/* ambcorral from ambcomp.c */
	if ( ra ) {
		if ( pg ) {
			if ( ra->x * (d = fabsf( pg->x ) ) > 1.0f )
				ra->x = 1.0f / d;
			if ( ra->y * (d = fabsf( pg->y ) ) > 1.0f )
				ra->y = 1.0f / d;
			if ( ra->x > ra->y )
				ra->x = ra->y;
		}
		if ( ra->x < minarad ) {
			ra->x = minarad;
			if ( ra->y < minarad )
				ra->y = minarad;
		}
		*ra *= 1.0f / sqrtf( prd.result.weight );
		if ( ra->y > 2.0f * ra->x )
			ra->y = 2.0f * ra->x;
		if ( ra->y > maxarad ) {
			ra->y = maxarad;
			if ( ra->x > maxarad )
				ra->x = maxarad;
		}

#ifdef CORRAL
		if ( crlp ) {
			unsigned int flgs = 0u;
			const float2 r = *ra * ambacc;
			avg_d *= 4.0f / ( hp->ns * hp->ns );
			if ( ( hp->ns >= 12 ) && ( avg_d * r.x < 1.0f )	&& ( avg_d < max_d ) ) {
						/* else circle around perimeter */
				for ( i = 0; i < corral_count; i++ ) {
					float2 u = ab * corral_u(i);
					if ( ( r.x*r.x * u.x*u.x + r.y*r.y * u.y*u.y ) * corral_d(i) <= dot( u, u ) )
						continue;	/* occluder outside ellipse */
					float ang = atan2f( u.y, u.x );	/* else set direction flags */
					for ( float a1 = ang - ang_res; a1 <= ang + ang_res; a1 += ang_step )
						flgs |= 1L<<(int)( 16.0f * M_1_PIf * ( a1 + 2.0f * M_PIf * ( a1 < 0.0f ) ) );
				}
				*crlp = flgs;
			}
		}
#endif /* CORRAL */

		if ( pg ) {	/* cap gradient if necessary */
			d = pg->x*pg->x * ra->x*ra->x + pg->y*pg->y * ra->y*ra->y;
			if ( d > 1.0f )
				*pg *= 1.0f / sqrtf(d);
		}
	}
#else /* AMB_SAVE_MEM */
#ifdef AMB_SUPER_SAMPLE
	if (hp->sampOK < 64)
		return(1);		/* insufficient for super-sampling */
	n = ambssamp * wt + 0.5f;
	if (n > 8) {			/* perform super-sampling? */
		ambsupersamp(hp, n, normal, hit);
		*rcol = hp->acol;
	}
#endif
#endif /* AMB_SAVE_MEM */

	return( 1 );			/* all is well */
}

RT_METHOD int ambsample(AMBHEMI *hp, AmbientSample *ap, const unsigned int& i, const unsigned int& j, const unsigned int& n, const float3& normal, const float3& hit)
{
#ifdef AMB_PARALLEL
	if (!n) {
		if (ap->d == -1.0f) // An exception occurred
			rtThrow((int)(ap->v.x) | RT_RETHROWN_EXCEPTION);
		if (ap->d == 0.0f) // No exception, but bad data
			return(0);

		ap->v *= hp->acoef;	/* apply coefficient */
		hp->acol += ap->v;	/* add to our sum */
#ifdef DAYSIM_COMPATIBLE
		DaysimCoef sample_dc = make_uint3(0, i + hp->ns * j, prd.dc.z);
		sample_dc = daysimNext(sample_dc); // Skip ahead one
		daysimAddScaled(prd.dc, sample_dc, hp->acoef.x);
#endif
#ifdef RAY_COUNT
		prd.result.ray_count += ap->ray_count;
#endif
#ifdef HIT_COUNT
		prd.result.hit_count += ap->hit_count;
#endif
		return(1);
	}
#endif /* AMB_PARALLEL */
#if defined AMB_SUPER_SAMPLE || !defined AMB_PARALLEL
	PerRayData_radiance new_prd;
	float b2;
					/* generate hemispherical sample */
					/* ambient coefficient for weight */
	if (ambacc > FTINY)
		b2 = AVGREFL; // Reusing this variable
	else
		b2 = fminf(fmaxf(hp->acoef), 1.0f);
	new_prd.weight = prd.result.weight * b2;
	if (new_prd.weight < minweight) //if (rayorigin(&ar, AMBIENT, r, ar.rcoef) < 0)
		return(0);
	//if (ambacc > FTINY) {
	//	rcoef *= h->acoef;
	//	rcoef *= 1.0f / AVGREFL; // This all seems unnecessary
	//}
	//hlist[0] = hp->rp->rno;
	//hlist[1] = j;
	//hlist[2] = i;
	//multisamp(spt, 2, urand(ilhash(hlist,3)+n));
	float2 spt = make_float2( curand_uniform( prd.state ), curand_uniform( prd.state ) );
	if (!n && i > 0 && i < hp->ns - 1 && j > 0 && j < hp->ns - 1) /* avoid coincident samples */
		spt = 0.1f + 0.8f * spt;
	SDsquare2disk( spt, (j+spt.y) / hp->ns, (i+spt.x) / hp->ns );
	float zd = sqrtf( 1.0f - dot( spt, spt ) );
	float3 rdir = normalize( spt.x*hp->ux + spt.y*hp->uy + zd*normal );
	//dimlist[ndims++] = AI(hp,i,j) + 90171;

	new_prd.depth = prd.result.lvl + 1;//prd.depth + 1;
	new_prd.ambient_depth = prd.result.lvl + 1;//prd.ambient_depth + 1;
	new_prd.tmax = RAY_END;
	//new_prd.seed = prd.seed;//lcg( prd.seed );
	new_prd.state = prd.state;
#ifdef CONTRIB
#ifdef CONTRIB_DOUBLE
	new_prd.rcoef = make_contrib3(prd.result.weight * hp->acoef); //TODO This is not exact, but it's probably not used
#else
	new_prd.rcoef = prd.result.weight * hp->acoef; //TODO This is not exact, but it's probably not used
#endif
#endif
#ifdef ANTIMATTER
	new_prd.mask = 0u; //TODO check if we are in an antimatter volume
	new_prd.inside = 0;
#endif
#ifdef DAYSIM_COMPATIBLE
	new_prd.dc = daysimNext(prd.dc);
#if defined AMB_PARALLEL && defined AMB_SUPER_SAMPLE
	new_prd.dc = daysimNext(new_prd.dc); // Skip ahead one
#endif /* AMB_PARALLEL && AMB_SUPER_SAMPLE */
#endif /* DAYSIM_COMPATIBLE */
	setupPayload(new_prd);
	Ray amb_ray = make_Ray(hit, rdir, RADIANCE_RAY, ray_start(hit, rdir, normal, RAY_START), new_prd.tmax);
	rtTrace(top_object, amb_ray, new_prd);
#ifdef RAY_COUNT
	prd.result.ray_count += new_prd.ray_count;
#endif
#ifdef HIT_COUNT
	prd.result.hit_count += new_prd.hit_count;
#endif

	//ndims--;
	if ( isnan( new_prd.result ) ) // TODO How does this happen?
		return(0);
	if ( new_prd.distance <= FTINY )
		return(0);		/* should never happen */
	new_prd.result *= hp->acoef;	/* apply coefficient */
	if (!n || new_prd.distance * ap->d < 1.0f )		/* new/closer distance? */
		ap->d = 1.0f / new_prd.distance;
	if (!n) {			/* record first vertex & value */
		if (new_prd.distance > 50.0f * maxarad + 1000.0f) // 10 * thescene.cusize + 1000
			new_prd.distance = 50.0f * maxarad + 1000.0f;
		ap->p = hit + rdir * new_prd.distance;
		ap->v = new_prd.result; // only one AmbientSample, otherwise would need +=
#ifdef DAYSIM_COMPATIBLE
		daysimAddScaled(prd.dc, new_prd.dc, hp->acoef.x);
#endif
#ifdef AMB_SUPER_SAMPLE
	} else {			/* else update recorded value */
		hp->acol -= ap->v;
		zd = 1.0f / (n+1);
		ap->v *= n * zd;
		ap->v += new_prd.result * zd;
#ifdef DAYSIM_COMPATIBLE
#ifdef AMB_PARALLEL
		DaysimCoef sample_dc = make_uint3(0, i + hp->ns * j, prd.dc.z);
		sample_dc = daysimNext(sample_dc); // Skip ahead one
		daysimAddScaled(prd.dc, sample_dc, -hp->acoef.x);
		daysimRunningAverage(sample_dc, new_prd.dc, n);
		daysimAddScaled(prd.dc, sample_dc, hp->acoef.x);
#endif /* AMB_PARALLEL */
		// TODO Daysim compatible solution if not parallel ambient calculation
#endif /* DAYSIM_COMPATIBLE */
#endif /* AMB_SUPER_SAMPLE */
	}
	hp->acol += ap->v;	/* add to our sum */
#endif /* AMB_SUPER_SAMPLE || !AMB_PARALLEL */
	return(1);
}

#ifdef AMB_SUPER_SAMPLE
/* Estimate variance based on ambient division differences */
RT_METHOD void getambdiffs(AMBHEMI *hp)
{
	const float normf = 1.0f / bright(hp->acoef);

	/* compute squared neighbor diffs */
	for (unsigned int i = 0u; i < hp->ns; i++)
		for (unsigned int j = 0u; j < hp->ns; j++) {
			earr(i, j) = 0.0f;
			float b = bright(ambsam(i, j).v);
			if (i) {		/* from above */
				float b1 = bright(ambsam(i - 1, j).v);
				float d2 = b - b1;
				d2 *= d2 * normf / (b + b1);
				earr(i, j) += d2;
				earr(i - 1, j) += d2;
			}
			if (!j) continue;
			/* from behind */
			float b1 = bright(ambsam(i, j - 1).v);
			float d2 = b - b1;
			d2 *= d2 * normf / (b + b1);
			earr(i, j) += d2;
			earr(i, j - 1) += d2;
			if (!i) continue;
			/* diagonal */
			b1 = bright(ambsam(i - 1, j - 1).v);
			d2 = b - b1;
			d2 *= d2 * normf / (b + b1);
			earr(i, j) += d2;
			earr(i - 1, j - 1) += d2;
		}

	/* correct for number of neighbors */
	earr(0, 0) *= 8.0f / 3.0f;
	earr(0, hp->ns - 1) *= 8.0f / 3.0f;
	earr(hp->ns - 1, 0) *= 8.0f / 3.0f;
	earr(hp->ns - 1, hp->ns - 1) *= 8.0f / 3.0f;
	for (unsigned int i = 1u; i < hp->ns - 1; i++) {
		earr(i, 0) *= 8.0f / 5.0f;
		earr(i, hp->ns - 1) *= 8.0f / 5.0f;
		earr(0, i) *= 8.0f / 5.0f;
		earr(hp->ns - 1, i) *= 8.0f / 5.0f;
	}
}

/* Perform super-sampling on hemisphere (introduces bias) */
RT_METHOD void ambsupersamp(AMBHEMI *hp, int cnt, const float3& normal, const float3& hit)
{
	getambdiffs(hp);
	float e2rem = 0.0f;

	/* accumulate estimated variances */
	for (unsigned int i = hp->ns; i--; )
		for (unsigned int j = hp->ns; j--; )
			e2rem += earr(i, j);

	/* perform super-sampling */
	for (unsigned int i = 0u; i < hp->ns; i++)
		for (unsigned int j = 0u; j < hp->ns; j++) {
			if (e2rem <= FTINY)
				return;	/* nothing left to do */
			const float ep = earr(i, j);
			const int nss = ep / e2rem * cnt + curand_uniform(prd.state);
			for (int n = 1; n <= nss && ambsample(hp, &ambsam(i, j), i, j, n, normal, hit); n++)
				if (!--cnt) return;
			e2rem -= ep;		/* update remainder */
		}
}
#endif /* AMB_SUPER_SAMPLE */

/* Return brightness of farthest ambient sample */
RT_METHOD float back_ambval( const AmbientSample *n1, const AmbientSample *n2, const AmbientSample *n3 )
{
	if (n1->d <= n2->d) {
		if (n1->d <= n3->d)
			return(n1->v.y);
		return(n3->v.y);
	}
	if (n2->d <= n3->d)
		return(n2->v.y);
	return(n3->v.y);
}

/* Compute vectors and coefficients for Hessian/gradient calcs */
RT_METHOD void comp_fftri( FFTRI *ftp, const AmbientSample *n0, const AmbientSample *n1, const float3& hit )
{
	ftp->r_i = n0->p - hit;
	ftp->r_i1 = n1->p - hit;
	ftp->e_i = n1->p - n0->p;
	ftp->rcp = cross( ftp->r_i, ftp->r_i1 );
	const float rdot_cp = 1.0f / dot( ftp->rcp, ftp->rcp );
	const float dot_e = dot( ftp->e_i, ftp->e_i );
	const float dot_er = dot( ftp->e_i, ftp->r_i );
	const float rdot_r = 1.0f / dot( ftp->r_i, ftp->r_i );
	const float rdot_r1 = 1.0f / dot( ftp->r_i1, ftp->r_i1 );
	ftp->I1 = acosf(clamp(dot(ftp->r_i, ftp->r_i1) * sqrtf(rdot_r * rdot_r1), -1.0f, 1.0f)) * sqrtf(rdot_cp);
	ftp->I2 = ( dot( ftp->e_i, ftp->r_i1 ) * rdot_r1 - dot_er * rdot_r + dot_e * ftp->I1 ) * 0.5f * rdot_cp;
	const float J2 =  ( 0.5f * ( rdot_r - rdot_r1 ) - dot_er * ftp->I2 ) / dot_e;
	ftp->rI2_eJ2 = ftp->I2 * ftp->r_i + J2 * ftp->e_i;
}

/* Compose 3x3 matrix from two vectors */
RT_METHOD optix::Matrix<3,3> compose_matrix( const float3& va, const float3& vb )
{
	optix::Matrix<3,3> mat;
	mat.setRow( 0, va * vb.x + vb * va.x );
	mat.setRow( 1, va * vb.y + vb * va.y );
	mat.setRow( 2, va * vb.z + vb * va.z );
	//mat += mat.transpose();
	return mat;
}

/* Compute partial 3x3 Hessian matrix for edge */
RT_METHOD optix::Matrix<3,3> comp_hessian( FFTRI *ftp, const float3& normal )
{
					/* compute intermediate coefficients */
	float d1 = 1.0f / dot( ftp->r_i, ftp->r_i );
	float d2 = 1.0f / dot( ftp->r_i1, ftp->r_i1 );
	float d3 = 1.0f / dot( ftp->e_i, ftp->e_i );
	float d4 = dot( ftp->e_i, ftp->r_i );
	const float I3 = ( dot( ftp->e_i, ftp->r_i1 ) * d2 * d2 - d4 * d1 * d1 + 3.0f / d3 * ftp->I2 ) / ( 4.0f * dot( ftp->rcp, ftp->rcp ) );
	const float J3 = 0.25f * d3 * ( d1 * d1 - d2 * d2 ) - d4 * d3 * I3;
	const float K3 = d3 * ( ftp->I2 - I3 / d1 - 2.0f * d4 * J3);
					/* intermediate matrices */
	const float3 ncp = cross( normal, ftp->e_i );
	const optix::Matrix<3,3> m1 = compose_matrix( ncp, ftp->rI2_eJ2 );
	const optix::Matrix<3,3> m2 = compose_matrix( ftp->r_i, ftp->r_i );
	const optix::Matrix<3,3> m3 = compose_matrix( ftp->e_i, ftp->e_i );
	const optix::Matrix<3,3> m4 = compose_matrix( ftp->r_i, ftp->e_i );
	d1 = dot( normal, ftp->rcp );
	d2 = -d1 * ftp->I2;
	d1 *= 2.0f;
					/* final matrix sum */
	optix::Matrix<3,3> hess = m1 + d1 * ( I3 * m2 + K3 * m3 + 2.0f * J3 * m4 );
	hess += d2 * Matrix<3,3>::identity();
	hess *= -M_1_PIf;
	return hess;
}

/* Compute partial displacement form factor gradient for edge */
RT_METHOD float3 comp_gradient( FFTRI *ftp, const float3& normal )
{
	const float f1 = 2.0f * dot( normal, ftp->rcp );
	const float3 ncp = cross( normal, ftp->e_i );
	return ( 0.5f * M_1_PIf ) * ( ftp->I1 * ncp + f1 * ftp->rI2_eJ2 );
}

/* Compute anisotropic radii and eigenvector directions */
RT_METHOD optix::Matrix<2,2> eigenvectors( optix::Matrix<2,3> *uv, float2 *ra, optix::Matrix<3,3> *hessian )
{
					/* project Hessian to sample plane */
	const optix::Matrix<2,2> hess2 = *uv * *hessian * uv->transpose();
					/* compute eigenvalue(s) */
	float2 evalue;
	const unsigned int i = quadratic( &evalue, 1.0f, -hess2[0] - hess2[3], hess2[0] * hess2[3] - hess2[1] * hess2[2] );
	//if (i == 1u)			/* double-root (circle) */
	//	evalue.y = evalue.x;
	if (!i || ((evalue.x = fabsf(evalue.x)) <= FTINY*FTINY) | ((evalue.y = fabsf(evalue.y)) <= FTINY*FTINY) ) {
		*ra = make_float2( maxarad );
		return optix::Matrix<2,2>::identity();
	}
	float slope1;
	if ( evalue.x > evalue.y ) {
		*ra = sqrtf( sqrtf ( 4.0f / evalue ) );
		slope1 = evalue.y;
	} else {
		*ra = make_float2( sqrtf( sqrtf ( 4.0f / evalue.y ) ), sqrtf( sqrtf ( 4.0f / evalue.x ) ) );
		slope1 = evalue.x;
	}
					/* compute unit eigenvectors */
	if ( fabsf( hess2[1] ) <= FTINY )
		return optix::Matrix<2,2>::identity();			/* uv OK as is */
	slope1 = ( slope1 - hess2[0] ) / hess2[1];
	const float xmag1 = sqrtf( 1.0f / ( 1.0f + slope1 * slope1 ) );
	optix::Matrix<2,2> ab;
	ab[0] = ab[3] = slope1 * xmag1;
	ab[1] = -xmag1;
	ab[2] = xmag1;
	*uv = ab * *uv;

	/* needed for ambcorral */
	return ab;
}

#ifndef AMB_SAVE_MEM
RT_METHOD void ambHessian( AMBHEMI *hp, optix::Matrix<2,3> *uv, float2 *ra, float2 *pg, const float3& normal, const float3& hit )
{
	optix::Matrix<3,3> hessian;
	float3 gradient = make_float3( 0.0f );
	hessian.setRow( 0, gradient ); // Set zero matrix
	hessian.setRow( 1, gradient );
	hessian.setRow( 2, gradient );
	FFTRI fftr;
					/* be sure to assign unit vectors */
	uv->setRow( 0, hp->ux );
	uv->setRow( 1, hp->uy );
			/* clock-wise vertex traversal from sample POV */
	//if (ra != NULL) {		/* initialize Hessian row buffer */
	//	hessrow = (FVECT (*)[3])malloc(sizeof(FVECT)*3*(hp->ns-1)); //TODO set memory size
	//	if (hessrow == NULL)
	//		error(SYSTEM, memerrmsg);
	//	memset(hessian, 0, sizeof(hessian));
	//} else if (pg == NULL)		/* bogus call? */
	//	return;
	//if (pg != NULL) {		/* initialize form factor row buffer */
	//	gradrow = (FVECT *)malloc(sizeof(FVECT)*(hp->ns-1));
	//	if (gradrow == NULL)
	//		error(SYSTEM, memerrmsg);
	//	memset(gradient, 0, sizeof(gradient));
	//}
					/* compute first row of edges */
	for (unsigned int j = 0; j < hp->ns-1; j++) {
		comp_fftri(&fftr, &ambsam(0, j), &ambsam(0, j + 1), hit);
		if (ra != NULL)
			hessrow(j) = comp_hessian( &fftr, normal );
		if (pg != NULL)
			gradrow(j) = comp_gradient( &fftr, normal );
	}
					/* sum each row of triangles */
	for (unsigned int i = 0; i < hp->ns - 1; i++) {
	    optix::Matrix<3,3> hesscol;	/* compute first vertical edge */
	    float3 gradcol;
		comp_fftri(&fftr, &ambsam(i, 0), &ambsam(i + 1, 0), hit);
		if (ra != NULL)
			hesscol = comp_hessian( &fftr, normal );
		if (pg != NULL)
			gradcol = comp_gradient( &fftr, normal );
		for (unsigned int j = 0; j < hp->ns - 1; j++) {
			optix::Matrix<3,3> hessdia;	/* compute triangle contributions */
			float3 graddia;
			float backg = back_ambval(&ambsam(i, j), &ambsam(i, j + 1), &ambsam(i + 1, j));
						/* diagonal (inner) edge */
			comp_fftri(&fftr, &ambsam(i, j + 1), &ambsam(i + 1, j), hit);
			if (ra != NULL) {
				hessdia = comp_hessian( &fftr, normal );
				//hesscol = -hesscol;
				hessian += backg * ( hessrow(j) + hessdia - hesscol );
			}
			if (pg != NULL) {
				graddia = comp_gradient( &fftr, normal );
				//gradcol = -gradcol;
				gradient += backg * ( gradrow(j) + graddia - gradcol );
			}
						/* initialize edge in next row */
			comp_fftri(&fftr, &ambsam(i + 1, j + 1), &ambsam(i + 1, j), hit);
			if (ra != NULL)
				hessrow(j) = comp_hessian( &fftr, normal );
			if (pg != NULL)
				gradrow(j) = comp_gradient( &fftr, normal );
						/* new column edge & paired triangle */
			backg = back_ambval(&ambsam(i + 1, j + 1), &ambsam(i + 1, j), &ambsam(i, j + 1));
			comp_fftri(&fftr, &ambsam(i, j + 1), &ambsam(i + 1, j + 1), hit);
			if (ra != NULL) {
				hesscol = comp_hessian( &fftr, normal );
				//hessdia = -hessdia;
				hessian += backg * ( hessrow(j) - hessdia + hesscol );
				if ( i < hp->ns-2 )
					hessrow(j) *= -1.0f;
			}
			if (pg != NULL) {
				gradcol = comp_gradient( &fftr, normal );
				//graddia = -graddia;
				gradient += backg * ( gradrow(j) - graddia + gradcol );
				if ( i < hp->ns-2 )
					gradrow(j) = -gradrow(j);
			}
	    }
	}
					/* release row buffers */
	//if (hessrow != NULL) free(hessrow);
	//if (gradrow != NULL) free(gradrow);
	
	if (ra != NULL)			/* extract eigenvectors & radii */
		eigenvectors( uv, ra, &hessian );
	if (pg != NULL) {		/* tangential position gradient */
		*pg = *uv * gradient;
	}
}

/* Compute direction gradient from a hemispherical sampling */
RT_METHOD void ambdirgrad( AMBHEMI *hp, const float3& u, const float3& v, float2 *dg, const float3& normal, const float3& hit )
{
	float2 dgsum = make_float2( 0.0f );	/* sum values times -tan(theta) */
	for (unsigned int i = 0; i < hp->ns; i++)
		for (unsigned int j = 0; j < hp->ns; j++) {
			const AmbientSample ap = ambsam(i, j);
					/* use vector for azimuth + 90deg */
			float3 vd = ap.p - hit;
					/* brightness over cosine factor */
			float gfact = dot(normal, vd);
			if (gfact < FTINY)
				gfact = FTINY;
			gfact = ap.v.y / gfact;
					/* sine = proj_radius/vd_length */
			dgsum.x -= dot( v, vd ) * gfact;
			dgsum.y += dot( u, vd ) * gfact;
		}
	*dg = dgsum / (hp->ns*hp->ns);
}

/* Compute potential light leak direction flags for cache value */
RT_METHOD unsigned int ambcorral( AMBHEMI *hp, optix::Matrix<2,3> *uv, const float2& r, const float3& hit )
{
	const float max_d = 1.0f / ( minarad * ambacc + 0.001f );
	const float ang_res = M_PI_2f / hp->ns;
	const float ang_step = ang_res / ( (int)( 16.0f * M_1_PIf * ang_res ) + ( 1.01f ) );
	float avg_d = 0.0f;
	unsigned int flgs = 0u;
					/* don't bother for a few samples */
	if (hp->ns < 8)
		return(0u);
					/* check distances overhead */
	for (unsigned int i = hp->ns * 3 / 4; i-- > hp->ns >> 2;)
		for (unsigned int j = hp->ns * 3 / 4; j-- > hp->ns >> 2;)
			avg_d += ambsam(i, j).d;
	avg_d *= 4.0f / ( hp->ns * hp->ns );
	if ( avg_d * r.x >= 1.0f )		/* ceiling too low for corral? */
		return(0u);
	if ( avg_d >= max_d )		/* insurance */
		return(0u);
					/* else circle around perimeter */
	for (unsigned int i = 0; i < hp->ns; i++)
		for (unsigned int j = 0; j < hp->ns; j += !i | (i == hp->ns - 1) ? 1 : hp->ns - 1) {
			const AmbientSample ap = ambsam(i, j);
			if ( ( ap.d <= FTINY ) | ( ap.d >= max_d ) )
				continue;	/* too far or too near */
			const float2 u = *uv * ( ap.p - hit );
			if ( ( r.x*r.x * u.x*u.x + r.y*r.y * u.y*u.y ) * ap.d*ap.d <= u.x*u.x + u.y*u.y )
				continue;	/* occluder outside ellipse */
			const float ang = atan2f( u.y, u.x );	/* else set direction flags */
			for ( float a1 = ang - ang_res; a1 <= ang + ang_res; a1 += ang_step )
				flgs |= 1L<<(int)( 16.0f * M_1_PIf * ( a1 + 2.0f * M_PIf * ( a1 < 0.0f ) ) );
	    }
	return(flgs);
}
#endif /* AMB_SAVE_MEM */

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
