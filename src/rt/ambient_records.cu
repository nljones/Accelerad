/*
 * Copyright (c) 2013-2016 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "optix_shader_common.h"

using namespace optix;

rtBuffer<AmbientRecord> ambient_records;
#ifdef DAYSIM_COMPATIBLE
rtBuffer<DC, 2> ambient_dc;
#endif

rtDeclareVariable(float,        ambacc, , ); /* Ambient accuracy (aa). This value will approximately equal the error from indirect illuminance interpolation */
rtDeclareVariable(float,        minarad, , ); /* minimum ambient radius */

//rtDeclareVariable(float3, ambient_value, attribute ambient_value, );
//rtDeclareVariable(float, weight, attribute weight_attribute, );
//rtDeclareVariable(float, extrapolation, attribute extrapolation_attribute, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData_ambient, prd, rtPayload, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );


#ifndef OLDAMB
RT_METHOD int plugaleak(const AmbientRecord* record, const float3& anorm, const float3& normal, float ang);
#endif


// Ignore the intersection so that the intersection program will continue to run for all overlapping recrods.
RT_PROGRAM void ambient_record_any_hit()
{
	//prd.wsum += weight;

	//if (extrapolation > 0.0f) {
	//	prd.result += ambient_value * (extrapolation * weight);
	//}
	rtIgnoreIntersection();
}

// based on makeambient from ambient.c
RT_PROGRAM void ambient_miss()
{
	//if ( prd.wsum == 0.0f )
	//	rtThrow( RT_EXCEPTION_USER );
}

// based on sumambient from ambient.c
RT_PROGRAM void ambient_record_intersect( int primIdx )
{
#ifdef HIT_COUNT
	prd.hit_count++;
#endif

	const AmbientRecord record = ambient_records[primIdx];

	/* Ambient level test. */
	if ( record.lvl > prd.ambient_depth )
		return;
	if (record.lvl == prd.ambient_depth && record.weight < 0.9f * prd.weight)
		return;

	const float3 normal = ray.direction;

#ifndef OLDAMB
	/* Direction test using unperturbed normal */
	float3 w = decodedir( record.ndir );
	float d = dot(w, normal); // Ray direction is unperturbed surface normal
	if ( d <= 0.0f )		/* >= 90 degrees */
		return;
	if (d > 1.0f)
		d = 1.0f;

	float delta_r2 = 2.0f - 2.0f * d;	/* approx. radians^2 */
	const float minangle = 10.0f * M_PIf / 180.0f;
	float maxangle = minangle + ambacc;
					/* adjust maximum angle */
	if (prd.weight < 0.6f)
		maxangle = (maxangle - M_PI_2f) * powf(prd.weight, 0.13f) + M_PI_2f;
	if ( delta_r2 >= maxangle * maxangle )
		return;

	/* Modified ray behind test */
	float3 ck0 = ray.origin - record.pos;
	d = dot( ck0, w );
	if ( d < -minarad * ambacc - 0.001f )
		return;
	d /= record.rad.x;
	float delta_t2 = d * d;
	if ( delta_t2 >= ambacc * ambacc )
		return;
	
	/* Elliptical radii test based on Hessian */
	float3 u = decodedir( record.udir );
	float3 v = cross( w, u );
	float uu, vv;
	d = (uu = dot( ck0, u )) / record.rad.x;
	delta_t2 += d * d;
	d = (vv = dot( ck0, v )) / record.rad.y;
	delta_t2 += d * d;
	if ( delta_t2 >= ambacc * ambacc )
		return;
	
	/* Test for potential light leak */
	if (record.corral && plugaleak(&record, w, normal, atan2f(vv, uu)))
		return;

	/* Extrapolate value and compute final weight (hat function) */
	/* This is extambient from ambient.c */
	/* gradient due to translation */
	d = 1.0f + dot( ck0, record.gpos.x * u + record.gpos.y * v );

	/* gradient due to rotation */
	ck0 = cross( w, prd.surface_normal );
	d += dot( ck0, record.gdir.x * u + record.gdir.y * v );

	//if (d < min_d)			/* should not use if we can avoid it */
	//	d = min_d;
	if ( d <= 0.05f )
		return;

	if (rtPotentialIntersection(-dot(ck0, normal))) {
		float wt = ( 1.0f - sqrtf(delta_r2) / maxangle ) * ( 1.0f - sqrtf(delta_t2) / ambacc );
		prd.wsum += wt;

		// This assignment to the prd would take place in the any-hit program if there were one
		prd.result += record.val * ( d * wt );
#ifdef DAYSIM_COMPATIBLE
		if (ambient_dc.size().x)
			daysimAddScaled(prd.dc, &ambient_dc[make_uint2(0, primIdx)], d * wt);
#endif

		rtReportIntersection( 0 ); // There is only one material for ambient geometry group
	}
#else /* OLDAMB */
	/* Ambient radius test. */
	float3 ck0 = record.pos - ray.origin;
	float e1 = dot( ck0, ck0 ) / ( record.rad * record.rad );
	float acc = ambacc * ambacc * 1.21f;
	if ( e1 > acc )
		return;

	/* Direction test using closest normal. */
	float d = dot( record.dir, normal ); // Ray direction is unperturbed surface normal
	//if (rn != r->ron) {
	//	rn_dot = DOT(av->dir, rn);
	//	if (rn_dot > 1.0-FTINY)
	//		rn_dot = 1.0-FTINY;
	//	if (rn_dot >= d-FTINY) {
	//		d = rn_dot;
	//		rn_dot = -2.0;
	//	}
	//}
	float e2 = (1.0f - d) * prd.weight;
	if (e2 < 0.0f)
		e2 = 0.0f;
	else if (e1 + e2 > acc)
		return;

	/* Ray behind test. */
	d = dot( ck0, record.dir + normal );
	if (d * 0.5f > minarad * ambacc + 0.001f )
		return;

	/* Jittering final test reduces image artifacts. */
	e1 = sqrtf(e1);
	e2 = sqrtf(e2);
	float wt = e1 + e2;
	if (wt > ambacc * ( 0.9f + 0.2f * curand_uniform( prd.state ) ) )
		return;

	if (rtPotentialIntersection(dot(ck0, normal))) {
		/* Recompute directional error using perturbed normal */
		//if (rn_dot > 0.0) {
		//	e2 = sqrtf( ( 1.0f - rn_dot ) * prd.weight);
		//	wt = e1 + e2;
		//}
		if (wt <= 1e-3f)
			wt = 1e3f;
		else
			wt = 1.0f / wt;
		prd.wsum += wt; // This assignment to the prd would take place in the any-hit program if there were one

		/* This is extambient from ambient.c */
		//float d = 1.0f;			/* zeroeth order */

		/* gradient due to translation */
		d = 1.0f - dot( record.gpos, ck0 );

		/* gradient due to rotation */
		ck0 = cross( record.dir, prd.surface_normal );
		d += dot( record.gdir, ck0 );

		if (d > 0.0f) {
			// This assignment to the prd would take place in the any-hit program if there were one
			prd.result += record.val * (d * wt);
		}

		rtReportIntersection( 0 ); // There is only one material for ambient geometry group
	}
#endif /* OLDAMB */
}

RT_PROGRAM void ambient_record_bounds (int primIdx, float result[6])
{
	optix::Aabb* aabb = (optix::Aabb*)result;
	const AmbientRecord record = ambient_records[primIdx];

#ifndef OLDAMB
	const float2 rad = record.rad * ambacc; // Acceleration structure becomes dirty when ambacc is increased.

	if( rad.x > FTINY && isfinite(rad.y) ) {
		const float3 w = decodedir( record.ndir );
		const float3 u = decodedir( record.udir );
		const float3 v = cross( w, u );
		const float3 dims = sqrtf( u*u*(rad.x*rad.x) + v*v*(rad.y*rad.y) );// + FTINY; // Expanding by FTINY seems to help prevent misses
		//const float3 dims = sqrtf( u*u*(rad.x*rad.x) + v*v*(rad.y*rad.y) + w*w*(rad.x*rad.x) );// + FTINY;
		aabb->m_min = record.pos - dims;
		aabb->m_max = record.pos + dims;
	} else {
		aabb->invalidate();
	}
#else /* OLDAMB */
	const float rad = record.rad * 1.1f * ambacc; // Acceleration structure becomes dirty when ambacc is increased.

	if( rad > FTINY && isfinite(rad) ) {
		const float3 dims = rad * sqrtf( 1.0f - record.dir * record.dir );// + FTINY; // Expanding by FTINY seems to help prevent misses
		aabb->m_min = record.pos - dims;
		aabb->m_max = record.pos + dims;
	} else {
		aabb->invalidate();
	}
#endif /* OLDAMB */
}

#ifndef OLDAMB
/* Plug a potential leak where ambient cache value is occluded */
RT_METHOD int plugaleak(const AmbientRecord* record, const float3& anorm, const float3& normal, float ang)
{
	const float cost70sq = 0.1169778f;	/* cos(70deg)^2 */
	float2 t;

	ang += 2.0f * M_PIf * (ang < 0);			/* check direction flags */
	if ( !(record->corral>>(int)(ang * 16.0f * M_1_PIf) & 1) )
		return(0);
	/*
	 * Generate test ray, targeting 20 degrees above sample point plane
	 * along surface normal from cache position.  This should be high
	 * enough to miss local geometry we don't really care about.
	 */
	float3 vdif = record->pos - ray.origin;
	float normdot = dot(anorm, normal);
	float ndotd = dot(vdif, normal);
	float nadotd = dot( vdif, anorm );
	float a = normdot * normdot - cost70sq;
	float b = 2.0f * ( normdot * ndotd - nadotd * cost70sq );
	float c = ndotd * ndotd - dot( vdif, vdif ) * cost70sq;
	if ( quadratic( &t, a, b, c ) != 2 )
		return(1);			/* should rarely happen */
	//if ( t.y <= FTINY )
		return(0);			/* should fail behind test */

	/* Can't shoot rays from an intersection program. */
	//float3 rdir = vdif + anorm * t.y;	/* further dist. > plane */
	//Ray shadow_ray = make_Ray( ray.origin, normalize( rdir ), shadow_ray_type, RAY_START, length( rdir ) );
	//PerRayData_shadow shadow_prd;
	//shadow_prd.result = make_float3( 1.0f );
#ifdef ANTIMATTER
	//shadow_prd.mask = prd.mask;
	//shadow_prd.inside = prd.inside;
#endif
	//rtTrace( top_object, shadow_ray, shadow_prd );
	//return( dot( shadow_prd.result, shadow_prd.result ) < 1.0f );	/* check for occluder */
}
#endif /* OLDAMB */