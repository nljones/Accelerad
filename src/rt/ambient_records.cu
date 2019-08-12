/*
 *  ambient_records.cu - intersection testing for irradiance caching on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "optix_shader_ray.h"
#include "optix_shader_ambient.h"

using namespace optix;

rtBuffer<AmbientRecord> ambient_records;
#ifdef DAYSIM_COMPATIBLE
rtBuffer<DC, 2> ambient_dc;
#endif

rtDeclareVariable(float,        ambacc, , ); /* Ambient accuracy (aa). This value will approximately equal the error from indirect illuminance interpolation */
rtDeclareVariable(float,        minarad, , ); /* minimum ambient radius */

rtDeclareVariable(int, record_id, attribute record_id_attribute, );
rtDeclareVariable(float3, w, attribute w_attribute, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData_ambient, prd, rtPayload, );


RT_METHOD void sumambient();
RT_METHOD int plugaleak(const AmbientRecord* record, const float3& anorm, const float3& normal, float ang);


// Ignore the intersection so that the intersection program will continue to run for all overlapping recrods.
RT_PROGRAM void ambient_record_any_hit()
{
#ifdef HIT_COUNT
	prd.hit_count++;
#endif
	sumambient();
	rtIgnoreIntersection(); // Continue checking other intersections
}

// based on sumambient from ambient.c
RT_METHOD void sumambient()
{
	const AmbientRecord record = ambient_records[record_id];

	/* Ambient level test. */
	if ( record.lvl > prd.ambient_depth )
		return;
	if (record.lvl == prd.ambient_depth && record.weight < 0.9f * prd.weight)
		return;

	const float3 normal = ray.direction;

	/* Direction test using unperturbed normal */
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
	float3 ck0 = prd.surface_point - record.pos;
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

	float wt = (1.0f - sqrtf(delta_r2) / maxangle) * (1.0f - sqrtf(delta_t2) / ambacc);
	prd.wsum += wt;

	// This assignment to the prd would take place in the any-hit program if there were one
	prd.result += record.val * (d * wt);
#ifdef DAYSIM_COMPATIBLE
	if (ambient_dc.size().x && prd.dc.x)
		daysimAddScaled(prd.dc, &ambient_dc[make_uint2(0, primIdx)], d * wt);
#endif
}

// based on makeambient from ambient.c
RT_PROGRAM void ambient_miss()
{
	//if ( prd.wsum == 0.0f )
	//	rtThrow( RT_EXCEPTION_CUSTOM );
}

RT_PROGRAM void ambient_record_intersect( int primIdx )
{
	const AmbientRecord record = ambient_records[primIdx];

	/* Check for intersection with plane */
	const float3 disk_normal = decodedir(record.ndir);

	const float d = dot(disk_normal, ray.direction); // Ray direction is unperturbed surface normal
	if (d <= 0.0f)		/* >= 90 degrees */
		return;

	const float t = dot(disk_normal, record.pos - ray.origin) / d;

	if (rtPotentialIntersection(t)) {
		w = disk_normal;
		record_id = primIdx;
		rtReportIntersection(0); // There is only one material for ambient geometry group
	}
}

RT_PROGRAM void ambient_record_bounds (int primIdx, float result[6])
{
	optix::Aabb* aabb = (optix::Aabb*)result;
	const AmbientRecord record = ambient_records[primIdx];
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
}

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
	float3 vdif = record->pos - prd.surface_point;
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
	//Ray shadow_ray = make_Ray( prd.surface_point, normalize( rdir ), SHADOW_RAY, RAY_START, length( rdir ) );
	//PerRayData_shadow shadow_prd;
	//shadow_prd.result = make_float3( 1.0f );
#ifdef CONTRIB
	//shadow_prd.rcoef = make_contrib3(0.0f);
#endif
#ifdef ANTIMATTER
	//shadow_prd.mask = prd.mask;
	//shadow_prd.inside = prd.inside;
#endif
	//rtTrace( top_object, shadow_ray, shadow_prd );
	//return( dot( shadow_prd.result, shadow_prd.result ) < 1.0f );	/* check for occluder */
}
