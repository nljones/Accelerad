/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix_world.h>
#include "optix_shader_common.h"
#include "optix_ambient_common.h"

#ifdef AMB_PARALLEL

using namespace optix;

/* Contex variables */
rtBuffer<PointDirection, 1>     cluster_buffer; /* input */
rtBuffer<AmbientSample, 3>      amb_samp_buffer; /* ambient sample output */
rtDeclareVariable(rtObject,     top_object, , );
rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, level, , ) = 0u;
rtDeclareVariable(unsigned int, segment_offset, , ) = 0u;

rtDeclareVariable(float,        ambacc, , ); /* Ambient accuracy (aa). This value will approximately equal the error from indirect illuminance interpolation */
rtDeclareVariable(float,        maxarad, , ); /* maximum ambient radius */
rtDeclareVariable(float,        minweight, , ); /* minimum ray weight (lw) */

/* OptiX variables */
rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim,   rtLaunchDim, );

// Initialize the random state
RT_METHOD void init_state( PerRayData_radiance* prd )
{
	rand_state state;
	prd->state = &state;
	curand_init(launch_index.x + launch_dim.x * (launch_index.y + launch_dim.y * (launch_index.z + launch_dim.z * level)), 0, 0, prd->state);
}

RT_PROGRAM void ambient_sample_camera()
{
	uint3 index = launch_index;
	index.z += segment_offset;
	PointDirection cluster = cluster_buffer[index.z];

	PerRayData_radiance prd;
	init_state(&prd);
	float b2;
					/* generate hemispherical sample */
					/* ambient coefficient for weight */
//	if (ambacc > FTINY)
		b2 = AVGREFL; // Reusing this variable
//	else
//		b2 = fmaxf(hp->acoef); //TODO

	prd.weight = b2;
	for ( int i = level; i--; )
		prd.weight *= AVGREFL; // Compute weight as in makeambient() from ambient.c

	if (prd.weight < minweight) { //if (rayorigin(&ar, AMBIENT, r, ar.rcoef) < 0)
		amb_samp_buffer[index].d = 0.0f;
		return;
	}
	//if (ambacc > FTINY) {
	//	rcoef *= h->acoef;
	//	rcoef *= 1.0f / AVGREFL; // This all seems unnecessary
	//}
	//hlist[0] = hp->rp->rno;
	//hlist[1] = j;
	//hlist[2] = i;
	//multisamp(spt, 2, urand(ilhash(hlist,3)+n));
	float3 ux = getperpendicular(cluster.dir); // Can't be random, must be same for all threads for this point
	float3 uy = cross(cluster.dir, ux);
					/* avoid coincident samples */
	float2 spt = 0.1f + 0.8f * make_float2(curand_uniform(prd.state), curand_uniform(prd.state));
	SDsquare2disk(spt, (launch_index.y + spt.y) / launch_dim.y, (launch_index.x + spt.x) / launch_dim.x);
	float zd = sqrtf(1.0f - dot(spt, spt));
	float3 rdir = normalize(spt.x * ux + spt.y * uy + zd * cluster.dir);
	//dimlist[ndims++] = AI(hp,i,j) + 90171;

	prd.depth = level + 1;//prd.depth + 1;
	prd.ambient_depth = level + 1;//prd.ambient_depth + 1;
#ifdef DAYSIM_COMPATIBLE
	prd.dc = make_uint3(0, launch_index.x + launch_dim.x * launch_index.y, launch_index.z);
	prd.dc = daysimNext(prd.dc); // Skip ahead one
	daysimSet(prd.dc, 0.0f);
#endif
	setupPayload(prd, 1);
	Ray ray = make_Ray(cluster.pos, rdir, radiance_ray_type, ray_start(cluster.pos, rdir, cluster.dir, RAY_START), RAY_END);
	rtTrace(top_object, ray, prd);
#ifdef RAY_COUNT
	amb_samp_buffer[index].ray_count = prd.ray_count;
#endif
#ifdef HIT_COUNT
	amb_samp_buffer[index].hit_count = prd.hit_count;
#endif

	//ndims--;
	checkFinite(prd.result);
	if (prd.distance <= FTINY) {
		amb_samp_buffer[index].d = 0.0f;
		return;
	}

	//if ( new_prd.distance * ap->d < 1.0f )		/* new/closer distance? */ //TODO where did this value come from?
		amb_samp_buffer[index].d = 1.0f / prd.distance;
	//if (!n) {			/* record first vertex & value */
		if ( prd.distance > 10.0f * maxarad ) // 10 * thescene.cusize
			prd.distance = 10.0f * maxarad;
		amb_samp_buffer[index].p = cluster.pos + rdir * prd.distance;
		amb_samp_buffer[index].v = prd.result; // only one AmbientSample, otherwise would need +=
	//} else {			/* else update recorded value */
	//	hp->acol -= ap->v;
	//	zd = 1.0f / (float)(n+1);
	//	prd.result *= zd;
	//	zd *= (float)n;
	//	ambient_sample_buffer[index].v *= zd;
	//	ambient_sample_buffer[index].v += new_prd.result;
	//}
}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf("Caught exception 0x%X at launch index (%d,%d,%d)\n", code, launch_index.x, launch_index.y, launch_index.z);
	uint3 index = launch_index;
	index.z += segment_offset;
	amb_samp_buffer[index].d = -1.0f;
	amb_samp_buffer[index].v = exceptionToFloat3( code );
	amb_samp_buffer[index].p = exceptionToFloat3( code );
#ifdef RAY_COUNT
	amb_samp_buffer[index].ray_count = 0;
#endif
#ifdef HIT_COUNT
	amb_samp_buffer[index].hit_count = 0;
#endif
}

#endif /* AMB_PARALLEL */
