/*
 * Copyright (c) 2013-2016 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_common.h"
#include "optix_point_common.h"

using namespace optix;

/* Material variables */
rtDeclareVariable(float, spec, , ) = 0.0f;	/* The material specularity given by the rad file "plastic", "metal", or "trans" object */
rtDeclareVariable(float, rough, , ) = 0.0f;	/* The material roughness given by the rad file "plastic", "metal", or "trans" object */
rtDeclareVariable(float, transm, , ) = 0.0f;	/* The material transmissivity given by the rad file "trans" object */
rtDeclareVariable(float, tspecu, , ) = 0.0f;	/* The material transmitted specular component given by the rad file "trans" object */
rtDeclareVariable(unsigned int, ambincl, , ) = 1u;	/* Flag to skip ambient calculation and use default (ae, aE, ai, aI) */

/* OptiX variables */
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_point_cloud, prd, rtPayload, );

/* Attributes */
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
#ifdef ANTIMATTER
rtDeclareVariable(int, mat_id, attribute mat_id, );

/* Context variables */
rtDeclareVariable(rtObject, top_object, , );
#endif

/* Context variables */
rtBuffer<PointDirection, 3>      seed_buffer; /* output */

rtDeclareVariable(float, specthresh, , );	/* This is the minimum fraction of reflection or transmission, under which no specular sampling is performed */
rtDeclareVariable(float, specjitter, , );	/* specular sampling (ss) */

#ifdef AMBIENT_CELL
rtDeclareVariable(float3, cuorg, , ); /* bounding box minimum */
rtDeclareVariable(float, cell_size, , ); /* cell side dimension */
rtDeclareVariable(unsigned int, level, , ) = 0u;

rtDeclareVariable(unsigned int, ambient_ray_type, , );
rtDeclareVariable(rtObject, top_ambient, , );


RT_METHOD uint2 cell_hash(const float3& pos, const float3& dir)
{
	uint2 cell;
	float3 absdir = make_float3(fabsf(dir.x), fabsf(dir.y), fabsf(dir.z));
	if (absdir.x > absdir.y) {
		if (absdir.x > absdir.z)
			cell.x = dir.x > 0 ? 0 : 0x10000;
		else
			cell.x = dir.z > 0 ? 0x40000 : 0x50000;
	}
	else {
		if (absdir.y > absdir.z)
			cell.x = dir.y > 0 ? 0x20000 : 0x30000;
		else
			cell.x = dir.z > 0 ? 0x40000 : 0x50000;
	}
	float3 cell_index = (pos - cuorg) / cell_size;
	cell.x += ((unsigned int)cell_index.x) & 0xffff;
	cell.y = (((unsigned int)cell_index.y) << 16) + (((unsigned int)cell_index.z) & 0xffff);
	return cell;
}

RT_METHOD int occupied(const float3& pos, const float3& dir, const float3& world)
{
	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));

	PerRayData_ambient ambient_prd;
	ambient_prd.result = make_float3(0.0f);
	ambient_prd.surface_normal = faceforward(world_shading_normal, -ray.direction, world);
	ambient_prd.ambient_depth = level;
	ambient_prd.wsum = 0.0f;
	ambient_prd.weight = 1.0f;
	for ( int i = level; i--; )
		ambient_prd.weight *= AVGREFL; // Compute weight as in makeambient() from ambient.c

#ifdef OLDAMB
	ambient_prd.state = prd.state;
#endif
#ifdef DAYSIM_COMPATIBLE
	ambient_prd.dc = make_uint3(0u); // Mark as null (TODO check this)
#endif
#ifdef HIT_COUNT
	ambient_prd.hit_count = 0;
#endif
	const float tmin = ray_start(pos, AMBIENT_RAY_LENGTH);
	Ray ambient_ray = make_Ray(pos, dir, ambient_ray_type, -tmin, tmin);
	rtTrace(top_ambient, ambient_ray, ambient_prd);
#ifdef HIT_COUNT
	prd.hit_count += ambient_prd.hit_count;
#endif
	return ambient_prd.wsum > FTINY;
}
#else /* AMBIENT_CELL */
RT_METHOD float3 sample_hemisphere(const float3& uz)
{
	float3 ux = getperpendicular(uz);
	float3 uy = normalize(cross(uz, ux));
	float zd = sqrtf(curand_uniform(prd.state));
	float phi = 2.0f*M_PIf * curand_uniform(prd.state);
	float xd = cosf(phi) * zd;
	float yd = sinf(phi) * zd;
	zd = sqrtf(1.0f - zd*zd);
	return normalize(xd*ux + yd*uy + zd*uz);
}
#endif /* AMBIENT_CELL */

RT_PROGRAM void closest_hit_point_cloud_glass()
{
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 snormal = faceforward(world_geometric_normal, -ray.direction, world_geometric_normal);

#ifdef ANTIMATTER
	if (prd.mask & (1 << mat_id)) {
		prd.inside += dot(world_geometric_normal, ray.direction) < 0.0f ? 1 : -1;

		/* Continue the ray */
		Ray new_ray = make_Ray(ray.origin, ray.direction, ray.ray_type, ray_start(ray.origin + t_hit * ray.direction, ray.direction, snormal, RAY_START) + t_hit, RAY_END);
		rtTrace(top_object, new_ray, prd);
		return;
	}
#endif /* ANTIMATTER */

	prd.point = ray.origin + t_hit * ray.direction;

	/* Transmission */
#ifdef AMBIENT_CELL
	prd.forward = ray.direction;
#else
	prd.forward = sample_hemisphere(-snormal);
#endif

	/* Reflection */
#ifdef AMBIENT_CELL
	prd.reverse = reflect(ray.direction, snormal);
#else
	prd.reverse = sample_hemisphere(snormal);
#endif
}

RT_PROGRAM void closest_hit_point_cloud_normal()
{
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	float3 snormal = faceforward(world_geometric_normal, -ray.direction, world_geometric_normal);

#ifdef ANTIMATTER
	if (prd.mask & (1 << mat_id)) {
		prd.inside += dot(world_geometric_normal, ray.direction) < 0.0f ? 1 : -1;

		/* Continue the ray */
		Ray new_ray = make_Ray(ray.origin, ray.direction, ray.ray_type, ray_start(ray.origin + t_hit * ray.direction, ray.direction, snormal, RAY_START) + t_hit, RAY_END);
		rtTrace(top_object, new_ray, prd);
		return;
	}
#endif /* ANTIMATTER */

	float trans = transm * (1.0f - spec);
	float tspec = trans * tspecu;
	float alpha2 = rough * rough;

	/* Record new origin */
	prd.point = ray.origin + t_hit * ray.direction;

	/* Transmitted ambient */
	if (ambincl && trans - tspec > FTINY && prd.index.z < prd.seeds) {
#ifdef AMBIENT_CELL
		if (!occupied(prd.point, -snormal, world_geometric_normal)) {
			seed_buffer[prd.index].cell = cell_hash(prd.point, -snormal);
#endif

			/* Store seed point */
			seed_buffer[prd.index].pos = prd.point;
			seed_buffer[prd.index].dir = -snormal;
			prd.index.z++;

#ifdef AMBIENT_CELL
		}
#endif
	}

	/* Reflected ambient */
	if (ambincl && 1.0f - trans - spec > FTINY && prd.index.z < prd.seeds) {
#ifdef AMBIENT_CELL
		if (!occupied(prd.point, snormal, world_geometric_normal)) {
			seed_buffer[prd.index].cell = cell_hash(prd.point, snormal);
#endif

			/* Store seed point */
			seed_buffer[prd.index].pos = prd.point;
			seed_buffer[prd.index].dir = snormal;
			prd.index.z++;

#ifdef AMBIENT_CELL
		}
#endif
	}

	if (prd.index.z >= prd.seeds) return;

	/* Transmitted ray */
	if (tspec > FTINY && (alpha2 <= FTINY || specthresh < tspec - FTINY)) {
#ifdef AMBIENT_CELL
		prd.forward = ray.direction;

		if (alpha2 > FTINY) {
			float3 u = getperpendicular(-snormal); //TODO should be pnormal
			float3 v = cross(-snormal, u);
			float2 rv = make_float2(curand_uniform(prd.state), curand_uniform(prd.state)); // should be evenly distributed in both dimensions
			float d = 2.0f * M_PIf * rv.x;
			float cosp = cosf(d);
			float sinp = sinf(d);
			if ((0.0f <= specjitter) && (specjitter < 1.0f))
				rv.y = 1.0f - specjitter * rv.y;
			if (rv.y <= FTINY)
				d = 1.0f;
			else
				d = sqrtf(alpha2 * -logf(rv.y));
			float3 h = d * (cosp * u + sinp * v) - snormal; //TODO should be pnormal
			d = -2.0f * dot(h, prd.forward) / (1.0f + d*d);
			h = prd.forward + h * d;

			/* sample rejection test */
			if (dot(h, snormal) < -FTINY)
				prd.forward = h;
		}
#else
		prd.forward = sample_hemisphere(-snormal);
#endif
	}

	/* Reflected ray */
	if (spec > FTINY && (alpha2 <= FTINY || specthresh < spec - FTINY)) {
#ifdef AMBIENT_CELL
		prd.reverse = reflect(ray.direction, snormal);

		if (alpha2 > FTINY) {
			float3 u = getperpendicular(snormal); //TODO should be pnormal
			float3 v = cross(snormal, u);
			float2 rv = make_float2(curand_uniform(prd.state), curand_uniform(prd.state)); // should be evenly distributed in both dimensions
			float d = 2.0f * M_PIf * rv.x;
			float cosp = cosf(d);
			float sinp = sinf(d);
			if ((0.0f <= specjitter) && (specjitter < 1.0f))
				rv.y = 1.0f - specjitter * rv.y;
			if (rv.y <= FTINY)
				d = 1.0f;
			else
				d = sqrtf(alpha2 * -logf(rv.y));
			float3 h = d * (cosp * u + sinp * v) + snormal; //TODO should be pnormal
			d = -2.0f * dot(h, prd.reverse) / (1.0f + d*d);
			h = prd.reverse + h * d;

			/* sample rejection test */
			if (dot(h, snormal) > FTINY)
				prd.reverse = h;
		}
#else
		prd.reverse = sample_hemisphere(snormal);
#endif
	}
}

RT_PROGRAM void closest_hit_point_cloud_light()
{
#ifdef ANTIMATTER
	if (prd.mask & (1 << mat_id)) {
		float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
		prd.inside += dot(world_geometric_normal, ray.direction) < 0.0f ? 1 : -1;

		/* Continue the ray */
		float3 snormal = faceforward(world_geometric_normal, -ray.direction, world_geometric_normal);
		Ray new_ray = make_Ray(ray.origin, ray.direction, ray.ray_type, ray_start(ray.origin + t_hit * ray.direction, ray.direction, snormal, RAY_START) + t_hit, RAY_END);
		rtTrace(top_object, new_ray, prd);
		return;
	}
#endif /* ANTIMATTER */
}
