/*
 *  point_cloud_normal.cu - hit programs for geometry sampling on GPUs.
 */

#include "accelerad_copyright.h"

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optix_shader_ray.h"
#include "optix_point_common.h"

using namespace optix;

/* Context variables */
rtBuffer<PointDirection, 3>      seed_buffer; /* output */

rtDeclareVariable(float, specthresh, , );	/* This is the minimum fraction of reflection or transmission, under which no specular sampling is performed */
#ifdef AMBIENT_CELL
rtDeclareVariable(float, specjitter, , );	/* specular sampling (ss) */

rtDeclareVariable(float3, cuorg, , ); /* bounding box minimum */
rtDeclareVariable(float, cell_size, , ); /* cell side dimension */
rtDeclareVariable(unsigned int, level, , ) = 0u;

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
	ambient_prd.surface_point = pos;
	ambient_prd.surface_normal = faceforward(world_shading_normal, -ray.direction, world);
	ambient_prd.ambient_depth = level;
	ambient_prd.wsum = 0.0f;
	ambient_prd.weight = 1.0f;
	for ( int i = level; i--; )
		ambient_prd.weight *= AVGREFL; // Compute weight as in makeambient() from ambient.c

#ifdef DAYSIM_COMPATIBLE
	ambient_prd.dc = make_uint3(0u); // Mark as null (TODO check this)
#endif
#ifdef HIT_COUNT
	ambient_prd.hit_count = 0;
#endif
	const float tmax = ray_start(pos, AMBIENT_RAY_LENGTH);
	Ray ambient_ray = make_Ray(pos - dir * tmax, dir, AMBIENT_RAY, 0.0f, 2.0f * tmax);
	rtTrace(top_ambient, ambient_ray, ambient_prd, RT_VISIBILITY_ALL, RT_RAY_FLAG_DISABLE_CLOSESTHIT);
#ifdef HIT_COUNT
	prd.hit_count += ambient_prd.hit_count;
#endif
	return ambient_prd.wsum > FTINY;
}
#else /* AMBIENT_CELL */
RT_METHOD float3 sample_hemisphere(const float3& uz, rand_state* state)
{
	const float3 ux = getperpendicular(uz);
	const float3 uy = normalize(cross(uz, ux));
	float zd = sqrtf(curand_uniform(state));
	const float phi = 2.0f*M_PIf * curand_uniform(state);
	const float xd = cosf(phi) * zd;
	const float yd = sinf(phi) * zd;
	zd = sqrtf(1.0f - zd*zd);
	return normalize(xd*ux + yd*uy + zd*uz);
}
#endif /* AMBIENT_CELL */

RT_CALLABLE_PROGRAM PerRayData_point_cloud closest_hit_glass_point_cloud(IntersectData const&data, PerRayData_point_cloud prd)
{
	float3 snormal = faceforward(data.world_geometric_normal, -data.ray_direction, data.world_geometric_normal);

	prd.point = data.hit;

	/* Transmission */
#ifdef AMBIENT_CELL
	prd.forward = data.ray_direction;
#else
	prd.forward = sample_hemisphere(-snormal, prd.state);
#endif

	/* Reflection */
#ifdef AMBIENT_CELL
	prd.reverse = reflect(data.ray_direction, snormal);
#else
	prd.reverse = sample_hemisphere(snormal, prd.state);
#endif

	return prd;
}

RT_CALLABLE_PROGRAM PerRayData_point_cloud closest_hit_normal_point_cloud(IntersectData const&data, PerRayData_point_cloud prd)
{
	float3 snormal = faceforward(data.world_geometric_normal, -data.ray_direction, data.world_geometric_normal);

	float trans = data.mat.params.n.trans * (1.0f - data.mat.params.n.spec);
	float tspec = trans * data.mat.params.n.tspec;
	float alpha2 = data.mat.params.n.rough * data.mat.params.n.rough;

	/* Record new origin */
	prd.point = data.hit;

	/* Transmitted ambient */
	if (data.mat.params.n.ambincl && trans - tspec > FTINY && prd.index.z < prd.seeds) {
#ifdef AMBIENT_CELL
		if (!occupied(prd.point, -snormal, data.world_geometric_normal)) {
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
	if (data.mat.params.n.ambincl && 1.0f - trans - data.mat.params.n.spec > FTINY && prd.index.z < prd.seeds) {
#ifdef AMBIENT_CELL
		if (!occupied(prd.point, snormal, data.world_geometric_normal)) {
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

	if (prd.index.z >= prd.seeds) return prd;

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
		prd.forward = sample_hemisphere(-snormal, prd.state);
#endif
	}

	/* Reflected ray */
	if (data.mat.params.n.spec > FTINY && (alpha2 <= FTINY || specthresh < data.mat.params.n.spec - FTINY)) {
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
		prd.reverse = sample_hemisphere(snormal, prd.state);
#endif
	}

	return prd;
}
