/*
 *  triangle_mesh.cu - intersection program for triangles on GPUs.
 */

#include "accelerad_copyright.h"

/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "optix_shader_common.h"

#define RTX

using namespace optix;

/* Program variables */
rtDeclareVariable(unsigned int, backvis, , ); /* backface visibility (bv) */

/* Instance variables */
rtDeclareVariable(int, sole_material, , ); /* sole material index to use for all objects, or -1 to use per-face material index */

/* Contex variables */
rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<uint3>  vindex_buffer;    // position indices 
//rtBuffer<uint3>  nindex_buffer;    // normal indices
//rtBuffer<uint3>  tindex_buffer;    // texcoord indices
rtBuffer<unsigned int>    material_buffer; // per-face material index

/* OptiX variables */
rtDeclareVariable(Ray, ray, rtCurrentRay, );

/* Attributes */
rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(int, surface_id, attribute surface_id, );
rtDeclareVariable(int, mat_id, attribute mat_id, );


#ifdef RTX
RT_PROGRAM void mesh_attribute()
{
	const uint3 v_idx = vindex_buffer[rtGetPrimitiveIndex()];

	const float3 v0 = vertex_buffer[v_idx.x];
	const float3 v1 = vertex_buffer[v_idx.y];
	const float3 v2 = vertex_buffer[v_idx.z];
	const float3 Ng = optix::cross(v1 - v0, v2 - v0);

	geometric_normal = optix::normalize(Ng);

	const float2 barycentrics = rtGetTriangleBarycentrics();
	texcoord = make_float3(barycentrics.x, barycentrics.y, 0.0f);

	if (normal_buffer.size() == 0)
	{
		shading_normal = geometric_normal;
	}
	else
	{
		shading_normal = normal_buffer[v_idx.y] * barycentrics.x + normal_buffer[v_idx.z] * barycentrics.y
			+ normal_buffer[v_idx.x] * (1.0f - barycentrics.x - barycentrics.y);
	}

	if (texcoord_buffer.size() == 0)
	{
		texcoord = make_float3(0.0f, 0.0f, 0.0f);
	}
	else
	{
		const float2 t0 = texcoord_buffer[v_idx.x];
		const float2 t1 = texcoord_buffer[v_idx.y];
		const float2 t2 = texcoord_buffer[v_idx.z];
		texcoord = make_float3(t1*barycentrics.x + t2*barycentrics.y + t0*(1.0f - barycentrics.x - barycentrics.y));
	}

	surface_id = v_idx.x; // Not necessarily unique per triangle, but different for each surface

	int mat = sole_material;
	if (mat < 0) /* Use per-face material index */
		mat = material_buffer[rtGetPrimitiveIndex()];
	mat_id = mat;
}
#else
RT_PROGRAM void mesh_intersect(unsigned int primIdx)
{
	uint3 v_idx = vindex_buffer[primIdx];

	float3 p0 = vertex_buffer[ v_idx.x ];
	float3 p1 = vertex_buffer[ v_idx.y ];
	float3 p2 = vertex_buffer[ v_idx.z ];

	// Intersect ray with triangle
	float3 n;
	float  t, beta, gamma;
	if( intersect_triangle( ray, p0, p1, p2, n, t, beta, gamma ) && ( backvis || dot( n, ray.direction ) < 0) ) {

		int mat = sole_material;
		if ( mat < 0 ) /* Use per-face material index */
			mat = material_buffer[primIdx];

		if ( rtPotentialIntersection( t ) ) {

			//int3 n_idx = nindex_buffer[ primIdx ];
			geometric_normal = normalize( n );
			if ( normal_buffer.size() == 0 ) { //|| n_idx.x < 0 || n_idx.y < 0 || n_idx.z < 0 ) {
				shading_normal = geometric_normal;
			} else {
				float3 n0 = normal_buffer[ v_idx.x ];
				float3 n1 = normal_buffer[ v_idx.y ];
				float3 n2 = normal_buffer[ v_idx.z ];
				shading_normal = normalize( n1*beta + n2*gamma + n0*(1.0f-beta-gamma) );
				if ( !isfinite( shading_normal ) )
					shading_normal = geometric_normal;
			}

			//int3 t_idx = tindex_buffer[ primIdx ];
			if ( texcoord_buffer.size() == 0 ) { //|| t_idx.x < 0 || t_idx.y < 0 || t_idx.z < 0 ) {
				texcoord = make_float3( 0.0f, 0.0f, 0.0f );
			} else {
				float2 t0 = texcoord_buffer[ v_idx.x ];
				float2 t1 = texcoord_buffer[ v_idx.y ];
				float2 t2 = texcoord_buffer[ v_idx.z ];
				texcoord = make_float3( t1*beta + t2*gamma + t0*(1.0f-beta-gamma) );
			}

			surface_id = v_idx.x; // Not necessarily unique per triangle, but different for each surface
			mat_id = mat;

			rtReportIntersection(0);
		}
	}
}

RT_PROGRAM void mesh_bounds(unsigned int primIdx, float result[6])
{  
	const uint3 v_idx = vindex_buffer[primIdx];

	const float3 v0   = vertex_buffer[ v_idx.x ];
	const float3 v1   = vertex_buffer[ v_idx.y ];
	const float3 v2   = vertex_buffer[ v_idx.z ];
	const float  area = length(cross(v1-v0, v2-v0));

	Aabb* aabb = (Aabb*)result;
  
	if(area > 0.0f && !isinf(area)) {
		aabb->m_min = fminf( fminf( v0, v1), v2 );
		aabb->m_max = fmaxf( fmaxf( v0, v1), v2 );
	} else {
		aabb->invalidate();
	}
}
#endif
