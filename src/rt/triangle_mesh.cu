
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

using namespace optix;

// This is to be plugged into an RTgeometry object to represent
// a triangle mesh with a vertex buffer of triangle soup (triangle list)
// with an interleaved position, normal, texturecoordinate layout.

/* Program variables */
rtDeclareVariable(unsigned int, backvis, , ); /* backface visibility (bv) */

/* Contex variables */
rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<uint3>  vindex_buffer;    // position indices 
//rtBuffer<uint3>  nindex_buffer;    // normal indices
//rtBuffer<uint3>  tindex_buffer;    // texcoord indices
rtBuffer<int>    material_buffer; // per-face material index
rtBuffer<int2>   material_alt_buffer; // per-material alternate material indices
rtDeclareVariable(unsigned int, radiance_primary_ray_type, , );
rtDeclareVariable(unsigned int, diffuse_primary_ray_type, , ) = ~0;	/* Not always defined */
rtDeclareVariable(unsigned int, shadow_ray_type, , );

/* OptiX variables */
rtDeclareVariable(Ray, ray, rtCurrentRay, );

/* Attributes */
rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(int, surface_id, attribute surface_id, );

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

		int mat = material_buffer[primIdx];
		if ( mat < 0 || mat >= material_alt_buffer.size() ) /* Material void or missing */
			return;
		if (ray.ray_type == radiance_primary_ray_type || ray.ray_type == diffuse_primary_ray_type) /* Lambert material for irradiance calculations */
			mat = material_alt_buffer[mat].x;
		else if (ray.ray_type != shadow_ray_type || ray.tmax - t > ray.tmax * 0.0002f) /* For materials whose type depends on ray type (such as illum and mirror) */
			mat = material_alt_buffer[mat].y;
		if ( mat < 0 || mat >= material_alt_buffer.size() ) /* Material void or missing */
			return;

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

			rtReportIntersection( mat );
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

