/*
 * Copyright (c) 2013-2014 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <stdio.h>
#include <stdlib.h>

#include "optix_world.h"
#include "optix_common.h"

#include "kmeans.h"

//#define PRINT_CUDA

#ifdef __cplusplus
extern "C" {
#endif

// Ambient sample distribution based on Wang et al. (2009) "An efficient GPU-based approach for interactive global illumination"
__device__ inline static
PointDirection average_point_direction(const PointDirection& a, const PointDirection& b, const PointDirection& c, const PointDirection& d)
{
	PointDirection average;
	average.pos = (a.pos + b.pos + c.pos + d.pos) / 4.0f;
	//average.dir = optix::normalize(a.dir + b.dir + c.dir + d.dir);
	average.dir = a.dir + b.dir + c.dir + d.dir;
	const float length = optix::length(average.dir);
	if ( length > 0.0f )
		average.dir /= length;
	return average;
}

__device__ inline static
float geometric_error(const PointDirection& a, const PointDirection& b, const float alpha)
{
	return alpha * optix::length(a.pos - b.pos) + sqrtf(2.0f * fmaxf(1.0f - optix::dot(a.dir, b.dir), 0.0f));
}

__device__ inline static
void reduce(float *error, const int level, const int idX, const int idY, const int width)
{
	int tid = idX + idY * width;
	unsigned int stride = 1u;
	float err = error[tid];

	for (int i = 0; i < level; i++) {
		unsigned int stride2 = stride << 1;
		if (!(idX % stride2) && !(idY % stride2)) {
			err += error[tid + stride];
			err += error[tid + stride * width];
			err += error[tid + stride * (width + 1)];
		
			error[tid] = err;
		}
		stride = stride2;
		__syncthreads();
	}
}

// Ambient sample distribution
__global__ static
void geometric_variation(PointDirection *deviceHits, int *seed,
				   const unsigned int width, const unsigned int height, const unsigned int levels, const float alpha)
{
	extern __shared__ PointDirection blockSharedMemory[];

	unsigned int idX = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int idY = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int tid = idX + idY * width;
	unsigned int sid = blockDim.x * threadIdx.y + threadIdx.x;

	float *err = (float*)malloc(levels * sizeof(float));
	unsigned int stride = 1u;

	PointDirection hit = deviceHits[tid];
	unsigned int valid = optix::dot(hit.dir, hit.dir) > 0.0f && optix::dot(hit.pos, hit.pos) >= 0.0f;
	if (!valid)
		hit.pos.x = hit.pos.y = hit.pos.z = hit.dir.x = hit.dir.y = hit.dir.z = 0.0f;
	PointDirection accum = hit;
	blockSharedMemory[sid] = hit;

	__syncthreads();

	/* Calculate geometric error for each hit point to each quad-tree node. */
	for (int i = 0; i < levels; i++) {
		unsigned int stride2 = stride << 1;

		if (!(idX % stride2) && !(idY % stride2)) {
			accum = average_point_direction(
				accum,
				blockSharedMemory[sid + stride],
				blockSharedMemory[sid + stride * width],
				blockSharedMemory[sid + stride * (width + 1)]
			);

			blockSharedMemory[sid] = accum;
		}

		__syncthreads();

		err[i] = valid ? geometric_error(hit, blockSharedMemory[sid - idX % stride2 - (idY % stride2) * blockDim.x], alpha) : 0.0f;
		stride = stride2;
	}

	__syncthreads();

	float *error = (float *)blockSharedMemory;

	for (int i = levels; i--; ) {
		unsigned int stride2 = stride >> 1;

		/* Calculate geometric error average at each quad-tree node. */
		error[tid] = err[i];

		__syncthreads();

		if (i) 
			reduce(error, i, idX, idY, width); // sum errors at this quad tree node

		/* Divide the pool proportinally to error at each quad-tree node. */
		if (!(idX % stride) && !(idY % stride)) {
			float err0 = error[tid];
			float err1 = error[tid + stride2];
			float err2 = error[tid + stride2 * width];
			float err3 = error[tid + stride2 * (width + 1)];
			float errSum = err0 + err1 + err2 + err3;
			int seedSum = seed[tid];
			float scoreSum = errSum > 0.0f ? seedSum / errSum : 0.0f;

			int s[4];
			s[0] = scoreSum * err0 + 0.5f;
			s[1] = scoreSum * err1 + 0.5f;
			s[2] = scoreSum * err2 + 0.5f;
			s[3] = scoreSum * err3 + 0.5f;
			int diff = seedSum - s[0] - s[1] - s[2] - s[3];
			if (diff && errSum > 0.0f) {
				int maxIndex = err0 > err1 ?
								err0 > err2 ?
									err0 > err3 ? 0 : 3 :
									err2 > err3 ? 2 : 3 :
								err1 > err2 ?
									err1 > err3 ? 1 : 3 :
									err2 > err3 ? 2 : 3;
				s[maxIndex] += diff;
			}
			seed[tid]                         = s[0];
			seed[tid + stride2]               = s[1];
			seed[tid + stride2 * width]       = s[2];
			seed[tid + stride2 * (width + 1)] = s[3];
		}

		__syncthreads();

		stride = stride2;
	}

	free(err);
}

static unsigned int __cdecl calc_block_dim(const unsigned int maxThreadsPerBlock, const unsigned int levels)
{
	unsigned int blockDim = 1u;
	unsigned int size = maxThreadsPerBlock << 1;
	while ( size >>= 2 )
		blockDim <<= 1;
	if ( blockDim > (1u << levels) )
		blockDim = 1 << levels;
	return blockDim;
}

/* Score the relative need for an irradiance cache entry at each hit point */
void __cdecl cuda_score_hits(PointDirection *hits, int *seeds, const unsigned int width, const unsigned int height, const float weight, const unsigned int seed_count)
{
	PointDirection *deviceHits;
	int *deviceSeeds;
	
	/* Calculate number of levels */
	unsigned int levels = 0;
	unsigned int size = width > height ? width : height;
	while ( size >>= 1 )
		levels++;
	fprintf(stderr, "Levels: %i\n", levels);

	/* Determine block size */
	cudaDeviceProp deviceProp;
	int deviceNum;
	cudaGetDevice(&deviceNum);
	cudaGetDeviceProperties(&deviceProp, deviceNum);

	/* To support reduction, blockDim *must* be a power of two. */
	const unsigned int blockDim = calc_block_dim(deviceProp.maxThreadsPerBlock, levels);
	const unsigned int blocksX = (width - 1) / blockDim + 1;
	const unsigned int blocksY = (height - 1) / blockDim + 1;
	const unsigned int blockSharedMemorySize = blockDim * blockDim * sizeof(PointDirection);

	if (blockSharedMemorySize > deviceProp.sharedMemPerBlock)
		err("WARNING: Your CUDA hardware has insufficient block shared memory %u (%u needed).\n", deviceProp.sharedMemPerBlock, blockSharedMemorySize);

	const dim3 dimGrid(blocksX, blocksY);
	const dim3 dimBlock(blockDim, blockDim);
	fprintf(stderr, "Block %i x %i, Grid %i x %i, Shared %i\n", blockDim, blockDim, blocksX, blocksY, blockSharedMemorySize);

	/* Allocate memory on the GPU */
	size = width * height;
	checkCuda(cudaMalloc(&deviceHits, size * sizeof(PointDirection)));
	checkCuda(cudaMalloc(&deviceSeeds, size * sizeof(int)));

	/* Copy data to GPU */
	seeds[0] = seed_count;
	fprintf(stderr, "Target total score: %i\n", seed_count);
	checkCuda(cudaMemcpy(deviceHits, hits, size * sizeof(PointDirection), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(deviceSeeds, seeds, sizeof(int), cudaMemcpyHostToDevice)); // transfer only first entry

	/* Run kernel */
	geometric_variation <<< dimGrid, dimBlock, blockSharedMemorySize >>>
			(deviceHits, deviceSeeds, width, height, levels, weight);
	
	cudaDeviceSynchronize(); checkLastCudaError();

	/* Copy results from GPU */
	checkCuda(cudaMemcpy(seeds, deviceSeeds, size * sizeof(int), cudaMemcpyDeviceToHost));

	/* Free memory on the GPU */
	checkCuda(cudaFree(deviceHits));
	checkCuda(cudaFree(deviceSeeds));
}


// Ambient sample distribution for large images
__global__ static
void mip_map_hits(PointDirection *deviceHits, PointDirection *deviceMipMap,
				   const unsigned int width, const unsigned int height)
{
	extern __shared__ PointDirection blockSharedMemory[];

	unsigned int idX = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int idY = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int tid = idX + idY * width;
	unsigned int sid = blockDim.x * threadIdx.y + threadIdx.x;
	unsigned int offset = 0u;

	unsigned int stride = 1u;

	PointDirection hit = deviceHits[tid];
	unsigned int valid = optix::dot(hit.dir, hit.dir) > 0.0f && optix::dot(hit.pos, hit.pos) >= 0.0f;
	if (!valid)
		hit.pos.x = hit.pos.y = hit.pos.z = hit.dir.x = hit.dir.y = hit.dir.z = 0.0f;
	PointDirection accum = hit;
	blockSharedMemory[sid] = hit;

	__syncthreads();

	/* Calculate geometric error for each hit point to each quad-tree node. */
	while (stride < blockDim.x) {
#ifdef PRINT_CUDA
		if (!tid)
			printf("mip_map_hits stride=%i, offset=%i, accum=%g,%g,%g\n", stride, offset, accum.pos.x, accum.pos.y, accum.pos.z);
#endif
		unsigned int stride2 = stride << 1;

		if (!(idX % stride2) && !(idY % stride2)) {
			accum = average_point_direction(
				accum,
				blockSharedMemory[sid + stride],
				blockSharedMemory[sid + stride * blockDim.x],
				blockSharedMemory[sid + stride * (blockDim.x + 1)]
			);

			blockSharedMemory[sid] = accum;
			deviceMipMap[offset + (idX + idY * width / stride2) / stride2] = accum;
		}
#ifdef PRINT_CUDA
		if (!tid)
			printf("mip_map_hits accum=%g,%g,%g\n", accum.pos.x, accum.pos.y, accum.pos.z);
#endif

		__syncthreads();

		stride = stride2;
		offset += (width * height) / (stride2 * stride2);
	}
}

__global__ static
void calc_error(PointDirection *devicePointDirections, PointDirection *deviceMipMap, float *error,
				   const unsigned int width, const unsigned int height, const unsigned int levels, float alpha)
{
	unsigned int idX = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int idY = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int tid = idX + idY * width;

	unsigned int stride = 1u;

	PointDirection hit = devicePointDirections[tid];
	unsigned int valid = optix::dot(hit.dir, hit.dir) > 0.0f && optix::dot(hit.pos, hit.pos) >= 0.0f;
	if (!valid)
		hit.pos.x = hit.pos.y = hit.pos.z = hit.dir.x = hit.dir.y = hit.dir.z = 0.0f;

	PointDirection *mipMapLevel = deviceMipMap;

	/* Calculate geometric error for each hit point to each quad-tree node. */
	for (unsigned int i = 0u; i < levels; i++) {
#ifdef PRINT_CUDA
		if (!tid)
			printf("calc_error stride=%i, i=%i, valid=%i\n", stride, i, valid);
#endif
		stride <<= 1;

		error[tid + i * width * height] = valid ? geometric_error(hit, mipMapLevel[idX / stride + (idY / stride) * (width / stride)], alpha) : 0.0f;
		mipMapLevel += (width * height) / (stride * stride);
	}
}

__global__ static
void reduce_error(float *error, const unsigned int width, const unsigned int height, const unsigned int levels, const unsigned int scale)
{
	unsigned int idX = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int idY = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int tid = (idX + idY * width) * scale;

	for (unsigned int j = 1u; j < levels; j++) {
		tid += width * height;
		float err = error[tid];

		unsigned int stride = 1u;

		while (stride < (1 << j) && stride < blockDim.x) {
#ifdef PRINT_CUDA
			if (!(tid % (width * height)))
				printf("reduce_error stride=%i, j=%i, scale=%i, err=%g\n", stride, j, scale, err);
#endif
			unsigned int stride2 = stride << 1;
			if (!(idX % stride2) && !(idY % stride2)) {
				err += error[tid + stride * scale];
				err += error[tid + stride * scale * width];
				err += error[tid + stride * scale * (width + 1)];
		
				error[tid] = err;
			}
			stride = stride2;
			__syncthreads();
		}
	}
}

__global__ static
void calc_score(float *error, int *seed, const unsigned int width, const unsigned int height, const unsigned int levels, const unsigned int scale)
{
	unsigned int idX = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int idY = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int tid = (idX + idY * width) * scale;

	unsigned int stride = 1 << levels;

	for (int i = levels; i--; ) {
		unsigned int stride2 = stride >> 1;

		/* Divide the pool proportinally to error at each quad-tree node. */
		if (!(idX % stride) && !(idY % stride)) {
			unsigned int lid = tid + width * height * i;
			float err0 = error[lid];
			float err1 = error[lid + stride2 * scale];
			float err2 = error[lid + stride2 * scale * width];
			float err3 = error[lid + stride2 * scale * (width + 1)];
			float errSum = err0 + err1 + err2 + err3;
			int seedSum = seed[tid];
			float scoreSum = errSum > 0.0f ? seedSum / errSum : 0.0f;

			int s[4];
			s[0] = scoreSum * err0 + 0.5f;
			s[1] = scoreSum * err1 + 0.5f;
			s[2] = scoreSum * err2 + 0.5f;
			s[3] = scoreSum * err3 + 0.5f;
			int diff = seedSum - s[0] - s[1] - s[2] - s[3];
#ifdef PRINT_CUDA
			if (!tid)
				printf("calc_score stride=%i, i=%i, lid=%i, scale=%i, errSum=%g, seedSum=%i, scoreSum=%g, diff=%i\n", stride, i, lid, scale, errSum, seedSum, scoreSum, diff);
#endif
			if (diff && errSum > 0.0f) {
				int maxIndex = err0 > err1 ?
								err0 > err2 ?
									err0 > err3 ? 0 : 3 :
									err2 > err3 ? 2 : 3 :
								err1 > err2 ?
									err1 > err3 ? 1 : 3 :
									err2 > err3 ? 2 : 3;
				s[maxIndex] += diff;
			}
			seed[tid]                                 = s[0];
			seed[tid + stride2 * scale]               = s[1];
			seed[tid + stride2 * scale * width]       = s[2];
			seed[tid + stride2 * scale * (width + 1)] = s[3];
		}

		__syncthreads();

		stride = stride2;
	}
}

/* Calculate average of hits at each quad tree node */
static void __cdecl cuda_mip_map_hits_recursive(PointDirection *deviceHits, PointDirection *deviceMipMap,
	const unsigned int width, const unsigned int height, const unsigned int levels, const unsigned int maxThreadsPerBlock, dim3 dimGrid, dim3 dimBlock, unsigned int blockSharedMemorySize)
{
	/* Calculate average of hits at each quad tree node */
	mip_map_hits <<< dimGrid, dimBlock, blockSharedMemorySize >>>
			(deviceHits, deviceMipMap, width, height);

	cudaDeviceSynchronize(); checkLastCudaError();

	if ( dimBlock.x < (1u << levels) ) {
		unsigned int complete = 1u;
		unsigned int offset = 0u;
		unsigned int nextOffset = (width * height) >> 2; // TODO assumes size is a power of 4
		for (unsigned int i = 1u; i < dimBlock.x / 2u; i <<= 1) {
			complete++;
			offset += nextOffset;
			nextOffset >>= 2;
		}

		const unsigned int blockDim = calc_block_dim(maxThreadsPerBlock, levels - complete);
		const unsigned int blocksX = (dimGrid.x - 1) / blockDim + 1;
		const unsigned int blocksY = (dimGrid.y - 1) / blockDim + 1;
		const dim3 dimSuperGrid(blocksX, blocksY);
		const dim3 dimSuperBlock(blockDim, blockDim);

		cuda_mip_map_hits_recursive(deviceMipMap + offset, deviceMipMap + offset + nextOffset, dimGrid.x, dimGrid.y, levels - complete, maxThreadsPerBlock, dimSuperGrid, dimSuperBlock, dimSuperBlock.x * dimSuperBlock.y * sizeof(PointDirection));
	}
}

/* Calculate average geometric variation for each quad tree node */
static void __cdecl cuda_score_hits_recursive(float *deviceError, int *deviceSeeds,
	const unsigned int width, const unsigned int height, unsigned int levels, const unsigned int scale, const unsigned int maxThreadsPerBlock, dim3 dimGrid, dim3 dimBlock)
{
	/* Perform reduction on error */
	reduce_error <<< dimGrid, dimBlock >>>
			(deviceError, width, height, levels, scale);

	cudaDeviceSynchronize(); checkLastCudaError();

	/* Recruse if block not large enough for reduction */
	if ( dimBlock.x < (1u << levels) ) {
		unsigned int complete = 0u;
		for (unsigned int i = 1u; i < dimBlock.x; i <<= 1)
			complete++;

		const unsigned int blockDim = calc_block_dim(maxThreadsPerBlock, levels - complete);
		const unsigned int blocksX = (dimGrid.x - 1) / blockDim + 1;
		const unsigned int blocksY = (dimGrid.y - 1) / blockDim + 1;
		const dim3 dimSuperGrid(blocksX, blocksY);
		const dim3 dimSuperBlock(blockDim, blockDim);

		cuda_score_hits_recursive(deviceError + width * height * complete, deviceSeeds, width, height, levels - complete, scale * dimBlock.x, maxThreadsPerBlock, dimSuperGrid, dimSuperBlock);
		levels = complete;
	}

	/* Calculate score for each leaf node based on error */
	calc_score <<< dimGrid, dimBlock >>>
			(deviceError, deviceSeeds, width, height, levels, scale);

	cudaDeviceSynchronize(); checkLastCudaError();
}

/* Score the relative need for an irradiance cache entry at each hit point */
void __cdecl cuda_score_hits_big(PointDirection *hits, int *seeds, const unsigned int width, const unsigned int height, const float weight, const unsigned int seed_count)
{
	PointDirection *deviceHits, *deviceMipMap;
	float *deviceError;
	int *deviceSeeds;
	
	/* Calculate number of levels */
	unsigned int levels = 0;
	unsigned int size = width > height ? width : height;
	while ( size >>= 1 )
		levels++;
	fprintf(stderr, "Levels: %i\n", levels);

	/* Determine block size */
	cudaDeviceProp deviceProp;
	int deviceNum;
	cudaGetDevice(&deviceNum);
	cudaGetDeviceProperties(&deviceProp, deviceNum);

	/* To support reduction, blockDim *must* be a power of two. */
	const unsigned int blockDim = calc_block_dim(deviceProp.maxThreadsPerBlock, levels);
	const unsigned int blocksX = (width - 1) / blockDim + 1;
	const unsigned int blocksY = (height - 1) / blockDim + 1;
	const unsigned int blockSharedMemorySize = blockDim * blockDim * sizeof(PointDirection);

	if (blockSharedMemorySize > deviceProp.sharedMemPerBlock)
		err("WARNING: Your CUDA hardware has insufficient block shared memory %u (%u needed).\n", deviceProp.sharedMemPerBlock, blockSharedMemorySize);

	const dim3 dimGrid(blocksX, blocksY);
	const dim3 dimBlock(blockDim, blockDim);
	fprintf(stderr, "Block %i x %i, Grid %i x %i, Shared %i, Weight %g\n", blockDim, blockDim, blocksX, blocksY, blockSharedMemorySize, weight);

	/* Allocate memory on the GPU */
	size = width * height;
	checkCuda(cudaMalloc(&deviceHits, size * sizeof(PointDirection)));
	checkCuda(cudaMalloc(&deviceMipMap, size * sizeof(PointDirection) / 3u)); // Storage requirement for mip map is 1/3 or original data
	checkCuda(cudaMalloc(&deviceError, size * levels * sizeof(float)));

	/* Copy data to GPU */
	checkCuda(cudaMemcpy(deviceHits, hits, size * sizeof(PointDirection), cudaMemcpyHostToDevice));

	/* Calculate average of hits at each quad tree node */
	cuda_mip_map_hits_recursive(deviceHits, deviceMipMap, width, height, levels, deviceProp.maxThreadsPerBlock, dimGrid, dimBlock, blockSharedMemorySize);

	/* Calculate geometric variation at each quad tree node */
	calc_error <<< dimGrid, dimBlock >>>
			(deviceHits, deviceMipMap, deviceError, width, height, levels, weight);

	cudaDeviceSynchronize(); checkLastCudaError();

	/* Free memory on the GPU */
	checkCuda(cudaFree(deviceHits));
	checkCuda(cudaFree(deviceMipMap));

	/* Allocate memory on the GPU */
	checkCuda(cudaMalloc(&deviceSeeds, size * sizeof(int)));

	/* Copy data to GPU */
	seeds[0] = seed_count;
	fprintf(stderr, "Target total score: %i\n", seed_count);
	checkCuda(cudaMemcpy(deviceSeeds, seeds, sizeof(int), cudaMemcpyHostToDevice)); // transfer only first entry

	/* Calculate average geometric variation for each quad tree node */
	cuda_score_hits_recursive(deviceError, deviceSeeds, width, height, levels, 1u, deviceProp.maxThreadsPerBlock, dimGrid, dimBlock);

	/* Copy results from GPU */
	checkCuda(cudaMemcpy(seeds, deviceSeeds, size * sizeof(int), cudaMemcpyDeviceToHost));

	/* Free memory on the GPU */
	checkCuda(cudaFree(deviceError));
	checkCuda(cudaFree(deviceSeeds));
}


#ifdef __cplusplus
}
#endif
