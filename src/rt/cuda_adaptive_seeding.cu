/*
 *  cuda_adaptive_seeding.cu - routines for adaptive seeding on GPUs.
 */

#include "accelerad_copyright.h"

#include <stdio.h>
#include <stdlib.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include "kmeans.h"

//#define PRINT_CUDA
#define MULTI_BLOCK

#define VALID_HORIZONTAL	0x01	/* Horizontal neighbor quad tree node is valid. */
#define VALID_VERTICAL		0x10	/* Vertical neighbor quad tree node is valid. */

#ifdef CAP_REGISTERS_PER_THREAD
#include "accelerad.h"
/* This is the maximum number of registers used by any cuda kernel in this in this file,
found by using the flag "-Xptxas -v" to compile in nvcc. This should be updated when
changes are made to the kernels. */
#ifdef RTX
#define REGISTERS_PER_THREAD	36	/* Registers per thread under CUDA 10.0 */
#else
#define REGISTERS_PER_THREAD	23	/* Registers per thread under CUDA 7.5 */
#endif
#endif

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
unsigned int valid_hit(const PointDirection& hit)
{
	return optix::dot(hit.dir, hit.dir) > 0.0f && optix::dot(hit.pos, hit.pos) >= 0.0f;
}

static int CCALL isPowerOfTwo(unsigned int x)
{
  return ((x != 0) && !(x & (x - 1)));
}

static unsigned int CCALL calc_block_dim(const unsigned int maxThreadsPerBlock, const unsigned int levels)
{
	unsigned int blockDim = 1u;
	unsigned int size = maxThreadsPerBlock << 1;
	while ( size >>= 2 )
		blockDim <<= 1;
	if ( blockDim > (1u << levels) )
		blockDim = 1u << levels;
	return blockDim;
}

#ifndef MULTI_BLOCK
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

	PointDirection hit;
	unsigned int valid = idX < width && idY < height;
	if (valid) {
		hit = deviceHits[tid];
		valid = valid_hit(hit);
	}
	if (!valid)
		hit.pos.x = hit.pos.y = hit.pos.z = hit.dir.x = hit.dir.y = hit.dir.z = 0.0f;
	PointDirection accum = hit;
	blockSharedMemory[sid] = hit;
#ifdef PRINT_CUDA
		if (!tid)
			printf("mip_map_hits width=%i, height=%i, accum=%g,%g,%g, %g,%g,%g, valid=%i\n", width, height, accum.pos.x, accum.pos.y, accum.pos.z, accum.dir.x, accum.dir.y, accum.dir.z, valid);
#endif

	__syncthreads();

	/* Calculate geometric error for each hit point to each quad-tree node. */
	for (int i = 0; i < levels; i++) {
#ifdef PRINT_CUDA
		if (!tid)
			printf("mip_map_hits stride=%i, accum=%g,%g,%g\n", stride, accum.pos.x, accum.pos.y, accum.pos.z);
#endif
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
		if (idX < width && idY < height && !(idX % stride) && !(idY % stride)) {
			valid = 0u;
			if (idX + stride2 < width)
				valid |= VALID_HORIZONTAL;
			if (idY + stride2 < height)
				valid |= VALID_VERTICAL;
			float err[4];
			err[0] = error[tid];
			err[1] = (valid2 & VALID_HORIZONTAL) ? error[tid + stride2] : 0.0f;
			err[2] = (valid2 & VALID_VERTICAL) ? error[tid + stride2 * width] : 0.0f;
			err[3] = (valid2 & (VALID_HORIZONTAL | VALID_VERTICAL)) == (VALID_HORIZONTAL | VALID_VERTICAL) ? error[tid + stride2 * (width + 1)] : 0.0f;
			float errSum = err[0] + err[1] + err[2] + err[3];
			int seedSum = seed[tid];
			float scoreSum = errSum > 0.0f ? seedSum / errSum : 0.0f;

			int s[4];
			s[0] = scoreSum * err[0];
			s[1] = scoreSum * err[1];
			s[2] = scoreSum * err[2];
			s[3] = scoreSum * err[3];
			int diff = seedSum - s[0] - s[1] - s[2] - s[3];
#ifdef PRINT_CUDA
			if (!tid)
				printf("calc_score stride=%i, i=%i, errSum=%g, seedSum=%i, scoreSum=%g, diff=%i\n", stride, i, errSum, seedSum, scoreSum, diff);
#endif
			if (diff && errSum > 0.0f) {
				float max[3] = { 0.0f, 0.0f, 0.0f }; // Will store up to 3 maximum values in err[]
				int maxi[3] = { -1, -1, -1 }; // Will store the indices of up to 3 maximum values in err[]
				for (int j = 0; j < 4; j++) { // Find 3 largest values
					if (err[j] > max[0]) {
						max[2] = max[1]; maxi[2] = maxi[1];
						max[1] = max[0]; maxi[1] = maxi[0];
						max[0] = err[j]; maxi[0] = j;
					}
					else if (err[j] > max[1]) {
						max[2] = max[1]; maxi[2] = maxi[1];
						max[1] = err[j]; maxi[1] = j;
					}
					else if (err[j] > max[2]) {
						max[2] = err[j]; maxi[2] = j;
					}
				}
				if (diff > 2 && max[2] > 0.0f) {
					s[maxi[2]] += 1;
					diff -= 1;
				}
				if (diff > 1 && max[1] > 0.0f) {
					s[maxi[1]] += 1;
					diff -= 1;
				}
				if (diff && max[0] > 0.0f) {
					s[maxi[0]] += diff;
				}
			}

			seed[tid] = s[0];
			if (valid & VALID_HORIZONTAL)
				seed[tid + stride2] = s[1];
			if (valid & VALID_VERTICAL) {
				seed[tid + stride2 * width] = s[2];
				if (valid & VALID_HORIZONTAL)
					seed[tid + stride2 * (width + 1)] = s[3];
			}
		}

		__syncthreads();

		stride = stride2;
	}

	free(err);
}
#else /* MULTI_BLOCK */

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
	unsigned int levelWidth = width;
	unsigned int levelHeight = height;

	PointDirection hit;
	unsigned int valid = idX < width && idY < height;
	if (valid) {
		hit = deviceHits[tid];
		valid = valid_hit(hit);
	}
	if (!valid)
		hit.pos.x = hit.pos.y = hit.pos.z = hit.dir.x = hit.dir.y = hit.dir.z = 0.0f;
	PointDirection accum = hit;
	blockSharedMemory[sid] = hit;
#ifdef PRINT_CUDA
		if (!tid)
			printf("mip_map_hits width=%i, height=%i, accum=%g,%g,%g, %g,%g,%g, valid=%i\n", width, height, accum.pos.x, accum.pos.y, accum.pos.z, accum.dir.x, accum.dir.y, accum.dir.z, valid);
#endif

	__syncthreads();

	/* Calculate geometric error for each hit point to each quad-tree node. */
	while (stride < blockDim.x) {
#ifdef PRINT_CUDA
		if (!tid)
			printf("mip_map_hits stride=%i, offset=%i, accum=%g,%g,%g\n", stride, offset, accum.pos.x, accum.pos.y, accum.pos.z);
#endif
		unsigned int stride2 = stride << 1;
		levelWidth = (levelWidth - 1) / 2 + 1;
		levelHeight = (levelHeight - 1) / 2 + 1;

		if (!(idX % stride2) && !(idY % stride2)) {
			accum = average_point_direction(
				accum,
				blockSharedMemory[sid + stride],
				blockSharedMemory[sid + stride * blockDim.x],
				blockSharedMemory[sid + stride * (blockDim.x + 1)]
			);

			blockSharedMemory[sid] = accum;
			deviceMipMap[offset + (idX + idY * levelWidth) / stride2] = accum;
		}
#ifdef PRINT_CUDA
		if (!tid)
			printf("mip_map_hits width=%i, height=%i, accum=%g,%g,%g\n", levelWidth, levelHeight, accum.pos.x, accum.pos.y, accum.pos.z);
#endif

		__syncthreads();

		stride = stride2;
		offset += levelWidth * levelHeight;
	}
}

__global__ static
void calc_error(PointDirection *deviceHits, PointDirection *deviceMipMap, float *error,
				   const unsigned int width, const unsigned int height, const unsigned int levels, float alpha)
{
	unsigned int idX = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int idY = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int tid = idX + idY * width;

	unsigned int stride = 1u;
	unsigned int levelWidth = width;
	unsigned int levelHeight = height;

	if (idX < width && idY < height) {
		PointDirection hit = deviceHits[tid];
		unsigned int valid = valid_hit(hit);
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
			levelWidth = (levelWidth - 1) / 2 + 1;
			levelHeight = (levelHeight - 1) / 2 + 1;

			error[tid + i * width * height] = valid ? geometric_error(hit, mipMapLevel[idX / stride + idY / stride * levelWidth], alpha) : 0.0f;
			mipMapLevel += levelWidth * levelHeight;
		}
	}
}

__global__ static
void reduce_error(float *error, const unsigned int width, const unsigned int height, const unsigned int levels, const unsigned int scale)
{
	unsigned int idX = scale * (blockDim.x * blockIdx.x + threadIdx.x);
	unsigned int idY = scale * (blockDim.y * blockIdx.y + threadIdx.y);
	unsigned int tid = idX + idY * width;
	unsigned int valid = idX < width && idY < height; 

	for (unsigned int j = 1u; j < levels; j++) {
		tid += width * height;
		float err = valid ? error[tid] : 0.0f;

		unsigned int stride = scale;

		while (stride < (scale << j) && stride < blockDim.x * scale) {
#ifdef PRINT_CUDA
			if (!(tid % (width * height)))
				printf("reduce_error stride=%i, j=%i, scale=%i, err=%g\n", stride, j, scale, err);
#endif
			unsigned int stride2 = stride << 1;
			if (valid && !(idX % stride2) && !(idY % stride2)) {
				if (idX + stride < width)
					err += error[tid + stride];
				if (idY + stride < height) {
					err += error[tid + stride * width];
					if (idX + stride < width)
						err += error[tid + stride * (width + 1)];
				}
		
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
	unsigned int idX = scale * (blockDim.x * blockIdx.x + threadIdx.x);
	unsigned int idY = scale * (blockDim.y * blockIdx.y + threadIdx.y);
	unsigned int tid = idX + idY * width;
	unsigned int valid = idX < width && idY < height; 

	unsigned int stride = scale << levels;

	for (int i = levels; i--; ) {
		unsigned int stride2 = stride >> 1;

		/* Divide the pool proportinally to error at each quad-tree node. */
		if (valid && !(idX % stride) && !(idY % stride)) {
			unsigned int valid2 = 0u;
			if (idX + stride2 < width)
				valid2 |= VALID_HORIZONTAL;
			if (idY + stride2 < height)
				valid2 |= VALID_VERTICAL;
			unsigned int lid = tid + width * height * i;
			float err[4];
			err[0] = error[lid];
			err[1] = (valid2 & VALID_HORIZONTAL) ? error[lid + stride2] : 0.0f;
			err[2] = (valid2 & VALID_VERTICAL) ? error[lid + stride2 * width] : 0.0f;
			err[3] = (valid2 & (VALID_HORIZONTAL | VALID_VERTICAL)) == (VALID_HORIZONTAL | VALID_VERTICAL) ? error[lid + stride2 * (width + 1)] : 0.0f;
			float errSum = err[0] + err[1] + err[2] + err[3];
			int seedSum = seed[tid];
			float scoreSum = errSum > 0.0f ? seedSum / errSum : 0.0f;

			int s[4];
			s[0] = scoreSum * err[0];
			s[1] = scoreSum * err[1];
			s[2] = scoreSum * err[2];
			s[3] = scoreSum * err[3];
			int diff = seedSum - s[0] - s[1] - s[2] - s[3];
#ifdef PRINT_CUDA
			if (!tid)
				printf("calc_score stride=%i, i=%i, tid=%i, lid=%i, scale=%i, errSum=%g, seedSum=%i, scoreSum=%g, diff=%i\n", stride, i, tid, lid, scale, errSum, seedSum, scoreSum, diff);
#endif
			if (diff && errSum > 0.0f) {
				float max[3] = { 0.0f, 0.0f, 0.0f }; // Will store up to 3 maximum values in err[]
				int maxi[3] = { -1, -1, -1 }; // Will store the indices of up to 3 maximum values in err[]
				for (int j = 0; j < 4; j++) { // Find 3 largest values
					if (err[j] > max[0]) {
						max[2] = max[1]; maxi[2] = maxi[1];
						max[1] = max[0]; maxi[1] = maxi[0];
						max[0] = err[j]; maxi[0] = j;
					}
					else if (err[j] > max[1]) {
						max[2] = max[1]; maxi[2] = maxi[1];
						max[1] = err[j]; maxi[1] = j;
					}
					else if (err[j] > max[2]) {
						max[2] = err[j]; maxi[2] = j;
					}
				}
				if (diff > 2 && max[2] > 0.0f) {
					s[maxi[2]] += 1;
					diff -= 1;
				}
				if (diff > 1 && max[1] > 0.0f) {
					s[maxi[1]] += 1;
					diff -= 1;
				}
				if (diff && max[0] > 0.0f) {
					s[maxi[0]] += diff;
				}
			}

			seed[tid] = s[0];
			if (valid2 & VALID_HORIZONTAL)
				seed[tid + stride2] = s[1];
			if (valid2 & VALID_VERTICAL) {
				seed[tid + stride2 * width] = s[2];
				if (valid2 & VALID_HORIZONTAL)
					seed[tid + stride2 * (width + 1)] = s[3];
			}
		}

		__syncthreads();

		stride = stride2;
	}
}

/* Calculate average of hits at each quad tree node */
static void CCALL cuda_mip_map_hits_recursive(PointDirection *deviceHits, PointDirection *deviceMipMap,
	const unsigned int width, const unsigned int height, const unsigned int levels, const unsigned int maxThreadsPerBlock, dim3 dimGrid, dim3 dimBlock, size_t blockSharedMemorySize)
{
	/* Calculate average of hits at each quad tree node */
	mip_map_hits <<< dimGrid, dimBlock, blockSharedMemorySize >>>
			(deviceHits, deviceMipMap, width, height);

	cudaDeviceSynchronize(); checkLastCudaError();

	if ( dimBlock.x < (1u << levels) ) {
		unsigned int complete = 1u;
		unsigned int offset = 0u;
		unsigned int levelWidth = (width - 1) / 2 + 1;
		unsigned int levelHeight = (height - 1) / 2 + 1;
		for (unsigned int i = 1u; i < dimBlock.x / 2u; i <<= 1) {
			complete++;
			offset += levelWidth * levelHeight;
			levelWidth = (levelWidth - 1) / 2 + 1;
			levelHeight = (levelHeight - 1) / 2 + 1;
		}

		const unsigned int blockDim = calc_block_dim(maxThreadsPerBlock, levels - complete);
		const unsigned int blocksX = (levelWidth - 1) / blockDim + 1;
		const unsigned int blocksY = (levelHeight - 1) / blockDim + 1;
		const dim3 dimSuperGrid(blocksX, blocksY);
		const dim3 dimSuperBlock(blockDim, blockDim);

#ifdef PRINT_CUDA
		fprintf(stderr, "cuda_mip_map_hits_recursive: offset %i, width %i, height %i, levels %i\n", offset, levelWidth, levelHeight, levels - complete);
#endif

		cuda_mip_map_hits_recursive(deviceMipMap + offset, deviceMipMap + offset + levelWidth * levelHeight,
			levelWidth, levelHeight, levels - complete, maxThreadsPerBlock, dimSuperGrid, dimSuperBlock, dimSuperBlock.x * dimSuperBlock.y * sizeof(PointDirection));
	}
}

/* Calculate average geometric variation for each quad tree node */
static void CCALL cuda_score_hits_recursive(float *deviceError, int *deviceSeeds,
	const unsigned int width, const unsigned int height, unsigned int levels, const unsigned int scale, const unsigned int maxThreadsPerBlock, dim3 dimGrid, dim3 dimBlock)
{
	/* Perform reduction on error */
	reduce_error <<< dimGrid, dimBlock, 0 >>>
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
	calc_score <<< dimGrid, dimBlock, 0 >>>
			(deviceError, deviceSeeds, width, height, levels, scale);

	cudaDeviceSynchronize(); checkLastCudaError();
}
#endif /* MULTI_BLOCK */

/* Score the relative need for an irradiance cache entry at each hit point */
void CCALL cuda_score_hits(PointDirection *hits, int *seeds, const unsigned int width, const unsigned int height, const float weight, const unsigned int seed_count)
{
	PointDirection *deviceHits;
#ifdef MULTI_BLOCK
	PointDirection *deviceMipMap;
	float *deviceError;
#endif
	int *deviceSeeds;
	
	/* Calculate number of levels */
	unsigned int levels = 0;
	unsigned int size = width > height ? width : height;
	if ( !isPowerOfTwo(size) )
		levels++;
	while ( size >>= 1 )
		levels++;

	/* Determine block size */
	cudaDeviceProp deviceProp;
	int deviceNum;
	cudaGetDevice(&deviceNum);
	cudaGetDeviceProperties(&deviceProp, deviceNum);

#ifdef CAP_REGISTERS_PER_THREAD
	const unsigned int registersPerBlock = deviceProp.regsPerBlock;
	unsigned int threadsPerBlock = deviceProp.maxThreadsPerBlock;
	while (registersPerBlock / threadsPerBlock < REGISTERS_PER_THREAD)
		threadsPerBlock >>= 1;
#else
	const unsigned int threadsPerBlock = deviceProp.maxThreadsPerBlock;
#endif

	/* To support reduction, blockDim *must* be a power of two. */
	const unsigned int blockDim = calc_block_dim(threadsPerBlock, levels);
	const unsigned int blocksX = (width - 1) / blockDim + 1;
	const unsigned int blocksY = (height - 1) / blockDim + 1;
	const size_t blockSharedMemorySize = blockDim * blockDim * sizeof(PointDirection);

#ifndef MULTI_BLOCK
	if (blocksX != 1u || blocksY != 1u)
		err("Your CUDA hardware has insufficient block size %u threads (%u x %u blocks needed). Recompile with MULTI_BLOCK flag.", deviceProp.maxThreadsPerBlock, blocksX, blocksY);
#endif
	if (blockSharedMemorySize > deviceProp.sharedMemPerBlock)
		err("Your CUDA hardware has insufficient block shared memory %" PRIu64 " (%" PRIu64 " needed).", deviceProp.sharedMemPerBlock, blockSharedMemorySize);

	const dim3 dimGrid(blocksX, blocksY);
	const dim3 dimBlock(blockDim, blockDim);
#ifdef PRINT_CUDA
	fprintf(stderr, "Adaptive sampling: Block %i x %i, Grid %i x %i, Shared %i, Levels %i, Weight %g\n", blockDim, blockDim, blocksX, blocksY, blockSharedMemorySize, levels, weight);
#endif

	/* Allocate memory and copy hits to the GPU */
	size = width * height;
	checkCuda(cudaMalloc(&deviceHits, size * sizeof(PointDirection)));
	checkCuda(cudaMemcpy(deviceHits, hits, size * sizeof(PointDirection), cudaMemcpyHostToDevice));

#ifdef MULTI_BLOCK
	/* Allocate memory on the GPU */
	unsigned int mipMapSize = 0u;
	unsigned int levelWidth = width;
	unsigned int levelHeight = height;
	while (levelWidth > 1u || levelHeight > 1u) {
		levelWidth = (levelWidth - 1) / 2 + 1;
		levelHeight = (levelHeight - 1) / 2 + 1;
		mipMapSize += levelWidth * levelHeight;
	}
	checkCuda(cudaMalloc(&deviceMipMap, mipMapSize * sizeof(PointDirection))); // Storage requirement for mip map is 1/3 or original data
	checkCuda(cudaMalloc(&deviceError, size * levels * sizeof(float)));

	/* Calculate average of hits at each quad tree node */
	cuda_mip_map_hits_recursive(deviceHits, deviceMipMap, width, height, levels, threadsPerBlock, dimGrid, dimBlock, blockSharedMemorySize);

	/* Calculate geometric variation at each quad tree node */
	calc_error <<< dimGrid, dimBlock, 0 >>>
			(deviceHits, deviceMipMap, deviceError, width, height, levels, weight);

	cudaDeviceSynchronize(); checkLastCudaError();

	/* Free memory on the GPU */
	checkCuda(cudaFree(deviceHits));
	checkCuda(cudaFree(deviceMipMap));
#endif /* MULTI_BLOCK */

	/* Allocate memory and copy first seed to the GPU */
	seeds[0] = seed_count;
#ifdef PRINT_CUDA
	fprintf(stderr, "Target total score: %i\n", seed_count);
#endif
	checkCuda(cudaMalloc(&deviceSeeds, size * sizeof(int)));
	checkCuda(cudaMemcpy(deviceSeeds, seeds, sizeof(int), cudaMemcpyHostToDevice)); // transfer only first entry

#ifdef MULTI_BLOCK
	/* Calculate average geometric variation for each quad tree node */
	cuda_score_hits_recursive(deviceError, deviceSeeds, width, height, levels, 1u, threadsPerBlock, dimGrid, dimBlock);

	/* Free memory on the GPU */
	checkCuda(cudaFree(deviceError));
#else /* MULTI_BLOCK */
	/* Run kernel */
	geometric_variation <<< dimGrid, dimBlock, blockSharedMemorySize >>>
			(deviceHits, deviceSeeds, width, height, levels, weight);
	
	cudaDeviceSynchronize(); checkLastCudaError();

	/* Free memory on the GPU */
	checkCuda(cudaFree(deviceHits));
#endif /* MULTI_BLOCK */

	/* Copy results from GPU and free memory */
	checkCuda(cudaMemcpy(seeds, deviceSeeds, size * sizeof(int), cudaMemcpyDeviceToHost));
	checkCuda(cudaFree(deviceSeeds));
}

static void printDevProp(const cudaDeviceProp *devProp)
{
	fprintf(stderr, "Revision number:                    %d.%d\n", devProp->major, devProp->minor);
	fprintf(stderr, "Name:                               %s\n", devProp->name);
	fprintf(stderr, "Total global memory:                %" PRIu64 " bytes\n", devProp->totalGlobalMem);
	fprintf(stderr, "Total constant memory:              %" PRIu64 " bytes\n", devProp->totalConstMem);
	fprintf(stderr, "L2 cache size:                      %u bytes\n", devProp->l2CacheSize);
	fprintf(stderr, "Maximum threads per block:          %d\n", devProp->maxThreadsPerBlock);
	fprintf(stderr, "Shared memory per block:            %" PRIu64 " bytes\n", devProp->sharedMemPerBlock);
	fprintf(stderr, "Registers per block:                %d\n", devProp->regsPerBlock);
	fprintf(stderr, "Maximum threads per multiprocessor: %d\n", devProp->maxThreadsPerMultiProcessor);
	fprintf(stderr, "Shared mem per multiprocessor:      %" PRIu64 " bytes\n", devProp->sharedMemPerMultiprocessor);
	fprintf(stderr, "Registers per multiprocessor:       %d\n", devProp->regsPerMultiprocessor);
	fprintf(stderr, "Warp size:                          %d\n", devProp->warpSize);
	fprintf(stderr, "Maximum memory pitch:               %" PRIu64 " bytes\n", devProp->memPitch);
	for (int i = 0; i < 3; ++i)
		fprintf(stderr, "Maximum dimension %d of block:       %d\n", i, devProp->maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		fprintf(stderr, "Maximum dimension %d of grid:        %d\n", i, devProp->maxGridSize[i]);
	fprintf(stderr, "Global memory bus width:            %d bits\n", devProp->memoryBusWidth);
	fprintf(stderr, "Peak memory clock frequency:        %d kHz\n", devProp->memoryClockRate);
	fprintf(stderr, "Clock rate:                         %d kHz\n", devProp->clockRate);
	fprintf(stderr, "Texture alignment:                  %" PRIu64 "\n", devProp->textureAlignment);
	fprintf(stderr, "Texture pitch alignment:            %" PRIu64 "\n", devProp->texturePitchAlignment);
	fprintf(stderr, "Concurrent kernels:                 %s\n", devProp->concurrentKernels ? "Yes" : "No");
	fprintf(stderr, "Concurrent copy and execution:      %s\n", devProp->deviceOverlap ? "Yes" : "No");
	fprintf(stderr, "Number of async engines:            %d\n", devProp->asyncEngineCount);
	fprintf(stderr, "Number of multiprocessors:          %d\n", devProp->multiProcessorCount);
	fprintf(stderr, "Kernel execution timeout:           %s\n", devProp->kernelExecTimeoutEnabled ? "Yes" : "No");
	fprintf(stderr, "Unified addressing with host:       %s\n", devProp->unifiedAddressing ? "Yes" : "No");
	fprintf(stderr, "Device can map host memory:         %s\n", devProp->canMapHostMemory ? "Yes" : "No");
	fprintf(stderr, "Device supports managed memory:     %s\n", devProp->managedMemory ? "Yes" : "No");
	return;
}
 
void printCUDAProp()
{
	// Number of CUDA devices
	int devCount;
	cudaGetDeviceCount(&devCount);
	fprintf(stderr, "CUDA Device Query...\n");
	fprintf(stderr, "There are %d CUDA devices.\n", devCount);
 
	// Iterate through devices
	for (int i = 0; i < devCount; ++i)
	{
		// Get device properties
		fprintf(stderr, "\nCUDA Device #%d\n", i);
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		printDevProp(&devProp);
	}
}

#ifdef __cplusplus
}
#endif
