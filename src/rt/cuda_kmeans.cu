/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         cuda_kmeans.cu  (CUDA version)                            */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Copyright (c) 2005 Wei-keng Liao
// Copyright (c) 2011 Serban Giuroiu
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// -----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"
#ifdef RANDOM_SEEDS
#include "random.h"
#endif /* RANDOM_SEEDS */

#ifdef CAP_REGISTERS_PER_THREAD
#include "accelerad.h"
/* This is the maximum number of registers used by any cuda kernel in this in this file,
found by using the flag "-Xptxas -v" to compile in nvcc. This should be updated when
changes are made to the kernels. */
#ifdef RTX
#define REGISTERS_PER_THREAD	40	/* Registers per thread under CUDA 10.0 */
#else
#define REGISTERS_PER_THREAD	26	/* Registers per thread under CUDA 7.5 */
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

static inline int nextPowerOfTwo(int n) {
	n--;

	n = n >>  1 | n;
	n = n >>  2 | n;
	n = n >>  4 | n;
	n = n >>  8 | n;
	n = n >> 16 | n;
//  n = n >> 32 | n;    //  For 64-bit ints

	return ++n;
}

#ifdef IC_WEIGHT
/*----< ic_error >-----------------------------------------------------------*/
/* error metric from Wang et al. "An Efficient GPU-based Approach for        */
/* Interactive Global Illumination" Eq. 2                                    */
/* added by Nathaniel Jones 1/23/2014                                        */
__host__ __device__ inline static
float ic_error(int    numCoords,
			   int    numObjs,
			   int    numClusters,
			   float *objects,     // [numCoords][numObjs]
			   float *clusters,    // [numCoords][numClusters]
			   int    objectId,
			   int    clusterId,
			   float  alpha)
{
	int i;
	float ans=0.0f, ans1;

	for (i = 0; i < 3; i++) {
		ans1 = objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId];
		ans += ans1 * ans1;
	}

	ans1=1.0f;
	for ( ; i < numCoords; i++) {
		ans1 -= objects[numObjs * i + objectId] * clusters[numClusters * i + clusterId];
	}
	if (ans1 < 0.0f)
		ans1 = 0.0f;
	return alpha * sqrtf(ans) + sqrtf(2.0f*ans1);
}
#else /* IC_WEIGHT */
/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ inline static
float euclid_dist_2(int    numCoords,
					int    numObjs,
					int    numClusters,
					float *objects,     // [numCoords][numObjs]
					float *clusters,    // [numCoords][numClusters]
					int    objectId,
					int    clusterId)
{
	int i;
	float ans=0.0;

	for (i = 0; i < numCoords; i++) {
		ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
			   (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
	}

	return(ans);
}
#endif /* IC_WEIGHT */

/*----< find_nearest_cluster() >---------------------------------------------*/
__global__ static
void find_nearest_cluster(int numCoords,
						  int numObjs,
						  int numClusters,
#ifdef IC_WEIGHT
						  float alpha,
#endif
						  float *objects,           //  [numCoords][numObjs]
						  float *deviceClusters,    //  [numCoords][numClusters]
						  int *membership,          //  [numObjs]
#ifdef RETURN_DISTANCE
						  float *distance,          //  [numObjs]
#endif
						  unsigned int *intermediates)
{
	extern __shared__ COUNTER sharedMemory[];

	//  The type chosen for membershipChanged must be large enough to support
	//  reductions! There are blockDim.x elements, one for each thread in the
	//  block. See numThreadsPerClusterBlock in cuda_kmeans().
	COUNTER *membershipChanged = (COUNTER *)sharedMemory;
#if BLOCK_SHARED_MEM_OPTIMIZATION
	float *clusters = (float *)(sharedMemory + blockDim.x);
#else
	float *clusters = deviceClusters;
#endif

	membershipChanged[threadIdx.x] = 0;

#if BLOCK_SHARED_MEM_OPTIMIZATION
	//  BEWARE: We can overrun our shared memory here if there are too many
	//  clusters or too many coordinates! For reference, a Tesla C1060 has 16
	//  KiB of shared memory per block, and a GeForce GTX 480 has 48 KiB of
	//  shared memory per block.
	for (int i = threadIdx.x; i < numClusters; i += blockDim.x) {
		for (int j = 0; j < numCoords; j++) {
			clusters[numClusters * j + i] = deviceClusters[numClusters * j + i];
		}
	}
	__syncthreads();
#endif

	int objectId = blockDim.x * blockIdx.x + threadIdx.x;

	if (objectId < numObjs) {
		int   index, i;
		float dist, min_dist;

		/* find the cluster id that has min distance to object */
		index    = 0;
#ifdef IC_WEIGHT
		min_dist = ic_error(numCoords, numObjs, numClusters, objects, clusters, objectId, 0, alpha);
#else
		min_dist = euclid_dist_2(numCoords, numObjs, numClusters, objects, clusters, objectId, 0);
#endif

		for (i=1; i<numClusters; i++) {
#ifdef IC_WEIGHT
			dist = ic_error(numCoords, numObjs, numClusters, objects, clusters, objectId, i, alpha);
#else
			dist = euclid_dist_2(numCoords, numObjs, numClusters, objects, clusters, objectId, i);
#endif
			/* no need square root */
			if (isnan(min_dist) || dist < min_dist) { /* find the min and its array index */
				min_dist = dist;
				index    = i;
			}
		}

		if (membership[objectId] != index) {
			membershipChanged[threadIdx.x] = 1;
		}

		/* assign the membership to object objectId */
		membership[objectId] = index;
#ifdef RETURN_DISTANCE
		distance[objectId] = min_dist;
#endif

		__syncthreads();    //  For membershipChanged[]

		//  blockDim.x *must* be a power of two!
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (threadIdx.x < s) {
				membershipChanged[threadIdx.x] += membershipChanged[threadIdx.x + s];
			}
			__syncthreads();
		}

		if (threadIdx.x == 0) {
			intermediates[blockIdx.x] = membershipChanged[0];
		}
	}
}

__global__ static
void compute_delta(unsigned int *deviceIntermediates,
				   unsigned int numIntermediates)    //  The actual number of intermediates
{
	//  The number of elements in this array should be equal to
	//  numIntermediates2, the number of threads launched. It *must* be a power
	//  of two!
	extern __shared__ unsigned int intermediates[];

	//  Copy global intermediate values into shared memory.
	int objectId = blockDim.x * blockIdx.x + threadIdx.x;
	intermediates[threadIdx.x] = (objectId < numIntermediates) ? deviceIntermediates[objectId] : 0;

	__syncthreads();

	//  blockDim.x *must* be a power of two!
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		deviceIntermediates[blockDim.x * blockIdx.x] = intermediates[0];
	}
}

/*----< cuda_kmeans() >-------------------------------------------------------*/
//
//  ----------------------------------------
//  DATA LAYOUT
//
//  objects         [numObjs][numCoords]
//  clusters        [numClusters][numCoords]
//  dimObjects      [numCoords][numObjs]
//  dimClusters     [numCoords][numClusters]
//  newClusters     [numCoords][numClusters]
//  deviceObjects   [numCoords][numObjs]
//  deviceClusters  [numCoords][numClusters]
//  ----------------------------------------
//
/* return an array of cluster centers of size [numClusters][numCoords]       */
float** CCALL cuda_kmeans(float **objects,      /* in: [numObjs][numCoords] */
						  int     numCoords,    /* no. features */
						  int     numObjs,      /* no. objects */
						  int     numClusters,  /* no. clusters */
						  int     max_iterations,	/* maximum k-means iterations */
						  float   threshold,    /* % objects change membership */
#ifdef IC_WEIGHT
						  float   weight,       /* relative weighting of position */
#endif
#ifdef RANDOM_SEEDS
						  int     randomSeeds,  /* use randomly selected cluster centers (boolean) */
#endif
						  int    *membership,   /* out: [numObjs] */
#ifdef RETURN_DISTANCE
						  float  *distance,     /* out: [numObjs] */
#endif
						  int    *loop_iterations)
{
	int      i, j, index, step, loop=0;
	int     *newClusterSize; /* [numClusters]: no. objects assigned in each
								new cluster */
	float    delta;          /* % of objects change their clusters */
	float  **dimObjects;
	float  **clusters;       /* out: [numClusters][numCoords] */
	float  **dimClusters;
	float  **newClusters;    /* [numCoords][numClusters] */

	float *deviceObjects;
	float *deviceClusters;
	int *deviceMembership;
#ifdef RETURN_DISTANCE
	float *deviceDistance;
#endif
	unsigned int *deviceIntermediates;

	//  Copy objects given in [numObjs][numCoords] layout to new
	//  [numCoords][numObjs] layout
	malloc2D(dimObjects, numCoords, numObjs, float);
	for (i = 0; i < numCoords; i++) {
		for (j = 0; j < numObjs; j++) {
			dimObjects[i][j] = objects[j][i];
		}
	}

	/* pick first numClusters elements of objects[] as initial cluster centers*/
	malloc2D(dimClusters, numCoords, numClusters, float);
	//step = numObjs / numClusters;
	//for (i = 0; i < numCoords; i++) {
	//	for (j = 0; j < numClusters; j++) {
	//		dimClusters[i][j] = dimObjects[i][j * step];
	//	}
	//}
#ifdef RANDOM_SEEDS
	if (randomSeeds)
		for (j = 0; j < numClusters; j++) {
			step = (int)((j + frandom()) * numObjs / numClusters);
			for (i = 0; i < numCoords; i++) {
				dimClusters[i][j] = dimObjects[i][step];
			}
		}
	else
#endif /* RANDOM_SEEDS */
	for (i = 0; i < numCoords; i++) {
		for (j = 0; j < numClusters; j++) {
			dimClusters[i][j] = dimObjects[i][j];
		}
	}

	/* initialize membership[] */
	//for (i=0; i<numObjs; i++) membership[i] = -1;

	/* need to initialize newClusterSize and newClusters[0] to all 0 */
	newClusterSize = (int*) calloc(numClusters, sizeof(int));
	assert(newClusterSize != NULL);

	malloc2D(newClusters, numCoords, numClusters, float);
	memset(newClusters[0], 0, numCoords * numClusters * sizeof(float));

	cudaDeviceProp deviceProp;
	int deviceNum;
	cudaGetDevice(&deviceNum);
	cudaGetDeviceProperties(&deviceProp, deviceNum);

	//  To support reduction, numThreadsPerClusterBlock *must* be a power of
	//  two, and it *must* be no larger than the number of bits that will
	//  fit into an unsigned char, the type used to keep track of membership
	//  changes in the kernel.
#ifdef CAP_REGISTERS_PER_THREAD
	const unsigned int numRegistersPerClusterBlock = deviceProp.regsPerBlock;
	unsigned int numThreadsPerClusterBlock = deviceProp.maxThreadsPerBlock;
	while (numRegistersPerClusterBlock / numThreadsPerClusterBlock < REGISTERS_PER_THREAD)
		numThreadsPerClusterBlock >>= 1;
#else
	const unsigned int numThreadsPerClusterBlock = deviceProp.maxThreadsPerBlock;//128;
#endif
	const unsigned int numClusterBlocks = (numObjs - 1) / numThreadsPerClusterBlock + 1;
#if BLOCK_SHARED_MEM_OPTIMIZATION
	const size_t clusterBlockSharedDataSize = numThreadsPerClusterBlock * sizeof(COUNTER) + numClusters * numCoords * sizeof(float);

	if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
		err("Your CUDA hardware has insufficient block shared memory %llu (%llu needed). "
			"You need to recompile with BLOCK_SHARED_MEM_OPTIMIZATION=0.",
			deviceProp.sharedMemPerBlock, clusterBlockSharedDataSize);
	}
#else
	const size_t clusterBlockSharedDataSize = numThreadsPerClusterBlock * sizeof(COUNTER);
#endif

	const unsigned int numReductionBlocks = (numClusterBlocks - 1) / numThreadsPerClusterBlock + 1;
	const unsigned int numReductionThreads = nextPowerOfTwo(numClusterBlocks / numReductionBlocks); // per block
	//const unsigned int numReductionThreads = nextPowerOfTwo(numClusterBlocks);
	const unsigned int reductionBlockSharedDataSize = numReductionThreads * sizeof(unsigned int);

	checkCuda(cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(float)));
	checkCuda(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(float)));
	checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
#ifdef RETURN_DISTANCE
	checkCuda(cudaMalloc(&deviceDistance, numObjs*sizeof(float)));
#endif
	checkCuda(cudaMalloc(&deviceIntermediates, numReductionBlocks*numReductionThreads*sizeof(unsigned int)));

	checkCuda(cudaMemcpy(deviceObjects, dimObjects[0], numObjs*numCoords*sizeof(float), cudaMemcpyHostToDevice));
	//checkCuda(cudaMemcpy(deviceMembership, membership, numObjs*sizeof(int), cudaMemcpyHostToDevice));
	checkCuda(cudaMemset(deviceMembership, -1, numObjs*sizeof(int)));

	const unsigned int reducedLength = (numReductionBlocks - 1) * numReductionThreads + 1;
	int* reducedSums;
	if (numReductionBlocks > 1u)
		reducedSums = (int*)malloc(reducedLength * sizeof(int));

	do {
		checkCuda(cudaMemcpy(deviceClusters, dimClusters[0], numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice));

		find_nearest_cluster <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
			(numCoords, numObjs, numClusters,
#ifdef IC_WEIGHT
			weight,
#endif
			deviceObjects, deviceClusters, deviceMembership,
#ifdef RETURN_DISTANCE
			deviceDistance,
#endif
			deviceIntermediates);

		cudaDeviceSynchronize(); checkLastCudaError();

		compute_delta <<< numReductionBlocks, numReductionThreads, reductionBlockSharedDataSize >>>
			(deviceIntermediates, numClusterBlocks);

		cudaDeviceSynchronize(); checkLastCudaError();

		if (numReductionBlocks == 1u) {
			int d;
			checkCuda(cudaMemcpy(&d, deviceIntermediates, sizeof(int), cudaMemcpyDeviceToHost));
			delta = (float)d;
		} else {
			checkCuda(cudaMemcpy(reducedSums, deviceIntermediates, reducedLength * sizeof(int), cudaMemcpyDeviceToHost));
			unsigned int reducedSum = 0u;
			for (i=0; i<numReductionBlocks; i++)
				reducedSum += reducedSums[i * numReductionThreads];
			delta = (float)reducedSum;
			//int d;
			//long reductionSum = 0L;
			//for (i=0; i<numReductionBlocks; i++) {
			//	checkCuda(cudaMemcpy(&d, deviceIntermediates + i * numReductionThreads, sizeof(int), cudaMemcpyDeviceToHost));
			//	reductionSum += d;
			//}
			//delta = (float)reductionSum;
		}

		checkCuda(cudaMemcpy(membership, deviceMembership, numObjs*sizeof(int), cudaMemcpyDeviceToHost));

		for (i=0; i<numObjs; i++) {
			/* find the array index of nestest cluster center */
			index = membership[i];

			/* update new cluster centers : sum of objects located within */
			newClusterSize[index]++;
			for (j=0; j<numCoords; j++)
				newClusters[j][index] += objects[i][j];
		}

		//  TODO: Flip the nesting order
		//  TODO: Change layout of newClusters to [numClusters][numCoords]
		/* average the sum and replace old cluster centers with newClusters */
		for (i=0; i<numClusters; i++) {
			for (j=0; j<numCoords; j++) {
				if (newClusterSize[i] > 0)
					dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
				newClusters[j][i] = 0.0;   /* set back to 0 */
			}
			newClusterSize[i] = 0;   /* set back to 0 */
		}

		delta /= numObjs;
	} while (++loop < max_iterations && delta > threshold);

	*loop_iterations = loop;

	/* allocate a 2D space for returning variable clusters[] (coordinates
	   of cluster centers) */
	malloc2D(clusters, numClusters, numCoords, float);
	for (i = 0; i < numClusters; i++) {
		for (j = 0; j < numCoords; j++) {
			clusters[i][j] = dimClusters[j][i];
		}
	}

#ifdef RETURN_DISTANCE
	checkCuda(cudaMemcpy(distance, deviceDistance, numObjs*sizeof(float), cudaMemcpyDeviceToHost));
#endif

	checkCuda(cudaFree(deviceObjects));
	checkCuda(cudaFree(deviceClusters));
	checkCuda(cudaFree(deviceMembership));
	checkCuda(cudaFree(deviceDistance));
#ifdef RETURN_DISTANCE
	checkCuda(cudaFree(deviceIntermediates));
#endif

	free(dimObjects[0]);
	free(dimObjects);
	free(dimClusters[0]);
	free(dimClusters);
	free(newClusters[0]);
	free(newClusters);
	free(newClusterSize);
	if (numReductionBlocks > 1u)
		free(reducedSums);

	return clusters;
}

#ifdef __cplusplus
}
#endif
