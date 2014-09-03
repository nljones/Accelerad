/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         kmeans.h   (an OpenMP version)                            */
/*   Description:  header file for a simple k-means clustering program       */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department Northwestern University                         */
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

#ifndef _H_KMEANS
#define _H_KMEANS

//#define BLOCK_SHARED_MEM_OPTIMIZATION 1

#define IC_WEIGHT
#define RETURN_DISTANCE

#include <assert.h>

typedef unsigned int COUNTER;

#define msg(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); } while (0)
#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

#define malloc2D(name, xDim, yDim, type) do {               \
	name = (type **)malloc(xDim * sizeof(type *));          \
	assert(name != NULL);                                   \
	name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
	assert(name[0] != NULL);                                \
	for (size_t i = 1; i < xDim; i++)                       \
		name[i] = name[i-1] + yDim;                         \
} while (0)

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDACC__
//inline void checkCuda(cudaError_t e) {
//	if (e != cudaSuccess) {
//		// cudaGetErrorString() isn't always very helpful. Look up the error
//		// number in the cudaError enum in driver_types.h in the CUDA includes
//		// directory for a better explanation.
//		err("CUDA Error %d: %s\n(%s:%d)\n", e, cudaGetErrorString(e), __FILE__, __LINE__);
//	}
//}
//
//inline void checkLastCudaError() {
//	checkCuda(cudaGetLastError());
//}

#define checkCuda( func ) do {	\
	cudaError_t e = func;		\
	if (e != cudaSuccess)		\
		err("CUDA Error %d: %s\n(%s:%d)\n", e, cudaGetErrorString(e), __FILE__, __LINE__); } while(0)

#define checkLastCudaError()	\
	checkCuda(cudaGetLastError())
#endif

//float** omp_kmeans(int, float**, int, int, int, float, int*);
//float** seq_kmeans(float**, int, int, int, float, int*, int*);
float** __cdecl cuda_kmeans(float**, int, int, int, int, float, float, int*,
#ifdef RETURN_DISTANCE
	float*,
#endif
	int*);

//float** file_read(int, char*, int*, int*);
//int     file_write(char*, int, int, int, float**, int*);
//
//
//double  wtime(void);
//
//extern int _debug;

void __cdecl cuda_score_hits(PointDirection *hits, int *seeds, const unsigned int width, const unsigned int height, const float weight, const unsigned int seed_count);

#ifdef __cplusplus
}
#endif

#endif
