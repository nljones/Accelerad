/*
 *  optix_shader_random.h - shader routines for generating random numbers on GPUs.
 */

#ifndef OPTIX_RANDOM_HEADER
#define OPTIX_RANDOM_HEADER

#include "accelerad_copyright.h"

#define RANDOM
#ifdef RANDOM
#include <curand_kernel.h>

typedef curandState_t rand_state;
#else
typedef float rand_state;
#endif

RT_METHOD void init_rand(rand_state** pointer, const unsigned int& seed);
RT_METHOD float3 getperpendicular(const float3& v, rand_state* state);

#ifdef RANDOM
rtDeclareVariable(unsigned int, random_seed, , ) = 0u; /* random seed to generate different results on each run */

RT_METHOD void init_rand(rand_state** pointer, const unsigned int& seed)
{
	rand_state state;
	*pointer = &state;
	curand_init(seed + random_seed, 0, 0, *pointer);
}
#else /* RANDOM */
RT_METHOD float curand_uniform(rand_state* state);

/* Initialize the non-random number generator with zero. */
RT_METHOD void init_rand(rand_state** pointer, const unsigned int& seed)
{
	rand_state state = 0.0f;
	*pointer = &state;
}

/* Return the value of the non-random number. */
RT_METHOD float curand_uniform(rand_state* state)
{
	return *state;
}
#endif /* RANDOM */

/* Choose random perpedicular direction */
RT_METHOD float3 getperpendicular(const float3& v, rand_state* state)
{
	return optix::normalize(optix::cross(fmaxf(cross_direction(v) * 2.0f - 1.0f,
		make_float3(curand_uniform(state), curand_uniform(state), curand_uniform(state)) - 0.5f), v));
}

#endif /* OPTIX_RANDOM_HEADER */
