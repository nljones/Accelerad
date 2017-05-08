/*
 *  optix_shader_daysim.h - shader routines for generating daylight coefficients on GPUs.
 */

#ifndef OPTIX_DAYSIM_HEADER
#define OPTIX_DAYSIM_HEADER

#include "accelerad_copyright.h"

#ifdef DAYSIM_COMPATIBLE
typedef uint3 DaysimCoef;
typedef float DC;

RT_METHOD DaysimCoef daysimNext(const DaysimCoef& prev);
RT_METHOD void daysimCopy(DC* destin, const DaysimCoef& source);
RT_METHOD void daysimCopy(DaysimCoef& destin, const DaysimCoef& source);
RT_METHOD void daysimSet(DaysimCoef& coef, const DC& value);
RT_METHOD void daysimScale(DaysimCoef& coef, const DC& scaling);
RT_METHOD void daysimAdd(DaysimCoef& result, const DC& value);
RT_METHOD void daysimAdd(DaysimCoef& result, const DaysimCoef& add);
RT_METHOD void daysimMult(DaysimCoef& result, const DaysimCoef& mult);
RT_METHOD void daysimSetCoef(DaysimCoef& result, const unsigned int& index, const DC& value);
RT_METHOD void daysimAddCoef(DaysimCoef& result, const unsigned int& index, const DC& add);
RT_METHOD void daysimAddScaled(DaysimCoef& result, const DC* add, const DC& scaling);
RT_METHOD void daysimAddScaled(DaysimCoef& result, const DaysimCoef& add, const DC& scaling);
RT_METHOD void daysimAssignScaled(DaysimCoef& result, const DaysimCoef& source, const DC& scaling);
RT_METHOD void daysimRunningAverage(DaysimCoef& result, const DaysimCoef& source, const int& count);
RT_METHOD void daysimCheck(const DaysimCoef& daylightCoef, const DC& value, const int& error);

rtDeclareVariable(unsigned int, daylightCoefficients, , ) = 0; /* number of daylight coefficients */
rtBuffer<DC, 3> dc_scratch_buffer; /* scratch space for local storage of daylight coefficients */

#define DC_ptr(index)	&dc_scratch_buffer[index]

RT_METHOD DaysimCoef daysimNext(const DaysimCoef& prev)
{
	DaysimCoef next = prev;
	if (daylightCoefficients) {
		next.x += daylightCoefficients;
		if (next.x >= dc_scratch_buffer.size().x)
			rtThrow(RT_EXCEPTION_INDEX_OUT_OF_BOUNDS); // TODO handle overflow
	}
	return next;
}

/* Copies a daylight coefficient set */
RT_METHOD void daysimCopy(DC* destin, const DaysimCoef& source)
{
	if (daylightCoefficients)
		memcpy(destin, DC_ptr(source), daylightCoefficients * sizeof(DC));
}

/* Copies a daylight coefficient set */
RT_METHOD void daysimCopy(DaysimCoef& destin, const DaysimCoef& source)
{
	if (daylightCoefficients)
		memcpy(DC_ptr(destin), DC_ptr(source), daylightCoefficients * sizeof(DC));
}

/* Initialises all daylight coefficients with 'value' */
RT_METHOD void daysimSet(DaysimCoef& coef, const DC& value)
{
	if (daylightCoefficients) {
		DC* ptr = DC_ptr(coef);

		for (unsigned int i = 0u; i < daylightCoefficients; i++)
			ptr[i] = value;
	}
}

/* Scales the daylight coefficient set by the value 'scaling' */
RT_METHOD void daysimScale(DaysimCoef& coef, const DC& scaling)
{
	if (daylightCoefficients) {
		DC* ptr = DC_ptr(coef);

		for (unsigned int i = 0u; i < daylightCoefficients; i++)
			ptr[i] *= scaling;
	}
}

/* Adds a value to all daylight coefficients */
RT_METHOD void daysimAdd(DaysimCoef& result, const DC& value)
{
	if (daylightCoefficients) {
		DC* ptr = DC_ptr(result);

		for (unsigned int i = 0u; i < daylightCoefficients; i++)
			ptr[i] += value;
	}
}

/* Adds two daylight coefficient sets: result[i] = result[i] + add[i] */
RT_METHOD void daysimAdd(DaysimCoef& result, const DaysimCoef& add)
{
	if (daylightCoefficients) {
		DC* result_prt = DC_ptr(result);
		const DC* add_ptr = DC_ptr(add);

		for (unsigned int i = 0u; i < daylightCoefficients; i++)
			result_prt[i] += add_ptr[i];
	}
}

/* Multiply two daylight coefficient sets: result[i] = result[i] * add[i] */
RT_METHOD void daysimMult(DaysimCoef& result, const DaysimCoef& mult)
{
	if (daylightCoefficients) {
		DC* result_prt = DC_ptr(result);
		const DC* mult_ptr = DC_ptr(mult);

		for (unsigned int i = 0u; i < daylightCoefficients; i++)
			result_prt[i] *= mult_ptr[i];
	}
}

/* Sets the daylight coefficient at position 'index' to 'value' */
RT_METHOD void daysimSetCoef(DaysimCoef& result, const unsigned int& index, const DC& value)
{
	if (index < daylightCoefficients)
		(DC_ptr(result))[index] = value;
}

/* Adds 'value' to the daylight coefficient at position 'index' */
RT_METHOD void daysimAddCoef(DaysimCoef& result, const unsigned int& index, const DC& add)
{
	if (index < daylightCoefficients)
		(DC_ptr(result))[index] += add;
}

/* Adds the elements of 'source' scaled by 'scaling'  to 'result' */
RT_METHOD void daysimAddScaled(DaysimCoef& result, const DC* add, const DC& scaling)
{
	if (daylightCoefficients) {
		DC* ptr = DC_ptr(result);

		for (unsigned int i = 0u; i < daylightCoefficients; i++)
			ptr[i] += add[i] * scaling;
	}
}

/* Adds the elements of 'source' scaled by 'scaling'  to 'result' */
RT_METHOD void daysimAddScaled(DaysimCoef& result, const DaysimCoef& add, const DC& scaling)
{
	if (daylightCoefficients) {
		DC* result_prt = DC_ptr(result);
		const DC* add_ptr = DC_ptr(add);

		for (unsigned int i = 0u; i < daylightCoefficients; i++)
			result_prt[i] += add_ptr[i] * scaling;
	}
}

/* Assign the coefficients of 'source' scaled by 'scaling' to 'result' */
RT_METHOD void daysimAssignScaled(DaysimCoef& result, const DaysimCoef& source, const DC& scaling)
{
	if (daylightCoefficients) {
		DC* result_prt = DC_ptr(result);
		const DC* source_ptr = DC_ptr(source);

		for (unsigned int i = 0u; i < daylightCoefficients; i++)
			result_prt[i] = source_ptr[i] * scaling;
	}
}

/* Perform a running average by roling in 'source' to an average of 'count' elements already in 'result' */
RT_METHOD void daysimRunningAverage(DaysimCoef& result, const DaysimCoef& source, const int& count)
{
	if (daylightCoefficients) {
		DC* result_prt = DC_ptr(result);
		const DC* source_ptr = DC_ptr(source);
		const DC b = 1.0f / (count + 1);
		const DC a = count * b;

		for (unsigned int i = 0u; i < daylightCoefficients; i++)
			result_prt[i] = result_prt[i] * a + source_ptr[i] * b;
	}
}

/* Check that the sum of daylight coefficients equals the red color channel */
RT_METHOD void daysimCheck(const DaysimCoef& daylightCoef, const DC& value, const int& error)
{
	if (!daylightCoefficients)
		return;

	DC* ptr = DC_ptr(daylightCoef);
	DC ratio, sum = 0.0f;

	for (unsigned int k = 0u; k < daylightCoefficients; k++)
		sum += ptr[k];

	if (sum >= value) { /* test whether the sum of daylight coefficients corresponds to value for red */
		if (sum == 0.0f) return;
		ratio = value / sum;
	}
	else {
		if (value == 0.0f) return;
		ratio = sum / value;
	}
	if (ratio < 0.9999f)
		rtThrow(RT_EXCEPTION_USER + error);
}
#endif /* DAYSIM_COMPATIBLE */

#endif /* OPTIX_DAYSIM_HEADER */
