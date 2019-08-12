/*
 *  optix_shader_ambient.h - shader routines for ambient sampling on GPUs.
 */

#pragma once

#include "accelerad_copyright.h"

#define	DCSCALE		11585.2f		/* (1<<13)*sqrt(2) */
#define FXNEG		01
#define FYNEG		02
#define FZNEG		04
#define F1X		010
#define F2Z		020
#define F1SFT		5
#define F2SFT		18
#define FMASK		0x1fff

RT_METHOD int encodedir(const float3& dv);
RT_METHOD float3 decodedir(const int& dc);
RT_METHOD unsigned int quadratic(float2* r, const float& a, const float& b, const float& c);

/* Encode a normalized direction vector. */
RT_METHOD int encodedir(const float3& dv)
{
	int dc = 0;
	int	cd[3], cm;

	if (dv.x < 0.0f) {
		cd[0] = (int)(dv.x * -DCSCALE);
		dc |= FXNEG;
	}
	else
		cd[0] = (int)(dv.x * DCSCALE);
	if (dv.y < 0.0f) {
		cd[1] = (int)(dv.y * -DCSCALE);
		dc |= FYNEG;
	}
	else
		cd[1] = (int)(dv.y * DCSCALE);
	if (dv.z < 0.0f) {
		cd[2] = (int)(dv.z * -DCSCALE);
		dc |= FZNEG;
	}
	else
		cd[2] = (int)(dv.z * DCSCALE);
	if (!(cd[0] | cd[1] | cd[2]))
		return(0);		/* zero normal */
	if (cd[0] <= cd[1]) {
		dc |= F1X | cd[0] << F1SFT;
		cm = cd[1];
	}
	else {
		dc |= cd[1] << F1SFT;
		cm = cd[0];
	}
	if (cd[2] <= cm)
		dc |= F2Z | cd[2] << F2SFT;
	else
		dc |= cm << F2SFT;
	if (!dc)	/* don't generate 0 code normally */
		dc = F1X;
	return(dc);
}

/* Decode a normalized direction vector. */
RT_METHOD float3 decodedir(const int& dc)
{
	if (!dc)		/* special code for zero normal */
		return make_float3(0.0f);

	float3 dv;
	const float2 d = (make_float2((dc >> F1SFT & FMASK), (dc >> F2SFT & FMASK)) + 0.5f) * (1.0f / DCSCALE);
	const float der = sqrtf(1.0f - optix::dot(d, d));
	if (dc & F1X) {
		if (dc & F2Z) dv = make_float3(d.x, der, d.y);
		else dv = make_float3(d, der);
	}
	else {
		if (dc & F2Z) dv = make_float3(der, d);
		else dv = make_float3(d.y, d.x, der);
	}
	if (dc & FXNEG) dv.x = -dv.x;
	if (dc & FYNEG) dv.y = -dv.y;
	if (dc & FZNEG) dv.z = -dv.z;
	return dv;
}

/* find real roots of quadratic equation (from zeros.c) */
RT_METHOD unsigned int quadratic(float2* r, const float& a, const float& b, const float& c)
{
	int  first;

	if (a < -FTINY)
		first = 1;
	else if (a > FTINY)
		first = 0;
	else if (fabsf(b) > FTINY) {	/* solve linearly */
		*r = make_float2(-c / b);
		return(1);
	}
	else {
		*r = make_float2(0.0f);
		return(0);		/* equation is c == 0 ! */
	}

	float b2 = b * 0.5f;		/* simplifies formula */

	float disc = b2*b2 - a*c;	/* discriminant */

	if (disc < -FTINY*FTINY) {	/* no real roots */
		*r = make_float2(0.0f);
		return(0);
	}

	if (disc <= FTINY*FTINY) {	/* double root */
		*r = make_float2(-b2 / a);
		return(1);
	}

	disc = sqrtf(disc);

	if (first)
		*r = make_float2((-b2 + disc) / a, (-b2 - disc) / a);
	else
		*r = make_float2((-b2 - disc) / a, (-b2 + disc) / a);
	return(2);
}
