/**
 *  Daysim header file
 *
 * @author Augustinus Topor (topor@ise.fhg.de)
 */

#ifndef DAYSIM_H
#define DAYSIM_H

#ifdef DAYSIM

#include "fvect.h"



#define DAYSIM_MAX_COEFS		148
//#define DAYSIM_MAX_COEFS		2306

/**
 * daylight coefficient
 */
typedef float DaysimNumber;

typedef DaysimNumber DaysimCoef[DAYSIM_MAX_COEFS];

/* necessary for pmap which is built no matter if PHOTON_MAP is set or not */
typedef unsigned char DaysimSourcePatch;


/** */
extern double daysimLuminousSkySegments;

/** */
extern int daysimSortMode;

/** */
extern int NumberOfSensorsInDaysimFile;

/** */
extern int *DaysimSensorUnits;

/** */
int daysimInit( const int coefficients );

/** returns the number of coefficients */
const int daysimGetCoefficients();

/** Copies a daylight coefficient set */
void daysimCopy( DaysimCoef destin, DaysimCoef source );

/** Initialises all daylight coefficients with 'value' */
void daysimSet( DaysimCoef coef, const double value );

/** Scales the daylight coefficient set by the value 'scaling' */
void daysimScale( DaysimCoef coef, const double scaling );

/** Adds two daylight coefficient sets:
	result[i]= result[i] + add[i] */
void daysimAdd( DaysimCoef result, DaysimCoef add );

/** Multiply two daylight coefficient sets:
	result[i]= result[i] * add[i] */
void daysimMult( DaysimCoef result, DaysimCoef mult );

/** Sets the daylight coefficient at position 'index' to 'value' */
void daysimSetCoef( DaysimCoef result, const int index, const double value );

/** Adds 'value' to the daylight coefficient at position 'index' */
void daysimAddCoef( DaysimCoef result, const int index, const double add );

/** Adds the elements of 'source' scaled by 'scaling'  to 'result' */
void daysimAddScaled( DaysimCoef result, DaysimCoef add, const double scaling );

/** Assign the coefficients of 'source' scaled by 'scaling' to result */
void daysimAssignScaled( DaysimCoef result, DaysimCoef source, const double scaling );

/** Check that the sum of daylight coefficients equals the red color channel */
void daysimCheck(DaysimCoef daylightCoef, const double value, const char* where);

#endif /* DAYSIM */


#endif /* DAYSIM_H */
