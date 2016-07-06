/*
 * Copyright (c) 2013-2016 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#ifndef ACCELERAD
/* Compiler setting to use OptiX ray tracing */
#define ACCELERAD
/* Compiler setting to allow real-time progressive OptiX ray tracing */
#if defined(HAS_QT) || (defined(_WIN32) && defined(__cplusplus)) // TODO need a better QT test for C++
#define ACCELERAD_RT
#endif
/* Compiler setting to allow debugging of OptiX ray tracing */
#define ACCELERAD_DEBUG
#endif

#ifndef DAYSIM
/* Compiler setting to calculate Daysim daylight coefficients. Should only be used with rtrace. */
//#define DAYSIM
#endif
