/*
 * Copyright (c) 2013-2016 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#ifndef ACCELERAD
/* Compiler setting to use OptiX ray tracing */
#define ACCELERAD
/* Compiler setting to allow real-time progressive OptiX ray tracing */
#if defined(HAS_QT) && defined(HAS_QWT)
#define ACCELERAD_RT
#endif
/* Compiler setting to allow debugging of OptiX ray tracing */
//#define ACCELERAD_DEBUG
/* Compiler setting to allow RTX ray tracing introduced in OptiX 6.0.0 */
#define RTX
#ifndef RTX
/* Compiler setting to allow remote rendering witn Nvidia VCA */
#define REMOTE_VCA
#endif
#endif

#ifndef DAYSIM
/* Compiler setting to calculate Daysim daylight coefficients. Should only be used with rtrace. */
//#define DAYSIM
#endif
