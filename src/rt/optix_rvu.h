/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

/* from optix_radiance.c */
extern void renderOptixIterative(const VIEW* view, const int width, const int height, const int moved, const int greyscale, const double exposure, const double scale, const int decades, const double mask, const double alarm);
extern void endOptix();

/* from rvmain.c */
extern double scale;			/* maximum of scale for falsecolor images, zero for regular tonemapping (-s) */
extern int decades;				/* number of decades for log scale, zero for standard scale (-log) */
extern double mask;				/* minimum value to display in falsecolor images (-m) */
extern double ralrm;			/* seconds between reports (-t) */
