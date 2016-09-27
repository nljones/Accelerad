/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#ifdef __cplusplus
extern "C" {
#endif

/* from optix_rvu.c */
extern void renderOptixIterative(const VIEW* view, const int width, const int height, const int moved, const int greyscale, const double exposure, const double scale, const int decades, const double mask, const double alarm, void fpaint(int, int, int, int, const unsigned char *), void fplot(double *));
extern void retreiveOptixImage(const int width, const int height, const double exposure, COLR* colrs);
extern void endOptix();

/* from rvmain.c */
extern double scale;			/* maximum of scale for falsecolor images, zero for regular tonemapping (-s) */
extern int decades;				/* number of decades for log scale, zero for standard scale (-log) */
extern double mask;				/* minimum value to display in falsecolor images (-m) */
extern double ralrm;			/* seconds between reports (-t) */

#ifdef __cplusplus
}
#endif