/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#ifdef __cplusplus
extern "C" {
#endif

/* from optix_rvu.c */
extern void renderOptixIterative(const VIEW* view, const int width, const int height, const int moved, const double alarm, void fpaint(int, int, int, int, const unsigned char *), void fplot(double *));
extern void retreiveOptixImage(const int width, const int height, const double exposure, COLR* colrs);
extern void updateOctree(char* path);
extern void endOptix();

extern void setBackfaceVisibility(const int back);
extern void setIrradiance(const int irrad);

extern void setExposure(const double expose);
extern void setGreyscale(const int grey);
extern void setScale(const double maximum);
extern void setDecades(const int decade);
extern void setMask(const double masking);

extern void setTaskArea(const int x, const int y, const double omega);
extern void setHighArea(const int x, const int y, const double omega);
extern void setLowArea(const int x, const int y, const double omega);
extern void setAreaFlags(const unsigned int flags);


/* from rvmain.c */
extern double ralrm;			/* seconds between reports (-t) */

#ifdef __cplusplus
}
#endif