/* Copyright (c) 1997 Silicon Graphics, Inc. */

/* SCCSid "$SunId$ SGI" */

/*
 * Header file for holodeck device driver routines.
 */

#include "view.h"

extern struct driver {
	char	*name;		/* holodeck name or title */
	VIEW	v;		/* preferred view parameters */
	int	hres, vres;	/* device resolution */
	int	ifd;		/* input file descriptor (for select) */
} odev;			/* our open device */

			/* dev_input() return flags */
#define	DEV_SHUTDOWN	01	/* user shutdown request */
#define	DEV_NEWVIEW	02	/* view change (new view in odev.v) */
#define	DEV_NEWSIZE	04	/* device resolution change */
#define	DEV_WAIT	010	/* pause computation and wait for input */
#define	DEV_RESUME	020	/* resume after pause */
#define	DEV_REDRAW	040	/* redraw from server */
#define DEV_PUTVIEW	0100	/* print out current view */
#define DEV_LASTVIEW	0200	/* restore previous view */


/************************************************************************
 * Driver routines (all are required):


void
dev_open(nm)		: prepare the device
char	*nm;		: appropriate title bar annotation

Sets fields of odev structure and prepares the display for i/o.
The view type, horizontal and vertical view angles and other default
parameters in odev.v should also be assigned.


int
dev_view(nv)		: set display view parameters
VIEW	*nv;		: the new view

Updates the display for the given view change.
Look for nv==&odev.v when making view current after dev_input()
returns DEV_NEWVIEW flag.  Return 1 on success, or 0 if the
new view would conflict with device requirements.  In the latter
case, reset parameters in nv to make it more agreeable, calling
error(COMMAND, "appropriate user warning").


void
dev_value(c, p, v)	: register new point of light
COLR	c;		: pixel color (RGBE)
FVECT	p;		: world intersection point
FVECT	v;		: ray direction vector

Add the given color point to the display output queue.


int
dev_flush()		: flush the output and prepare for select call

Updates display, taking any pending action required before select(2) call.
Returns non-zero if there is device input available.


int
dev_input()		: process pending display input

Called when odev struct file descriptor shows input is ready.
Returns flags indicating actions to take in the control process.
If the DEV_NEWVIEW or DEV_NEWSIZE flag is returned, the odev
structure must be updated beforehand.


void
dev_close()		: close the display

Close the display device and free up resources in preparation for exit.
Set odev.v.type=0 and odev.hres=odev.vres=0 when done.


 ************************************************************************/
