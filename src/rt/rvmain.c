#ifndef lint
static const char	RCSid[] = "$Id: rvmain.c,v 2.17 2016/08/18 00:52:48 greg Exp $";
#endif
/*
 *  rvmain.c - main for rview interactive viewer
 */

#include "copyright.h"

#include  <signal.h>
#include  <time.h>

#include  "platform.h"
#include  "ray.h"
#include  "source.h"
#include  "ambient.h"
#include  "rpaint.h"
#include  "random.h"
#include  "paths.h"
#include  "view.h"
#include  "pmapray.h"

extern char  *progname;			/* global argv[0] */

#ifdef ACCELERAD
int  xt = 0, yt = 0;			/* position of task area (-T) */
double  omegat = 0.0;			/* opening angle of task area in radians (-T) */
int  xh = 0, yh = 0;			/* position of contrast high luminance area (-C) */
double  omegah = 0.0;			/* opening angle of contrast high luminance area (-C) */
int  xl = 0, yl = 0;			/* position of contrast high luminance area (-C) */
double  omegal = 0.0;			/* opening angle of contrast high luminance area (-C) */
int  do_lum = 1;				/* show luminance rather than radiance */
int  fc = 1;					/* use falsecolor tonemapping, zero for natural tonemapping (-f) */
double  scale = 0.0;			/* maximum of scale for falsecolor images, zero auto-scaling (-s) */
int  decades = 0;				/* number of decades for log scale, zero for linear scale (-log) */
int  base = 10;					/* base for log scale (-base) */
double  masking = 0.0;			/* minimum value to display in falsecolor images (-m) */

double dstrpix = 0.0;			/* pixel jitter (-pj) */
#endif

VIEW  ourview = STDVIEW;		/* viewing parameters */
#ifdef ACCELERAD
int  hresolu = 0, vresolu = 0;	/* image resolution */
#else
int  hresolu, vresolu;			/* image resolution */
#endif

int  psample = 8;			/* pixel sample size */
double	maxdiff = .15;			/* max. sample difference */

int  greyscale = 0;			/* map colors to brightness? */
char  *dvcname = dev_default;		/* output device name */

double	exposure = 1.0;			/* exposure for scene */

int  newparam = 1;			/* parameter setting changed */
 
struct driver  *dev = NULL;		/* driver functions */

char  rifname[128];			/* rad input file name */

VIEW  oldview;				/* previous view parameters */

PNODE  ptrunk;				/* the base of our image */
RECT  pframe;				/* current frame boundaries */
int  pdepth;				/* image depth in current frame */

char  *errfile = NULL;			/* error output file */

int  nproc = 1;				/* number of processes */

char  *sigerr[NSIG];			/* signal error messages */

static void onsig(int  signo);
static void sigdie(int  signo, char  *msg);
static void printdefaults(void);

int
main(int argc, char *argv[])
{
#define	 check(ol,al)		if (argv[i][ol] || \
				badarg(argc-i-1,argv+i+1,al)) \
				goto badopt
#define	 check_bool(olen,var)		switch (argv[i][olen]) { \
				case '\0': var = !var; break; \
				case 'y': case 'Y': case 't': case 'T': \
				case '+': case '1': var = 1; break; \
				case 'n': case 'N': case 'f': case 'F': \
				case '-': case '0': var = 0; break; \
				default: goto badopt; }
	char  *octnm = NULL;
	char  *err;
	int  rval;
	int  i;
					/* global program name */
	progname = argv[0] = fixargv0(argv[0]);
					/* set our defaults */
	shadthresh = .1;
	shadcert = .25;
	directrelay = 0;
	vspretest = 128;
	srcsizerat = 0.;
	specthresh = .3;
	specjitter = 1.;
	maxdepth = 6;
	minweight = 1e-2;
	ambacc = 0.3;
	ambres = 32;
	ambdiv = 256;
	ambssamp = 64;
					/* option city */
	for (i = 1; i < argc; i++) {
						/* expand arguments */
		while ((rval = expandarg(&argc, &argv, i)) > 0)
			;
		if (rval < 0) {
			sprintf(errmsg, "cannot expand '%s'", argv[i]);
			error(SYSTEM, errmsg);
		}
		if (argv[i] == NULL || argv[i][0] != '-')
			break;			/* break from options */
		if (!strcmp(argv[i], "-version")) {
			puts(VersionID);
			quit(0);
		}
		if (!strcmp(argv[i], "-defaults") ||
				!strcmp(argv[i], "-help")) {
			printdefaults();
			quit(0);
		}
		if (!strcmp(argv[i], "-devices")) {
			printdevices();
			quit(0);
		}
		rval = getrenderopt(argc-i, argv+i);
		if (rval >= 0) {
			i += rval;
			continue;
		}
		rval = getviewopt(&ourview, argc-i, argv+i);
		if (rval >= 0) {
			i += rval;
			continue;
		}
		switch (argv[i][1]) {
		case 'n':				/* # processes */
			check(2,"i");
			nproc = atoi(argv[++i]);
			if (nproc <= 0)
				error(USER, "bad number of processes");
			break;
		case 'v':				/* view file */
			if (argv[i][2] != 'f')
				goto badopt;
			check(3,"s");
			rval = viewfile(argv[++i], &ourview, NULL);
			if (rval < 0) {
				sprintf(errmsg,
				"cannot open view file \"%s\"",
						argv[i]);
				error(SYSTEM, errmsg);
			} else if (rval == 0) {
				sprintf(errmsg,
					"bad view file \"%s\"",
						argv[i]);
				error(USER, errmsg);
			}
			break;
		case 'b':				/* grayscale */
#ifdef ACCELERAD
			if (argv[i][2] == 'a') { /* base for color scale */
				check(5, "i");
				base = atoi(argv[++i]);
				break;
			}
#endif
			check_bool(2, greyscale);
			break;
		case 'p':				/* pixel */
			switch (argv[i][2]) {
			case 's':				/* sample */
				check(3,"i");
				psample = atoi(argv[++i]);
				break;
			case 't':				/* threshold */
				check(3,"f");
				maxdiff = atof(argv[++i]);
				break;
#ifdef ACCELERAD
			case 'j':				/* jitter */
				check(3, "f");
				dstrpix = atof(argv[++i]);
				break;
#endif
			case 'e':				/* exposure */
				check(3,"f");
				exposure = atof(argv[++i]);
				if (argv[i][0] == '+' || argv[i][0] == '-')
					exposure = pow(2.0, exposure);
				break;
			default:
				goto badopt;
			}
			break;
		case 'w':				/* warnings */
			rval = erract[WARNING].pf != NULL;
			check_bool(2,rval);
			if (rval) erract[WARNING].pf = wputs;
			else erract[WARNING].pf = NULL;
			break;
		case 'e':				/* error file */
			check(2,"s");
			errfile = argv[++i];
			break;
		case 'o':				/* output device */
			check(2,"s");
			dvcname = argv[++i];
			break;
		case 'R':				/* render input file */
			check(2,"s");
			strcpy(rifname, argv[++i]);
			break;
#ifdef ACCELERAD
		case 'x':				/* x resolution */
			check(2, "i");
			hresolu = atoi(argv[++i]);
			break;
		case 'y':				/* y resolution */
			check(2, "i");
			vresolu = atoi(argv[++i]);
			break;
		case 'T':				/* task area luminance */
			check(2, "iif");
			xt = atoi(argv[++i]);
			yt = atoi(argv[++i]);
			omegat = atof(argv[++i]);
			break;
		case 'C':				/* contrast area luminance */
			check(2, "iifiif");
			xh = atoi(argv[++i]);
			yh = atoi(argv[++i]);
			omegah = atof(argv[++i]);
			xl = atoi(argv[++i]);
			yl = atoi(argv[++i]);
			omegal = atof(argv[++i]);
			break;
		case 'f':				/* use falsecolor images */
			check_bool(2, fc);
			break;
		case 's':				/* scale for falsecolor images */
			check(2, "f");
			scale = atof(argv[++i]);
			break;
		case 'l':				/* decades in log scale for falsecolor images */
			check(4, "i");
			decades = atoi(argv[++i]);
			break;
		case 'm':				/* minimum value for falsecolor images */
			check(2, "f");
			masking = atof(argv[++i]);
			break;
		case 't':				/* timer */
			check(2, "f");
			error(WARNING, "GPU callback time (-t) is depricated.");
			++i;
			break;
#endif
		default:
			goto badopt;
		}
	}
#ifdef ACCELERAD_RT
	if (use_optix) {
		if (nproc > 1) /* Don't allow multiple processes to access the graphics card. */
			error(USER, "multiprocessing incompatible with GPU implementation");
		if (ambacc > FTINY) {
			ambacc = 0.0;
			error(WARNING, "ambient accuracy set to zero for progressive path tracing");
		}
		if (ambdiv > 1 || optix_amb_fill > 1) {
			ambdiv = optix_amb_fill = 1;
			error(WARNING, "ambient divisions set to one for progressive path tracing");
		}
	}
#endif
	err = setview(&ourview);	/* set viewing parameters */
	if (err != NULL)
		error(USER, err);
						/* set up signal handling */
	sigdie(SIGINT, "Interrupt");
	sigdie(SIGTERM, "Terminate");
#if !defined(_WIN32) && !defined(_WIN64)
	sigdie(SIGHUP, "Hangup");
	sigdie(SIGPIPE, "Broken pipe");
	sigdie(SIGALRM, "Alarm clock");
#endif
					/* open error file */
	if (errfile != NULL) {
		if (freopen(errfile, "a", stderr) == NULL)
			quit(2);
		fprintf(stderr, "**************\n*** PID %5d: ",
				getpid());
		printargs(argc, argv, stderr);
		putc('\n', stderr);
		fflush(stderr);
	}
#ifdef	NICE
	nice(NICE);			/* lower priority */
#endif
					/* get octree */
	if (i == argc)
		octnm = NULL;
	else if (i == argc-1)
		octnm = argv[i];
	else
		goto badopt;
	if (octnm == NULL)
		error(USER, "missing octree argument");
					/* set up output & start process(es) */
	SET_FILE_BINARY(stdout);
	
	ray_init(octnm);		/* also calls ray_init_pmap() */
	
/* temporary shortcut, until winrview is refactored into a "device" */
#ifndef WIN_RVIEW
	rview();			/* run interactive viewer */


	devclose();			/* close output device */
#endif

	/* PMAP: free photon maps */
	ray_done_pmap();
	
#ifdef WIN_RVIEW
	return 1;
#endif
	quit(0);

badopt:
	sprintf(errmsg, "command line error at '%s'", argv[i]);
	error(USER, errmsg);
	return 1; /* pro forma return */

#undef	check
#undef	check_bool
}


void
wputs(				/* warning output function */
	char	*s
)
{
	int  lasterrno = errno;
	eputs(s);
	errno = lasterrno;
}


void
eputs(				/* put string to stderr */
	char  *s
)
{
	static int  midline = 0;

	if (!*s)
		return;
	if (!midline++) {
		fputs(progname, stderr);
		fputs(": ", stderr);
	}
	fputs(s, stderr);
	if (s[strlen(s)-1] == '\n') {
		fflush(stderr);
		midline = 0;
	}
}


static void
onsig(				/* fatal signal */
	int  signo
)
{
	static int  gotsig = 0;

	if (gotsig++)			/* two signals and we're gone! */
		_exit(signo);

#if !defined(_WIN32) && !defined(_WIN64)
	alarm(15);			/* allow 15 seconds to clean up */
	signal(SIGALRM, SIG_DFL);	/* make certain we do die */
#endif
	eputs("signal - ");
	eputs(sigerr[signo]);
	eputs("\n");
	devclose();
	quit(3);
}


static void
sigdie(			/* set fatal signal */
	int  signo,
	char  *msg
)
{
	if (signal(signo, onsig) == SIG_IGN)
		signal(signo, SIG_IGN);
	sigerr[signo] = msg;
}


static void
printdefaults(void)			/* print default values to stdout */
{
	printf("-n %-2d\t\t\t\t# number of rendering processes\n", nproc);
	printf(greyscale ? "-b+\t\t\t\t# greyscale on\n" :
			"-b-\t\t\t\t# greyscale off\n");
	printf("-vt%c\t\t\t\t# view type %s\n", ourview.type,
			ourview.type==VT_PER ? "perspective" :
			ourview.type==VT_PAR ? "parallel" :
			ourview.type==VT_HEM ? "hemispherical" :
			ourview.type==VT_ANG ? "angular" :
			ourview.type==VT_CYL ? "cylindrical" :
			ourview.type==VT_PLS ? "planisphere" :
			"unknown");
	printf("-vp %f %f %f\t# view point\n",
			ourview.vp[0], ourview.vp[1], ourview.vp[2]);
	printf("-vd %f %f %f\t# view direction\n",
			ourview.vdir[0], ourview.vdir[1], ourview.vdir[2]);
	printf("-vu %f %f %f\t# view up\n",
			ourview.vup[0], ourview.vup[1], ourview.vup[2]);
	printf("-vh %f\t\t\t# view horizontal size\n", ourview.horiz);
	printf("-vv %f\t\t\t# view vertical size\n", ourview.vert);
	printf("-vo %f\t\t\t# view fore clipping plane\n", ourview.vfore);
	printf("-va %f\t\t\t# view aft clipping plane\n", ourview.vaft);
	printf("-vs %f\t\t\t# view shift\n", ourview.hoff);
	printf("-vl %f\t\t\t# view lift\n", ourview.voff);
	printf("-pe %f\t\t\t# pixel exposure\n", exposure);
	printf("-ps %-9d\t\t\t# pixel sample\n", psample);
	printf("-pt %f\t\t\t# pixel threshold\n", maxdiff);
#ifdef ACCELERAD
	printf("-pj %f\t\t\t# pixel jitter\n", dstrpix);
#endif
	printf("-o %s\t\t\t\t# output device\n", dvcname);
	printf(erract[WARNING].pf != NULL ?
			"-w+\t\t\t\t# warning messages on\n" :
			"-w-\t\t\t\t# warning messages off\n");
#ifdef ACCELERAD
	printf("-x %-9d\t\t\t# x resolution\n", hresolu);
	printf("-y %-9d\t\t\t# y resolution\n", vresolu);
	printf("-T %-9d %-9d %f\t# position and opening angle of task area\n", xt, yt, omegat);
	printf("-C %-9d %-9d %f %-9d %-9d %f\t# position and opening angle of high and low contrast areas\n", xh, yh, omegah, xl, yl, omegal);
	printf("-s %f\t\t\t# scale for falsecolor images\n", scale);
	printf("-log %-9d\t\t\t# decades in log scale for falsecolor images\n", decades);
	printf("-m %f\t\t\t# minimum value for falsecolor images\n", masking);
#endif
	print_rdefaults();
}
