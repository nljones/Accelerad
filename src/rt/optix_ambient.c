/*
 *  optix_ambient.c - routines for irradiance caching on GPUs.
 */

#include "accelerad_copyright.h"

#include "ray.h" // required by ambient.h
#include "ambient.h"

#include "optix_radiance.h"


#ifdef ACCELERAD

#define ADAPTIVE_SEED_SAMPLING
#define CLUSTER_GRID_WIDTH	1u
#define length_squared(v)	(((v).x*(v).x)+((v).y*(v).y)+((v).z*(v).z))
#define is_nan(v)			(((v).x!=(v).x)||((v).y!=(v).y)||((v).z!=(v).z))

static void updateAmbientCache( const RTcontext context, const unsigned int level );
static void createPointCloudCamera(const RTcontext context, const VIEW* view);
#ifdef AMB_PARALLEL
static void createAmbientSamplingCamera(const RTcontext context);
#endif
#ifdef ITERATIVE_IC
static void createHemisphereSamplingCamera(const RTcontext context);
#endif
static void createAmbientRecordCamera(const RTcontext context);
static void createGeometryInstanceAmbient( const RTcontext context, RTgeometryinstance* instance, const unsigned int ambinet_record_count );
static void createAmbientAcceleration( const RTcontext context, const RTgeometryinstance instance );
static double ambientWeight(const unsigned int level);
static double ambientWeightAdjusted(const unsigned int level);
static double ambientAngle(const double weight);
static double ambientRadius(const double weight);
static unsigned int ambientDivisions(const double weight);
static void updateAmbientDynamicStorage(const RTcontext context, const RTsize size, const unsigned int level);
static unsigned int populateAmbientRecords( const RTcontext context, const int level );
static size_t chooseAmbientLocations(const RTcontext context, const unsigned int level, const RTsize width, const RTsize height, const unsigned int seeds_per_thread, const size_t cluster_count, RTbuffer seed_buffer, RTbuffer cluster_buffer, RTvariable segment_var);
#ifdef DAYSIM
static unsigned int gatherAmbientRecords( AMBTREE* at, AmbientRecord** records, float** dc, const int level );
static int saveAmbientRecords( AmbientRecord* record, float* dc );
static void calcAmbientValues(const RTcontext context, const unsigned int level, const unsigned int max_level, const size_t cluster_count, RTbuffer ambient_record_buffer, void (*freport)(double), RTbuffer ambient_dc_buffer, RTvariable segment_var);
#else
static unsigned int gatherAmbientRecords( AMBTREE* at, AmbientRecord** records, const int level );
static int saveAmbientRecords( AmbientRecord* record );
static void calcAmbientValues(const RTcontext context, const unsigned int level, const unsigned int max_level, const size_t cluster_count, RTbuffer ambient_record_buffer, void(*freport)(double));
#endif
#ifdef AMBIENT_CELL
static size_t createClusters(const size_t seed_count, PointDirection* seed_buffer_data, size_t* seed_index, const unsigned int level);
#else /* AMBIENT_CELL */
static size_t createKMeansClusters(const size_t seed_count, const size_t cluster_count, PointDirection* seed_buffer_data, PointDirection* cluster_buffer_data, const unsigned int level);
//static void sortKMeans( const unsigned int cluster_count, PointDirection* cluster_buffer_data );
//static int clusterComparator( const void* a, const void* b );

/* return an array of cluster centers of size [numClusters][numCoords] */
float** cuda_kmeans(float **objects,	/* in: [numObjs][numCoords] */
					int     numCoords,	/* no. features */
					int     numObjs,		/* no. objects */
					int     numClusters,	/* no. clusters */
					int     max_iterations,	/* maximum k-means iterations */
					float   threshold,	/* % objects change membership */
					float   weight,	/* relative weighting of position */
					int     randomSeeds,	/* use randomly selected cluster centers (boolean) */
					int    *membership,	/* out: [numObjs] */
					float  *distance,	/* out: [numObjs] */
					int    *loop_iterations);

#ifdef ADAPTIVE_SEED_SAMPLING
void cuda_score_hits(PointDirection *hits, int *seeds, const unsigned int width, const unsigned int height, const float weight, const unsigned int seed_count);
#endif /* ADAPTIVE_SEED_SAMPLING */
#endif /* AMBIENT_CELL */


/* from ambient.c */
extern AMBTREE	atrunk;		/* our ambient trunk node */

extern double  avsum;		/* computed ambient value sum (log) */
extern unsigned int  navsum;	/* number of values in avsum */
extern unsigned int  nambvals;	/* total number of indirect values */

extern void avsave(AMBVAL *av);

/* Handles to objects used in ambient calculation */
extern RTvariable avsum_var;
extern RTvariable navsum_var;

/* Handles to objects used repeatedly in iterative irradiance cache builds */
static RTbuffer ambient_record_input_buffer;
#ifdef DAYSIM
static RTbuffer ambient_dc_input_buffer;
extern RTbuffer dc_scratch_buffer;
#endif
static RTgeometry ambient_record_geometry;
static RTacceleration ambient_record_acceleration;
static RTbuffer hess_row_buffer, grad_row_buffer;
#ifdef AMB_SAVE_MEM
static RTbuffer amb_samp_buffer, corral_u_buffer, corral_d_buffer;
#else /* AMB_SAVE_MEM */
static RTbuffer amb_samp_buffer;
#ifdef AMB_SUPER_SAMPLE
static RTbuffer earr_buffer;
#endif
#endif /* AMB_SAVE_MEM */

/* Allow faster irradiance cache creation by leaving some amount of threads unused.
	For 1D launch:
		Quadro K4000: Optimal is 4
		Tesla K40c: Optimal is 16
	For 2D (1xN) launch:
		Optimal is 2
*/
const unsigned int thread_stride = 4u;

#ifdef HIT_COUNT
static size_t hit_total = 0;
#endif
#ifdef AMBIENT_CELL
static int radius_scale = 1;
#endif


void setupAmbientCache( const RTcontext context, const unsigned int level )
{
	/* Primary RTAPI objects */
	RTgeometryinstance  ambient_records;

	unsigned int ambient_record_count;

	/* Create the buffer of ambient records. */
#ifdef DAYSIM_COMPATIBLE
#ifndef DAYSIM
	RTbuffer ambient_dc_input_buffer;
#endif
	createBuffer2D(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 0, 0, &ambient_dc_input_buffer);
	applyContextObject(context, "ambient_dc", ambient_dc_input_buffer);
#endif

	createCustomBuffer1D( context, RT_BUFFER_INPUT, sizeof(AmbientRecord), 0, &ambient_record_input_buffer );
	applyContextObject( context, "ambient_records", ambient_record_input_buffer );

	/* Populate the buffer of ambient records. */
	ambient_record_count = populateAmbientRecords( context, level );
	createGeometryInstanceAmbient( context, &ambient_records, ambient_record_count );
	createAmbientAcceleration( context, ambient_records );
}

static void updateAmbientCache( const RTcontext context, const unsigned int level )
{
	unsigned int useful_record_count;
			
	/* Repopulate the buffer of ambient records. */
	useful_record_count = populateAmbientRecords( context, level );
	RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( ambient_record_geometry, useful_record_count ) );
	RT_CHECK_ERROR( rtAccelerationMarkDirty( ambient_record_acceleration ) );
}

static void createAmbientRecordCamera(const RTcontext context)
{
	RTprogram  program;

	/* Ray generation program */
	char *ptx = ptxString("ambient_cloud_generator");
	RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "ambient_cloud_camera", &program));
	RT_CHECK_ERROR(rtContextSetRayGenerationProgram(context, AMBIENT_ENTRY, program));

	/* Exception program */
	RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "exception", &program));
	RT_CHECK_ERROR(rtContextSetExceptionProgram(context, AMBIENT_ENTRY, program));
	free(ptx);

	/* Stride for these programs */
	applyContextVariable1ui(context, "stride", thread_stride);
}

static void createGeometryInstanceAmbient( const RTcontext context, RTgeometryinstance* instance, const unsigned int ambinet_record_count )
{
	RTprogram program;
	RTmaterial ambient_record_material;
	char *ptx;

	/* Create the geometry reference for OptiX. */
	RT_CHECK_ERROR( rtGeometryCreate( context, &ambient_record_geometry ) );
	RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( ambient_record_geometry, ambinet_record_count ) );

	RT_CHECK_ERROR( rtMaterialCreate( context, &ambient_record_material ) );

	ptx = ptxString("ambient_records");
	RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "ambient_record_bounds", &program));
	RT_CHECK_ERROR(rtGeometrySetBoundingBoxProgram(ambient_record_geometry, program));
	RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "ambient_record_intersect", &program));
	RT_CHECK_ERROR(rtGeometrySetIntersectionProgram(ambient_record_geometry, program));
	RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "ambient_record_any_hit", &program));
	RT_CHECK_ERROR(rtMaterialSetAnyHitProgram(ambient_record_material, AMBIENT_RAY, program));

	/* Miss program */
	RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "ambient_miss", &program));
	RT_CHECK_ERROR(rtContextSetMissProgram(context, AMBIENT_RAY, program));
	free(ptx);
	// TODO miss program could handle makeambient()

	/* Create the geometry instance containing the ambient records. */
	RT_CHECK_ERROR( rtGeometryInstanceCreate( context, instance ) );
	RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( *instance, ambient_record_geometry ) );
	RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( *instance, 1 ) );

	/* Apply this material to the geometry instance. */
	RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( *instance, 0, ambient_record_material ) );
}

static double ambientWeight(const unsigned int level)
{
	return pow(AVGREFL, level); //  Compute weight as in makeambient() from ambient.c
}

static double ambientWeightAdjusted(const unsigned int level)
{
	return pow(AVGREFL * cuda_kmeans_threshold, level);
}

static double ambientAngle(const double weight)
{
	/* initial limit is 10 degrees plus ambacc radians */
	const double minangle = 10.0 * PI / 180.0;
	double maxangle = minangle + ambacc;

	/* adjust maximum angle */
	if (weight < 0.6)
		maxangle = (maxangle - PI / 2.0) * pow(weight, 0.13) + PI / 2.0;
	return maxangle;
}

static double ambientRadius(const double weight)
{
	double radius = minarad / sqrt(weight);
#ifdef AMBIENT_CELL
	radius *= radius_scale;
#endif
	if (radius > maxarad)
		return maxarad;
	return radius;
}

static unsigned int ambientDivisions(const double weight)
{
	unsigned int divisions = (unsigned int)(sqrt(ambdiv * weight) + 0.5);
	unsigned int i = 1 + 5 * (ambacc > FTINY);	/* minimum number of samples */
	if (divisions < i)
		return i;
	return divisions;
}

void createAmbientDynamicStorage(const RTcontext context, const RTprogram program, const RTsize size)
{
	unsigned int n = ambientDivisions(1.0);

	/* Create GPU scratch space buffers in global GPU memory */
	createCustomBuffer2D(context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, 9 * sizeof(float), size ? n - 1 : 0, size, &hess_row_buffer); // Size of optix::Matrix<3, 3>
	applyProgramObject(context, program, "hess_row_buffer", hess_row_buffer);

	createBuffer2D(context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, size ? n - 1 : 0, size, &grad_row_buffer);
	applyProgramObject(context, program, "grad_row_buffer", grad_row_buffer);

#ifdef AMB_SAVE_MEM
	createCustomBuffer2D(context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, sizeof(AmbientSample), size ? n : 0, size, &amb_samp_buffer);
	applyProgramObject(context, program, "amb_samp_buffer", amb_samp_buffer);

	createBuffer2D(context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT2, size ? 4 * (n - 1) : 0, size, &corral_u_buffer);
	applyProgramObject(context, program, "corral_u_buffer", corral_u_buffer);

	createBuffer2D(context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, size ? 4 * (n - 1) : 0, size, &corral_d_buffer);
	applyProgramObject(context, program, "corral_d_buffer", corral_d_buffer);
#else /* AMB_SAVE_MEM */
#ifdef AMB_PARALLEL
	createCustomBuffer3D(context, RT_BUFFER_INPUT_OUTPUT, sizeof(AmbientSample), size ? n : 0, size ? n : 0, size, &amb_samp_buffer);
	applyContextObject(context, "amb_samp_buffer", amb_samp_buffer);
#else
	createCustomBuffer3D(context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, sizeof(AmbientSample), size ? n : 0, size ? n : 0, size, &amb_samp_buffer);
	applyProgramObject(context, program, "amb_samp_buffer", amb_samp_buffer);
#endif
#ifdef AMB_SUPER_SAMPLE
	if (!ambssamp)
		n = 0;
	createBuffer3D(context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, size ? n : 0, size ? n : 0, n ? size : 0, &earr_buffer);
	applyProgramObject(context, program, "earr_buffer", earr_buffer);
#endif
#endif /* AMB_SAVE_MEM */
}

static void updateAmbientDynamicStorage(const RTcontext context, const RTsize size, const unsigned int level)
{
	unsigned int n = ambientDivisions(ambientWeight(level));

	/* Create GPU scratch space buffers in global GPU memory */
	RT_CHECK_ERROR(rtBufferSetSize2D(hess_row_buffer, size ? n - 1 : 0, size));
	RT_CHECK_ERROR(rtBufferSetSize2D(grad_row_buffer, size ? n - 1 : 0, size));
#ifdef AMB_SAVE_MEM
	RT_CHECK_ERROR(rtBufferSetSize2D(amb_samp_buffer, size ? n : 0, size));
	RT_CHECK_ERROR(rtBufferSetSize2D(corral_u_buffer, size ? 4 * (n - 1) : 0, size));
	RT_CHECK_ERROR(rtBufferSetSize2D(corral_d_buffer, size ? 4 * (n - 1) : 0, size));
#else /* AMB_SAVE_MEM */
	RT_CHECK_ERROR(rtBufferSetSize3D(amb_samp_buffer, size ? n : 0, size ? n : 0, size));
#ifdef AMB_SUPER_SAMPLE
	if (ambssamp)
		RT_CHECK_ERROR(rtBufferSetSize3D(earr_buffer, size ? n : 0, size ? n : 0, size));
#endif
#endif /* AMB_SAVE_MEM */
}

static unsigned int populateAmbientRecords( const RTcontext context, const int level )
{
	//RTbuffer   ambient_record_input_buffer;
	void*      data;

	AmbientRecord* ambient_records, *ambient_records_ptr;
#ifdef DAYSIM
	float *ambient_dc, *ambient_dc_ptr;
#endif
	unsigned int useful_record_count = 0u;

	/* Check that there are records */
	if ( nambvals ) {
		/* Allocate memory for temporary storage of ambient records. */
		ambient_records = (AmbientRecord*)malloc(sizeof(AmbientRecord) * nambvals);
		if (ambient_records == NULL) goto parmemerr;
		ambient_records_ptr = ambient_records;
#ifdef DAYSIM
		ambient_dc = (float*)malloc(sizeof(float) * daysimGetCoefficients() * nambvals);
		if (ambient_dc == NULL) goto parmemerr;
		ambient_dc_ptr = ambient_dc;

		/* Get the ambient records from the octree structure. */
		useful_record_count = gatherAmbientRecords(&atrunk, &ambient_records_ptr, &ambient_dc_ptr, level);
#else
		/* Get the ambient records from the octree structure. */
		useful_record_count = gatherAmbientRecords( &atrunk, &ambient_records_ptr, level );
#endif
		vprintf("Using %u of %u ambient records up to level %i.\n", useful_record_count, nambvals, level);
	}

	/* Resize the buffer of ambient records. */
	RT_CHECK_ERROR( rtBufferSetSize1D( ambient_record_input_buffer, useful_record_count ) );
#ifdef DAYSIM
	RT_CHECK_ERROR(rtBufferSetSize2D(ambient_dc_input_buffer, useful_record_count ? daysimGetCoefficients() : 0, daysimGetCoefficients() ? useful_record_count : 0));
#endif

	if ( nambvals ) {
		/* Copy ambient records from temporary storage to buffer. */
		RT_CHECK_ERROR( rtBufferMap( ambient_record_input_buffer, &data ) );
		memcpy( data, ambient_records, sizeof(AmbientRecord) * useful_record_count );
		RT_CHECK_ERROR( rtBufferUnmap( ambient_record_input_buffer ) );

		/* Free the temporary storage. */
		free(ambient_records);

#ifdef DAYSIM
		/* Copy daylight coefficients from temporary storage to buffer. */
		RT_CHECK_ERROR(rtBufferMap(ambient_dc_input_buffer, &data));
		memcpy(data, ambient_dc, sizeof(float) * daysimGetCoefficients() * useful_record_count);
		RT_CHECK_ERROR(rtBufferUnmap(ambient_dc_input_buffer));

		/* Free the temporary storage. */
		free(ambient_dc);
#endif
	}

	return useful_record_count;
parmemerr:
	error(SYSTEM, "out of memory in populateAmbientRecords");
	return 0;
}

#ifdef DAYSIM
static unsigned int gatherAmbientRecords(AMBTREE* at, AmbientRecord** records, float** dc, const int level)
#else
static unsigned int gatherAmbientRecords( AMBTREE* at, AmbientRecord** records, const int level )
#endif
{
	AMBVAL* record;
	AMBTREE* child;

	unsigned int count, i;
	count = 0u;

	for (record = at->alist; record != NULL; record = record->next) {
		if ( record->lvl <= level ) {
			array2cuda3( (*records)->pos, record->pos );
			array2cuda3( (*records)->val, record->val );
			array2cuda2( (*records)->gpos, record->gpos );
			array2cuda2( (*records)->gdir, record->gdir );
			array2cuda2( (*records)->rad, record->rad );

			(*records)->ndir = record->ndir;
			(*records)->udir = record->udir;
			(*records)->corral = record->corral;
			(*records)->lvl = record->lvl;
			(*records)->weight = record->weight;
#ifdef DAYSIM
			daysimCopy(*dc, record->daylightCoef);
			(*dc) += daysimGetCoefficients();
#endif

			(*records)++;
			count++;
		}
	}

	child = at->kid;

	for (i = 0u; i < 8u; i++) {
		if (child != NULL) {
#ifdef DAYSIM
			count += gatherAmbientRecords(child++, records, dc, level);
#else
			count += gatherAmbientRecords( child++, records, level );
#endif
		}
	}
	return(count);
}

#ifdef DAYSIM
static int saveAmbientRecords(AmbientRecord* record, float* dc)
#else
static int saveAmbientRecords( AmbientRecord* record )
#endif
{
	AMBVAL amb;

	cuda2array3( amb.pos, record->pos );
	cuda2array3( amb.val, record->val );
	cuda2array2( amb.gpos, record->gpos );
	cuda2array2( amb.gdir, record->gdir );
	cuda2array2( amb.rad, record->rad );

	amb.ndir = record->ndir;
	amb.udir = record->udir;
	amb.corral = record->corral;
	amb.lvl = record->lvl;
	amb.weight = record->weight;
#ifdef DAYSIM
	daysimCopy(amb.daylightCoef, dc);
#endif
#ifdef RAY_COUNT
	nrays += record->ray_count;
#endif
#ifdef HIT_COUNT
	hit_total += record->hit_count;
#endif

	/* Check that an ambient record was created. */
	if (ambvalOK(&amb)) {
		avsave(&amb);
		return(1);
	}

#ifdef DEBUG_OPTIX
	/* See what went wrong. */
	if (record->weight == -1.0f)
		logException((RTexception)record->val.x);
#endif
	return(0);
}

static void createAmbientAcceleration( const RTcontext context, const RTgeometryinstance instance )
{
	RTgeometrygroup geometrygroup;
	//RTacceleration  ambient_record_acceleration;

	/* Create a geometry group to hold the geometry instance.  This will be used as the top level group. */
	RT_CHECK_ERROR( rtGeometryGroupCreate( context, &geometrygroup ) );
	RT_CHECK_ERROR( rtGeometryGroupSetChildCount( geometrygroup, 1 ) );
	RT_CHECK_ERROR( rtGeometryGroupSetChild( geometrygroup, 0, instance ) );

	/* Set the geometry group as the top level object. */
	applyContextObject( context, "top_ambient", geometrygroup );

	/* create acceleration object for group and specify some build hints*/
	RT_CHECK_ERROR( rtAccelerationCreate( context, &ambient_record_acceleration ) );
	RT_CHECK_ERROR( rtAccelerationSetBuilder( ambient_record_acceleration, "Bvh" ) );
	RT_CHECK_ERROR( rtAccelerationSetTraverser( ambient_record_acceleration, "Bvh" ) );
	//RT_CHECK_ERROR( rtAccelerationSetProperty( ambient_record_acceleration, "refit", "1" ) ); // For Bvh, MedianBvh, and Lbvh only
	//RT_CHECK_ERROR( rtAccelerationSetProperty( ambient_record_acceleration, "refine", "1" ) ); // For Bvh, MedianBvh, and Lbvh only
	RT_CHECK_ERROR( rtGeometryGroupSetAcceleration( geometrygroup, ambient_record_acceleration ) );

	/* mark acceleration as dirty */
	RT_CHECK_ERROR( rtAccelerationMarkDirty( ambient_record_acceleration ) );
}

void createAmbientRecords(const RTcontext context, const VIEW* view, const RTsize width, const RTsize height, void(*freport)(double))
{
	RTvariable     level_var, segment_var = NULL;
	RTbuffer       seed_buffer, ambient_record_buffer;
#ifdef DAYSIM_COMPATIBLE
	RTbuffer       ambient_dc_buffer;
#endif
#ifdef ITERATIVE_IC
	RTvariable     current_cluster_buffer;
	RTbuffer*      cluster_buffer;
	int*	       cluster_buffer_id;
#ifdef AMBIENT_CELL
	size_t*        cluster_counts;
	double         radius;
	RTvariable     cell_size_var;
#endif
#else
	RTbuffer       cluster_buffer;
#endif

	RTsize grid_width, grid_height;
	unsigned int level, max_level;

	/* Set number of iterations. */
	max_level = ambounce;
	if (maxdepth && abs(maxdepth) < ambounce)
		max_level = abs(maxdepth);
	if (minweight > 0)
		while (max_level > 1 && ambientWeight(max_level) < minweight) // if max_level is zero, objects won't get initialized
			max_level--;

	/* Set dimensions for first sample. */
	if (view && optix_amb_grid_size > 0) { // if -az argument greater than zero
		grid_width = optix_amb_grid_size / 2;
		grid_height = optix_amb_grid_size;
	} else if (view && optix_amb_scale > 0) { // if -al argument greater than zero
		grid_width = width / optix_amb_scale;
		grid_height = height / optix_amb_scale;
	} else {
		grid_width = width;
		grid_height = height;
	}

	createPointCloudCamera( context, view );
#ifdef AMB_PARALLEL
	createAmbientSamplingCamera( context );
#endif
#ifdef ITERATIVE_IC
	createHemisphereSamplingCamera( context );
#endif
	createAmbientRecordCamera(context);

#ifdef AMBIENT_CELL
	/* Set up cell hash. */
	applyContextVariable3f(context, "cuorg", (float)thescene.cuorg[0], (float)thescene.cuorg[1], (float)thescene.cuorg[2]);
	RT_CHECK_ERROR(rtContextDeclareVariable(context, "cell_size", &cell_size_var));
#endif

	/* Create buffer for retrieving potential seed points. */
	createCustomBuffer3D( context, RT_BUFFER_OUTPUT, sizeof(PointDirection), 0, 0, 0, &seed_buffer );
	applyContextObject( context, "seed_buffer", seed_buffer );

	/* Create buffer for inputting seed point clusters. */
#ifdef ITERATIVE_IC
	cluster_buffer = (RTbuffer*)malloc(max_level * sizeof(RTbuffer));
	cluster_buffer_id = (int*)malloc(max_level * sizeof(int));
	if (cluster_buffer == NULL || cluster_buffer_id == NULL) goto carmemerr;
	for (level = 0u; level < max_level; level++) {
		createCustomBuffer1D( context, RT_BUFFER_INPUT, sizeof(PointDirection), cuda_kmeans_clusters, &cluster_buffer[level] );
		RT_CHECK_ERROR(rtBufferGetId(cluster_buffer[level], &cluster_buffer_id[level]));
	}
	//current_cluster_buffer = applyContextObject(context, "cluster_buffer", cluster_buffer[0]);
	current_cluster_buffer = applyContextVariable(context, "cluster_buffer", sizeof(cluster_buffer_id[0]), &cluster_buffer_id[0]);
#ifdef AMBIENT_CELL
	cluster_counts = (size_t*)malloc(max_level * sizeof(size_t));
	if (cluster_counts == NULL) goto carmemerr;
	radius_scale = cuda_kmeans_clusters;
#endif
#else
	createCustomBuffer1D( context, RT_BUFFER_INPUT, sizeof(PointDirection), cuda_kmeans_clusters, &cluster_buffer );
	applyContextObject( context, "cluster_buffer", cluster_buffer );
#endif

	/* Create buffer for retrieving ambient records. */
	createCustomBuffer1D( context, RT_BUFFER_OUTPUT, sizeof(AmbientRecord), cuda_kmeans_clusters, &ambient_record_buffer );
	applyContextObject( context, "ambient_record_buffer", ambient_record_buffer );

	/* Create variable for offset into scratch space */
	segment_var = applyContextVariable1ui(context, "segment_offset", 0u);
#ifdef DAYSIM_COMPATIBLE
#ifdef DAYSIM
	createBuffer2D(context, RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, daysimGetCoefficients(), daysimGetCoefficients() ? cuda_kmeans_clusters : 0, &ambient_dc_buffer);
#else
	createBuffer2D(context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, 0, 0, &ambient_dc_buffer);
#endif
	applyContextObject(context, "ambient_dc_buffer", ambient_dc_buffer);
#endif

	/* Set additional variables used for ambient record generation */
	level_var = applyContextVariable1ui(context, "level", 0u); // Could be camera program variable

	/* Put any existing ambient records into GPU cache. There should be none. */
	setupAmbientCache( context, 0u ); // Do this now to avoid additional setup time later

#ifdef ITERATIVE_IC
	for (level = 0u; level < max_level; level++) {
#ifdef AMBIENT_CELL
		radius = ambientRadius(ambientWeightAdjusted(level)) * ambacc * cuda_kmeans_iterations; // TODO new variable
		vprintf("Size of cell: %g\nNumber of cells: %u\n", radius, (unsigned int)ceil(thescene.cusize / radius));

		RT_CHECK_ERROR(rtVariableSet1f(cell_size_var, (float)radius));
		cluster_counts[level] = chooseAmbientLocations(context, level, grid_width, grid_height, optix_amb_seeds_per_thread, level ? cluster_counts[level - 1] : 0, seed_buffer, cluster_buffer[level], segment_var);
#else
		chooseAmbientLocations(context, level, grid_width, grid_height, 1u, cuda_kmeans_clusters, seed_buffer, cluster_buffer[level], segment_var);
#endif

		/* Set input buffer index for next iteration */
		//RT_CHECK_ERROR( rtVariableSetObject( current_cluster_buffer, cluster_buffer[level] ) );
		RT_CHECK_ERROR(rtVariableSetUserData(current_cluster_buffer, sizeof(cluster_buffer_id[level]), &cluster_buffer_id[level]));
	}
#else /* ITERATIVE_IC */
	chooseAmbientLocations(context, 0u, grid_width, grid_height, optix_amb_seeds_per_thread, cuda_kmeans_clusters, seed_buffer, cluster_buffer, segment_var);
	level = max_level;
#endif /* ITERATIVE_IC */
#ifndef AMB_PARALLEL
#ifdef DAYSIM
	if (daysimGetCoefficients())
		RT_CHECK_ERROR(rtBufferSetSize3D(dc_scratch_buffer, daysimGetCoefficients() * abs(maxdepth) * 2, 1u, cuda_kmeans_clusters));
#endif
#endif /* AMB_PARALLEL */
#ifdef AMBIENT_CELL
	radius_scale = 1;
#endif

	while ( level-- ) {
		RT_CHECK_ERROR( rtVariableSet1ui( level_var, level ) );
#ifdef AMBIENT_CELL
		updateAmbientDynamicStorage(context, cluster_counts[level], level);
		RT_CHECK_ERROR(rtBufferSetSize1D(ambient_record_buffer, cluster_counts[level]));
#ifdef DAYSIM
		RT_CHECK_ERROR(rtBufferSetSize2D(ambient_dc_buffer, daysimGetCoefficients(), daysimGetCoefficients() ? cluster_counts[level] : 0));
		calcAmbientValues(context, level, max_level, cluster_counts[level], ambient_record_buffer, freport, ambient_dc_buffer, segment_var);
#else
		calcAmbientValues(context, level, max_level, cluster_counts[level], ambient_record_buffer, freport);
#endif

		/* Second Round */
		radius = ambientRadius(ambientWeightAdjusted(level)) * ambacc * cuda_kmeans_iterations; // TODO new variable
		vprintf("Size of cell: %g\nNumber of cells: %u\n", radius, (unsigned int)ceil(thescene.cusize / radius));

		if (level)
			//RT_CHECK_ERROR(rtVariableSetObject(current_cluster_buffer, cluster_buffer[level - 1]));
			RT_CHECK_ERROR(rtVariableSetUserData(current_cluster_buffer, sizeof(cluster_buffer_id[level - 1]), &cluster_buffer_id[level - 1]));
		RT_CHECK_ERROR(rtVariableSet1f(cell_size_var, (float)radius));
		cluster_counts[level] = chooseAmbientLocations(context, level, grid_width, grid_height, optix_amb_seeds_per_thread, level ? cluster_counts[level - 1] : 0, seed_buffer, cluster_buffer[level], segment_var);

		if (!cluster_counts[level]) continue;

		if (level)
			//RT_CHECK_ERROR(rtVariableSetObject(current_cluster_buffer, cluster_buffer[level]));
			RT_CHECK_ERROR(rtVariableSetUserData(current_cluster_buffer, sizeof(cluster_buffer_id[level]), &cluster_buffer_id[level]));

		updateAmbientDynamicStorage(context, cluster_counts[level], level);
		RT_CHECK_ERROR(rtBufferSetSize1D(ambient_record_buffer, cluster_counts[level]));
#ifdef DAYSIM
		RT_CHECK_ERROR(rtBufferSetSize2D(ambient_dc_buffer, daysimGetCoefficients(), daysimGetCoefficients() ? cluster_counts[level] : 0));
		calcAmbientValues(context, level, max_level, cluster_counts[level], ambient_record_buffer, freport, ambient_dc_buffer, segment_var);
#else
		calcAmbientValues(context, level, max_level, cluster_counts[level], ambient_record_buffer, freport);
#endif

#else /* AMBIENT_CELL */
#ifdef DAYSIM
		calcAmbientValues(context, level, max_level, cuda_kmeans_clusters, ambient_record_buffer, freport, ambient_dc_buffer, segment_var);
#else
		calcAmbientValues(context, level, max_level, cuda_kmeans_clusters, ambient_record_buffer, freport);
#endif
#endif /* AMBIENT_CELL */
#ifdef ITERATIVE_IC
		if ( level )
			//RT_CHECK_ERROR( rtVariableSetObject( current_cluster_buffer, cluster_buffer[level - 1] ) );
			RT_CHECK_ERROR(rtVariableSetUserData(current_cluster_buffer, sizeof(cluster_buffer_id[level - 1]), &cluster_buffer_id[level - 1]));
#endif
	}

#ifdef ITERATIVE_IC
	free(cluster_buffer);
	free(cluster_buffer_id);
#endif
#ifdef AMBIENT_CELL
	free(cluster_counts);
#endif
	return;
carmemerr:
	error(SYSTEM, "out of memory in createAmbientRecords");
}

static void createPointCloudCamera( const RTcontext context, const VIEW* view )
{
	RTprogram  program;

	/* Ray generation program */
	if ( view ) {
		ptxFile( path_to_ptx, "rpict_cloud_generator" );
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "point_cloud_camera", &program));

		if (optix_amb_grid_size > 0) // Ignore camera for bounds of sampling area
			applyProgramVariable1ui(context, program, "camera", 0u); // Hide context variable
	} else {
		ptxFile( path_to_ptx, "rtrace_cloud_generator" );
		RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "cloud_generator", &program));
	}
	RT_CHECK_ERROR(rtContextSetRayGenerationProgram(context, POINT_CLOUD_ENTRY, program));

	/* Exception program */
	RT_CHECK_ERROR(rtProgramCreateFromPTXFile(context, path_to_ptx, "exception", &program));
	RT_CHECK_ERROR(rtContextSetExceptionProgram(context, POINT_CLOUD_ENTRY, program));
}

#ifdef AMB_PARALLEL
static void createAmbientSamplingCamera( const RTcontext context )
{
	RTprogram  program;

	/* Ray generation program */
	char *ptx = ptxString("ambient_sample_generator");
	RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "ambient_sample_camera", &program));
	RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, AMBIENT_SAMPLING_ENTRY, program ) );

	/* Exception program */
	RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "exception", &program));
	RT_CHECK_ERROR( rtContextSetExceptionProgram( context, AMBIENT_SAMPLING_ENTRY, program ) );
	free(ptx);
}
#endif /* AMB_PARALLEL */

#ifdef ITERATIVE_IC
static void createHemisphereSamplingCamera( const RTcontext context )
{
	RTprogram  program;
	char *ptx;

	/* Ray generation program */
	ptx = ptxString("hemisphere_generator");
	RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "hemisphere_camera", &program));
	RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, HEMISPHERE_SAMPLING_ENTRY, program ) );

	/* Exception program */
	RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "exception", &program));
	RT_CHECK_ERROR( rtContextSetExceptionProgram( context, HEMISPHERE_SAMPLING_ENTRY, program ) );
	free(ptx);
}
#endif /* ITERATIVE_IC */

static size_t chooseAmbientLocations(const RTcontext context, const unsigned int level, const RTsize width, const RTsize height, const unsigned int seeds_per_thread, const size_t cluster_count, RTbuffer seed_buffer, RTbuffer cluster_buffer, RTvariable segment_var)
{
	size_t seed_count, i, multi_pass = 0u;
	PointDirection *seed_buffer_data = NULL, *cluster_buffer_data = NULL;
#ifdef AMBIENT_CELL
	size_t *seed_index = NULL;
#endif

	if (level) {
#ifdef ITERATIVE_IC
		/* Adjust output buffer size */
		unsigned int divisions = cluster_count ? ambientDivisions(ambientWeight(level)) : 0;
		const size_t bytes_per_cluster = sizeof(PointDirection) * divisions * divisions;
		const size_t clusters_per_segment = bytes_per_cluster ? min(cluster_count, INT_MAX / bytes_per_cluster) : cluster_count; // Limit imposed by OptiX
		seed_count = cluster_count * divisions * divisions;

		if ((multi_pass = (cluster_count - 1) / clusters_per_segment)) {
			mprintf("Processing seeds in %" PRIu64 " segments of %" PRIu64 ".\n", multi_pass + 1, clusters_per_segment);
			seed_buffer_data = (PointDirection *)malloc(seed_count * sizeof(PointDirection));
			if (seed_buffer_data == NULL) goto calmemerr;
		}

		RT_CHECK_ERROR(rtBufferSetSize3D(seed_buffer, divisions, divisions, clusters_per_segment));

		for (i = 0u; i < cluster_count; i += clusters_per_segment) {
			size_t current_count = min(cluster_count - i, clusters_per_segment);

			if (i != (unsigned int)i)
				eprintf(USER, "segment offset %" PRIu64 " out of range", i); // To get here, size of cluster_buffer must also be out of range, which should already have triggered an error
			RT_CHECK_ERROR(rtVariableSet1ui(segment_var, (unsigned int)i));

			/* Run kernel to gerate more seed points from cluster centers */
			runKernel3D(context, HEMISPHERE_SAMPLING_ENTRY, divisions, divisions, current_count); // stride?

			/* Retrieve potential seed points. */
			if (multi_pass) {
				PointDirection *segment_data;
				RT_CHECK_ERROR(rtBufferMap(seed_buffer, (void**)&segment_data));
				memcpy(seed_buffer_data + i * divisions * divisions, segment_data, current_count * bytes_per_cluster);
				RT_CHECK_ERROR(rtBufferUnmap(seed_buffer));
			}
		}
#endif /* ITERATIVE_IC */
	}
	else {
		/* Set output buffer size */
		seed_count = width * height * seeds_per_thread;
		RT_CHECK_ERROR(rtBufferSetSize3D(seed_buffer, width, height, seeds_per_thread));

		/* Run the kernel to get the first set of seed points. */
		runKernel2D(context, POINT_CLOUD_ENTRY, width, height);
	}

	/* Retrieve potential seed points. */
	if (!multi_pass)
		RT_CHECK_ERROR(rtBufferMap(seed_buffer, (void**)&seed_buffer_data));

#ifndef AMBIENT_CELL
#if defined(ITERATIVE_IC) && defined(ADAPTIVE_SEED_SAMPLING)
	if (!level) {
		clock_t kernel_clock;
		//int total = 0;
		size_t i;
		size_t missing = 0u;
		size_t si = cluster_count;
		size_t ci = 0u;
		int *score = (int*)malloc(seed_count * sizeof(int));
		PointDirection *temp_list = (PointDirection*)malloc(seed_count * sizeof(PointDirection));
		if (score == NULL || temp_list == NULL) goto calmemerr;

		kernel_clock = clock();
		cuda_score_hits(seed_buffer_data, score, (unsigned int)width, (unsigned int)height, (float)(cuda_kmeans_error / (ambacc * maxarad)), (unsigned int)cluster_count);
		tprint(clock() - kernel_clock, "Adaptive sampling");

		for (i = 0u; i < seed_count; i++) {
			if (score[i]) {
				missing += score[i] - 1;
				//total++;
				temp_list[ci++] = seed_buffer_data[i];
			}
			else if (missing) { // TODO need better way to randomly add extra cluster seeds
				missing--;
				temp_list[ci++] = seed_buffer_data[i];
			}
			else if (si < seed_count)
				temp_list[si++] = seed_buffer_data[i];
			else
				temp_list[ci++] = seed_buffer_data[i];
		}
		//mprintf("Score total: %i\n", total);

		memcpy(seed_buffer_data, temp_list, seed_count * sizeof(PointDirection));

		free(score);
		free(temp_list);
	}
#endif /* ITERATIVE_IC && ADAPTIVE_SEED_SAMPLING */

	/* Group seed points into clusters and add clusters to buffer */
	RT_CHECK_ERROR(rtBufferMap(cluster_buffer, (void**)&cluster_buffer_data));
	seed_count = createKMeansClusters(seed_count, cluster_count, seed_buffer_data, cluster_buffer_data, level);
	//sortKMeans( cluster_count, cluster_buffer_data );
	RT_CHECK_ERROR(rtBufferUnmap(cluster_buffer));
#else /* AMBIENT_CELL */

	/* Group seed points into clusters */
	seed_index = (size_t*)malloc(seed_count * sizeof(size_t));
	if (seed_index == NULL) goto calmemerr;
	seed_count = createClusters(seed_count, seed_buffer_data, seed_index, level);

	/* Add clusters to buffer */
	RT_CHECK_ERROR(rtBufferSetSize1D(cluster_buffer, seed_count));
	RT_CHECK_ERROR(rtBufferMap(cluster_buffer, (void**)&cluster_buffer_data));
	for (i = 0u; i < seed_count; i++)
		cluster_buffer_data[i] = seed_buffer_data[seed_index[i]];
	RT_CHECK_ERROR(rtBufferUnmap(cluster_buffer));

	/* Add previous round's ambient values to cache */
	if (nambvals && seed_count)
		updateAmbientCache(context, level + 1);

	free(seed_index);
#endif /* AMBIENT_CELL */
	if (multi_pass)
		free(seed_buffer_data);
	else
		RT_CHECK_ERROR(rtBufferUnmap(seed_buffer));
	return seed_count;
calmemerr:
	error(SYSTEM, "out of memory in chooseAmbientLocations");
	return 0;
}

#ifdef AMBIENT_CELL
/*static int compare_point_direction_hash(const PointDirection* a, const PointDirection* b)
{
	int ahash[6], bhash[6];
	memcpy(ahash, a, 6 * sizeof(float));
	memcpy(bhash, b, 6 * sizeof(float));
	return 0xffff & (ahash[0] - ahash[3] + ahash[1] - ahash[4] + ahash[2] - ahash[5]) - 0xffff & (bhash[0] - bhash[3] + bhash[1] - bhash[4] + bhash[2] - bhash[5]);
}*/

static int compare_point_directions(const void* a, const void* b)
{
	if (((PointDirection*)a)->cell.x < ((PointDirection*)b)->cell.x) return -1;
	if (((PointDirection*)a)->cell.x > ((PointDirection*)b)->cell.x) return 1;
	if (((PointDirection*)a)->cell.y < ((PointDirection*)b)->cell.y) return -1;
	if (((PointDirection*)a)->cell.y > ((PointDirection*)b)->cell.y) return 1;
	return 0;
}

static int sameCell(const PointDirection* a, const PointDirection* b)
{
	return a->cell.x == b->cell.x && a->cell.y == b->cell.y;
}

static unsigned int checkForOverlap(const PointDirection* a, const PointDirection* b, const double angle, const double radius)
{
	FVECT adir, bdir, apos, bpos, ck0, U, V;
	double d, delta_r2, delta_t2;

	/* Direction test using unperturbed normal */
	cuda2array3(adir, a->dir);
	cuda2array3(bdir, b->dir);
	d = DOT(adir, bdir);
	if (d <= 0.0)		/* >= 90 degrees */
		return 0;

	delta_r2 = 2.0 - 2.0*d;	/* approx. radians^2 */
	if (delta_r2 >= angle * angle)
		return 0;

	/* Modified ray behind test */
	cuda2array3(apos, a->pos);
	cuda2array3(bpos, b->pos);
	VSUB(ck0, bpos, apos);
	d = DOT(ck0, adir);
	if (d < -minarad * ambacc - 0.001)
		return 0;
	d /= radius;
	delta_t2 = d * d;
	if (delta_t2 >= ambacc * ambacc)
		return 0;

	/* Elliptical radii test based on Hessian */
	if (!getperpendicular(U, adir, 0))
		error(CONSISTENCY, "bad ray direction in checkForOverlap");
	VCROSS(V, adir, U);
	d = DOT(ck0, U) / radius;
	delta_t2 += d * d;
	d = DOT(ck0, V) / radius;
	delta_t2 += d * d;
	if (delta_t2 >= ambacc * ambacc)
		return 0;

	return 1;
}

static size_t createClusters(const size_t seed_count, PointDirection* seed_buffer_data, size_t* seed_index, const unsigned int level)
{
	size_t i, j, k, cluster_bin_start;
	unsigned int cell_counter = 0;
	const double weight = ambientWeightAdjusted(level);
	const double angle = ambientAngle(weight);
	const double radius = ambientRadius(weight) * ambacc * cuda_kmeans_error; // TODO new variable
	clock_t kernel_clock; // Timer in clock cycles for short jobs

	if (seed_count > 1e6)
		mprintf("Retrieved %" PRIu64 " samples at level %u. This could take a%s long time. Bad parameter combination?\n", seed_count, level, seed_count > 1e8 ? " really really" : seed_count > 1e7 ? " really" : "");

	kernel_clock = clock();

	/* Sort seeds into bins by 4D index */
	qsort(seed_buffer_data, seed_count, sizeof(PointDirection), compare_point_directions);

	/* Pick important seeds from each bin */
	for (i = j = 0u; i < seed_count; i++) {
		if (length_squared(seed_buffer_data[i].dir) < FTINY) {
#ifdef DEBUG_OPTIX
			logException((RTexception)seed_buffer_data[i].pos.x);
#endif
		}
		else if (!j || !sameCell(seed_buffer_data + i, seed_buffer_data + seed_index[j - 1])) {
			/* Start a new cell */
			cluster_bin_start = j;
			seed_index[j++] = i;
			cell_counter++;
		}
		else {
			/* Check against other members of cell */
			for (k = cluster_bin_start; k < j; k++)
				if (checkForOverlap(seed_buffer_data + seed_index[k], seed_buffer_data + i, angle, radius))
					break;
			if (k == j)
				seed_index[j++] = i;
		}
	}

	tprintf(clock() - kernel_clock, "Created %u occupied cells", cell_counter);
#ifdef DEBUG_OPTIX
	flushExceptionLog("ambient seeding");
#endif
	mprintf("Retrieved %" PRIu64 " potential seeds from %" PRIu64 " samples at level %u.\n\n", j, seed_count, level);

	return j;
}
#else /* AMBIENT_CELL */

static size_t createKMeansClusters( const size_t seed_count, const size_t cluster_count, PointDirection* seed_buffer_data, PointDirection* cluster_buffer_data, const unsigned int level )
{
	clock_t kernel_clock; // Timer in clock cycles for short jobs
	size_t good_seed_count, i, j;
	PointDirection **seeds, **clusters; // input and output for cuda_kmeans()
	int *membership, loops; // output from cuda_kmeans()
	float *distance, *min_distance; // output from cuda_kmeans()

	/* Eliminate bad seeds and copy addresses to new array */
	good_seed_count = 0u;
	seeds = (PointDirection**) malloc(seed_count * sizeof(PointDirection*));
	if (seeds == NULL) goto kmmemerr;
	//TODO Is there any point in filtering out values that are not valid (length(seed_buffer_data[i].dir) == 0)?
	for ( i = 0u; i < seed_count; i++) {
		if ( length_squared( seed_buffer_data[i].dir ) < FTINY ) {
#ifdef DEBUG_OPTIX
			logException((RTexception)seed_buffer_data[i].pos.x);
#endif
		}
		else
			seeds[good_seed_count++] = &seed_buffer_data[i];
	}
#ifdef DEBUG_OPTIX
	flushExceptionLog("ambient seeding");
#endif
	mprintf("Retrieved %" PRIu64 " of %" PRIu64 " potential seeds at level %u.\n", good_seed_count, seed_count, level);

	/* Check that enough seeds were found */
	if ( good_seed_count <= cluster_count ) {
		mprintf("Using all %" PRIu64 " seeds at level %u (%" PRIu64 " needed for k-means).\n\n", good_seed_count, level, cluster_count);
		for ( i = 0u; i < good_seed_count; i++ )
			cluster_buffer_data[i] = *seeds[i];
		for ( ; i < cluster_count; i++ )
			cluster_buffer_data[i].dir.x = cluster_buffer_data[i].dir.y = cluster_buffer_data[i].dir.z = 0.0f; // Don't use this value
		free(seeds);
		return good_seed_count;
	}

	/* Check that k-means should be used */
	if (cuda_kmeans_iterations < 1) {
		mprintf("Using first %" PRIu64 " seeds at level %u.\n\n", cluster_count, level);
		for ( i = 0u; i < cluster_count; i++ )
			cluster_buffer_data[i] = *seeds[i]; //TODO should randomly choose from array
		free(seeds);
		return good_seed_count;
	}

	/* Group the seeds into clusters with k-means */
	membership = (int*) malloc(good_seed_count * sizeof(int));
	distance = (float*) malloc(good_seed_count * sizeof(float));
	if (membership == NULL || distance == NULL)	goto kmmemerr;
	kernel_clock = clock();
#if defined(ITERATIVE_IC) && defined(ADAPTIVE_SEED_SAMPLING)
	clusters = (PointDirection**)cuda_kmeans((float**)seeds, sizeof(PointDirection) / sizeof(float), (unsigned int)good_seed_count, (unsigned int)cluster_count, cuda_kmeans_iterations, (float)cuda_kmeans_threshold, (float)(cuda_kmeans_error / (ambacc * maxarad)), level, membership, distance, &loops);
#else
	clusters = (PointDirection**)cuda_kmeans((float**)seeds, sizeof(PointDirection) / sizeof(float), (unsigned int)good_seed_count, (unsigned int)cluster_count, cuda_kmeans_iterations, (float)cuda_kmeans_threshold, (float)(cuda_kmeans_error / (ambacc * maxarad)), 1u, membership, distance, &loops);
#endif
	tprintf(clock() - kernel_clock, "K-means performed %u loop iterations", loops);

	/* Populate buffer of seed point clusters. */
	min_distance = (float*) malloc(cluster_count * sizeof(float));
	if (min_distance == NULL) goto kmmemerr;
	for ( i = 0u; i < cluster_count; i++ )
		min_distance[i] = FHUGE;
	for ( i = 0u; i < good_seed_count; i++ ) {
		j = membership[i];
		if ( distance[i] < min_distance[j] ) { // || length_squared( cluster_buffer_data[j].dir ) > FTINY ) {
			min_distance[j] = distance[i];
			cluster_buffer_data[j] = *(seeds[i]);
		}
#ifdef DEBUG_OPTIX
		else if ( min_distance[j] != min_distance[j] )
			mprintf("NaN distance from seed %" PRIu64 " to cluster %" PRIu64 "\n", i, j);
#endif
	}
	j = 0u;
	for ( i = 0u; i < cluster_count; i++ ) {
		if (min_distance[i] == FHUGE) {
			j++;
			cluster_buffer_data[i].dir.x = cluster_buffer_data[i].dir.y = cluster_buffer_data[i].dir.z = 0.0f; // Don't use this value
		}
#ifdef DEBUG_OPTIX
		else if ( is_nan(cluster_buffer_data[i].pos) || is_nan(cluster_buffer_data[i].dir) )
			mprintf("NaN in cluster %" PRIu64 " (%g, %g, %g) (%g, %g, %g)\n", i, cluster_buffer_data[i].pos.x, cluster_buffer_data[i].pos.y, cluster_buffer_data[i].pos.z, cluster_buffer_data[i].dir.x, cluster_buffer_data[i].dir.y, cluster_buffer_data[i].dir.z);
		else if ( length_squared( cluster_buffer_data[i].dir ) < FTINY)
			mprintf("Zero direction in cluster %" PRIu64 " (%g, %g, %g) (%g, %g, %g)\n", i, cluster_buffer_data[i].pos.x, cluster_buffer_data[i].pos.y, cluster_buffer_data[i].pos.z, cluster_buffer_data[i].dir.x, cluster_buffer_data[i].dir.y, cluster_buffer_data[i].dir.z);
#endif
	}
	mprintf("K-means produced %" PRIu64 " of %" PRIu64 " clusters at level %u.\n\n", cluster_count - j, cluster_count, level);

	/* Free memory */
	free(min_distance);
	free(clusters[0]); // allocated inside cuda_kmeans()
	free(clusters);
	free(distance);
	free(membership);
	free(seeds);

	return good_seed_count;
kmmemerr:
	error(SYSTEM, "out of memory in createKMeansClusters");
	return 0;
}
#endif /* AMBIENT_CELL */

#ifdef DAYSIM
static void calcAmbientValues(const RTcontext context, const unsigned int level, const unsigned int max_level, const size_t cluster_count, RTbuffer ambient_record_buffer, void (*freport)(double), RTbuffer ambient_dc_buffer, RTvariable segment_var)
#else
static void calcAmbientValues(const RTcontext context, const unsigned int level, const unsigned int max_level, const size_t cluster_count, RTbuffer ambient_record_buffer, void(*freport)(double))
#endif
{
	AmbientRecord *ambient_record_buffer_data;
	size_t record_count, i;
#ifdef DAYSIM
	float *ambient_dc_buffer_data;
#endif

#ifdef AMB_PARALLEL
	unsigned int divisions = ambientDivisions(ambientWeight(level));
#ifdef DAYSIM
	/* Determine how large the scratch space can be */
	const size_t bytes_per_cluster = sizeof(float) * daysimGetCoefficients() * abs(maxdepth) * 2 * divisions * divisions;
	const size_t clusters_per_segment = bytes_per_cluster ? min(cluster_count, INT_MAX / bytes_per_cluster) : cluster_count; // Limit imposed by OptiX
	if (cluster_count > clusters_per_segment)
		mprintf("Processing ambient records in %" PRIu64 " segments of %" PRIu64 ".\n", (cluster_count - 1) / clusters_per_segment + 1, clusters_per_segment);

	if (daysimGetCoefficients())
		RT_CHECK_ERROR(rtBufferSetSize3D(dc_scratch_buffer, daysimGetCoefficients() * abs(maxdepth) * 2, divisions * divisions, clusters_per_segment));

	for (i = 0u; i < cluster_count; i += clusters_per_segment) {
		const size_t current_count = min(cluster_count - i, clusters_per_segment);

		RT_CHECK_ERROR(rtVariableSet1ui(segment_var, (unsigned int)i));

		/* Run */
		runKernel3D(context, AMBIENT_SAMPLING_ENTRY, divisions, divisions, current_count);
		runKernel2D(context, AMBIENT_ENTRY, CLUSTER_GRID_WIDTH, (current_count * thread_stride - 1) / CLUSTER_GRID_WIDTH + 1);
	}
#else /* DAYSIM */

	/* Run */
	runKernel3D(context, AMBIENT_SAMPLING_ENTRY, divisions, divisions, cluster_count);
	runKernel2D(context, AMBIENT_ENTRY, CLUSTER_GRID_WIDTH, (cluster_count * thread_stride - 1) / CLUSTER_GRID_WIDTH + 1);
#endif /* DAYSIM */
#else /* AMB_PARALLEL */

	/* Run */
	//runKernel1D( context, AMBIENT_ENTRY, cluster_count * thread_stride );
	runKernel2D(context, AMBIENT_ENTRY, CLUSTER_GRID_WIDTH, (cluster_count * thread_stride - 1) / CLUSTER_GRID_WIDTH + 1);
#endif /* AMB_PARALLEL */

	RT_CHECK_ERROR(rtBufferMap(ambient_record_buffer, (void**)&ambient_record_buffer_data));
#ifdef DAYSIM
	RT_CHECK_ERROR(rtBufferMap(ambient_dc_buffer, (void**)&ambient_dc_buffer_data));
#endif

	/* Copy the results to allocated memory. */
	//TODO the buffer could go directly to creating Bvh
	record_count = 0u;
	for (i = 0u; i < cluster_count; i++) {
#ifdef DAYSIM
		record_count += saveAmbientRecords(&ambient_record_buffer_data[i], &ambient_dc_buffer_data[i * daysimGetCoefficients()]);
#else
		record_count += saveAmbientRecords(&ambient_record_buffer_data[i]);
#endif
	}

	RT_CHECK_ERROR(rtBufferUnmap(ambient_record_buffer));
#ifdef DAYSIM
	RT_CHECK_ERROR(rtBufferUnmap(ambient_dc_buffer));
#endif

#ifdef DEBUG_OPTIX
	flushExceptionLog("ambient calculation");
#endif
	if (freport)
		(*freport)(100.0 * (max_level - level) / (max_level + 1));
#ifdef HIT_COUNT
	mprintf("Hit count %u (%f per ambient value).\n", hit_total, (double)hit_total / cluster_count);
	hit_total = 0;
#endif
	mprintf("Retrieved %" PRIu64 " ambient records from %" PRIu64 " queries at level %u.\n\n", record_count, cluster_count, level);

	/* Copy new ambient values into buffer for Bvh. */
	updateAmbientCache(context, level);

	/* Update averages */
	RT_CHECK_ERROR(rtVariableSet1f(avsum_var, (float)avsum));
	RT_CHECK_ERROR(rtVariableSet1ui(navsum_var, navsum));
}

//typedef struct {
//	PointDirection*	data;	/* pointer to data */
//	int	membership;			/* cluster id */
//}  CLUSTER;
//
//static void sortKMeans( const unsigned int cluster_count, PointDirection* cluster_buffer_data )
//{
//	clock_t method_clock; // Timer in clock cycles for short jobs
//	clock_t kernel_clock; // Timer in clock cycles for short jobs
//	PointDirection **clusters, **groups, *temp;
//	unsigned int group_count, i;
//	CLUSTER* sortable;
//
//	int *membership, loops; // output from cuda_kmeans()
//	float *distance, *min_distance; // output from cuda_kmeans()
//
//	group_count = cluster_count / 4u;
//
//	method_clock = clock();
//
//	/* Copy addresses to new array */
//	clusters = (PointDirection**) malloc(cluster_count * sizeof(PointDirection*));
//	for ( i = 0u; i < cluster_count; i++ )
//		clusters[i] = &cluster_buffer_data[i];
//
//	/* Group the seeds into clusters with k-means */
//	membership = (int*) malloc(cluster_count * sizeof(int));
//	distance = (float*) malloc(cluster_count * sizeof(float));
//	kernel_clock = clock();
//	groups = (PointDirection**)cuda_kmeans((float**)clusters, sizeof(PointDirection) / sizeof(float), cluster_count, group_count, (float)cuda_kmeans_threshold, (float)(cuda_kmeans_error / (ambacc * maxarad)), random_seeds, membership, distance, &loops);
//	kernel_clock = clock() - kernel_clock;
//	mprintf("Kmeans performed %u loop iterations in %" PRIu64 " milliseconds.\n", loops, MILLISECONDS(kernel_clock));
//
//	temp = (PointDirection*) malloc(cluster_count * sizeof(PointDirection));
//	sortable = (CLUSTER*) malloc(cluster_count * sizeof(CLUSTER));
//	for ( i = 0u; i < cluster_count; i++ ) {
//		sortable[i].data = clusters[i];
//		sortable[i].membership = membership[i];
//	}
//	qsort( sortable, cluster_count, sizeof(CLUSTER), clusterComparator );
//	for ( i = 0u; i < cluster_count; i++ ) {
//		//clusters[i] = sortable[i].data;
//		temp[i] = *sortable[i].data;
//	}
//	memcpy(cluster_buffer_data, temp, cluster_count * sizeof(PointDirection));
//
//	/* Free memory */
//	free(sortable);
//	free(temp);
//	free(groups[0]); // allocated inside cuda_kmeans()
//	free(groups);
//	free(distance);
//	free(membership);
//	free(clusters);
//
//	method_clock = clock() - method_clock;
//	mprintf("Sorting took %" PRIu64 " milliseconds.\n", MILLISECONDS(method_clock));
//}
//
//static int clusterComparator( const void* a, const void* b )
//{
//	return( ( (CLUSTER*) a )->membership - ( (CLUSTER*) b )->membership );
//}
#endif /* ACCELERAD */
