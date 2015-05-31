/*
 * Copyright (c) 2013-2015 Nathaniel Jones
 * Massachusetts Institute of Technology
 */

#include <stdio.h>

#include "ray.h" // required by ambient.h
#include "ambient.h"

#include "optix_radiance.h"


#ifdef ACCELERAD

static void updateAmbientCache( const RTcontext context, const unsigned int level );
static void createAmbientRecordCamera( const RTcontext context, const VIEW* view );
static void createGeometryInstanceAmbient( const RTcontext context, RTgeometryinstance* instance, const unsigned int ambinet_record_count );
static void createAmbientAcceleration( const RTcontext context, const RTgeometryinstance instance );
static unsigned int populateAmbientRecords( const RTcontext context, const int level );
#ifdef DAYSIM
static unsigned int gatherAmbientRecords( AMBTREE* at, AmbientRecord** records, float** dc, const int level );
static int saveAmbientRecords( AmbientRecord* record, float* dc );
#else
static unsigned int gatherAmbientRecords( AMBTREE* at, AmbientRecord** records, const int level );
static int saveAmbientRecords( AmbientRecord* record );
#endif

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

/* Allow faster irradiance cache creation by leaving some amount of threads unused.
	For 1D launch:
		Quadro K4000: Optimal is 4
		Tesla K40c: Optimal is 16
	For 2D (1xN) launch:
		Optimal is 2
*/
const unsigned int thread_stride = 4u;

#ifdef HIT_COUNT
static unsigned long hit_total = 0uL;
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

static void createAmbientRecordCamera( const RTcontext context, const VIEW* view )
{
	RTprogram  ray_gen_program;
	RTprogram  exception_program;

	if ( view ) { // use camera to pick points
		/* Ray generation program */
		ptxFile( path_to_ptx, "ambient_generator" );
		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "ambient_camera", &ray_gen_program ) );

		/* Exception program */
		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "exception", &exception_program ) );
	} else { // read input buffer for points
		/* Ray generation program */
		ptxFile( path_to_ptx, "ambient_cloud_generator" );
		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "ambient_cloud_camera", &ray_gen_program ) );

		/* Exception program */
		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "exception", &exception_program ) );

		/* Stride for these programs */
		applyContextVariable1ui(context, "stride", thread_stride);
	}
	RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, AMBIENT_ENTRY, ray_gen_program ) );
	RT_CHECK_ERROR( rtContextSetExceptionProgram( context, AMBIENT_ENTRY, exception_program ) );

	/* Define ray types */
	applyContextVariable1ui( context, "ambient_record_ray_type", AMBIENT_RECORD_RAY );
}

static void createGeometryInstanceAmbient( const RTcontext context, RTgeometryinstance* instance, const unsigned int ambinet_record_count )
{
	RTprogram  ambient_record_intersection_program;
	RTprogram  ambient_record_bounding_box_program;
	RTprogram  ambient_record_any_hit_program;
	RTprogram  ambient_record_miss_program;
	RTmaterial ambient_record_material;

	/* Create the geometry reference for OptiX. */
	RT_CHECK_ERROR( rtGeometryCreate( context, &ambient_record_geometry ) );
	RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( ambient_record_geometry, ambinet_record_count ) );

	RT_CHECK_ERROR( rtMaterialCreate( context, &ambient_record_material ) );

	/* Define ray types */
	applyContextVariable1ui( context, "ambient_ray_type", AMBIENT_RAY );

	ptxFile( path_to_ptx, "ambient_records" );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "ambient_record_bounds", &ambient_record_intersection_program ) );
	RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( ambient_record_geometry, ambient_record_intersection_program ) );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "ambient_record_intersect", &ambient_record_bounding_box_program ) );
	RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( ambient_record_geometry, ambient_record_bounding_box_program ) );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "ambient_record_any_hit", &ambient_record_any_hit_program ) );
	RT_CHECK_ERROR( rtMaterialSetAnyHitProgram( ambient_record_material, AMBIENT_RAY, ambient_record_any_hit_program ) );

	/* Miss program */
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "ambient_miss", &ambient_record_miss_program ) );
	RT_CHECK_ERROR( rtContextSetMissProgram( context, AMBIENT_RAY, ambient_record_miss_program ) );
	// TODO miss program could handle makeambient()

	/* Create the geometry instance containing the ambient records. */
	RT_CHECK_ERROR( rtGeometryInstanceCreate( context, instance ) );
	RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( *instance, ambient_record_geometry ) );
	RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( *instance, 1 ) );

	/* Apply this material to the geometry instance. */
	RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( *instance, 0, ambient_record_material ) );
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
		if (ambient_records == NULL)
			error(SYSTEM, "out of memory in populateAmbientRecords");
		ambient_records_ptr = ambient_records;
#ifdef DAYSIM
		ambient_dc = (float*)malloc(sizeof(float) * daysimGetCoefficients() * nambvals);
		if (ambient_dc == NULL)
			error(SYSTEM, "out of memory in populateAmbientRecords");
		ambient_dc_ptr = ambient_dc;

		/* Get the ambient records from the octree structure. */
		useful_record_count = gatherAmbientRecords(&atrunk, &ambient_records_ptr, &ambient_dc_ptr, level);
#else
		/* Get the ambient records from the octree structure. */
		useful_record_count = gatherAmbientRecords( &atrunk, &ambient_records_ptr, level );
#endif
		vprintf("Using %u of %u ambient records\n", useful_record_count, nambvals);
	}

	/* Resize the buffer of ambient records. */
	RT_CHECK_ERROR( rtBufferSetSize1D( ambient_record_input_buffer, useful_record_count ) );
#ifdef DAYSIM
	RT_CHECK_ERROR(rtBufferSetSize2D(ambient_dc_input_buffer, useful_record_count ? daysimGetCoefficients() : 0, useful_record_count));
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
#ifndef OLDAMB
			array2cuda2( (*records)->gpos, record->gpos );
			array2cuda2( (*records)->gdir, record->gdir );
			array2cuda2( (*records)->rad, record->rad );

			(*records)->ndir = record->ndir;
			(*records)->udir = record->udir;
			(*records)->corral = record->corral;
#else
			array2cuda3( (*records)->dir, record->dir );
			array2cuda3( (*records)->gpos, record->gpos );
			array2cuda3( (*records)->gdir, record->gdir );

			(*records)->rad = record->rad;
#endif
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
#ifndef OLDAMB
	cuda2array2( amb.gpos, record->gpos );
	cuda2array2( amb.gdir, record->gdir );
	cuda2array2( amb.rad, record->rad );

	amb.ndir = record->ndir;
	amb.udir = record->udir;
	amb.corral = record->corral;
#else
	cuda2array3( amb.dir, record->dir );
	cuda2array3( amb.gpos, record->gpos );
	cuda2array3( amb.gdir, record->gdir );

	amb.rad = record->rad;
#endif
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

#ifndef KMEANS_IC
void createAmbientRecords( const RTcontext context, const VIEW* view, const int width, const int height, const double alarm )
{
	RTvariable     level_var;
	RTbuffer       ambient_record_buffer;
	AmbientRecord* ambient_record_buffer_data;

	unsigned int scaled_width, scaled_height, generated_record_count, useful_record_count, level, i;

	if ( optix_amb_scale ) {
		scaled_width = width / optix_amb_scale;
		scaled_height = height / optix_amb_scale;
	} else {
		scaled_width = width;
		scaled_height = height;
	}
	generated_record_count = scaled_width * scaled_height * optix_amb_semgents;
	level = ambounce;

	createAmbientRecordCamera( context, view );

	RT_CHECK_ERROR( rtContextDeclareVariable( context, "level", &level_var ) ); // Could be camera program variable

	/* Create buffer for retrieving ambient records. */
	createCustomBuffer3D( context, RT_BUFFER_OUTPUT, sizeof(AmbientRecord), scaled_width, scaled_height, optix_amb_semgents, &ambient_record_buffer );
	applyContextObject( context, "ambient_record_buffer", ambient_record_buffer );
	applyContextVariable1ui( context, "segments", optix_amb_semgents );

	/* Put any existing ambient records into GPU cache. There should be none. */
	setupAmbientCache( context, 0u ); // Do this now to avoid additional setup time later

	while ( level-- ) {
		RT_CHECK_ERROR( rtVariableSet1ui( level_var, level ) );

		//ambdone();

		/* Run */
		runKernel2D( context, AMBIENT_ENTRY, scaled_width, scaled_height );

		RT_CHECK_ERROR( rtBufferMap( ambient_record_buffer, (void**)&ambient_record_buffer_data ) );
		RT_CHECK_ERROR( rtBufferUnmap( ambient_record_buffer ) );

		/* Copy the results to allocated memory. */
		//TODO the buffer could go directoy to creating Bvh
		useful_record_count = 0u;
		for (i = 0u; i < generated_record_count; i++) {
			useful_record_count += saveAmbientRecords( &ambient_record_buffer_data[i] );
		}
#ifdef DEBUG_OPTIX
		flushExceptionLog("ambient calculation");
#endif
#ifdef RAY_COUNT
		reportProgress( 100.0 * (ambounce - level) / (ambounce + 1), alarm );
#endif
#ifdef HIT_COUNT
		mprintf("Hit count %u (%f per ambient value).\n", hit_total, (float)hit_total / generated_record_count);
		hit_total = 0;
#endif
		mprintf("Retrieved %u ambient records from %u queries at level %u.\n\n", useful_record_count, generated_record_count, level);

		// Populate ambinet records
		updateAmbientCache( context, level );

		/* Update ambient average. */
		RT_CHECK_ERROR( rtVariableSet1f( avsum_var, avsum ) );
		RT_CHECK_ERROR( rtVariableSet1ui( navsum_var, navsum ) );
	}

}
#else /* KMEANS_IC */
#define CLUSTER_GRID_WIDTH	1u
#define length_squared(v)	(v.x*v.x)+(v.y*v.y)+(v.z*v.z)
#define is_nan(v)			(v.x!=v.x)||(v.y!=v.y)||(v.z!=v.z)

static unsigned int ambientDivisions(unsigned int level);
static void createPointCloudCamera( const RTcontext context, const VIEW* view );
#ifdef AMB_PARALLEL
static void createAmbientSamplingCamera( const RTcontext context );
#endif
#ifdef ITERATIVE_KMEANS_IC
static void createHemisphereSamplingCamera( const RTcontext context );
#endif
static unsigned int chooseAmbientLocations(const RTcontext context, const unsigned int level, const unsigned int width, const unsigned int height, const unsigned int seeds_per_thread, RTbuffer seed_buffer, RTbuffer cluster_buffer);
static unsigned int createKMeansClusters(const unsigned int seed_count, const unsigned int cluster_count, PointDirection* seed_buffer_data, PointDirection* cluster_buffer_data, const unsigned int level);
#ifdef DAYSIM
static void calcAmbientValues(const RTcontext context, const unsigned int level, const double alarm, RTbuffer ambient_record_buffer, RTbuffer ambient_dc_buffer, RTvariable segment_var);
#else
static void calcAmbientValues(const RTcontext context, const unsigned int level, const double alarm, RTbuffer ambient_record_buffer);
#endif
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

#define ADAPTIVE_SEED_SAMPLING
#ifdef ADAPTIVE_SEED_SAMPLING
void cuda_score_hits(PointDirection *hits, int *seeds, const unsigned int width, const unsigned int height, const float weight, const unsigned int seed_count);
#endif /* ADAPTIVE_SEED_SAMPLING */

void createAmbientRecords( const RTcontext context, const VIEW* view, const int width, const int height, const double alarm )
{
	RTvariable     level_var;
	RTbuffer       seed_buffer, ambient_record_buffer;
#ifdef DAYSIM_COMPATIBLE
	RTbuffer       ambient_dc_buffer;
#endif
#ifdef DAYSIM
	RTvariable     segment_var = NULL;
#endif
#ifdef ITERATIVE_KMEANS_IC
	RTvariable     current_cluster_buffer;
	RTbuffer*      cluster_buffer;
#else
	RTbuffer       cluster_buffer;
#endif

	unsigned int grid_width, grid_height, level;

	if ( view && optix_amb_grid_size ) { // if -az argument greater than zero
		grid_width = optix_amb_grid_size / 2;
		grid_height = optix_amb_grid_size;
	} else if ( view && optix_amb_scale ) { // if -al argument greater than zero
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
#ifdef ITERATIVE_KMEANS_IC
	createHemisphereSamplingCamera( context );
#endif
	createAmbientRecordCamera( context, NULL ); // Use input from kmeans to sample points instead of camera

	/* Create buffer for retrieving potential seed points. */
	createCustomBuffer3D( context, RT_BUFFER_OUTPUT, sizeof(PointDirection), 0, 0, 0, &seed_buffer );
	applyContextObject( context, "seed_buffer", seed_buffer );

	/* Create buffer for inputting seed point clusters. */
#ifdef ITERATIVE_KMEANS_IC
	cluster_buffer = (RTbuffer*) malloc(ambounce * sizeof(RTbuffer));
	if (cluster_buffer == NULL)
		error(SYSTEM, "out of memory in createAmbientRecords");
	for ( level = 0u; level < ambounce; level++ ) {
		createCustomBuffer1D( context, RT_BUFFER_INPUT, sizeof(PointDirection), cuda_kmeans_clusters, &cluster_buffer[level] );
	}
	current_cluster_buffer = applyContextObject(context, "cluster_buffer", cluster_buffer[0]);
#else
	createCustomBuffer1D( context, RT_BUFFER_INPUT, sizeof(PointDirection), cuda_kmeans_clusters, &cluster_buffer );
	applyContextObject( context, "cluster_buffer", cluster_buffer );
#endif

	/* Create buffer for retrieving ambient records. */
	createCustomBuffer1D( context, RT_BUFFER_OUTPUT, sizeof(AmbientRecord), cuda_kmeans_clusters, &ambient_record_buffer );
	applyContextObject( context, "ambient_record_buffer", ambient_record_buffer );
#ifdef DAYSIM_COMPATIBLE
#ifdef DAYSIM
#ifdef AMB_PARALLEL
	/* Create variable for offset into scratch space */
	RT_CHECK_ERROR(rtContextDeclareVariable(context, "segment_offset", &segment_var));
#endif
	createBuffer2D(context, RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, daysimGetCoefficients(), cuda_kmeans_clusters, &ambient_dc_buffer);
#else
	createBuffer2D(context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, 0, 0, &ambient_dc_buffer);
#endif
	applyContextObject(context, "ambient_dc_buffer", ambient_dc_buffer);
#endif

	/* Set additional variables used for ambient record generation */
	level_var = applyContextVariable1ui(context, "level", 0u); // Could be camera program variable

	/* Put any existing ambient records into GPU cache. There should be none. */
	setupAmbientCache( context, 0u ); // Do this now to avoid additional setup time later

#ifdef ITERATIVE_KMEANS_IC
	for ( level = 0u; level < ambounce; level++ ) {
		chooseAmbientLocations(context, level, grid_width, grid_height, 1u, seed_buffer, cluster_buffer[level]);

		/* Set input buffer index for next iteration */
		RT_CHECK_ERROR( rtVariableSetObject( current_cluster_buffer, cluster_buffer[level] ) );
	}
#else /* ITERATIVE_KMEANS_IC */
	chooseAmbientLocations(context, 0u, grid_width, grid_height, optix_amb_seeds_per_thread, seed_buffer, cluster_buffer);
	level = ambounce;
#endif /* ITERATIVE_KMEANS_IC */
#ifndef AMB_PARALLEL
#ifdef DAYSIM
	RT_CHECK_ERROR(rtBufferSetSize3D(dc_scratch_buffer, daysimGetCoefficients() * maxdepth * 2, 1u, cuda_kmeans_clusters));
#endif
#endif /* AMB_PARALLEL */

	while ( level-- ) {
		RT_CHECK_ERROR( rtVariableSet1ui( level_var, level ) );
#ifdef DAYSIM
		calcAmbientValues(context, level, alarm, ambient_record_buffer, ambient_dc_buffer, segment_var);
#else
		calcAmbientValues(context, level, alarm, ambient_record_buffer);
#endif
#ifdef ITERATIVE_KMEANS_IC
		if ( level )
			RT_CHECK_ERROR( rtVariableSetObject( current_cluster_buffer, cluster_buffer[level - 1] ) );
#endif
	}

#ifdef ITERATIVE_KMEANS_IC
	free(cluster_buffer);
#endif
}

static unsigned int ambientDivisions(unsigned int level)
{
	unsigned int i, divisions;
	double weight = 1.0; //TODO set by level?
	for ( i = level; i--; )
		weight *= AVGREFL; //  Compute weight as in makeambient() from ambient.c
	divisions = sqrt(ambdiv * weight) + 0.5;
	i = 1 + 5 * (ambacc > FTINY);	/* minimum number of samples */
	if (divisions < i)
		return i;
	return divisions;
}

static void createPointCloudCamera( const RTcontext context, const VIEW* view )
{
	RTprogram  ray_gen_program;
	RTprogram  exception_program;
	RTprogram  miss_program;

	/* Ray generation program */
	if ( view ) {
		ptxFile( path_to_ptx, "point_cloud_generator" );
		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "point_cloud_camera", &ray_gen_program ) );

		if (optix_amb_grid_size) // Ignore camera for bounds of sampling area
			applyProgramVariable1ui(context, ray_gen_program, "camera", 0u); // Hide context variable
	} else {
		ptxFile( path_to_ptx, "sensor_cloud_generator" );
		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "cloud_generator", &ray_gen_program ) );
	}
	RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, POINT_CLOUD_ENTRY, ray_gen_program ) );

	/* Exception program */
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "exception", &exception_program ) );
	RT_CHECK_ERROR( rtContextSetExceptionProgram( context, POINT_CLOUD_ENTRY, exception_program ) );

	/* Define ray types */
	applyContextVariable1ui( context, "point_cloud_ray_type", POINT_CLOUD_RAY );

	/* Miss program */
	ptxFile( path_to_ptx, "point_cloud_normal" );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "point_cloud_miss", &miss_program ) );
	RT_CHECK_ERROR( rtContextSetMissProgram( context, POINT_CLOUD_RAY, miss_program ) );
}

#ifdef AMB_PARALLEL
static void createAmbientSamplingCamera( const RTcontext context )
{
	RTprogram  program;

	/* Ray generation program */
	ptxFile( path_to_ptx, "ambient_sample_generator" );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "ambient_sample_camera", &program ) );
	RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, AMBIENT_SAMPLING_ENTRY, program ) );

	/* Exception program */
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "exception", &program ) );
	RT_CHECK_ERROR( rtContextSetExceptionProgram( context, AMBIENT_SAMPLING_ENTRY, program ) );
}
#endif /* AMB_PARALLEL */

#ifdef ITERATIVE_KMEANS_IC
static void createHemisphereSamplingCamera( const RTcontext context )
{
	RTprogram  program;

	/* Ray generation program */
	ptxFile( path_to_ptx, "hemisphere_generator" );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "hemisphere_camera", &program ) );
	RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, HEMISPHERE_SAMPLING_ENTRY, program ) );

	/* Exception program */
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "exception", &program ) );
	RT_CHECK_ERROR( rtContextSetExceptionProgram( context, HEMISPHERE_SAMPLING_ENTRY, program ) );
}
#endif /* ITERATIVE_KMEANS_IC */

static unsigned int chooseAmbientLocations(const RTcontext context, const unsigned int level, const unsigned int width, const unsigned int height, const unsigned int seeds_per_thread, RTbuffer seed_buffer, RTbuffer cluster_buffer)
{
	unsigned int seed_count;
	PointDirection *seed_buffer_data, *cluster_buffer_data;

	if (level) {
		/* Adjust output buffer size */
		unsigned int divisions = ambientDivisions(level);
		seed_count = divisions * divisions * cuda_kmeans_clusters;
		RT_CHECK_ERROR(rtBufferSetSize3D(seed_buffer, divisions, divisions, cuda_kmeans_clusters));

		/* Run kernel to gerate more seed points from cluster centers */
		runKernel3D(context, HEMISPHERE_SAMPLING_ENTRY, divisions, divisions, cuda_kmeans_clusters); // stride?
	}
	else {
		/* Set output buffer size */
		seed_count = width * height * seeds_per_thread;
		RT_CHECK_ERROR(rtBufferSetSize3D(seed_buffer, width, height, seeds_per_thread));

		/* Run the kernel to get the first set of seed points. */
		runKernel2D(context, POINT_CLOUD_ENTRY, width, height);
	}

	/* Retrieve potential seed points. */
	RT_CHECK_ERROR(rtBufferMap(seed_buffer, (void**)&seed_buffer_data));
	RT_CHECK_ERROR(rtBufferUnmap(seed_buffer));

#ifdef ITERATIVE_KMEANS_IC
#ifdef ADAPTIVE_SEED_SAMPLING
	if (!level) {
		clock_t kernel_clock;
		//int total = 0;
		unsigned int i;
		unsigned int missing = 0u;
		unsigned int si = cuda_kmeans_clusters;
		unsigned int ci = 0u;
		int *score = (int*)malloc(seed_count * sizeof(int));
		PointDirection *temp_list = (PointDirection*)malloc(seed_count * sizeof(PointDirection));
		if (score == NULL || temp_list == NULL)
			error(SYSTEM, "out of memory in chooseAmbientLocations");

		kernel_clock = clock();
		cuda_score_hits(seed_buffer_data, score, width, height, cuda_kmeans_error / (ambacc * maxarad), cuda_kmeans_clusters);
		kernel_clock = clock() - kernel_clock;
		mprintf("Adaptive sampling: %llu milliseconds.\n", kernel_clock * 1000uLL / CLOCKS_PER_SEC);

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
#endif /* ADAPTIVE_SEED_SAMPLING */
#endif /* ITERATIVE_KMEANS_IC */

	/* Group seed points into clusters and add clusters to buffer */
	RT_CHECK_ERROR(rtBufferMap(cluster_buffer, (void**)&cluster_buffer_data));
	seed_count = createKMeansClusters(seed_count, cuda_kmeans_clusters, seed_buffer_data, cluster_buffer_data, level);
	//sortKMeans( cuda_kmeans_clusters, cluster_buffer_data );
	RT_CHECK_ERROR(rtBufferUnmap(cluster_buffer));
	return seed_count;
}

static unsigned int createKMeansClusters( const unsigned int seed_count, const unsigned int cluster_count, PointDirection* seed_buffer_data, PointDirection* cluster_buffer_data, const unsigned int level )
{
	clock_t kernel_clock; // Timer in clock cycles for short jobs
	unsigned int good_seed_count, i, j;
	PointDirection **seeds, **clusters; // input and output for cuda_kmeans()
	int *membership, loops; // output from cuda_kmeans()
	float *distance, *min_distance; // output from cuda_kmeans()

	/* Eliminate bad seeds and copy addresses to new array */
	good_seed_count = 0u;
	seeds = (PointDirection**) malloc(seed_count * sizeof(PointDirection*));
	if (seeds == NULL)
		error(SYSTEM, "out of memory in createKMeansClusters");
	//TODO Is there any point in filtering out values that are not valid (length(seed_buffer_data[i].dir) == 0)?
	for ( i = 0u; i < seed_count; i++) {
		if ( length_squared( seed_buffer_data[i].dir ) < FTINY ) {
#ifdef DEBUG_OPTIX
			if (length_squared(seed_buffer_data[i].pos) > FTINY)
				logException((RTexception)seed_buffer_data[i].pos.x);
#endif
		} else if ( is_nan( seed_buffer_data[i].pos ) || is_nan( seed_buffer_data[i].dir ) )
			mprintf("NaN in seed %u (%g, %g, %g) (%g, %g, %g)\n", i, seed_buffer_data[i].pos.x, seed_buffer_data[i].pos.y, seed_buffer_data[i].pos.z, seed_buffer_data[i].dir.x, seed_buffer_data[i].dir.y, seed_buffer_data[i].dir.z);
		else
			seeds[good_seed_count++] = &seed_buffer_data[i];
	}
#ifdef DEBUG_OPTIX
	flushExceptionLog("ambient seeding");
#endif
	mprintf("Retrieved %u of %u potential seeds at level %u.\n", good_seed_count, seed_count, level);

	/* Check that enough seeds were found */
	if ( good_seed_count <= cluster_count ) {
		mprintf("Using all %u seeds at level %u (%u needed for k-means).\n\n", good_seed_count, level, cluster_count);
		for ( i = 0u; i < good_seed_count; i++ )
			cluster_buffer_data[i] = *seeds[i];
		for ( ; i < cluster_count; i++ )
			cluster_buffer_data[i].dir.x = cluster_buffer_data[i].dir.y = cluster_buffer_data[i].dir.z = 0.0f; // Don't use this value
		free(seeds);
		return good_seed_count;
	}

	/* Check that k-means should be used */
	if ( !cuda_kmeans_iterations ) {
		mprintf("Using first %u seeds at level %u.\n\n", cluster_count, level);
		for ( i = 0u; i < cluster_count; i++ )
			cluster_buffer_data[i] = *seeds[i]; //TODO should randomly choose from array
		free(seeds);
		return good_seed_count;
	}

	/* Group the seeds into clusters with k-means */
	membership = (int*) malloc(good_seed_count * sizeof(int));
	distance = (float*) malloc(good_seed_count * sizeof(float));
	if (membership == NULL || distance == NULL)
		error(SYSTEM, "out of memory in createKMeansClusters");
	kernel_clock = clock();
	clusters = (PointDirection**)cuda_kmeans((float**)seeds, sizeof(PointDirection) / sizeof(float), good_seed_count, cluster_count, cuda_kmeans_iterations, cuda_kmeans_threshold, cuda_kmeans_error / (ambacc * maxarad), level, membership, distance, &loops);
	kernel_clock = clock() - kernel_clock;
	mprintf("K-means performed %u loop iterations in %llu milliseconds.\n", loops, kernel_clock * 1000uLL / CLOCKS_PER_SEC);

	/* Populate buffer of seed point clusters. */
	min_distance = (float*) malloc(cluster_count * sizeof(float));
	if (min_distance == NULL)
		error(SYSTEM, "out of memory in createKMeansClusters");
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
			mprintf("NaN distance from seed %u to cluster %u\n", i, j);
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
			mprintf("NaN in cluster %u (%g, %g, %g) (%g, %g, %g)\n", i, cluster_buffer_data[i].pos.x, cluster_buffer_data[i].pos.y, cluster_buffer_data[i].pos.z, cluster_buffer_data[i].dir.x, cluster_buffer_data[i].dir.y, cluster_buffer_data[i].dir.z);
		else if (length_squared( cluster_buffer_data[i].dir ) < FTINY)
			mprintf("Zero direction in cluster %u (%g, %g, %g) (%g, %g, %g)\n", i, cluster_buffer_data[i].pos.x, cluster_buffer_data[i].pos.y, cluster_buffer_data[i].pos.z, cluster_buffer_data[i].dir.x, cluster_buffer_data[i].dir.y, cluster_buffer_data[i].dir.z);
#endif
	}
	mprintf("K-means produced %u of %u clusters at level %u.\n\n", cluster_count - j, cluster_count, level);

	/* Free memory */
	free(min_distance);
	free(clusters[0]); // allocated inside cuda_kmeans()
	free(clusters);
	free(distance);
	free(membership);
	free(seeds);

	return good_seed_count;
}

#ifdef DAYSIM
static void calcAmbientValues(const RTcontext context, const unsigned int level, const double alarm, RTbuffer ambient_record_buffer, RTbuffer ambient_dc_buffer, RTvariable segment_var)
#else
static void calcAmbientValues(const RTcontext context, const unsigned int level, const double alarm, RTbuffer ambient_record_buffer)
#endif
{
	AmbientRecord *ambient_record_buffer_data;
	unsigned int record_count, i;
#ifdef DAYSIM
	float *ambient_dc_buffer_data;
#endif

#ifdef AMB_PARALLEL
	unsigned int divisions = ambientDivisions(level);
#ifdef DAYSIM
	/* Determine how large the scratch space can be */
	unsigned int kmeans_clusters_per_segment = cuda_kmeans_clusters;
	size_t bytes_per_kmeans_cluster = sizeof(float) * daysimGetCoefficients() * maxdepth * 2 * divisions * divisions;
	while (bytes_per_kmeans_cluster * kmeans_clusters_per_segment > INT_MAX) { // Limit imposed by OptiX
		kmeans_clusters_per_segment = (kmeans_clusters_per_segment - 1) / 2 + 1;
	}

	RT_CHECK_ERROR(rtBufferSetSize3D(dc_scratch_buffer, daysimGetCoefficients() * maxdepth * 2, divisions * divisions, kmeans_clusters_per_segment));

	for (i = 0u; i < cuda_kmeans_clusters; i += kmeans_clusters_per_segment) {
		RT_CHECK_ERROR(rtVariableSet1ui(segment_var, i));

		/* Run */
		runKernel3D(context, AMBIENT_SAMPLING_ENTRY, divisions, divisions, kmeans_clusters_per_segment);
		runKernel2D(context, AMBIENT_ENTRY, CLUSTER_GRID_WIDTH, (kmeans_clusters_per_segment * thread_stride - 1) / CLUSTER_GRID_WIDTH + 1);
	}
#else /* DAYSIM */

	/* Run */
	runKernel3D(context, AMBIENT_SAMPLING_ENTRY, divisions, divisions, cuda_kmeans_clusters);
	runKernel2D(context, AMBIENT_ENTRY, CLUSTER_GRID_WIDTH, (cuda_kmeans_clusters * thread_stride - 1) / CLUSTER_GRID_WIDTH + 1);
#endif /* DAYSIM */
#else /* AMB_PARALLEL */

	/* Run */
	//runKernel1D( context, AMBIENT_ENTRY, cuda_kmeans_clusters * thread_stride );
	runKernel2D(context, AMBIENT_ENTRY, CLUSTER_GRID_WIDTH, (cuda_kmeans_clusters * thread_stride - 1) / CLUSTER_GRID_WIDTH + 1);
#endif /* AMB_PARALLEL */

	RT_CHECK_ERROR(rtBufferMap(ambient_record_buffer, (void**)&ambient_record_buffer_data));
	RT_CHECK_ERROR(rtBufferUnmap(ambient_record_buffer));
#ifdef DAYSIM
	RT_CHECK_ERROR(rtBufferMap(ambient_dc_buffer, (void**)&ambient_dc_buffer_data));
	RT_CHECK_ERROR(rtBufferUnmap(ambient_dc_buffer));
#endif

	/* Copy the results to allocated memory. */
	//TODO the buffer could go directly to creating Bvh
	record_count = 0u;
	for (i = 0u; i < cuda_kmeans_clusters; i++) {
#ifdef DAYSIM
		record_count += saveAmbientRecords(&ambient_record_buffer_data[i], &ambient_dc_buffer_data[i * daysimGetCoefficients()]);
#else
		record_count += saveAmbientRecords(&ambient_record_buffer_data[i]);
#endif
	}
#ifdef DEBUG_OPTIX
	flushExceptionLog("ambient calculation");
#endif
#ifdef RAY_COUNT
	reportProgress(100.0 * (ambounce - level) / (ambounce + 1), alarm);
#endif
#ifdef HIT_COUNT
	mprintf("Hit count %u (%f per ambient value).\n", hit_total, (float)hit_total / cuda_kmeans_clusters);
	hit_total = 0;
#endif
	mprintf("Retrieved %u ambient records from %u queries at level %u.\n\n", record_count, cuda_kmeans_clusters, level);

	/* Copy new ambient values into buffer for Bvh. */
	updateAmbientCache(context, level);

	/* Update averages */
	RT_CHECK_ERROR(rtVariableSet1f(avsum_var, avsum));
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
//	groups = (PointDirection**)cuda_kmeans((float**)clusters, sizeof(PointDirection) / sizeof(float), cluster_count, group_count, cuda_kmeans_threshold, cuda_kmeans_error / (ambacc * maxarad), random_seeds, membership, distance, &loops);
//	kernel_clock = clock() - kernel_clock;
//	mprintf("Kmeans performed %u loop iterations in %llu milliseconds.\n", loops, kernel_clock * 1000uLL / CLOCKS_PER_SEC);
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
//	mprintf("Sorting took %llu milliseconds.\n", method_clock * 1000uLL / CLOCKS_PER_SEC);
//}
//
//static int clusterComparator( const void* a, const void* b )
//{
//	return( ( (CLUSTER*) a )->membership - ( (CLUSTER*) b )->membership );
//}
#endif /* KMEANS_IC */
#endif /* ACCELERAD */
