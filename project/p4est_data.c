/*  
Datatree
Ana C. Perez-Gea
Code adapted from p4est example file p4est_step1.c
*/
#ifndef P4_TO_P8
#include <p4est_vtk.h>
#else
#include <p8est_vtk.h>
#endif

#ifdef P4_TO_P8
static const p4est_qcoord_t eighth = P4EST_QUADRANT_LEN (3);
#endif

/* Parameters to adjust */
#define max_pow0 7 // no deeper than this level (power of 2)
int max_size = 500; // keep splitting if we have more than this many data points
int data_base = 1000; // data point power base (for simulating data)
int max_data = 10000000; //max points to simulate, adjust so it covers all your points

/* Parameters not to touch */
float *array, *cols, *rows, max_col, min_col, max_row, min_row;
int count = 0; // actual number read from file
#define max_quad0 (1<<max_pow0)
static const int max_pow = max_pow0;
static const int max_quad = max_quad0; 

/** Callback function to decide on refinement. */
static int refine_fn (p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t * quadrant){
  int tilelen, inside, ql;
  int i, j, offsi, offsj;
  float dx, dy, qx, qy;

  /* We do not want to refine deeper than a given maximum level. */
  if (quadrant->level > max_pow) {
    return 0;
  }
  #ifdef P4_TO_P8
  /* In 3D we extrude the 2D image in the z direction between [3/8, 5/8]. */
  if (quadrant->level >= 3 && (quadrant->z < 3 * eighth || quadrant->z >= 5 * eighth)) {
    return 0;
  }
  #endif

  inside = 0;
  qx = quadrant->x;
  qy = quadrant->y;
  ql = quadrant->level;
  offsi = quadrant->x / P4EST_QUADRANT_LEN (max_pow);
  offsj = quadrant->y / P4EST_QUADRANT_LEN (max_pow);
  qx = (float)offsi / (float)max_quad;
  qy = (float)offsj / (float)max_quad;
  for(i=0; i<count; i++){
    dx = (rows[i]-min_row) / (max_row-min_row);
    dy = (cols[i]-min_col) / (max_col-min_col);
    //printf("dx in rows[%d]=%f\n", i, dx);
    //printf("dy in cols[%d]=%f\n", i, dy);
    if(qx<dx && dx<(qx+1.0/(float)(ql+1))){
      //printf("condition in rows < %f + %f\n", qx,1.0/(float)(ql+1));
      if(qy<dy && dy<(qy+1.0/(float)(ql+1))){
        //printf("condition in cols < %f + %f\n", qy,1.0/(float)(ql+1));
        inside++;
        if(inside > max_size){
          //printf("WIN rows[%d]=%f\n", i, rows[i]);
          //printf("WIN cols[%d]=%f\n", i, cols[i]);
          return 1;
        }
      }
    }
  }
  return 0;
}

/* Main function creates a connectivity and forest, refines it, and writes a VTK file. */
int main (int argc, char **argv) {
  int                 mpiret, i;
  int                 recursive, partforcoarsen, balance;
  sc_MPI_Comm         mpicomm;
  p4est_t            *p4est;
  p4est_connectivity_t *conn;
  double t1, t2; 

  /* Open file we will use to simulate data points */
  FILE *fp;
  int row,col,inc;
  float data, u1, u2;
  char ch;
  //array = (float*)malloc(sizeof(float)*ple*ple);
  cols = (float*)malloc(sizeof(float)*max_data);
  rows = (float*)malloc(sizeof(float)*max_data);
  fp = fopen("example/steps/logo.txt","r");
  row = col = 0;
  while(EOF!=(inc=fscanf(fp,"%f%c", &data, &ch)) && inc == 2){
      //array[count] = data;
      for(i=0;i<(int)pow(data_base,(0.6549-data));i++){
        u1 = (float)rand() / (float)RAND_MAX ;
        rows[count] = row+u1;
        if(rows[count]<min_row || 0==count){
          min_row = rows[count];
        }
        if(rows[count]>max_row || 0==count){
          max_row = rows[count];
        }
        u2 = (float)rand() / (float)RAND_MAX ;
        cols[count] = col+u2;
        if(cols[count]<min_col || 0==count){
          min_col = cols[count];
        }
        if(cols[count]>max_col || 0==count){
          max_col = cols[count];
        }
        ++count;
        if(count==max_data){
          goto exit;
        }
    }
    ++col;
      if(ch == '\n'){
          ++row;
          col = 0;
      } else if(ch != ','){
          fprintf(stderr, "Different separator (%c) of row at %d \n", ch, row);
          goto exit;
      }
  }
  exit:
    fclose(fp);

  printf("Min row: %f\n", min_row);
  printf("Max row: %f\n", max_row);
  printf("Min col: %f\n", min_col);
  printf("Max col: %f\n", max_col);
  printf("Total points: %d\n", count);

  /* Initialize MPI */
  mpiret = sc_MPI_Init (&argc, &argv);
  SC_CHECK_MPI (mpiret);
  mpicomm = sc_MPI_COMM_WORLD;
  // Get the rank of the process
  int world_rank;
  sc_MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  t1 = sc_MPI_Wtime();

  /* Store the MPI rank as a static variable so subsequent global p4est log messages are only issued from processor zero.  */
  sc_init (mpicomm, 1, 1, NULL, SC_LP_ESSENTIAL);
  p4est_init (NULL, SC_LP_PRODUCTION);
  P4EST_GLOBAL_PRODUCTIONF
    ("This is the p4est %dD data\n", P4EST_DIM);

  /* Create a forest that consists of just one quadtree/octree. */
#ifndef P4_TO_P8
  conn = p4est_connectivity_new_unitsquare ();
#else
  conn = p8est_connectivity_new_unitcube ();
#endif

  /* Create a forest that is not refined; it consists of the root octant. */
  p4est = p4est_new (mpicomm, conn, 0, NULL, NULL);

  /* Refine the forest recursively in parallel. */
  recursive = 1;
  p4est_refine (p4est, recursive, refine_fn, NULL);

  /* Partition: The quadrants are redistributed for equal element count. */
  partforcoarsen = 1;
  p4est_partition (p4est, partforcoarsen, NULL);

  /* If we call the 2:1 balance we ensure that neighbors do not differ in size by more than a factor of 2.  */
  balance = 1;
  if (balance) {
    p4est_balance (p4est, P4EST_CONNECT_FACE, NULL);
    p4est_partition (p4est, partforcoarsen, NULL);
  }
  printf("Hi this is processor %d\n", world_rank);
  /* Write the forest to disk for visualization, one file per processor. */
  p4est_vtk_write_file (p4est, NULL, P4EST_STRING "_data");

  /* Destroy the p4est and the connectivity structure. */
  p4est_destroy (p4est);
  p4est_connectivity_destroy (conn);

  /* Verify that allocations internal to p4est and sc do not leak memory. */
  sc_finalize ();

  t2 = sc_MPI_Wtime(); 
  printf( "Elapsed time is %f\n", t2 - t1 ); 

  /* This is standard MPI programs.  Without --enable-mpi, this is a dummy. */
  mpiret = sc_MPI_Finalize ();
  SC_CHECK_MPI (mpiret);
  return 0;
}