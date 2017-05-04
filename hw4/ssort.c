/* Parallel sample sort
Ana C. Perez-Gea
High Performance Computing
Used code from Greog Stadler
 */
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>


static int compare(const void *a, const void *b)
{
  int *da = (int *)a;
  int *db = (int *)b;

  if (*da > *db)
    return 1;
  else if (*da < *db)
    return -1;
  else
    return 0;
}

int main( int argc, char *argv[])
{
  int rank, numtasks;
  int i, N, NN;
  int *vec, *svec, *svec0, *ivec, *ivecn, *rivec, *rivecn;
  double start, end;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  start = MPI_Wtime();

  /* Number of random numbers per processor */
  N = atoi(argv[1]);

  vec = calloc(N, sizeof(int));
  /* seed random number generator differently on every core */
  srand((unsigned int) (rank + 393919));

  /* fill vector with random integers */
  for (i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  //printf("rank: %d, first entry: %d\n", rank, vec[0]);

  /* sort locally */
  qsort(vec, N, sizeof(int), compare);
  //printf("rank: %d, first entry: %d\n", rank, vec[0]);

  /* randomly sample s entries from vector or select local splitters,
   * i.e., every N/P-th entry of the sorted vector */
  svec = calloc(numtasks-1, sizeof(int));
  for (i = 0; i < numtasks-1; i++) {
    svec[i] = vec[(i+1)*N/numtasks-1];
    //printf("Rank %d %d\n",rank,svec[i]);
  }

  /* every processor communicates the selected entries
   * to the root processor; use for instance an MPI_Gather */
  if(rank==0){
    svec0 = (int *)malloc((numtasks-1)*numtasks* sizeof(int));
  }
  MPI_Gather(svec, numtasks-1, MPI_INT, svec0, numtasks-1, MPI_INT, 0, MPI_COMM_WORLD);

  /* root processor does a sort, determinates splitters that
   * split the data into P buckets of approximately the same size */
  if(rank==0){
    qsort(svec0, (numtasks-1)*numtasks, sizeof(int), compare);
    for (i = 0; i < numtasks-1; i++) {
      svec[i] = svec0[(i+1)*(numtasks-1)-1];
    }
  }

  /* root process broadcasts splitters */
  MPI_Bcast(svec, numtasks-1, MPI_INT, 0, MPI_COMM_WORLD);

  /* every processor uses the obtained splitters to decide
   * which integers need to be sent to which other processor (local bins) */
  ivec = calloc(numtasks, sizeof(int));
  ivecn = calloc(numtasks, sizeof(int));
  for (i=0; i<numtasks-1; ++i){
    while (vec[ivec[i+1]] < svec[i]){
      ivec[i+1]++;
    }
    ivecn[i] = ivec[i+1];
  }
  ivecn[numtasks-1] = N;
  for(i=numtasks-1; i>0; i--){
    ivecn[i] -= ivecn[i-1];
    //printf("Rank %d offset %d w %d elements\n",rank,ivec[i],ivecn[i]);
  }

  /* send and receive: either use an MPI_Alltoall to share
   * with every processor how many integers it should expect,
   * and then use MPI_Alltoallv to exchange the data */
  
  rivecn = (int *)malloc(numtasks*sizeof(int));
  MPI_Alltoall(ivecn, 1, MPI_INT, rivecn, 1, MPI_INT, MPI_COMM_WORLD);
  NN = 0;
  for (i = 0; i < numtasks; i++) {
    NN += rivecn[i];
  }
  //printf("Rank %d needs %d elements\n",rank,NN);
  svec0 = (int *)malloc(NN*sizeof(int));
  ivec = calloc(numtasks, sizeof(int));
  rivec = calloc(numtasks, sizeof(int));
  for (i=1; i<numtasks; i++){
    ivec[i] = ivec[i-1]+ivecn[i-1];
    rivec[i] = rivec[i-1]+rivecn[i-1];
  }
  MPI_Alltoallv(vec, ivecn, ivec, MPI_INT, svec0, rivecn, rivec, MPI_INT, MPI_COMM_WORLD);

  /* do a local sort */
  qsort(svec0, NN, sizeof(int), compare);

  /* time it */
  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();
  if(rank==0)
    printf("Runtime = %f\n", end-start);

  /* every processor writes its result to a file */
  {
    FILE* fd = NULL;
    char filename[256];
    snprintf(filename, 256, "output%02d.txt", rank);
    fd = fopen(filename,"w+");

    if(NULL == fd)
    {
      printf("Error opening file \n");
      return 1;
    }

    fprintf(fd, "rank %d has the integers:\n", rank);
    for(i = 0; i < NN; ++i)
      fprintf(fd, "  %d\n", svec0[i]);

    fclose(fd);
  }

  free(vec);
  free(svec);
  free(svec0);
  free(ivec);
  free(ivecn);
  MPI_Finalize();
  return 0;
}
