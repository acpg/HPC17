/* Communication ping-pong:
 * Exchange between messages between mpirank
 * 0 <-> 1, 2 <-> 3, ....
 */
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include "util.h"

int main( int argc, char *argv[])
{
  int i, size, rank, tag, origin, destination;
  int message_out = 0;
  timestamp_type time1, time2;
  get_timestamp(&time1);
  MPI_Status status;

  //char hostname[1024];
  //gethostname(hostname, 1024);

  if(argc != 2) {
    fprintf(stderr, "Function needs an input argument!\n");
    abort();
  }
  int n = atoi(argv[1]);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //printf("My rank %d of %d\n", rank, size );
  destination = (rank + 1) % size;
  origin = rank - 1;
  tag = 99;

  if (rank == 0){
    origin += size;
    for (i=0; i<n; i++){
      message_out += rank;
      int message_in = -1;
      MPI_Send(&message_out, 1, MPI_INT, destination, tag, MPI_COMM_WORLD);
      MPI_Recv(&message_in,  1, MPI_INT, origin,      tag, MPI_COMM_WORLD, &status);
      message_out = message_in;
      printf("rank %d received from %d the message %d\n", rank, origin, message_in);
    }
  } else{
    for (i=0; i<n; i++){
      int message_in = -1;
      MPI_Recv(&message_in,  1, MPI_INT, origin,      tag, MPI_COMM_WORLD, &status);
      message_out = message_in;
      message_out += rank;
      MPI_Send(&message_out, 1, MPI_INT, destination, tag, MPI_COMM_WORLD);
      printf("rank %d received from %d the message %d\n", rank, origin, message_in);
    }
  }
  MPI_Finalize();
  get_timestamp(&time2);
  double elapsed = timestamp_diff_in_seconds(time1,time2);
  printf("Time elapsed is %f seconds.\n", elapsed);
  return 0;
}
