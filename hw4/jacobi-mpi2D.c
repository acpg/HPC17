/*  
Solves the Laplace equation in two space dimensions
using Jacobi update
using MPI
Ana C. Perez-Gea
used code from Georg Stadler
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "util.h"
#include "mpi.h"

int main (int argc, char **argv)
{
	long k, i, j, n, nj, nl, n2, max_k;
	double h, *u, *u0;

	if(argc != 2) {
		fprintf(stderr, "Function needs vector size as input argument!\n");
		abort();
	}
	n = atoi(argv[1]);

	int   numtasks, rank, len, buffer, root, rc, source, dest;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	if (rank == 0 ) {
	  if (fmod(log(numtasks)/log(4), 1.0) != 0) {
	    printf("ERROR: Number of tasks must be a power of 4. Quitting.\n");
	    MPI_Abort(MPI_COMM_WORLD, rc);
	  }
		nj = log(numtasks)/log(4);
		nl = ceil(n/(nj+1));
		n2 = (nl+2)*(nl+2);
		max_k = 1000;
		h = 1./(nl+1.);
		h *= h;
		u0 = (double *) malloc(sizeof(double) * n2);
		u = (double *) malloc(sizeof(double) * n2);	

		/* fill vector u0 */
		for(i=0; i<n2; ++i){
			u0[i] = 0.0;
		}
		u = u0;

		/* Jacobi */
		for(k = 0; k < max_k; k++){
			for(i=1; i<=nl; ++i){
				for(j=1; j<=nl; ++j){
					u[nl*i+j] = h+u0[(i-1)*nl+j]+u0[i*nl+(j-1)]+u0[(i+1)*nl+j]+u0[i*nl+(j+1)];
					u[nl*i+j] /= 4.0;
				}
				if(nj>0){
					dest = rank + 1;
					MPI_Send(&u[nl*i+j], 1, MPI_DOUBLE, dest, k, MPI_COMM_WORLD);
				}
			}
			if(nj>0){
				dest = rank + sqrt(pow(4,nj));
				source = rank+sqrt(pow(4,nj));
				for(j=1; j<=nl; ++j){
					MPI_Send(&u[nl*i+j], 1, MPI_DOUBLE, dest, k, MPI_COMM_WORLD);
					MPI_Recv(&u[nl*(i+1)+j], 1, MPI_DOUBLE, source, k, MPI_COMM_WORLD, &status);
				}
			}
			u0 = u;
			//printf("Iter %ld with midpoint value %f. \n",k,u[n2/2]);
		}
	}

	if (rank > 0 ) {
		nj = log(numtasks)/log(4);
		nl = ceil(n/(nj+1));
		n2 = (nl+2)*(nl+2);
		max_k = 1000;
		h = 1./(nl+1.);
		h *= h;
		u0 = (double *) malloc(sizeof(double) * n2);
		u = (double *) malloc(sizeof(double) * n2);	

		/* fill vector u0 */
		for(i=0; i<n2; ++i){
			u0[i] = 0.0;
		}
		u = u0;

		for(k = 0; k < max_k; k++){
		}
	}

	printf("Jacobi done in %ld iterations. \n",k);

	MPI_Finalize();
}