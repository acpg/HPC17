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
	int p, k, i, j, nj, nl, n2, k_max;
	double h, *u, *u0;

	if(argc != 2) {
		fprintf(stderr, "Function needs vector size as input argument!\n");
		abort();
	}
	nl = atoi(argv[1]);
	n2 = (nl+2)*(nl+2);
	h = 1./(nl+1.);
	h *= h;
	k_max = 2;

	int   numtasks, rank, len, buffer, root, rc, source, dest;
	MPI_Init(&argc, &argv);
	MPI_Status status;
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	printf ("MPI task %d has started...\n", rank);
	p = numtasks;

	u0 = (double *) malloc(sizeof(double) * n2);
	u = (double *) malloc(sizeof(double) * n2);	

	/* fill vector u0 */
	for(i=0; i<n2; ++i){
		u0[i] = 0.0;
	}
	u = u0;

	/* Processor 0 will do main work */
	if (rank == 0 ) {
		if (fmod(log(numtasks)/log(4), 1.0) != 0) {
	    	printf("ERROR: Number of tasks must be a power of 4. Quitting.\n");
	    	MPI_Abort(MPI_COMM_WORLD, rc);
	    }

		for(k = 0; k < k_max; ++k){
			if(k>0){
				/* if there is a right side , get points */
				for(j=1; j<=nl; ++j){
					MPI_Recv(&u0[(nl+2)*(nl+1)+j], 1, MPI_DOUBLE, rank+1, k, MPI_COMM_WORLD, &status);
					printf("MPI task %d has received right side point %d.\n", rank,(nl+2)*(nl+1)+j);
				}
				/* if there is a top side , get points */
				for(i=1; i<=nl; ++i){
					MPI_Recv(&u0[i*(nl+2)+nl+1], 1, MPI_DOUBLE, (int) rank+sqrt(p), k, MPI_COMM_WORLD, &status);
					printf("MPI task %d has received top side point %d.\n", rank, i*(nl+2)+nl+1);
				}
			}
			/* Jacobi */
			for(i=1; i<=nl; ++i){
				for(j=1; j<=nl; ++j){
					u[(nl+2)*i+j] = (h+u0[(nl+2)*(i-1)+j]+u0[(nl+2)*i+(j-1)]+u0[(nl+2)*(i+1)+j]+u0[(nl+2)*i+j+1])/4.0;
				}
			}
			/* if there is a right side , send points */
			for(j=1; j<=nl; ++j){
				MPI_Send(&u[(nl+2)*nl+j], 1, MPI_DOUBLE, rank+1, k+1, MPI_COMM_WORLD);
				printf("MPI task %d has sent right side point %d.\n", rank, (nl+2)*nl+j);
			}
			/* if there is a top side , send points */
			for(i=1; i<=nl; ++i){
				MPI_Send(&u[i*(nl+2)+nl], 1, MPI_DOUBLE, (int) rank + sqrt(p), k+1, MPI_COMM_WORLD);
				printf("MPI task %d has sent top side point %d.\n", rank, i*(nl+2)+nl);
			}
			/* save as old point to start new iteration */
			u0 = u;
			printf("Iter %ld in processor %d with some value %f. \n",k,rank,u[nl*nl]);
		}
	}

	/* other processors */
	if (rank > 0 ) {
		for(k = 0; k < k_max; ++k){
			if(k>0){
				/* if there is a left side , get points */
				if(fmod(rank,sqrt(p)) > 0){
					for(j=1; j<=nl; ++j){
						MPI_Recv(&u0[j], 1, MPI_DOUBLE, rank-1, k, MPI_COMM_WORLD, &status);
						printf("MPI task %d has received left side point.\n", rank);
					}
				}
				/* if there is a right side , get points */
				if(fmod(rank,sqrt(p)) < sqrt(p)-1){
					for(j=1; j<=nl; ++j){
						MPI_Recv(&u0[(nl+2)*(nl+1)+j], 1, MPI_DOUBLE, rank+1, k, MPI_COMM_WORLD, &status);
						printf("MPI task %d has received right side point.\n", rank);
					}
				}
				/* if there is a bottom side , get points */
				if(rank > (sqrt(p)-1)){
					for(i=1; i<=nl; ++i){
						MPI_Recv(&u0[i*(nl+2)], 1, MPI_DOUBLE, (int) rank-sqrt(p), k, MPI_COMM_WORLD, &status);
						printf("MPI task %d has received bottom side point.\n", rank);
					}
				}
				/* if there is a top side , get points */
				if(rank < (p-sqrt(p))){
					for(i=1; i<=nl; ++i){
						MPI_Recv(&u0[i*(nl+2)+nl+1], 1, MPI_DOUBLE, (int) rank+sqrt(p), k, MPI_COMM_WORLD, &status);
						printf("MPI task %d has received top side point.\n", rank);
					}
				}
			}
			/* Jacobi */
			for(i=1; i<=nl; ++i){
				for(j=1; j<=nl; ++j){
					u[(nl+2)*i+j] = (h+u0[(nl+2)*(i-1)+j]+u0[(nl+2)*i+(j-1)]+u0[(nl+2)*(i+1)+j]+u0[(nl+2)*i+j+1])/4.0;
				}
			}
			/* if there is a left side , send points */
			if(fmod(rank,sqrt(p)) > 0){
				for(j=1; j<=nl; ++j){
					MPI_Send(&u[nl+2+j], 1, MPI_DOUBLE, rank-1, k+1, MPI_COMM_WORLD);
					printf("MPI task %d has sent left side point.\n", rank);
				}
			}
			/* if there is a right side , send points */
			if(fmod(rank,sqrt(p)) < (sqrt(p)-1)){
				for(j=1; j<=nl; ++j){
					MPI_Send(&u[(nl+2)*nl+j], 1, MPI_DOUBLE, rank+1, k+1, MPI_COMM_WORLD);
					printf("MPI task %d has sent right side point.\n", rank);
				}
			}
			/* if there is a bottom side , send points */
			if(rank > (sqrt(p)-1)){
				for(i=1; i<=nl; ++i){
					MPI_Send(&u[i*(nl+2)+1], 1, MPI_DOUBLE, (int) rank-sqrt(p), k+1, MPI_COMM_WORLD);
					printf("MPI task %d has sent bottom side point.\n", rank);
				}
			}
			/* if there is a top side , send points */
			if(rank < (p-sqrt(p))){
				for(i=1; i<=nl; ++i){
					MPI_Send(&u[i*(nl+2)+nl], 1, MPI_DOUBLE, (int) rank+sqrt(p), k+1, MPI_COMM_WORLD);
					printf("MPI task %d has sent top side point.\n", rank);
				}
			}
			/* save as old point to start new iteration */
			u0 = u;
			printf("Iter %ld in processor %d with some value %f. \n",k,rank,u[nl*nl]);
		}
	}

	printf("Jacobi done in %ld iterations. \n",k);

	MPI_Finalize();
}