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
#include <mpi.h>

int main (int argc, char **argv)
{
	int k, i, j, nl;
	double start, end;

	if(argc == 2){
		nl = atoi(argv[1]);
	}
	else{
		nl = 100;
	}
	double h = 1.0/(nl+1);
	double hsq = h*h;
	int k_max = 100;

	int numtasks, rank, len, buffer, root, rc, source, dest;
	MPI_Init(&argc, &argv);
	MPI_Status status;
	MPI_Request in_right, in_left, in_top, in_bottom, out_right, out_left, out_top, out_bottom;
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	start = MPI_Wtime();
	int sqrt_numtasks = (int)sqrt(numtasks);
	//printf ("MPI task %d has started...\n", rank);
	if (fmod(log(numtasks)/log(4), 1.0) != 0) {
		printf("ERROR: Number of tasks must be a power of 4. Quitting.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}

	double* u0 = (double *) calloc(sizeof(double), (nl+2)*(nl+2));
	double* u = (double *) calloc(sizeof(double), (nl+2)*(nl+2));	
	double* utemp;

	/* Processor 0 will do main work */
	if (rank == 0 ) {
		printf ("MPI started with %d tasks.\n", numtasks);
		for(k = 0; k < k_max; ++k){
			if(numtasks > 1){
				/* if there is a right side , get and send points */
				for(j=1; j<=nl; ++j){
					MPI_Irecv(&u[(nl+2)*(nl+1)+j], 1, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &in_right);
					//printf("MPI task %d has received right side point %d from rank %d.\n", rank, (nl+2)*(nl+1)+j, rank+1);
					MPI_Isend(&u[(nl+2)*nl+j], 1, MPI_DOUBLE, rank+1, 2, MPI_COMM_WORLD, &out_right);
					//printf("MPI task %d has sent right side point %d.\n", rank, (nl+2)*nl+j);
				}
				/* if there is a top side , get and send points */
				for(i=1; i<=nl; ++i){
					MPI_Irecv(&u[i*(nl+2)+nl+1], 1, MPI_DOUBLE, rank+sqrt_numtasks, 3, MPI_COMM_WORLD, &in_top);
					//printf("MPI task %d has received top side point %d from rank %d.\n", rank, i*(nl+2)+nl+1, rank+sqrt_numtasks);
					MPI_Isend(&u[i*(nl+2)+nl], 1, MPI_DOUBLE, rank+sqrt_numtasks, 4, MPI_COMM_WORLD, &out_top);
					//printf("MPI task %d has sent top side point %d.\n", rank, i*(nl+2)+nl);
				}
			}
			/* Jacobi */
			for(i=1; i<=nl; ++i){
				for(j=1; j<=nl; ++j){
					u[(nl+2)*i+j] = (hsq+u0[(nl+2)*(i-1)+j]+u0[(nl+2)*i+(j-1)]+u0[(nl+2)*(i+1)+j]+u0[(nl+2)*i+j+1])/4.0;
				}
			}
			if(numtasks > 1){
				/* check if Isend/Irecv are done */
				MPI_Wait(&out_right, &status);
				MPI_Wait(&in_right, &status);
				MPI_Wait(&out_top, &status);
				MPI_Wait(&in_top, &status);
			}
			/* save as old point to start new iteration */
			utemp = u0; u0 = u; u = utemp;
			//printf("Iter %ld in processor %d with some value %f. \n",k,rank,u[nl*nl]);
		}
	}

	/* other processors */
	if (rank > 0 ) {
		for(k = 0; k < k_max; ++k){
			/* if there is a left side , get and send points */
			if((int)fmod(rank,sqrt_numtasks) > 0){
				//printf("Rank %d has left side points.\n", rank);
				for(j=1; j<=nl; ++j){
					MPI_Irecv(&u[j], 1, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD, &in_left);
					//printf("MPI task %d has received left side point %d from rank %d.\n", rank, j, rank-1);
					MPI_Isend(&u[nl+2+j], 1, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &out_left);
					//printf("MPI task %d has sent left side point %d.\n", rank, nl+2+j);
				}
			}
			/* if there is a right side , get and send points */
			if((int)fmod(rank,sqrt_numtasks) < sqrt_numtasks-1){
				//printf("Rank %d has right side points.\n", rank);
				for(j=1; j<=nl; ++j){
					MPI_Irecv(&u[(nl+2)*(nl+1)+j], 1, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &in_right);
					//printf("MPI task %d has received right side point %d.\n", rank, (nl+2)*(nl+1)+j);
					MPI_Isend(&u[(nl+2)*nl+j], 1, MPI_DOUBLE, rank+1, 2, MPI_COMM_WORLD, &out_right);
					//printf("MPI task %d has sent right side point %d to rank %d.\n", rank, (nl+2)*nl+j, rank+1);
				}
			}
			/* if there is a bottom side , get and send points */
			if(rank > (sqrt_numtasks-1)){
				//printf("Rank %d has bottom side points.\n", rank);
				for(i=1; i<=nl; ++i){
					MPI_Irecv(&u[i*(nl+2)], 1, MPI_DOUBLE, rank-sqrt_numtasks, 4, MPI_COMM_WORLD, &in_bottom);
					//printf("MPI task %d has received bottom side point %d.\n", rank, i*(nl+2));
					MPI_Isend(&u[i*(nl+2)+1], 1, MPI_DOUBLE, rank-sqrt_numtasks, 3, MPI_COMM_WORLD, &out_bottom);
					//printf("MPI task %d has sent bottom side point %d to rank %d.\n", rank, i*(nl+2)+1, rank-sqrt_numtasks);
				}
			}
			/* if there is a top side , get and send points */
			if(rank < (numtasks-sqrt_numtasks)){
				//printf("Rank %d has top side points.\n", rank);
				for(i=1; i<=nl; ++i){
					MPI_Irecv(&u[i*(nl+2)+nl+1], 1, MPI_DOUBLE, rank+sqrt_numtasks, 3, MPI_COMM_WORLD, &in_top);
					//printf("MPI task %d has received top side point %d from rank %d.\n", rank, i*(nl+2)+nl+1, rank+sqrt_numtasks);
					MPI_Isend(&u[i*(nl+2)+nl], 1, MPI_DOUBLE, rank+sqrt_numtasks, 4, MPI_COMM_WORLD, &out_top);
					//printf("MPI task %d has sent top side point %d to rank %d.\n", rank, i*(nl+2)+nl), rank+sqrt_numtasks;
				}
			}
			/* Jacobi */
			for(i=1; i<=nl; ++i){
				for(j=1; j<=nl; ++j){
					u[(nl+2)*i+j] = (hsq+u0[(nl+2)*(i-1)+j]+u0[(nl+2)*i+(j-1)]+u0[(nl+2)*(i+1)+j]+u0[(nl+2)*i+j+1])/4.0;
				}
			}
			/* if there is a left side , check if Isend/Irecv are done */
			if((int)fmod(rank,sqrt_numtasks) > 0){
				MPI_Wait(&out_left, &status);
				MPI_Wait(&in_left, &status);
			}
			/* if there is a right side , check if Isend/Irecv are done */
			if((int)fmod(rank,sqrt_numtasks) < sqrt_numtasks-1){
				MPI_Wait(&out_right, &status);
				MPI_Wait(&in_right, &status);
			}
			/* if there is a bottom side , check if Isend/Irecv are done */
			if(rank > (sqrt_numtasks-1)){
				MPI_Wait(&out_bottom, &status);
				MPI_Wait(&in_bottom, &status);
			}
			/* if there is a top side , check if Isend/Irecv are done */
			if(rank < (numtasks-sqrt_numtasks)){
				MPI_Wait(&out_top, &status);
				MPI_Wait(&in_top, &status);
			}
			/* save as old point to start new iteration */
			utemp = u0; u0 = u; u = utemp;
			//printf("Iter %ld in processor %d with some value %f. \n",k,rank,u[nl*nl]);
		}
	}
	/* time it */
	MPI_Barrier(MPI_COMM_WORLD);
	end = MPI_Wtime();
	if(0==rank){
		printf("Jacobi done in %ld iterations. \n",k);
		printf("Runtime = %f\n", end-start);
	}
	free(u);
	free(u0);
	MPI_Finalize();
}