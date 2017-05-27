/*  
/* Multigrid for solving -u''=f for x in (0,1)x(0,1)
using Jacobi update
using MPI
Ana C. Perez-Gea
used code from Georg Stadler
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

MPI_Request in_right, in_left, in_top, in_bottom, out_right, out_left, out_top, out_bottom;

/* compuate norm of residual */
double compute_norm(double *u, int N)
{
  int i;
  double norm = 0.0;
  for (i = 0; i < (N+1)*(N+1); i++)
    norm += u[i] * u[i];
  return sqrt(norm);
}

/* set vector to zero */
void set_zero (double *u, int N) {
  int i;
  for (i = 0; i < (N+1)*(N+1); i++)
    u[i] = 0.0;
}

/* debug function */
void output_to_screen (double *u, int N) {
  int i;
  for (i = 0; i < (N+1)*(N+1); i++)
    printf("%f ", u[i]);
  printf("\n");
}

/* coarsen uf from length (N+2)*(N+2) to length (N/2+2)*(N/2+2)
   assuming N = 2^l
*/
void coarsen(double *uf, double *uc, int N) {
  int ic, jc;
  for (ic = 1; ic < N/2; ++ic)
    for (jc = 1; jc < N/2; ++jc)
      uc[(N/2+1)*ic+jc] = 0.2 * uf[(N+1)*(ic+1)+2*jc] + 0.2 * (uf[(N+1)*ic+2*jc]+uf[(N+1)*(ic-1)+2*jc]+uf[(N+1)*(ic+1)+2*jc-1]+uf[(N+1)*(ic+1)+2*jc+1]);
}


/* refine u from length (N+2)*(N+2) to lenght (2N+2)*(2N+2)
   assuming N = 2^l, and add to existing uf
*/
void refine_and_add(double *u, double *uf, int N)
{
  int i, j;
  uf[1] += 0.5 * (u[0] + u[1]);
  for (i = 1; i < N; ++i) {
    for (j = 1; j < N; ++j) {
      uf[2*N*i+j] += u[(N+1)*i+j];
      uf[2*N*i+j+1] += 0.25 * (u[(N+1)*i+j] + u[(N+1)*i+j+1] + u[(N+1)*(i-1)+j] + u[(N+1)*(i-1)+j+1]);
    }
  }
}

/* compute residual vector */
void compute_residual(double *u, double *rhs, double *res, int N, double invhsq)
{
  int i,j;
  for (i = 1; i < N; i++){
    for (j = 1; j < N; j++)
      res[(N+1)*i+j] = (rhs[(N+1)*i+j] - (4.0*u[(N+1)*i+j] - u[(N+1)*(i+1)+j] - u[(N+1)*(i-1)+j] - u[(N+1)*i+j-1] - u[(N+1)*i+j+1]) * invhsq);
  }
}


/* compute residual and coarsen */
void compute_and_coarsen_residual(double *u, double *rhs, double *resc,
          int N, double invhsq)
{
  double *resf = calloc(sizeof(double), (N+1)*(N+1));
  compute_residual(u, rhs, resf, N, invhsq);
  coarsen(resf, resc, N);
  free(resf);
}

/* Perform Jacobi iterations on u */
void jacobi(double *u, double *rhs, int N, double hsq, int ssteps)
{
  int i, j, k;
  /* Jacobi damping parameter -- plays an important role in MG */
  double omega = 2./3.;
  double *unew = calloc(sizeof(double), (N+1)*(N+1));
  for (k = 0; k < ssteps; ++k) {
    for (i = 1; i < N; i++){
      for (j = 1; j < N; j++){
        unew[(N+1)*i+j]  = (1-omega)*u[(N+1)*i+j] +  omega * 0.25 * (hsq*rhs[(N+1)*i+j] + u[(N+1)*(i-1)+j] + u[(N+1)*i+(j-1)]+u[(N+1)*(i+1)+j]+u[(N+1)*i+(j+1)]);
      }
    }
    memcpy(u, unew, (N+1)*(N+1)*sizeof(double));
  }
  free (unew);
}

/* Get and send points from other processors */
void getsend(double *u, int N, int rank, int numtasks) {
	if(numtasks > 1){
		int i, j, k;
		int nl = N-1;
		int sqrt_numtasks = (int) sqrt(numtasks);
		/* if there is a left side , get and send points */
		if((int)fmod(rank,sqrt_numtasks) > 0){
			printf("Rank %d has left side points.\n", rank);
			for(j=1; j<=nl; ++j){
				MPI_Irecv(&u[j], 1, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD, &in_left);
				//printf("MPI task %d has received left side point %d from rank %d.\n", rank, j, rank-1);
				MPI_Isend(&u[nl+2+j], 1, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &out_left);
				//printf("MPI task %d has sent left side point %d.\n", rank, nl+2+j);
			}
		}
		/* if there is a right side , get and send points */
		if((int)fmod(rank,sqrt_numtasks) < sqrt_numtasks-1){
			printf("Rank %d has right side points.\n", rank);
			for(j=1; j<=nl; ++j){
				MPI_Irecv(&u[(nl+2)*(nl+1)+j], 1, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &in_right);
				//printf("MPI task %d has received right side point %d.\n", rank, (nl+2)*(nl+1)+j);
				MPI_Isend(&u[(nl+2)*nl+j], 1, MPI_DOUBLE, rank+1, 2, MPI_COMM_WORLD, &out_right);
				//printf("MPI task %d has sent right side point %d to rank %d.\n", rank, (nl+2)*nl+j, rank+1);
			}
		}
		/* if there is a bottom side , get and send points */
		if(rank > (sqrt_numtasks-1)){
			printf("Rank %d has bottom side points.\n", rank);
			for(i=1; i<=nl; ++i){
				MPI_Irecv(&u[i*(nl+2)], 1, MPI_DOUBLE, rank-sqrt_numtasks, 4, MPI_COMM_WORLD, &in_bottom);
				//printf("MPI task %d has received bottom side point %d.\n", rank, i*(nl+2));
				MPI_Isend(&u[i*(nl+2)+1], 1, MPI_DOUBLE, rank-sqrt_numtasks, 3, MPI_COMM_WORLD, &out_bottom);
				//printf("MPI task %d has sent bottom side point %d to rank %d.\n", rank, i*(nl+2)+1, rank-sqrt_numtasks);
			}
		}
		/* if there is a top side , get and send points */
		if(rank < (numtasks-sqrt_numtasks)){
			printf("Rank %d has top side points.\n", rank);
			for(i=1; i<=nl; ++i){
				MPI_Irecv(&u[i*(nl+2)+nl+1], 1, MPI_DOUBLE, rank+sqrt_numtasks, 3, MPI_COMM_WORLD, &in_top);
				//printf("MPI task %d has received top side point %d from rank %d.\n", rank, i*(nl+2)+nl+1, rank+sqrt_numtasks);
				MPI_Isend(&u[i*(nl+2)+nl], 1, MPI_DOUBLE, rank+sqrt_numtasks, 4, MPI_COMM_WORLD, &out_top);
				//printf("MPI task %d has sent top side point %d to rank %d.\n", rank, i*(nl+2)+nl), rank+sqrt_numtasks;
			}
		}
	}
}

void checkgetsend(double *u, int N, int rank, int numtasks) {
	if(numtasks > 1){
		MPI_Status status;
		int sqrt_numtasks = (int) sqrt(numtasks);
		/* if there is a left side , check if Isend/Irecv are done */
		if((int)fmod(rank,sqrt_numtasks) > 0){
			MPI_Wait(&out_left, &status);
			MPI_Wait(&in_left, &status);
			printf("Rank %d has left side points, checked.\n", rank);
		}
		/* if there is a right side , check if Isend/Irecv are done */
		if((int)fmod(rank,sqrt_numtasks) < sqrt_numtasks-1){
			MPI_Wait(&out_right, &status);
			MPI_Wait(&in_right, &status);
			printf("Rank %d has right side points, checked.\n", rank);
		}
		/* if there is a bottom side , check if Isend/Irecv are done */
		if(rank > (sqrt_numtasks-1)){
			MPI_Wait(&out_bottom, &status);
			MPI_Wait(&in_bottom, &status);
			printf("Rank %d has bottom side points, checked.\n", rank);
		}
		/* if there is a top side , check if Isend/Irecv are done */
		if(rank < (numtasks-sqrt_numtasks)){
			MPI_Wait(&out_top, &status);
			MPI_Wait(&in_top, &status);
			printf("Rank %d has top side points, checked.\n", rank);
		}
	}
}

int main (int argc, char **argv)
{
	int i, j, k, Nfine, l, iter, max_iters, levels, ssteps = 3;
	double start, end;

	if (argc < 3 || argc > 4) {
		fprintf(stderr, "Usage: ./multigrid_1d Nfine maxiter [s-steps]\n");
		fprintf(stderr, "Nfine: # of intervals, must be power of two number\n");
		fprintf(stderr, "s-steps: # jacobi smoothing steps (optional, default is 3)\n");
		abort();
	}
	sscanf(argv[1], "%d", &Nfine);
	sscanf(argv[2], "%d", &max_iters);
	if (argc > 3)
		sscanf(argv[3], "%d", &ssteps);

	/* compute number of multigrid levels */
	levels = floor(log2(Nfine));
	printf("Multigrid Solve using V-cycles for -u'' = f on (0,1)\n");
	printf("Number of intervals = %d, max_iters = %d\n", Nfine, max_iters);
	printf("Number of MG levels: %d \n", levels);

	int numtasks, rank, len, buffer, root, rc, source, dest;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	start = MPI_Wtime();

	printf ("MPI task %d has started...\n", rank);
	if (fmod(log(numtasks)/log(4), 1.0) != 0) {
		printf("ERROR: Number of tasks must be a power of 4. Quitting.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}

	/* Allocation of vectors, including left and right bdry points */
	double *u[levels], *rhs[levels];
	/* N, h*h and 1/(h*h) on each level */
	int *N = (int*) calloc(sizeof(int), levels);
	double *invhsq = (double* ) calloc(sizeof(double), levels);
	double *hsq = (double* ) calloc(sizeof(double), levels);
	double *res = (double *) calloc(sizeof(double), (Nfine+1)*(Nfine+1));
	for (l = 0; l < levels; ++l) {
		N[l] = Nfine / (int) pow(2,l);
		double h = 1.0 / N[l];
		hsq[l] = h * h;
		printf("MG level %2d, N = %8d\n", l, N[l]);
		invhsq[l] = 1.0 / hsq[l];
		u[l]    = (double *) calloc(sizeof(double), (N[l]+1)*(N[l]+1));
		rhs[l] = (double *) calloc(sizeof(double), (N[l]+1)*(N[l]+1));
	}
	/* rhs on finest mesh */
	for (i = 0; i <= N[0]*N[0]; ++i) {
		rhs[0][i] = 1.0;
	}
	double res_norm, res0_norm, tol = 1e-6;

	/* initial residual norm */
	compute_residual(u[0], rhs[0], res, N[0], invhsq[0]);
	res_norm = res0_norm = compute_norm(res, N[0]);
	printf("Initial Residual: %f\n", res0_norm); 

	if (rank == 0 ) 
		printf ("MPI started with %d tasks.\n", numtasks);
	for(k = 0; k < max_iters; ++k){
		/* V-cycle: Coarsening */
		for (l = 0; l < levels-1; ++l) {
			getsend(u[l], N[l], rank, numtasks);
			/* pre-smoothing and coarsen */
			jacobi(u[l], rhs[l], N[l], hsq[l], ssteps);
			compute_and_coarsen_residual(u[l], rhs[l], rhs[l+1], N[l], invhsq[l]);
			/* initialize correction for solution with zero */
			set_zero(u[l+1],N[l+1]);
			checkgetsend(u[l], N[l], rank, numtasks);
			printf("Level %dfinished on rank %d.\n", l, rank);
		}
		getsend(u[levels-1], N[levels-1], rank, numtasks);
		/* V-cycle: Solve on coarsest grid using many smoothing steps */
		jacobi(u[levels-1], rhs[levels-1], N[levels-1], hsq[levels-1], 50);
		checkgetsend(u[levels-1], N[levels-1], rank, numtasks);
		/* V-cycle: Refine and correct */
		for (l = levels-1; l > 0; --l) {
			getsend(u[l], N[l], rank, numtasks);
			/* refine and add to u */
			refine_and_add(u[l], u[l-1], N[l]);
			/* post-smoothing steps */
			jacobi(u[l-1], rhs[l-1], N[l-1], hsq[l-1], ssteps);
			checkgetsend(u[l], N[l], rank, numtasks);
		}
		if (0 == (iter % 1)) {
			compute_residual(u[0], rhs[0], res, N[0], invhsq[0]);
			res_norm = compute_norm(res, N[0]);
			printf("[Iter %d] Residual norm: %2.8f\n", iter, res_norm);
		}
	}

	/* time it */
	MPI_Barrier(MPI_COMM_WORLD);
	end = MPI_Wtime();
	if(0==rank){
		printf("Jacobi done in %ld iterations. \n",k);
		printf("Runtime = %f\n", end-start);
	}
	/* Clean up */
	free (hsq);
	free (invhsq);
	free (N);
	free(res);
	for (l = levels-1; l >= 0; --l) {
		free(u[l]);
		free(rhs[l]);
	}
	MPI_Finalize();
}