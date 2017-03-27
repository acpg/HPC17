/*  
Solves the Laplace equation in two space dimensions
using Gauss-Seidel method with red-black coloring
Ana C. Perez-Gea
used code from Georg Stadler
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "util.h"
#ifdef _OPENMP
#include <omp.h>
#endif

int main (int argc, char **argv)
{
	long k, i, j, n, n2, max_k;
	double h, *u, *u0;

	if(argc != 2) {
		fprintf(stderr, "Function needs vector size as input argument!\n");
		abort();
	}

	# pragma omp parallel
	{
	#ifdef _OPENMP
		int my_threadnum = omp_get_thread_num();
		int numthreads = omp_get_num_threads();
	#else
		int my_threadnum = 0;
		int numthreads = 1;
	#endif
	printf("Hello, I'm thread %d out of %d\n", my_threadnum, numthreads);
	}

	n = atoi(argv[1]);
	n2 = (n+1)*(n+1);
	max_k = 10;
	h = 1./(n+1.);
	h *= h;
	u0 = (double *) malloc(sizeof(double) * n2);	
	u = (double *) malloc(sizeof(double) * n2);	

	/* fill vector u0 */
	for(i=0; i<n2; ++i){
		u0[i] = 0.0;
	}
	u = u0;

	timestamp_type time1, time2;
	get_timestamp(&time1);

	/* GS with red-black coloring */
	for(k = 0; k < max_k; k++){
		#pragma omp parallel private(i,j) shared(u,u0)
		// red coloring
		for(i=1; i<=n; i++){
			for(j=2-(i % 2); j<=n; j=j+2){
				u[n*i+j] = h;
				u[n*i+j] += u0[(i-1)*n+j];
				u[n*i+j] += u0[i*n+(j-1)];
				u[n*i+j] += u0[(i+1)*n+j];
				u[n*i+j] += u0[i*n+(j+1)];
				u[n*i+j] /= 4.0;
				printf("red i= %ld j= %ld \n",i,j);
			}
		}

		#pragma omp parallel private(i,j) shared(u,u0)
		// black coloring
		for(i=1; i<=n; i++){
			for(j=1+(i % 2); j<=n; j=j+2){
				u[n*i+j] = h;
				u[n*i+j] += u[(i-1)*n+j];
				u[n*i+j] += u[i*n+(j-1)];
				u[n*i+j] += u[(i+1)*n+j];
				u[n*i+j] += u[i*n+(j+1)];
				u[n*i+j] /= 4.0;
				printf("black i= %ld j= %ld \n",i,j);
			}
		}

		u0 = u;
		printf("Iter %ld with midpoint value %f. \n",k,u[n2/2]);
	}
	get_timestamp(&time2);
	double elapsed = timestamp_diff_in_seconds(time1,time2);
	printf("Jacobi done. \n");
	printf("Time elapsed is %f seconds.\n", elapsed);

	/* print final vector u vs sol 
	printf("\n");
	for (i=0; i<n; ++i){
		printf("%f     %f", u[i], (i+1)*h*(1.-(i+1)*h)/2.);
	    printf("\n");
	 }*/

	free(u0);
	free(u);
	return 0;
}