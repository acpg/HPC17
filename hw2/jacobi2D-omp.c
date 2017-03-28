/*  
Solves the Laplace equation in two space dimensions
using Jacobi update
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

	n = atoi(argv[1]);
	n2 = (n+1)*(n+1);
	max_k = 1000;
	h = 1./(n+1.);
	h *= h;
	u0 = (double *) malloc(sizeof(double) * n2);
	u = (double *) malloc(sizeof(double) * n2);	

	/* fill vector u0 */
	for(i=0; i<n2; ++i){
		u0[i] = 0.0;
	}
	u = u0;

	// timestamp
	timestamp_type time1, time2;
	get_timestamp(&time1);

	/* Jacobi */
	for(k = 0; k < max_k; k++){
		omp_set_num_threads(25);
		#pragma omp parallel private(i,j) shared(u,u0)
		{
			#ifdef _OPENMP
				int my_threadnum = omp_get_thread_num();
				//int numthreads = omp_get_num_threads();
			#else
				int my_threadnum = 0;
				//int numthreads = 1;
			#endif
			for(i=1+my_threadnum; i<=n; i=i+1+my_threadnum){
				for(j=1; j<=n; ++j){
					u[n*i+j] = h;
					u[n*i+j] += u0[(i-1)*n+j];
					u[n*i+j] += u0[i*n+(j-1)];
					u[n*i+j] += u0[(i+1)*n+j];
					u[n*i+j] += u0[i*n+(j+1)];
					u[n*i+j] /= 4.0;
				}
			}
			//printf("Hello, I'm thread %d out of %d\n", my_threadnum, numthreads);
		}

		u0 = u;
		//printf("Iter %ld with midpoint value %f. \n",k,u[n2/2]);
	}
	get_timestamp(&time2);
	double elapsed = timestamp_diff_in_seconds(time1,time2);
	printf("Jacobi done in %ld iterations. \n",k);
	printf("Time elapsed is %f seconds.\n", elapsed);

	//free(u0);
	free(u);
	return 0;
}