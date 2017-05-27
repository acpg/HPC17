EXECS=multigrid_2D multigrid_openMP2D multigrid-mpi2D
CC=gcc
MPICC=mpicc
FLAGS=-O3 -Wall -g -lm

all: ${EXECS}

multigrid_2D: multigrid_2D.c
	${CC} ${FLAGS} $^ -o multigrid_2D

multigrid_openMP2D: multigrid_openMP2D.c
	clang-omp -O3 -fopenmp -Wall $^ -o multigrid_openMP2D

multigrid-mpi2D: multigrid-mpi2D.c
	${MPICC} -DUSE_MPI $^ -o multigrid-mpi2D

clean:
	rm -f ${EXECS}
