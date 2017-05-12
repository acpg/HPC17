EXECS=multigrid_2d multigrid_2d_openmp multigrid_2d_mpi
CC=gcc
MPICC=mpicc
FLAGS=-O3 -Wall -g -lm

all: ${EXECS}

multigrid_2d: multigrid_2d.c
	${CC} ${FLAGS} $^ -o multigrid_2d

multigrid_2d_openmp: multigrid_2d.c
	clang-omp -O3 -fopenmp -Wall $^ -o multigrid_2d_openmp

multigrid_2d_mpi: multigrid_2d.c
	${MPICC} -DUSE_MPI $^ -o multigrid_2d_mpi

clean:
	rm -f ${EXECS}
