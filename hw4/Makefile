EXECS=mpi_solved1 mpi_solved2 mpi_solved3 mpi_solved4 mpi_solved5 mpi_solved6 mpi_solved7 jacobi-mpi2D
MPICC=mpicc

all: ${EXECS}

mpi_solved1: mpi_solved1.c
	${MPICC} -o mpi_solved1 mpi_solved1.c

mpi_solved2: mpi_solved2.c
	${MPICC} -o mpi_solved2 mpi_solved2.c

mpi_solved3: mpi_solved3.c
	${MPICC} -o mpi_solved3 mpi_solved3.c

mpi_solved4: mpi_solved4.c
	${MPICC} -o mpi_solved4 mpi_solved4.c

mpi_solved5: mpi_solved5.c
	${MPICC} -o mpi_solved5 mpi_solved5.c

mpi_solved6: mpi_solved6.c
	${MPICC} -o mpi_solved6 mpi_solved6.c

mpi_solved7: mpi_solved7.c
	${MPICC} -o mpi_solved7 mpi_solved7.c

jacobi-mpi2D: jacobi-mpi2D.c
	${MPICC} -o jacobi-mpi2D jacobi-mpi2D.c

clean:
	rm -f ${EXECS}
