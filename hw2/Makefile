CC=clang-omp
FLAGS=-O3 -fopenmp -Wall
EXECS= omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp

all: ${EXECS}

omp_solved2: omp_solved2.c
	${CC} ${FLAGS} omp_solved2.c -o omp_solved2

omp_solved3: omp_solved3.c
	${CC} ${FLAGS} omp_solved3.c -o omp_solved3

omp_solved4: omp_solved4.c
	${CC} ${FLAGS} omp_solved4.c -o omp_solved4

omp_solved5: omp_solved5.c
	${CC} ${FLAGS} omp_solved5.c -o omp_solved5

omp_solved6: omp_solved6.c
	${CC} ${FLAGS} omp_solved6.c -o omp_solved6

jacobi2D-omp: jacobi2D-omp.c
	${CC} ${FLAGS} jacobi2D-omp.c -o jacobi2D-omp

gs2D-omp: gs2D-omp.c
	${CC} ${FLAGS} gs2D-omp.c -o gs2D-omp

clean:
	rm -f ${EXECS}
