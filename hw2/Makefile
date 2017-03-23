CC=clang-omp
FLAGS=-O3 -fopenmp -Wall
EXECS=omp2 omp3 omp4 omp5 omp6

all: ${EXECS}

omp2: omp_solved2.c
	${CC} ${FLAGS} omp_solved2.c -o omp_solved2

omp3: omp_solved3.c
	${CC} ${FLAGS} omp_solved3.c -o omp_solved3

omp4: omp_solved4.c
	${CC} ${FLAGS} omp_solved4.c -o omp_solved4

omp5: omp_solved5.c
	${CC} ${FLAGS} omp_solved5.c -o omp_solved5

omp6: omp_solved6.c
	${CC} ${FLAGS} omp_solved6.c -o omp_solved6

clean:
	rm -f ${EXECS}
