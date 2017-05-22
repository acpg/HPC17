EXECS=report_ap
CC=gcc
MPI=mpicc

all: ${EXECS}

p4est_data: p4est_data.c
	mpicc -DHAVE_CONFIG_H -I. -I./src   -I./src -Isc/src -I./sc/src   -g -O2 -MT p4est_data.o -MD -MP -MF -c -o p4est_data.o p4est_data.c
	mpicc  -g -O2   -o p4est_data p4est_data.o -I./src/libp4est.la -I./sc/src/libsc.la -llapack -lblas -llua -lz   
	mpicc -g -O2 -o .libs/p4est_data p4est_data.o  -I./src/.libs/libp4est.dylib -I./sc/src/.libs/libsc.dylib -llapack -lblas -llua -lz
	mpicc -DHAVE_CONFIG_H -I. -I./sc -I./src -Isc/src -I./sc/src -g p4est_data.c -o p4est_data -llapack -lblas -llua -lz

report_ap: report_ap.tex
	pdflatex $^

clean:
	rm -f *.out *.log *.aux *.DS_Store
