#include "mpi.h"
#include <stdio.h>

#include <string.h>
#include <stdlib.h>

#include <math.h>
#define MAXSIZE 1000000

int main(int argc, char **argv)
{
	int myid, numprocs;
	
	double data[MAXSIZE];
	
	double  mymin, mymax, max, min;
	mymin = -INFINITY;
	mymax = INFINITY;
	
	int i, x, low, high;

	char fn[255];
	FILE *fp;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	if(0 == myid) {
		/* open input file and intialize data */
		strcpy(fn, getenv("PWD"));
		strcat(fn, "../data/data.txt");
		if( NULL == (fp = fopen(fn, "r")) ) {
			printf("Can't open the input file: %s\n\n", fn);
			exit(1);
		}
		for(i=0; i<MAXSIZE; i++) {
			fscanf(fp, "%lf", &data[i]);
		}
	}

	/* broadcast data */
	MPI_Bcast(data, MAXSIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	/* compute min max portion for data */
	x = MAXSIZE/numprocs;	/* must be an integer */
	low = myid * x;
	high = low + x;
	for(i=low; i<high; i++) {
		if(data[i] < mymin)
			mymin = data[i];
		if(data[i] > mymax)
			mymax = data[i];

	}
	printf("I got min=%lf, max=%lf from %d\n", mymin, mymax, myid);

	/* compute global sum */
	MPI_Reduce(&mymin, &min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&mymax, &max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if(0 == myid) {
		printf("The min is %lf and max is %lf.\n", min, max);
	}

	MPI_Finalize();
}

	
