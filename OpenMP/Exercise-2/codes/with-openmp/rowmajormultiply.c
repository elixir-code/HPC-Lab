/* Matrix Multiplication for Row Major Matrices 

High Performance Computing Lab - Assignment 2
Author: R Mukesh (CED15I002), IIITDM Kancheepuram */

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define N 2048

int const NUM_THREADS[] = {1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32};

int main(void)
{
	printf("Dimension of Matrix = %d x %d\n", N, N);
	
	int i, j, k;

	// create and initialise matrix A
	double **matrixA;
	matrixA = (double **)malloc(sizeof(double *)*N);

	for(i=0; i<N; ++i)
	{
		*(matrixA + i) = (double *)malloc(sizeof(double)*N);
		
		for(j=0; j<N; ++j)
			*(*(matrixA + i) + j) = rand();
	}

	/*// display matrix A
	for(i=0; i<N; ++i)
	{
		for(j=0; j<N; ++j)
			printf("%lf ", *(*(matrixA + i) + j));
		printf("\n");
	}
	printf("\n");*/
	
	// create and initialise matrix B
	double **matrixB;
	matrixB = (double **)malloc(sizeof(double *)*N);

	for(i=0; i<N; ++i)
	{
		*(matrixB + i) = (double *)malloc(sizeof(double)*N);

		for(j=0; j<N; ++j)
			*(*(matrixB + i) + j) = rand();
	}

	
	/*// display matrix B
	for(i=0; i<N; ++i)
	{
		for(j=0; j<N; ++j)
			printf("%lf ", *(*(matrixB + i) + j));
		printf("\n");
	}
	printf("\n");*/
	

	// create and initialise matrix C
	double **matrixC;
	matrixC = (double **)malloc(sizeof(double *)*N);

	for(i=0; i<N; ++i)
		*(matrixC + i) = (double *)malloc(sizeof(double)*N);

	// perform matrix multiplication
	int num_threads_index;
	double start_time, end_time;


	for(num_threads_index=0; num_threads_index<sizeof(NUM_THREADS)/sizeof(int); ++num_threads_index)
	{

		// initialise result matrix C
		for(i=0; i<N; ++i)
			for(j=0; j<N; ++j)
				*(*(matrixC + i) + j) = 0;

		start_time = omp_get_wtime();

		#pragma omp parallel for default(none) private(i,j,k) shared(matrixA, matrixB, matrixC) num_threads(NUM_THREADS[num_threads_index])
		for(i=0; i<N; ++i)
			for(j=0; j<N; ++j)
				for(k=0; k<N; ++k)
					*(*(matrixC + i) + k) += *(*(matrixA + i) + j) * *(*(matrixB + j) + k);

		end_time = omp_get_wtime();
		printf("No. of Threads = %d, time taken = %lf\n", NUM_THREADS[num_threads_index], (end_time - start_time));

		/*// display result matrix C
		for(i=0; i<N; ++i)
		{
			for(j=0; j<N; ++j)
				printf("%lf ", *(*(matrixA + i) + j));
			printf("\n");
		}
		printf("\n");*/
		
	}
	
	return 0;
}
