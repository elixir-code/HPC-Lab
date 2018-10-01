/* Matrix Multiplication for Column Major Matrices

High Performance Computing Lab - Assignment 2
Author: R Mukesh (CED15I002), IIITDM Kancheepuram */

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define N 2048

int main(void)
{
	int i,j,k;

	double **matrixA, **matrixB, **matrixC;
	
	// create and initialises matrix A	
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
			printf("%lf ", *(*(matrixA + j) + i));
		printf("\n");
	}
	printf("\n");*/


	// create and initialises matrix B
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
			printf("%lf ", *(*(matrixB + j) + i));
		printf("\n");
	}
	printf("\n");*/

	// create matrix C
	matrixC = (double **)malloc(sizeof(double *)*N);
	for(i=0; i<N; ++i)
	{
		*(matrixC + i) = (double *)malloc(sizeof(double)*N);
		for(j=0; j<N; ++j)
				*(*(matrixC + i) + j) = 0;
	}

	double start_time, end_time;

	start_time = omp_get_wtime();

		// perform matrix multiplication
	for(i=0; i<N; ++i)
		for(j=0; j<N; ++j)
			for(k=0; k<N; ++k)
				 *(*(matrixC + i) + k) += *(*(matrixA + j) + k) * *(*(matrixB + i) + j);

	end_time = omp_get_wtime();		
	printf("Dimension = %dx%d, Time taken = %lf\n", N, N, (end_time - start_time));

	/*// display result matrix C
	for(i=0; i<N; ++i)
	{
		for(j=0; j<N; ++j)
			printf("%lf ", *(*(matrixC + j) + i));
		printf("\n");
	}
	printf("\n");*/
	
	return 0;
}