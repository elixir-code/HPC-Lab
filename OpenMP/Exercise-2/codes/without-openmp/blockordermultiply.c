/* Matrix Multiplication for Block Matrices

High Performance Computing Lab - Assignment 2
Author: R Mukesh (CED15I002), IIITDM Kancheepuram */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// size of matrix
#define N 2048
#define CACHE_BLOCK_SIZE 64
#define B (int)floor(sqrt(CACHE_BLOCK_SIZE))


// Perform matrix multiplication of row-majpr matrices A (mxn), B(nxp) and stor in C(mxp)
void matrixMultiplication(double *matrixA, double *matrixB, double *matrixC, int m, int n, int p);

// print matrix of dimension NxN and maximum block size BxB
void printMatrix(double ***matrix);

int main(void)
{
	printf("Dimension of Matrix = %d x %d\n", N, N);
	printf("Max Dimension of Matrix Block = %d x %d\n", CACHE_BLOCK_SIZE, CACHE_BLOCK_SIZE);

	int n_blocks = (int)ceil(N*1.0/B);
	double ***matrixA, ***matrixB, ***matrixC;

	int i,j,k;

	// create and initialise matrix A
	matrixA = (double ***)malloc(sizeof(double **)*n_blocks);
	for(i=0; i<n_blocks; ++i)
	{
		*(matrixA + i) = (double **)malloc(sizeof(double *)*n_blocks);
		for(j=0; j<n_blocks; ++j)
		{	
			*(*(matrixA+i)+j) = (double *)malloc(sizeof(double)*((i==n_blocks-1)&&(N%B!=0)?N%B:B)*((j==n_blocks-1)&&(N%B!=0)?N%B:B));
			for(k=0; k<((i==n_blocks-1)&&(N%B!=0)?N%B:B)*((j==n_blocks-1)&&(N%B!=0)?N%B:B); ++k)
				*(*(*(matrixA+i)+j)+k) = rand();
		}
	}

	// printMatrix(matrixA);

	// create and initialise matrix B
	matrixB = (double ***)malloc(sizeof(double **)*n_blocks);
	for(i=0; i<n_blocks; ++i)
	{
		*(matrixB + i) = (double **)malloc(sizeof(double *)*n_blocks);
		for(j=0; j<n_blocks; ++j)
		{	
			*(*(matrixB+i)+j) = (double *)malloc(sizeof(double)*((i==n_blocks-1)&&(N%B!=0)?N%B:B)*((j==n_blocks-1)&&(N%B!=0)?N%B:B));
			for(k=0; k<((i==n_blocks-1)&&(N%B!=0)?N%B:B)*((j==n_blocks-1)&&(N%B!=0)?N%B:B); ++k)
				*(*(*(matrixB+i)+j)+k) = rand();
		}
	}

	// printMatrix(matrixB);

	// create and initialise result matrix C
	matrixC = (double ***)malloc(sizeof(double **)*n_blocks);
	for(i=0; i<n_blocks; ++i)
	{
		*(matrixC + i) = (double **)malloc(sizeof(double *)*n_blocks);
		for(j=0; j<n_blocks; ++j)
		{	
			*(*(matrixC+i)+j) = (double *)malloc(sizeof(double)*((i==n_blocks-1)&&(N%B!=0)?N%B:B)*((j==n_blocks-1)&&(N%B!=0)?N%B:B));
			for(k=0; k<((i==n_blocks-1)&&(N%B!=0)?N%B:B)*((j==n_blocks-1)&&(N%B!=0)?N%B:B); ++k)
				*(*(*(matrixC+i)+j)+k) = 0;
		}
	}

	double start_time, end_time;

	start_time = omp_get_wtime();

	// Perform matrix multiplication C=AxB
	for(i=0; i<n_blocks; ++i)
		for(j=0; j<n_blocks; ++j)
			for(k=0; k<n_blocks; ++k)
				matrixMultiplication(*(*(matrixA+i)+k), *(*(matrixB+k)+j), *(*(matrixC+i)+j), (i==n_blocks-1)&&(N%B!=0)?N%B:B, (k==n_blocks-1)&&(N%B!=0)?N%B:B, (j==n_blocks-1)&&(N%B!=0)?N%B:B);

	end_time = omp_get_wtime();
	printf("Time taken = %lf\n", (end_time - start_time));

	// printMatrix(matrixC);

	return 0;
}

// Perform matrix multiplication of row-majpr matrices A (mxn), B(nxp) and stor in C(mxp)
void matrixMultiplication(double *matrixA, double *matrixB, double *matrixC, int m, int n, int p)
{	
	int i,j,k;

	for(i=0; i<m; ++i)
		for(j=0; j<p; ++j)
			for(k=0; k<n; ++k)
				*(matrixC + i*p + j) += *(matrixA + i*n  + k) * *(matrixB + k*p + j);
}

// print matrix of dimension NxN and maximum block size BxB
void printMatrix(double ***matrix)
{
	int n_blocks = (int)ceil(N*1.0/B);
	
	int i, j, k, l;
	for(i=0; i<n_blocks; ++i)
		// print contents of a row of blocks here
		for(j=0; j<((i==n_blocks-1)&&(N%B!=0)?N%B:B); ++j)
		{
			// print the jth row of all blocks in the ith row of blocks
			for(k=0; k<n_blocks; ++k)
				// print the ith row of block, kth block, jth row elements
				for(l=j*((k==n_blocks-1)&&(N%B!=0)?N%B:B); l<(j+1)*((k==n_blocks-1)&&(N%B!=0)?N%B:B); ++l)
					printf("%lf ", *(*(*(matrix + i) + k) + l));
			printf("\n");
		}

	printf("\n");		
		
}
