#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

// define the dimensions of the matrix
#define M 384
#define N 512

/* Read the numbers in the file as an m x n matrix 
Note: If the file has fewer elements than in matrix, the rest is filled with zeroes. */
int **readMatrix(const char *filename, int m, int n);

const int const NUM_THREADS[] = {1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32};

int main(void)
{
	int i, j;

	// read random image matrices from file 'red.txt', 'green.txt', 'blue.txt'
	int **redmatrix = readMatrix("../../data/red.txt", M, N);
	int **greenmatrix = readMatrix("../../data/green.txt", M, N);
	int **bluematrix = readMatrix("../../data/blue.txt", M, N);

	// create matrices to store mean filtering result for red, green, blue matrices
	int **redmeanfilteredmatrix, **greenmeanfilteredmatrix, **bluemeanfilteredmatrix;

	redmeanfilteredmatrix = (int **)malloc(sizeof(int *)*(M-4));
	greenmeanfilteredmatrix = (int **)malloc(sizeof(int *)*(M-4));
	bluemeanfilteredmatrix = (int **)malloc(sizeof(int *)*(M-4));

	for(i=0; i<(M-4); ++i)
	{
		*(redmeanfilteredmatrix+i) = (int *)malloc(sizeof(int)*(N-4));
		*(greenmeanfilteredmatrix+i) = (int *)malloc(sizeof(int)*(N-4));
		*(bluemeanfilteredmatrix+i) = (int *)malloc(sizeof(int)*(N-4));
	}


	int num_threads_index;
	double start_time, end_time;

	for(num_threads_index=0; num_threads_index<sizeof(NUM_THREADS)/sizeof(int); ++num_threads_index)
	{
		start_time = omp_get_wtime();

		// perform mean filtering on the matrices
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				#pragma omp parallel for default(none) shared(redmeanfilteredmatrix, redmatrix) private(i, j) num_threads(NUM_THREADS[num_threads_index]) collapse(2)
				for(i=0; i<(M-4); ++i)
					for(j=0; j<(N-4); ++j)
						*(*(redmeanfilteredmatrix + i) + j) = 0.04 * *(*(redmatrix+i)+j) + 0.04 * *(*(redmatrix+i)+j+1) + 0.04 * *(*(redmatrix+i)+j+2) + 0.04 * *(*(redmatrix+i)+j+3) + 0.04 * *(*(redmatrix+i)+j+4) + 0.04 * *(*(redmatrix+i+1)+j) + 0.04 * *(*(redmatrix+i+1)+j+1) + 0.04 * *(*(redmatrix+i+1)+j+2) + 0.04 * *(*(redmatrix+i+1)+j+3) + 0.04 * *(*(redmatrix+i+1)+j+4) + 0.04 * *(*(redmatrix+i+2)+j) + 0.04 * *(*(redmatrix+i+2)+j+1) + 0.04 * *(*(redmatrix+i+2)+j+2) + 0.04 * *(*(redmatrix+i+2)+j+3) + 0.04 * *(*(redmatrix+i+2)+j+4) + 0.04 * *(*(redmatrix+i+3)+j) + 0.04 * *(*(redmatrix+i+3)+j+1) + 0.04 * *(*(redmatrix+i+3)+j+2) + 0.04 * *(*(redmatrix+i+3)+j+3) + 0.04 * *(*(redmatrix+i+3)+j+4) + 0.04 * *(*(redmatrix+i+4)+j) + 0.04 * *(*(redmatrix+i+4)+j+1) + 0.04 * *(*(redmatrix+i+4)+j+2) + 0.04 * *(*(redmatrix+i+4)+j+3) + 0.04 * *(*(redmatrix+i+4)+j+4);
			}
			
			#pragma omp section
			{
				#pragma omp parallel for default(none) shared(greenmeanfilteredmatrix, greenmatrix) private(i, j) num_threads(NUM_THREADS[num_threads_index]) collapse(2)
				for(i=0; i<(M-4); ++i)
					for(j=0; j<(N-4); ++j)
						*(*(greenmeanfilteredmatrix + i) + j) = 0.04 * *(*(greenmatrix+i)+j) + 0.04 * *(*(greenmatrix+i)+j+1) + 0.04 * *(*(greenmatrix+i)+j+2) + 0.04 * *(*(greenmatrix+i)+j+3) + 0.04 * *(*(greenmatrix+i)+j+4) + 0.04 * *(*(greenmatrix+i+1)+j) + 0.04 * *(*(greenmatrix+i+1)+j+1) + 0.04 * *(*(greenmatrix+i+1)+j+2) + 0.04 * *(*(greenmatrix+i+1)+j+3) + 0.04 * *(*(greenmatrix+i+1)+j+4) + 0.04 * *(*(greenmatrix+i+2)+j) + 0.04 * *(*(greenmatrix+i+2)+j+1) + 0.04 * *(*(greenmatrix+i+2)+j+2) + 0.04 * *(*(greenmatrix+i+2)+j+3) + 0.04 * *(*(greenmatrix+i+2)+j+4) + 0.04 * *(*(greenmatrix+i+3)+j) + 0.04 * *(*(greenmatrix+i+3)+j+1) + 0.04 * *(*(greenmatrix+i+3)+j+2) + 0.04 * *(*(greenmatrix+i+3)+j+3) + 0.04 * *(*(greenmatrix+i+3)+j+4) + 0.04 * *(*(greenmatrix+i+4)+j) + 0.04 * *(*(greenmatrix+i+4)+j+1) + 0.04 * *(*(greenmatrix+i+4)+j+2) + 0.04 * *(*(greenmatrix+i+4)+j+3) + 0.04 * *(*(greenmatrix+i+4)+j+4);
			}

			#pragma omp section
			{
				#pragma omp parallel for default(none) shared(bluemeanfilteredmatrix, bluematrix) private(i, j) num_threads(NUM_THREADS[num_threads_index]) collapse(2)
				for(i=0; i<(M-4); ++i)
					for(j=0; j<(N-4); ++j)
						*(*(bluemeanfilteredmatrix + i) + j) = 0.04 * *(*(bluematrix+i)+j) + 0.04 * *(*(bluematrix+i)+j+1) + 0.04 * *(*(bluematrix+i)+j+2) + 0.04 * *(*(bluematrix+i)+j+3) + 0.04 * *(*(bluematrix+i)+j+4) + 0.04 * *(*(bluematrix+i+1)+j) + 0.04 * *(*(bluematrix+i+1)+j+1) + 0.04 * *(*(bluematrix+i+1)+j+2) + 0.04 * *(*(bluematrix+i+1)+j+3) + 0.04 * *(*(bluematrix+i+1)+j+4) + 0.04 * *(*(bluematrix+i+2)+j) + 0.04 * *(*(bluematrix+i+2)+j+1) + 0.04 * *(*(bluematrix+i+2)+j+2) + 0.04 * *(*(bluematrix+i+2)+j+3) + 0.04 * *(*(bluematrix+i+2)+j+4) + 0.04 * *(*(bluematrix+i+3)+j) + 0.04 * *(*(bluematrix+i+3)+j+1) + 0.04 * *(*(bluematrix+i+3)+j+2) + 0.04 * *(*(bluematrix+i+3)+j+3) + 0.04 * *(*(bluematrix+i+3)+j+4) + 0.04 * *(*(bluematrix+i+4)+j) + 0.04 * *(*(bluematrix+i+4)+j+1) + 0.04 * *(*(bluematrix+i+4)+j+2) + 0.04 * *(*(bluematrix+i+4)+j+3) + 0.04 * *(*(bluematrix+i+4)+j+4);	
			}
		}
		
		
		end_time = omp_get_wtime();
		printf("No. of Threads = %d, time taken = %lf\n", NUM_THREADS[num_threads_index], (end_time - start_time));
	}
	
	return 0;
}

int **readMatrix(const char *filename, int m, int n)
{
	int **matrix;
	
	// open the matrix file for reading
	FILE *fptr = fopen(filename, "r");

	if(fptr == NULL)
	{
		matrix = NULL;

		char *errormsg = (char *)malloc( sizeof(char)*(30 + strlen(filename) + 1) );
		sprintf(errormsg, "Failed to read and parse file %s", filename);
		perror(errormsg);
	}

	else
	{
		// create and read the matrix
		matrix = (int **)malloc(sizeof(int *)*m);
		
		int i, j;
		for(i=0; i<m; ++i)
		{
			*(matrix + i) = (int *)malloc(sizeof(int)*n);
			for(j=0; j<n; ++j)
				fscanf(fptr, "%d", *(matrix+i)+j);
		}

		fclose(fptr);
	}
	
	return matrix;

}