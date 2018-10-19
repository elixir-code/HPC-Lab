#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

// define the dimension of the matrices (MxN)
#define M 400
#define N 400

// iteratively calculate time and take average
#define N_ITERATIONS 100

int main(){

	int i,j,k,l;
	double start_time, end_time;
	long double total_time_taken;

	int n_threads[] = {1, 2, 4, 6, 8, 10, 12, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62};

	// create and initialise the input and output matrices
	long double A[M][N], B[M][N], C[M][N];

	for(i=0; i<M; ++i)
		for(j=0; j<N; ++j)
		{
			A[i][j] = rand();
			B[i][j] = rand();
		}

	for(k=0; k<20; ++k)
	{
		total_time_taken = 0;

		for(l=0; l<N_ITERATIONS; ++l)
		{
			start_time = omp_get_wtime();
			#pragma omp parallel for default(none) private(i,j) shared(A,B,C) num_threads(n_threads[k]) collapse(2)
					for(i=0; i<M; ++i)
						for(j=0; j<N; ++j)
							C[i][j] = A[i][j]+B[i][j];
			end_time = omp_get_wtime();

			total_time_taken += (end_time - start_time);
		}

		printf("No.of Threads = %d, Time Taken: %Lf\n", n_threads[k],total_time_taken/N_ITERATIONS);	
	}
	
	return 0;
}