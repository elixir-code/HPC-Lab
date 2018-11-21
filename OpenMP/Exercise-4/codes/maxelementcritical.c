/* Find maximum element among K matrices 
Author: R Mukesh (CED15I002)
*/

#define M 1000 // the number of rows in matrices
#define N 1000 // the number of columns in matrices

#define K 100 // the number of matrices to find max among

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <time.h>

int const NUM_THREADS[] = {1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32};

int main(void)
{
	int k, i, j, num_threads_index;
	clock_t start_clock, end_clock;

	// initialise K matrices
	int ***matrices;
	matrices = (int ***)malloc(sizeof(int **)*K);

	for(k=0; k<K; ++k)
	{
		*(matrices+k) = (int **)malloc(sizeof(int)*M);

		for(i=0; i<M; ++i)
		{
			*(*(matrices+k) + i) = (int *)malloc(sizeof(int)*N);
			
			for(j=0; j<N; ++j)
				*(*(*(matrices+k) + i)+j) = rand();
		}
	}
	
	// Finding max among K matrices using critical or atomic directive
	int local_max = INT_MIN, global_max = INT_MIN;

	for(num_threads_index=0; num_threads_index<sizeof(NUM_THREADS)/sizeof(int); ++num_threads_index)
	{	
		start_clock = clock();

		#pragma omp parallel for default(none) firstprivate(local_max) private(k, i, j) shared(matrices, global_max) num_threads(NUM_THREADS[num_threads_index])
		for(k=0; k<K; ++k)
		{
			for(i=0; i<M; ++i)
				for(j=0; j<N; ++j)
					if( *(*(*(matrices+k) + i)+j) > local_max)
						local_max = *(*(*(matrices+k) + i)+j);
			
			#pragma omp flush(global_max)
			if(local_max > global_max)
				#pragma omp critical
				if(local_max > global_max)
					global_max = local_max;
		}

		end_clock = clock();

		printf("Num Threads = %d, Time taken = %lf\n", NUM_THREADS[num_threads_index], (end_clock - start_clock)*1.00/CLOCKS_PER_SEC);
	}

	printf("Maximum element = %d\n", global_max);

	return 0;
}