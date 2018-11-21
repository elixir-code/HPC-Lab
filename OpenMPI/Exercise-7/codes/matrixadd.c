#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N_ROWS 4096
#define N_COLUMNS 4096

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

int main(int argc, char *argv[]) {
	
	int i, j, n_rows_per_worker, nextrarows, dest, source;
	
	MPI_Status status;

	// initialise the MPI environment argc, argv
	MPI_Init(&argc, &argv);
	
	// Get the number of processors
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	
	if(world_rank == MASTER)
	{
		double **matrixA, **matrixB, **matrixC;
		
		// initialize matrix A
		matrixA = (double **)malloc(sizeof(double *)*N_ROWS);
		for(i=0; i<N_ROWS; ++i)
		{
			*(matrixA + i) = (double *)malloc(sizeof(double)*N_COLUMNS);
			for(j=0; j<N_COLUMNS; ++j)
				*(*(matrixA + i)+j) = random()%100;
		}
		
		// initialize matrix B
		matrixB = (double **)malloc(sizeof(double *)*N_ROWS);
                for(i=0; i<N_ROWS; ++i)
                {
                        *(matrixB + i) = (double *)malloc(sizeof(double)*N_COLUMNS);
                        for(j=0; j<N_COLUMNS; ++j)
                                *(*(matrixB + i)+j) = random()%100;
                }
		
		// initialize matrix C
		matrixC = (double **)malloc(sizeof(double *)*N_ROWS);
		for(i=0; i<N_ROWS; ++i)
			*(matrixC + i) = (double *)malloc(sizeof(double)*N_COLUMNS);
		
		// send the respective rows of matrixA, matrixB to corresponding processes
		n_rows_per_worker = N_ROWS/(world_size-1);
		nextrarows = N_ROWS%(world_size-1);

		for(dest=1; dest<world_size; ++dest)
		{
			for(j=0; j<n_rows_per_worker; ++j)
			{
				MPI_Send(*(matrixA + (dest-1)*n_rows_per_worker + j), N_COLUMNS, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(*(matrixB + (dest-1)*n_rows_per_worker + j), N_COLUMNS, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
			}
		}
		
		// processes the extra rows addition in master node
		for(i=N_ROWS-nextrarows; i<N_ROWS; ++i)
			for(j=0; j<N_COLUMNS; ++j)
				*(*(matrixC + i)+j) = *(*(matrixA + i)+j) + *(*(matrixB + i)+j);
		
		// recieve and consolidate results from all workers
		for(source=1; source<world_size; ++source)
		{
			for(j=0; j<n_rows_per_worker; ++j)
				MPI_Recv(*(matrixC + (source-1)*n_rows_per_worker + j), N_COLUMNS, MPI_DOUBLE, source, FROM_WORKER, MPI_COMM_WORLD, &status);
		}

		// print the matrices A, B, C
		printf("Matrix A\n");
		for(i=0; i<N_ROWS; ++i)
		{
			for(j=0; j<N_COLUMNS; ++j)
				printf("%lf ", *(*(matrixA+i)+j));
			printf("\n");
		}
		printf("\n");
		
		printf("Matrix B\n");
		for(i=0; i<N_ROWS; ++i)
                {
                        for(j=0; j<N_COLUMNS; ++j)
                                printf("%lf ", *(*(matrixB+i)+j));
                        printf("\n");
                }
                printf("\n");
		printf("Matrix C\n");
		for(i=0; i<N_ROWS; ++i)
                {
                        for(j=0; j<N_COLUMNS; ++j)
                                printf("%lf ", *(*(matrixC+i)+j));
                        printf("\n");
                }
                printf("\n");
	}
	
	// Perform addition of rows in worker thread
	else if(world_rank > MASTER)
	{
		n_rows_per_worker = N_ROWS/(world_size-1);

		double **matrixA = (double **)malloc(sizeof(double *)*n_rows_per_worker);
		double **matrixB = (double **)malloc(sizeof(double *)*n_rows_per_worker);
		for(i=0; i<n_rows_per_worker; ++i)
		{
			*(matrixA + i) = (double *)malloc(sizeof(double)*N_COLUMNS);
			MPI_Recv(*(matrixA + i), N_COLUMNS, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
                        *(matrixB + i) = (double *)malloc(sizeof(double)*N_COLUMNS);
			MPI_Recv(*(matrixB + i), N_COLUMNS, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
		}
		
		// Perform addition of rows and store result in matrix A
		for(i=0; i<n_rows_per_worker; ++i)
			for(j=0; j<N_COLUMNS; ++j)
				*(*(matrixA + i) + j) += *(*(matrixB + i) + j);

		for(i=0; i<n_rows_per_worker; ++i)
			MPI_Send(*(matrixA + i), N_COLUMNS, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
	}

	// Finalize the MPI environment
	MPI_Finalize();
}
