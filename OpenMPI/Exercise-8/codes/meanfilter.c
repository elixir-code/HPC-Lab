#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N_ROWS 128
#define N_COLUMNS 128

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

int max(int a, int b)
{
	if(a > b)
		return a;

	else 
		return b;
}

int min(int a, int b)
{
	if(a < b)
		return a;

	else 
		return b;
}

// Perform Mean Filtering of a RGB Image of size n_rows x n_columns and store result in mean_filtered_image
void RGBMeanFiltering(int **image, int n_rows, int n_columns, int **mean_filtered_image);

int main(int argc, char *argv[])
{
	int i, j;
	
	int n_rows_per_worker, n_extra_rows, offset, row_start_index, worker_n_rows;
	MPI_Status status;

	//initialise the MPI environment
	MPI_Init(&argc, &argv);

	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the processes
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Compute the number of rows per worker and number of extra rows
	n_rows_per_worker = N_ROWS/(world_size-1);
	n_extra_rows = N_ROWS%(world_size-1);

	if(world_rank == MASTER)
	{	
		int dest, source;
		int output_offset;

		int **image, **mean_filtered_image;

		// Create and initialise a three channel (RGB) image of size N_ROWSxN_COLUMNS
		image = (int **)malloc(sizeof(int *)*N_ROWS);
		for(i=0; i<N_ROWS; ++i)
		{
			image[i] = (int *)malloc(sizeof(int)*3*N_COLUMNS);
			for(j=0; j<3*N_COLUMNS; ++j)
				image[i][j] = random()%256;
		}

		for(dest=1; dest<world_size; ++dest)
		{
			offset = -2;
			if((dest == 1) && (n_extra_rows < 2))
				offset = -1*n_extra_rows;

			row_start_index = n_extra_rows + (dest-1)*n_rows_per_worker;

			for(i=offset; i<n_rows_per_worker; ++i)
			{
				// printf("%d: Send row %d\n", dest, row_start_index + i);
				MPI_Send(image[row_start_index + i], 3*N_COLUMNS, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
			}
		}

		mean_filtered_image = (int **)malloc(sizeof(int *)*(N_ROWS-2));
		for(i=0; i<N_ROWS-2; ++i)
			mean_filtered_image[i] = (int *)malloc(sizeof(int)*3*(N_COLUMNS-2));

		// Perform mean filtering of extra rows in master
		if(n_extra_rows > 2)
			RGBMeanFiltering(image, n_extra_rows, N_COLUMNS, mean_filtered_image);


		// Recieve and consolidate mean filtered image from workers
		row_start_index = max(0, n_extra_rows-2);

		for(source=1; source<world_size; ++source)
		{
			if(source == 1)
				worker_n_rows = n_rows_per_worker + min(n_extra_rows, 2) - 2;
			else
				worker_n_rows = n_rows_per_worker;

			for(i=0; i<worker_n_rows; ++i)
				MPI_Recv(mean_filtered_image[row_start_index+i], 3*(N_COLUMNS-2), MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);

			row_start_index += worker_n_rows;
		}

		// print the image and mean filtered image
		printf("Image\n");
		for(i=0; i<N_ROWS; ++i)
		{
			for(j=0; j<N_COLUMNS; j+=3)
				printf("[%d %d %d] ", image[i][j], image[i][j+1], image[i][j+2]);
			printf("\n");
		}

		printf("\nMean Filtered Image\n");
		for(i=0; i<N_ROWS-2; ++i)
		{
			for(j=0; j<3*(N_COLUMNS-2); j+=3)
				printf("[%d %d %d] ", mean_filtered_image[i][j], mean_filtered_image[i][j+1], mean_filtered_image[i][j+2]);
			printf("\n");
		}
	}

	else if(world_rank > MASTER)
	{

		offset = -2;
		if((world_rank == 1) && (n_extra_rows < 2))
			offset = -1*n_extra_rows;

		worker_n_rows = n_rows_per_worker - offset;

		int **image, **mean_filtered_image;
		
		// Create and initialise a three channel (RGB) image of size N_ROWSxN_COLUMNS
		image = (int **)malloc(sizeof(int *) * worker_n_rows);
		for(i=0; i<worker_n_rows; ++i)
			image[i] = (int *)malloc(sizeof(int)*3*N_COLUMNS);

		for(i=0; i<worker_n_rows; ++i)
		{
			MPI_Recv(image[i], 3*N_COLUMNS, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
		}

		// Perform mean filtering of rows in worker
		mean_filtered_image = (int **)malloc(sizeof(int *)*(worker_n_rows-2));
		for(i=0; i<worker_n_rows; ++i)
			mean_filtered_image[i] = (int *)malloc(sizeof(int)*3*(N_COLUMNS-2));

		RGBMeanFiltering(image, worker_n_rows, N_COLUMNS, mean_filtered_image);

		// Send back mean filtered image to master
		for(i=0; i<worker_n_rows-2; ++i)
			MPI_Send(mean_filtered_image[i], 3*(N_COLUMNS-2), MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
	}

	// Finalize the MPI Environment
	MPI_Finalize();
}

void RGBMeanFiltering(int **image, int n_rows, int n_columns, int **mean_filtered_image)
{
	int i, j;

	for(i=0; i<n_rows-2; ++i)
		for(j=0; j<3*(n_columns-2); j+=3)
		{
			mean_filtered_image[i][j] = (image[i][j]+image[i][j+3]+image[i][j+6] + image[i+1][j]+image[i+1][j+3]+image[i+1][j+6] + image[i+2][j]+image[i+2][j+3]+image[i+2][j+6])/9;
			mean_filtered_image[i][j+1] = (image[i][j+1]+image[i][j+4]+image[i][j+7] + image[i+1][j+1]+image[i+1][j+4]+image[i+1][j+7] + image[i+2][j+1]+image[i+2][j+4]+image[i+2][j+7])/9;
			mean_filtered_image[i][j+2] = (image[i][j+2]+image[i][j+5]+image[i][j+8] + image[i+1][j+2]+image[i+1][j+5]+image[i+1][j+8] + image[i+2][j+2]+image[i+2][j+5]+image[i+2][j+8])/9;
		}
}