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

// Perform Median Filtering of a RGB Image of size n_rows x n_columns and store result in median_filtered_image
void RGBMedianFiltering(int **image, int n_rows, int n_columns, int **median_filtered_image);

// Compute the median of 9 numbers
int median(int num1, int num2, int num3, int num4, int num5, int num6, int num7, int num8, int num9);


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

		int **image, **median_filtered_image;

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

		median_filtered_image = (int **)malloc(sizeof(int *)*(N_ROWS-2));
		for(i=0; i<N_ROWS-2; ++i)
			median_filtered_image[i] = (int *)malloc(sizeof(int)*3*(N_COLUMNS-2));

		// Perform median filtering of extra rows in master
		if(n_extra_rows > 2)
			RGBMeanFiltering(image, n_extra_rows, N_COLUMNS, median_filtered_image);


		// Recieve and consolidate median filtered image from workers
		row_start_index = max(0, n_extra_rows-2);

		for(source=1; source<world_size; ++source)
		{
			if(source == 1)
				worker_n_rows = n_rows_per_worker + min(n_extra_rows, 2) - 2;
			else
				worker_n_rows = n_rows_per_worker;

			for(i=0; i<worker_n_rows; ++i)
				MPI_Recv(median_filtered_image[row_start_index+i], 3*(N_COLUMNS-2), MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);

			row_start_index += worker_n_rows;
		}

		// print the image and median filtered image
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
				printf("[%d %d %d] ", median_filtered_image[i][j], median_filtered_image[i][j+1], median_filtered_image[i][j+2]);
			printf("\n");
		}
	}

	else if(world_rank > MASTER)
	{

		offset = -2;
		if((world_rank == 1) && (n_extra_rows < 2))
			offset = -1*n_extra_rows;

		worker_n_rows = n_rows_per_worker - offset;

		int **image, **median_filtered_image;
		
		// Create and initialise a three channel (RGB) image of size N_ROWSxN_COLUMNS
		image = (int **)malloc(sizeof(int *) * worker_n_rows);
		for(i=0; i<worker_n_rows; ++i)
			image[i] = (int *)malloc(sizeof(int)*3*N_COLUMNS);

		for(i=0; i<worker_n_rows; ++i)
		{
			MPI_Recv(image[i], 3*N_COLUMNS, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
		}

		// Perform median filtering of rows in worker
		median_filtered_image = (int **)malloc(sizeof(int *)*(worker_n_rows-2));
		for(i=0; i<worker_n_rows; ++i)
			median_filtered_image[i] = (int *)malloc(sizeof(int)*3*(N_COLUMNS-2));

		RGBMeanFiltering(image, worker_n_rows, N_COLUMNS, median_filtered_image);

		// Send back median filtered image to master
		for(i=0; i<worker_n_rows-2; ++i)
			MPI_Send(median_filtered_image[i], 3*(N_COLUMNS-2), MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
	}

	// Finalize the MPI Environment
	MPI_Finalize();
}

void RGBMedianFiltering(int **image, int n_rows, int n_columns, int **median_filtered_image)
{
	int i, j;

	for(i=0; i<n_rows-2; ++i)
		for(j=0; j<3*(n_columns-2); j+=3)
		{
			median_filtered_image[i][j] = median(image[i][j],image[i][j+3],image[i][j+6], image[i+1][j],image[i+1][j+3],image[i+1][j+6], image[i+2][j],image[i+2][j+3],image[i+2][j+6]);
			median_filtered_image[i][j+1] = median(image[i][j+1],image[i][j+4],image[i][j+7], image[i+1][j+1],image[i+1][j+4],image[i+1][j+7], image[i+2][j+1],image[i+2][j+4],image[i+2][j+7]);
			median_filtered_image[i][j+2] = median(image[i][j+2],image[i][j+5],image[i][j+8], image[i+1][j+2],image[i+1][j+5],image[i+1][j+8], image[i+2][j+2],image[i+2][j+5],image[i+2][j+8]);
		}
}

int median(int num1, int num2, int num3, int num4, int num5, int num6, int num7, int num8, int num9)
{
	int numbers[] = {num1, num2, num3, num4, num5, num6, num7, num8, num9};

	int min, min_index;

	int i,j;
	for(i=0; i<5; ++i)
	{
		min = numbers[i];
		min_index = i;

		printf("i = %d, min = %d, min index = %d\n", i, min, min_index);

		for(j=i+1; j<9; ++j)
			if(numbers[j] < min)
			{
				min_index = j;
				min = numbers[j];
			}

		// swap numbers[i] and numbers[min_index]
		numbers[min_index] = numbers[i];
		numbers[i] = min;
	}

	return numbers[4];
}