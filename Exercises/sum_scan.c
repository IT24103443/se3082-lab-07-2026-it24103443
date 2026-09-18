#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = N / size;
    int *array = NULL;
    int *local_chunk = (int *)malloc(chunk_size * sizeof(int));

    /* Only root allocates and fills the full array */
    if (rank == 0) {
        array = (int *)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++)
            array[i] = i + 1;
        printf("Root filled array with values 1 to %d\n", N);
    }

    double start = MPI_Wtime();

    /* SCATTER: Root distributes chunks to all processes */
    MPI_Scatter(array, chunk_size, MPI_INT, local_chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    long long local_sum = 0;
    for (int i = 0; i < chunk_size; i++)
        local_sum += local_chunk[i];

    printf("Rank %d: local sum = %lld\n", rank, local_sum);

    long long prefix_sum = 0;

    /* SCAN: Each process gets the sum of local_sums from rank 0 to its own rank */
    MPI_Scan(&local_sum, &prefix_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    long long sum_before_me = prefix_sum - local_sum;
    
    printf("Rank %d: local_sum = %lld, prefix_sum = %lld, sum_before_me = %lld\n", 
           rank, local_sum, prefix_sum, sum_before_me);

    /* Verification for the last rank */
    if (rank == size - 1) {
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Scan] Final prefix_sum on last rank = %lld\n", prefix_sum);
        printf("[Scan] Correct? = %s\n", prefix_sum == expected ? "YES" : "NO");

        double end = MPI_Wtime();
        printf("[Scan] Time      = %f seconds\n", end - start);
    }

    if (rank == 0) free(array);

    free(local_chunk);
    MPI_Finalize();
    return 0;
}
