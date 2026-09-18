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

    long long *all_sums = NULL;
    if (rank == 0) {
        all_sums = (long long *)malloc(size * sizeof(long long));
    }

    /* GATHER: Collect all local sums into the all_sums array on root */
    MPI_Gather(&local_sum, 1, MPI_LONG_LONG, all_sums, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long long total_sum = 0;
        for (int i = 0; i < size; i++) {
            total_sum += all_sums[i];
        }

        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Gather] Total sum = %lld\n", total_sum);
        printf("[Gather] Correct?  = %s\n", total_sum == expected ? "YES" : "NO");
        
        free(all_sums);
        free(array);

        printf("[Gather] Time      = %f seconds\n", MPI_Wtime() - start);
    }

    free(local_chunk);
    MPI_Finalize();
    return 0;
}