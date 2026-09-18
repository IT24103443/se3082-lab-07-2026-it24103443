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

    long long total_sum = 0;

    /* ALLREDUCE: Sum all local_sums; EVERY process gets the total_sum */
    MPI_Allreduce(&local_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    double percentage = ((double)local_sum / total_sum) * 100.0;
    
    printf("Rank %d: local_sum = %lld, total_sum = %lld, contribution = %.2f%%\n", 
           rank, local_sum, total_sum, percentage);

    if (rank == 0) {
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Allreduce] Correct? = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Allreduce] Time     = %f seconds\n", MPI_Wtime() - start);
        free(array);
    }

    free(local_chunk);
    MPI_Finalize();
    return 0;
}
