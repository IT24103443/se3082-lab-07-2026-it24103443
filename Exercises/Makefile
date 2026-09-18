CC = mpicc
CFLAGS = -Wall -O2
EXECS = program sum_scatter sum_gather sum_reduce sum_allreduce sum_scan
NP = 4

all: $(EXECS)

program: program.c
	$(CC) $(CFLAGS) -o $@ $^

sum_scatter: sum_scatter.c
	$(CC) $(CFLAGS) -o $@ $^

sum_gather: sum_gather.c
	$(CC) $(CFLAGS) -o $@ $^

sum_reduce: sum_reduce.c
	$(CC) $(CFLAGS) -o $@ $^

sum_allreduce: sum_allreduce.c
	$(CC) $(CFLAGS) -o $@ $^

sum_scan: sum_scan.c
	$(CC) $(CFLAGS) -o $@ $^

run: all
	@echo "--- Running Ex 1: Bcast ---"
	mpirun -np $(NP) ./program
	@echo "\n--- Running Ex 2: Scatter ---"
	mpirun -np $(NP) ./sum_scatter
	@echo "\n--- Running Ex 3: Gather ---"
	mpirun -np $(NP) ./sum_gather
	@echo "\n--- Running Ex 4: Reduce ---"
	mpirun -np $(NP) ./sum_reduce
	@echo "\n--- Running Ex 5: Allreduce ---"
	mpirun -np $(NP) ./sum_allreduce
	@echo "\n--- Running Ex 6: Scan ---"
	mpirun -np $(NP) ./sum_scan

clean:
	rm -f $(EXECS)