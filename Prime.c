#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define N 100000000  // Upper limit to count primes
#define SQR_LIMIT 10000  // sqrt(N) for small primes

void simpleSieve(int limit, int primes[], int *primeCount) {
    int isPrime[limit + 1];
    for (int i = 0; i <= limit; i++) isPrime[i] = 1;
    isPrime[0] = isPrime[1] = 0;

    for (int p = 2; p * p <= limit; p++) {
        if (isPrime[p]) {
            for (int i = p * p; i <= limit; i += p) isPrime[i] = 0;
        }
    }

    *primeCount = 0;
    for (int p = 2; p <= limit; p++) {
        if (isPrime[p]) primes[(*primeCount)++] = p;
    }
}

int countPrimesSequential(int n) {
    int primes[SQR_LIMIT], primeCount;
    simpleSieve(SQR_LIMIT, primes, &primeCount);

    int count = (n >= 2) ? 1 : 0;

    for (int low = 2; low <= n; low += SQR_LIMIT) {
        int high = low + SQR_LIMIT - 1;
        if (high > n) high = n;

        int segmentSize = high - low + 1;
        int isPrimeArray[segmentSize];
        for (int i = 0; i < segmentSize; i++) isPrimeArray[i] = 1;

        for (int i = 0; i < primeCount; i++) {
            int p = primes[i];
            int start = (low / p) * p;
            if (start < low) start += p;
            if (start == p) start += p;

            for (int j = start; j <= high; j += p) {
                isPrimeArray[j - low] = 0;
            }
        }

        for (int i = 0; i < segmentSize; i++) {
            if (isPrimeArray[i]) count++;
        }
    }
    return count;
}

int countPrimesParallel(int n) {
    int primes[SQR_LIMIT], primeCount;
    simpleSieve(SQR_LIMIT, primes, &primeCount);

    int count = (n >= 2) ? 1 : 0;

    #pragma omp parallel
    {
        int local_count = 0;

        #pragma omp for schedule(dynamic)
        for (int low = 2; low <= n; low += SQR_LIMIT) {
            int high = low + SQR_LIMIT - 1;
            if (high > n) high = n;

            int segmentSize = high - low + 1;
            int isPrimeArray[segmentSize];
            for (int i = 0; i < segmentSize; i++) isPrimeArray[i] = 1;

            for (int i = 0; i < primeCount; i++) {
                int p = primes[i];
                int start = (low / p) * p;
                if (start < low) start += p;
                if (start == p) start += p;

                for (int j = start; j <= high; j += p) {
                    isPrimeArray[j - low] = 0;
                }
            }

            for (int i = 0; i < segmentSize; i++) {
                if (isPrimeArray[i]) local_count++;
            }
        }

        #pragma omp atomic
        count += local_count;
    }
    return count;
}

int main() {
    int num_threads = 4;
    omp_set_num_threads(num_threads);

    clock_t start, end;
    double time_seq, time_parallel;

    // Sequential Execution
    start = clock();
    int count_seq = countPrimesSequential(N);
    end = clock();
    time_seq = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Parallel Execution
    start = clock();
    int count_par = countPrimesParallel(N);
    end = clock();
    time_parallel = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Output results
    printf("Prime Count (Sequential): %d\n", count_seq);
    printf("Time Taken (Sequential): %.4f sec\n", time_seq);
    printf("Prime Count (Parallel, %d threads): %d\n", num_threads, count_par);
    printf("Time Taken (Parallel): %.4f sec\n", time_parallel);
    printf("Speedup: %.2fx\n", time_seq / time_parallel);

    return 0;
}
