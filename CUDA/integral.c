//
// module add cuda/11.6.0
// nvcc integral.cu -o integral
//

/**
 * fir.gpu.cu
 **/
#include <stdio.h>

/*
 * CUDA parameters
 */
#define NUM_BLOCKS 64
#define NUM_THREADS_PER_BLOCK 128
#define NUM_GENERATED_SAMPLES (NUM_BLOCKS * NUM_THREADS_PER_BLOCK)

/*
 * Integrated functions
 */
__device__ double d_my_func2(double x) {
    return (sin(x) + (2 * cos(2 * x)) + (15 * cos(3 * x))
            + ((10 * sin(0.1 * x)) * cos(x)))
           / (abs(sin(x)) + 1);
}

double my_func2(double x) {
    return (sin(x) + (2 * cos(2 * x)) + (15 * cos(3 * x))
            + ((10 * sin(0.1 * x)) * cos(x)))
           / (abs(sin(x)) + 1);
}

/*
 * x coordinates (points) generator
 *
 * params:
 *   rand_num_arr - array which will be filled with random numbers
 *   num_samples  - number of generated array elements
 *   left_bound   - minimal value of generated number
 *   right_bound  - max value of generated number
 */
void generate_random_doubles(double *rand_num_arr,
                             int num_samples,
                             double left_bound,
                             double right_bound) {
    double rand_base = 0;

    for (int i = 0; i < num_samples; ++i) {
        rand_base = (double) rand() / RAND_MAX;
        rand_num_arr[i] = left_bound + rand_base * (right_bound - left_bound);
    }
}

/*
 * monte carlo parallel (device) integration function
 * optimization idea: sum results from all threads inside block
 *                    then send block results to host - this way
 *                    host has less additions to do
 *
 * params:
 *   sample_in - x value of point on plane
 *   f_value   - function value in point defined by sample_in
 */
__global__ void monte_carlo(double *sample_in, double *f_value) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    f_value[index] = d_my_func2(sample_in[index]);
    __syncthreads();
}

/*
 * monte carlo sequential (host) integration function
 *
 * params:
 *   sample_in - x value of point on plane
 *   result    - function value in point defined by sample_in
 */
void monte_carlo_seq(double *sample_in, double *result) {
    for (int i = 0; i < NUM_GENERATED_SAMPLES; i++) {
        result[i] = my_func2(sample_in[i]);
    }
}

/*
 * host program
 */
int main(void) {

    //------------ variable declarations and definitions ------------//
    double *h_samples; // host copies of samples,
    double *d_samples; // device copies of samples
    double *h_fVals;   // host copies of function values
    double *d_fVals;   // device copies of function values

    int nBlk = NUM_BLOCKS;
    int nThx = NUM_THREADS_PER_BLOCK;
    int N = nBlk * nThx; // number of generated samples, same as NUM_GENERATED_SAMPLES
    int size_samples = N * sizeof(double);
    int size_fVals = N * sizeof(double);

    double result;
    double seq_samples[NUM_GENERATED_SAMPLES];
    double seq_result[NUM_GENERATED_SAMPLES];

    cudaEvent_t start, stop;
    float para_time, seq_time;

    double left_bound = 0;
    double right_bound = 20000;

    //---------------------------- setup ----------------------------//
    srand(time(NULL));

    // Allocate space for device copies of samples, function values
    cudaMalloc((void **) &d_samples, size_samples);
    cudaMalloc((void **) &d_fVals, size_fVals);

    // Alloc space for host copies of samples, function values
    h_samples = (double *) malloc(size_samples);
    h_fVals = (double *) malloc(size_fVals);

    // generate x values
    generate_random_doubles(h_samples, N, left_bound, right_bound);
    generate_random_doubles(seq_samples, N, left_bound, right_bound);

    // Copy input data to device
    cudaMemcpy(d_samples, h_samples, size_samples, cudaMemcpyHostToDevice);

    //--------------- parallel monte carlo integration ---------------//
    // time measurement start
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    monte_carlo<<<nBlk, nThx>>>(d_samples, d_fVals);

    // time measurement stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&para_time, start, stop);

    printf("%s\n", cudaGetErrorName(cudaGetLastError()));

    // Copy result back to host
    cudaMemcpy(h_fVals, d_fVals, size_fVals, cudaMemcpyDeviceToHost);

    // sum results
    result = 0;
    for (int i = 0; i < N; i++) {
        result = result + h_fVals[i];
    }
    result = result / NUM_GENERATED_SAMPLES;
    printf("[PARA] monte_carlo() GPU time is %f ms\n", para_time);
    printf("[PARA] result: %f\n", result);

    //--------------- sequential monte carlo integration ---------------//
    // time measurement start
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    monte_carlo_seq(seq_samples, seq_result);

    // time measurement stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&seq_time, start, stop);

    result = 0;
    for (int i = 0; i < N; i++) {
        result = result + seq_result[i];
    }
    result = result / NUM_GENERATED_SAMPLES;
    printf("[SEQ] monte_carlo() CPU time is %f ms\n", seq_time);
    printf("[SEQ] result: %f\n", result);


    //-------------------------- memory cleanup --------------------------//
    free(h_samples);
    free(h_fVals);
    cudaFree(d_samples);
    cudaFree(d_fVals);

    return 0;
}
