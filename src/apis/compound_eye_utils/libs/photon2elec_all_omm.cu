#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <curand.h>
#include <curand_kernel.h>

#define FPS 1000
#define ONE_TURN_OMM_NUM 8

#define SPLIT_SIZE 8  // split heavy parallel processing to sequencial processing (30000 parallel to 30000/SPLIT_SIZE parallel, each has SPLIT_SIZE sequential)

__constant__ int BumpDuration=30;  // bump for about 30 ms

// for test only, will take quite some time
void print_device_int_result(int *d_arr, int arr_len, int print_len=10000)
{
    int *outcome_counts;
    outcome_counts = (int *)calloc(arr_len, sizeof(int));
    cudaMemcpy(outcome_counts, d_arr, arr_len * sizeof(int), cudaMemcpyDeviceToHost);
    printf("generated random:");
    int cnt = 0;
    for(int i=0; i < print_len; i++)
    {
        cnt += outcome_counts[i];
        // printf("%d, ", outcome_counts[i]);
    }
    printf("%d \n", cnt);
    free(outcome_counts);
}

// CUDA kernel to generate multinomial random variables
__global__ void generate_multinomial(float* ph_input_d, int n, int time_steps, int* res, int seed) {
    /*
    ph_input_d: [ONE_TURN_OMM_NUM][time_steps]
    n: 30000
    time_steps: FPS * sim_time
    res: [ONE_TURN_OMM_NUM][n][time_steps]
    */
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread calculates one idx
    if (idx < time_steps * ONE_TURN_OMM_NUM) {
        int t = idx % time_steps;
        int omm_idx = idx / time_steps;
        curandState state;
        curand_init(seed+idx, 0, 0, &state);
        // Generate random numbers and convert them to outcomes
        for (int j = 0; j < int(ph_input_d[omm_idx * time_steps + t]); j++) {
            double random_num = 1 - curand_uniform(&state);  //curand_uniform: 1included, 0excluded
            int outcome = (int)(random_num * n);
            atomicAdd(&res[omm_idx * time_steps * n + outcome * time_steps + t], 1);
            // atomicAdd(&res[ONE_TURN_OMM_NUM*n*FPS-1], 1);
        }
        // printf("idx=%d, seeds=%d\n", idx, seed);
        // if (idx > time_steps * ONE_TURN_OMM_NUM - 10)
        // {
        //     printf("idx=%d, seeds=%d\n", idx, seed);
        // }
    }
    // if (idx > time_steps * ONE_TURN_OMM_NUM - 30000)
    // {
    //     printf("idx=%d, seeds=%d\n", idx, seed);
    // }
}

void generate_multinomial_host(float *ph_input_d, int n, int time_steps, int *ph_counts_d)
{
    // Start the clock
    // clock_t start = clock();

    cudaMemset(ph_counts_d, 0, n * time_steps * ONE_TURN_OMM_NUM * sizeof(int));

    // Define the number of threads per block
    int threads_per_block = 128;

    // Calculate the number of blocks needed
    int num_blocks = (time_steps*ONE_TURN_OMM_NUM + threads_per_block - 1) / threads_per_block;

    // Launch the CUDA kernel
    generate_multinomial<<<num_blocks, threads_per_block>>>(ph_input_d, n, time_steps, ph_counts_d, int(clock()));
    cudaDeviceSynchronize();
}


__device__ double ran_gamma(curandState localState, const double a, const double b)
{
    if (a < 1){
        double u = curand_uniform_double(&localState);
        return ran_gamma (localState, 1.0 + a, b) * pow (u, 1.0 / a);
    }
    {
        double x, v, u;
        double d = a - 1.0 / 3.0;
        double c = (1.0 / 3.0) / sqrt (d);

        while (1){
            do{
                x = curand_normal_double(&localState);
                v = 1.0 + c * x;
            } while (v <= 0);

            v = v * v * v;
            u = curand_uniform_double(&localState);

            if (u < 1 - 0.0331 * x * x * x * x)
                break;

            if (log (u) < 0.5 * x * x + d * (1 - v + log (v)))
                break;
        }
        return b * d * v;
    }
}

__global__ void photon_transduction_parallel(int n,
                                            int time_steps, int seed,
                                            float *I_in_d,
                                            int *photon_counts, float *res_arr)
{
    // res_arr: [ONE_TURN_OMM_NUM][n/SPLIT_SIZE][time_len]

    int idx = (threadIdx.x + blockIdx.x * blockDim.x) % ((n/SPLIT_SIZE)*ONE_TURN_OMM_NUM);
    int omm_idx_offset = idx / (n / SPLIT_SIZE);
    int mcv_unit_idx = idx % (n / SPLIT_SIZE);

    curandState state;
    curand_init((unsigned long long)(seed + idx), 0, 0, &state);

    int lrefrac[SPLIT_SIZE] = {0};  // 记录不应期的位置
    for (int iter_in_split=0; iter_in_split < SPLIT_SIZE; iter_in_split++)
    {
        for (int t=0; t < time_steps; t++)
        {
            int photon_num = photon_counts[omm_idx_offset*time_steps*n + (mcv_unit_idx*SPLIT_SIZE+iter_in_split)*time_steps + t];

            if (photon_num == 0)
            {
                continue;
            }

            // res_arr[omm_idx_offset*time_steps*(n/SPLIT_SIZE) + mcv_unit_idx*time_steps + t] = photon_num;

            if (lrefrac[iter_in_split] == 0 || t > lrefrac[iter_in_split])
            {
                // generate random latency and bumprefractory
                int latency = int(ran_gamma(state, 12, 2.5));
                int refr_period = int(ran_gamma(state, 18, 12));
                int lend = min(t + latency + BumpDuration, time_steps);
                for (int tmp=t+latency; tmp<lend; tmp++)
                {
                    res_arr[omm_idx_offset*time_steps*(n/SPLIT_SIZE) + mcv_unit_idx*time_steps + tmp] += I_in_d[int(tmp-t-latency+1)];
                }
                lrefrac[iter_in_split] = min(t+latency+BumpDuration+refr_period, time_steps-1);
                t = lrefrac[iter_in_split] - 1;
            }
        }
    }

}

__global__ void add_one_omm_parallel(int n, int time_steps, float *electric_mcv_ts, float *electric_omm_ts)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) % (time_steps*ONE_TURN_OMM_NUM);
    int omm_idx = idx / time_steps;
    int t_idx = idx % time_steps;

    for (int i=0; i < (n / SPLIT_SIZE); i++)
    {
        electric_omm_ts[omm_idx*time_steps + t_idx] += electric_mcv_ts[omm_idx * (n / SPLIT_SIZE) * time_steps + i * time_steps + t_idx];
    }
}

extern "C" void gamma_dis_func(float *I_in, float A=60, float alpha=9, float tau=1, int len=601)  // TODO: check these constants
{
    // A*(ti/tao).^(alpha-1).*exp(-ti./tao)./tao./gamma(alpha);
    for (int i=0; i < len; i++)
    {
        // I_in[i] = pow(A * (i/tau), alpha-1) * exp(-i/tau)/tau/tgamma(alpha);
        I_in[i] = A * pow((i/tau), alpha-1) * exp(-i/tau)/tau/40320;
    }
}

void multi_omm_processing(int n, int time_steps, int *ph_counts_d, float *electric_omm_ts)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error at the beggining of multi_omm: %s\n", cudaGetErrorString(err));
    }
    // TODO: Memory free not checked yet
    // clock_t start = clock();

    // call kernel function to process omm_num ommatidias
    // // generate needed parameters
    int I_in_len = 601;
    float* I_in = (float *)malloc(I_in_len * sizeof(float));
    // cudaMallocHost((void**)&I_in, I_in_len * sizeof(float));
    gamma_dis_func(I_in);  // checked
    float* I_in_d;
    cudaMalloc((void**)&I_in_d, I_in_len * sizeof(float));
    cudaMemcpy(I_in_d, I_in, I_in_len * sizeof(float), cudaMemcpyHostToDevice);
    // cudaFreeHost(I_in);

    // clock_t end_I = clock();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error before define mscv_ts_d: %s\n", cudaGetErrorString(err));
    }

    float *electric_mcv_ts_d;  // [ONE_TURN_OMM_NUM][n/SPLIT_SIZE][time_steps]
    cudaMalloc((void**)&electric_mcv_ts_d, n / SPLIT_SIZE * time_steps * ONE_TURN_OMM_NUM * sizeof(float));
    cudaMemset(electric_mcv_ts_d, 0, n / SPLIT_SIZE * time_steps * ONE_TURN_OMM_NUM * sizeof(float));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error before photon_transduction: %s\n", cudaGetErrorString(err));
    }

    int threads_per_block = 128;
    int num_blocks = (ONE_TURN_OMM_NUM * n / SPLIT_SIZE + threads_per_block - 1) / threads_per_block;
    // print_device_int_result(ph_counts_d, ONE_TURN_OMM_NUM * n * time_steps);
    photon_transduction_parallel<<<num_blocks, threads_per_block>>>(n, time_steps, int(clock()), I_in_d, ph_counts_d, electric_mcv_ts_d);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after photon_transduction: %s\n", cudaGetErrorString(err));
    }

    // clock_t end_ptp = clock();

    float *electric_omm_ts_d; // float [ONE_TURN_OMM_NUM][time_steps]
    cudaMalloc((void**)&electric_omm_ts_d, time_steps * ONE_TURN_OMM_NUM * sizeof(float));
    cudaMemset(electric_omm_ts_d, 0, time_steps * ONE_TURN_OMM_NUM * sizeof(float));
    threads_per_block = 128;
    num_blocks = (ONE_TURN_OMM_NUM * time_steps + threads_per_block - 1) / threads_per_block;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error before add: %s\n", cudaGetErrorString(err));
    }
    add_one_omm_parallel<<<num_blocks, threads_per_block>>>(n, time_steps, electric_mcv_ts_d, electric_omm_ts_d);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after add: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(electric_omm_ts, electric_omm_ts_d, time_steps * ONE_TURN_OMM_NUM * sizeof(float), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error at the before free in multi_omm: %s\n", cudaGetErrorString(err));
    }

    cudaFree(I_in_d);
    cudaFree(electric_mcv_ts_d);
    cudaFree(electric_omm_ts_d);

    // free(photon_ts_h);
    // cudaFreeHost(I_in);
    free(I_in);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error at the end of multi_omm_calling: %s\n", cudaGetErrorString(err));
    }
}

// CUDA kernel to sum along the 30000 axis
__global__ void sumAlongAxis(int *input, float *output, int dim1, int dim2, int dim3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim1 * dim3) {
        int i = idx / dim3; // Index for the first dimension (8)
        int k = idx % dim3; // Index for the third dimension (4000)
        int sum = 0;
        for (int j = 0; j < dim2; j++) {
            sum += input[i * dim2 * dim3 + j * dim3 + k];
        }
        output[i * dim3 + k] = static_cast<float>(sum); // Convert sum to float
    }
}

extern "C" void get_all_omm(float* ph_input, int omm_num, int time_steps, float* electric_output)
{
    /*
    ph_input: [omm_num][sim_time * FPS], the photon input in each time step
    omm_num: int, the total ommatidia numbers
    time_steps: int, the total simulation time steps, with dt=1ms
    electric_output: [omm_num][sim_time * FPS], the output electric signal
    */

//     printf("entered .so \n");
    int n = 30000;              // Number of possible outcomes

    float* ph_input_d;
    cudaMalloc((void**)&ph_input_d, omm_num * time_steps * sizeof(float));
    cudaMemcpy(ph_input_d, ph_input, omm_num * time_steps * sizeof(float), cudaMemcpyHostToDevice);

    // ***** 
    const char *filename = "output.csv";
    int run = 0;
    // ***** 

//     printf("started looping \n");
    for (int turn=0; turn < int(omm_num / ONE_TURN_OMM_NUM); turn++)
    {
//         printf("============%d\n", turn);
        int base_omm_idx = turn * ONE_TURN_OMM_NUM;
        // Allocate memory for outcome_counts on GPU
        int *ph_counts_d; // Pointer for GPU memory, the photon number [ONE_TURN_OMM_NUM][n][time_steps]
        cudaMalloc((void**)&ph_counts_d, ONE_TURN_OMM_NUM * n * time_steps * sizeof(int));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error before generation_multi: %s\n", cudaGetErrorString(err));
        }
        generate_multinomial_host(&ph_input_d[base_omm_idx * time_steps], n, time_steps, ph_counts_d);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error after generation_multi: %s\n", cudaGetErrorString(err));
        }

        run++;
        bool append = (run > 0); // Append mode for subsequent runs
        // processAndSaveToCSV(ph_counts_d, &electric_output[base_omm_idx * time_steps], filename, append);

        multi_omm_processing(n, time_steps, ph_counts_d, &electric_output[base_omm_idx * time_steps]);

        cudaFree(ph_counts_d);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error at the end of the loop: %s\n", cudaGetErrorString(err));
        }
    }


    int base_omm_idx = omm_num - ONE_TURN_OMM_NUM;
    // Allocate memory for outcome_counts on GPU
    int *ph_counts_d; // Pointer for GPU memory
    cudaMalloc((void**)&ph_counts_d, ONE_TURN_OMM_NUM * n * time_steps * sizeof(int));
    generate_multinomial_host(&ph_input_d[base_omm_idx * time_steps], n, time_steps, ph_counts_d);

    run++;
    bool append = (run > 0); // Append mode for subsequent runs
    // processAndSaveToCSV(ph_counts_d, &electric_output[base_omm_idx * time_steps], filename, append);
    multi_omm_processing(n, time_steps, ph_counts_d, &electric_output[base_omm_idx * time_steps]);

    cudaFree(ph_counts_d);

    // ph_input_d
    cudaFree(ph_input_d);

}