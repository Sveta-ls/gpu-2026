
#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void BlockGemmKernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * 32 + tx;
    int row = blockIdx.y * 32 + ty * 4;

    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int t = 0; t < n; t += 32) {

        #pragma unroll
        for (int i = 0; i < 4; i++) {

            if (row + i < n && t + tx < n) {
                tileA[ty * 4 + i][tx] =
                    a[(row + i) * n + t + tx];
            } else {
                tileA[ty * 4 + i][tx] = 0.0f;
            }

            if (t + ty * 4 + i < n && col < n) {
                tileB[ty * 4 + i][tx] =
                    b[(t + ty * 4 + i) * n + col];
            } else {
                tileB[ty * 4 + i][tx] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 32; k++) {
            float value = tileB[k][tx];

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                sum[i] += tileA[ty * 4 + i][k] * value;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (row + i < n && col < n) {
            c[(row + i) * n + col] = sum[i];
        }
    }
}


__global__ void GemmSmall(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0f;

    for (int k = 0; k < n; k++) {
        sum += a[row * n + k] * b[k * n + col];
    }

    c[row * n + col] = sum;
}


std::vector<float> BlockGemmCUDA(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int n
) {
    if (n <= 0) {
        return {};
    }

    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    static float* d_a = nullptr;
    static float* d_b = nullptr;
    static float* d_c = nullptr;
    static int capacity = 0;

    if (n > capacity) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);

        capacity = n;
    }

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(32, 8);

    if (n >= 32) {
        dim3 blocks(
            (n + 31) / 32,
            (n + 31) / 32
        );

        BlockGemmKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    } else {
        dim3 blocks(
            (n + 31) / 32,
            (n + 7) / 8
        );

        GemmSmall<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }

    std::vector<float> result(n * n);

    cudaMemcpy(result.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    return result;
}
