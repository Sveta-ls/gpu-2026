import operator
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


_MODULE = SourceModule(r"""
__global__ void layernorm_kernel(
    const float *__restrict__ x,
    const float *__restrict__ gamma,
    const float *__restrict__ beta,
    float *__restrict__ y,
    int width,
    float eps)
{
    extern __shared__ float scratch[];

    const unsigned int tid = threadIdx.x;
    const unsigned int threads = blockDim.x;
    const size_t offset = (size_t)blockIdx.x * (size_t)width;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    // Первый проход: сумма и сумма квадратов.
    for (size_t col = tid; col < (size_t)width; col += threads) {
        const float value = x[offset + col];
        sum += value;
        sum_sq += value * value;
    }

    scratch[tid] = sum;
    scratch[threads + tid] = sum_sq;
    __syncthreads();

    for (unsigned int step = threads / 2; step > 0; step >>= 1) {
        if (tid < step) {
            scratch[tid] += scratch[tid + step];
            scratch[threads + tid] += scratch[threads + tid + step];
        }
        __syncthreads();
    }

    const float mean = scratch[0] / width;
    const float variance =
        fmaxf(scratch[threads] / width - mean * mean, 0.0f);
    const float inv_std = rsqrtf(variance + eps);

    for (size_t col = tid; col < (size_t)width; col += threads) {
        const float normalized = (x[offset + col] - mean) * inv_std;
        y[offset + col] = normalized * gamma[col] + beta[col];
    }
}
""")

_KERNEL = _MODULE.get_function("layernorm_kernel")


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """Return flattened float32 output after row-wise LayerNorm."""
    row_size = operator.index(row_size)
    if row_size <= 0 or row_size > np.iinfo(np.int32).max:
        raise ValueError("row_size must be a positive int32 integer")

    eps = float(eps)
    if not np.isfinite(eps) or not 0 < eps <= np.finfo(np.float32).max:
        raise ValueError("eps must be finite and positive")

    eps32 = np.float32(eps)
    if eps32 == 0:
        raise ValueError("eps is too small for float32")

    x = np.asarray(input, dtype=np.float32).ravel()
    g = np.asarray(gamma, dtype=np.float32).ravel()
    b = np.asarray(beta, dtype=np.float32).ravel()

    if x.size % row_size:
        raise ValueError("input size must be divisible by row_size")
    if g.size != row_size or b.size != row_size:
        raise ValueError("gamma and beta must have length row_size")
    if x.size == 0:
        return np.empty(0, dtype=np.float32)

    h_x = drv.pagelocked_empty(x.size, np.float32)
    h_g = drv.pagelocked_empty(row_size, np.float32)
    h_b = drv.pagelocked_empty(row_size, np.float32)
    h_y = drv.pagelocked_empty(x.size, np.float32)

    np.copyto(h_x, x)
    np.copyto(h_g, g)
    np.copyto(h_b, b)

    stream = drv.Stream()
    allocations = []

    try:
        for host_array in (h_x, h_g, h_b, h_y):
            allocations.append(drv.mem_alloc(host_array.nbytes))

        d_x, d_g, d_b, d_y = allocations

        drv.memcpy_htod_async(d_x, h_x, stream)
        drv.memcpy_htod_async(d_g, h_g, stream)
        drv.memcpy_htod_async(d_b, h_b, stream)

        threads = min(256, max(32, 1 << (row_size - 1).bit_length()))

        _KERNEL(
            d_x, d_g, d_b, d_y,
            np.int32(row_size),
            eps32,
            block=(threads, 1, 1),
            grid=(x.size // row_size, 1, 1),
            shared=2 * threads * np.dtype(np.float32).itemsize,
            stream=stream,
        )

        drv.memcpy_dtoh_async(h_y, d_y, stream)
    finally:
        try:
            stream.synchronize()
        finally:
            for allocation in allocations:
                allocation.free()

    return h_y