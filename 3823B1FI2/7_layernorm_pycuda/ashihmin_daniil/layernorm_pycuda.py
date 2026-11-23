import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_CUDA_SOURCE = """
#define WARP_SIZE 32

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    int num_warps = blockDim.x / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warpReduceSum(val);
        if (threadIdx.x == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();

    return shared[0];
}

__inline__ __device__ float reduceSum(float val) {
    if (blockDim.x <= WARP_SIZE) {
        val = warpReduceSum(val);
        return __shfl_sync(0xffffffff, val, 0);
    }
    return blockReduceSum(val);
}

extern "C" __global__ void layernormSharedKernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int row_size,
    float eps)
{
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    size_t row_offset = static_cast<size_t>(row) * row_size;
    const float* row_in = input + row_offset;
    float* row_out = output + row_offset;

    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float val = row_in[i];
        sdata[i] = val;
        local_sum += val;
    }

    float mean = reduceSum(local_sum) / static_cast<float>(row_size);

    float local_var_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float diff = sdata[i] - mean;
        sdata[i] = diff;
        local_var_sum += diff * diff;
    }

    float var = reduceSum(local_var_sum) / static_cast<float>(row_size);
    float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < row_size; i += blockDim.x) {
        row_out[i] = (sdata[i] * inv_std) * __ldg(&gamma[i]) + __ldg(&beta[i]);
    }
}

extern "C" __global__ void layernormVectorKernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int row_size,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    size_t row_offset = static_cast<size_t>(row) * row_size;

    if (row_size >= 4) {
        const float4* row_in4 = reinterpret_cast<const float4*>(input + row_offset);
        const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
        const float4* beta4 = reinterpret_cast<const float4*>(beta);
        float4* row_out4 = reinterpret_cast<float4*>(output + row_offset);
        int vec_size = row_size / 4;

        float local_sum = 0.0f;
        for (int i = tid; i < vec_size; i += blockDim.x) {
            float4 v = row_in4[i];
            local_sum += (v.x + v.y) + (v.z + v.w);
        }
        float mean = reduceSum(local_sum) / static_cast<float>(row_size);

        float local_var_sum = 0.0f;
        for (int i = tid; i < vec_size; i += blockDim.x) {
            float4 v = row_in4[i];
            float dx = v.x - mean;
            float dy = v.y - mean;
            float dz = v.z - mean;
            float dw = v.w - mean;
            local_var_sum += (dx * dx + dy * dy) + (dz * dz + dw * dw);
        }
        float var = reduceSum(local_var_sum) / static_cast<float>(row_size);
        float inv_std = rsqrtf(var + eps);

        for (int i = tid; i < vec_size; i += blockDim.x) {
            float4 v = row_in4[i];
            float4 g = __ldg(&gamma4[i]);
            float4 b = __ldg(&beta4[i]);
            float4 out;
            out.x = ((v.x - mean) * inv_std) * g.x + b.x;
            out.y = ((v.y - mean) * inv_std) * g.y + b.y;
            out.z = ((v.z - mean) * inv_std) * g.z + b.z;
            out.w = ((v.w - mean) * inv_std) * g.w + b.w;
            row_out4[i] = out;
        }
    } else {
        const float* row_in = input + row_offset;
        float* row_out = output + row_offset;

        float local_sum = 0.0f;
        for (int i = tid; i < row_size; i += blockDim.x) {
            local_sum += row_in[i];
        }
        float mean = reduceSum(local_sum) / static_cast<float>(row_size);

        float local_var_sum = 0.0f;
        for (int i = tid; i < row_size; i += blockDim.x) {
            float diff = row_in[i] - mean;
            local_var_sum += diff * diff;
        }
        float var = reduceSum(local_var_sum) / static_cast<float>(row_size);
        float inv_std = rsqrtf(var + eps);

        for (int i = tid; i < row_size; i += blockDim.x) {
            row_out[i] = ((row_in[i] - mean) * inv_std) * __ldg(&gamma[i]) + __ldg(&beta[i]);
        }
    }
}
"""

_MODULE = SourceModule(_CUDA_SOURCE, options=["-O3"], no_extern_c=True)
_KERNEL_SHARED = _MODULE.get_function("layernormSharedKernel")
_KERNEL_VECTOR = _MODULE.get_function("layernormVectorKernel")

_MEMORY_CACHE = {
    "d_x": None,
    "d_g": None,
    "d_b": None,
    "d_out": None,
    "cap_elements": 0,
    "cap_row_size": 0,
}
_GPU_STREAM = None


def _ensure_gpu_buffers(total_elements, row_size):
    global _GPU_STREAM
    if _GPU_STREAM is None:
        _GPU_STREAM = cuda.Stream()

    if _MEMORY_CACHE["cap_elements"] < total_elements:
        if _MEMORY_CACHE["d_x"] is not None:
            _MEMORY_CACHE["d_x"].free()
            _MEMORY_CACHE["d_out"].free()
        _MEMORY_CACHE["d_x"] = cuda.mem_alloc(total_elements * 4)
        _MEMORY_CACHE["d_out"] = cuda.mem_alloc(total_elements * 4)
        _MEMORY_CACHE["cap_elements"] = total_elements

    if _MEMORY_CACHE["cap_row_size"] < row_size:
        if _MEMORY_CACHE["d_g"] is not None:
            _MEMORY_CACHE["d_g"].free()
            _MEMORY_CACHE["d_b"].free()
        _MEMORY_CACHE["d_g"] = cuda.mem_alloc(row_size * 4)
        _MEMORY_CACHE["d_b"] = cuda.mem_alloc(row_size * 4)
        _MEMORY_CACHE["cap_row_size"] = row_size


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.
    """
    orig_shape = np.shape(input)
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    if x.size == 0 or row_size <= 0:
        return x.reshape(orig_shape)

    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta, dtype=np.float32).ravel()

    total_elements = x.size
    row_count = total_elements // row_size

    _ensure_gpu_buffers(total_elements, row_size)
    d_x = _MEMORY_CACHE["d_x"]
    d_g = _MEMORY_CACHE["d_g"]
    d_b = _MEMORY_CACHE["d_b"]
    d_out = _MEMORY_CACHE["d_out"]

    cuda.memcpy_htod_async(d_x, x, _GPU_STREAM)
    cuda.memcpy_htod_async(d_g, g, _GPU_STREAM)
    cuda.memcpy_htod_async(d_b, b, _GPU_STREAM)

    if row_size <= 4096:
        threads = min(512, max(32, row_size))
        shared_bytes = row_size * 4
        _KERNEL_SHARED(
            d_x,
            d_g,
            d_b,
            d_out,
            np.int32(row_size),
            np.float32(eps),
            block=(threads, 1, 1),
            grid=(row_count, 1, 1),
            shared=shared_bytes,
            stream=_GPU_STREAM,
        )
    else:
        threads = 256
        _KERNEL_VECTOR(
            d_x,
            d_g,
            d_b,
            d_out,
            np.int32(row_size),
            np.float32(eps),
            block=(threads, 1, 1),
            grid=(row_count, 1, 1),
            shared=0,
            stream=_GPU_STREAM,
        )

    out = np.empty_like(x)

    cuda.memcpy_dtoh_async(out, d_out, _GPU_STREAM)
    _GPU_STREAM.synchronize()

    return out.reshape(orig_shape)