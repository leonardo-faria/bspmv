#include "ell.cuh"
#include "cudaMacros.cuh"
#include <stdio.h>


template<unsigned int blockSize, unsigned int beh>
__device__ void warpReduce_mxn(volatile double *sdata, unsigned int tid_0) {
	int i;
	if (blockSize >= 64)
		SUM_POSITIONS_H(beh, 32)
	if (blockSize >= 32)
		SUM_POSITIONS_H(beh, 16)
	if (blockSize >= 16)
		SUM_POSITIONS_H(beh, 8)
	if (blockSize >= 8)
		SUM_POSITIONS_H(beh, 4)
	if (blockSize >= 4)
		SUM_POSITIONS_H(beh, 2)
	if (blockSize >= 2) {
		SUM_POSITIONS_H(beh, 1)
	}
}

template<unsigned int blockSize, unsigned int beh>
__device__ void warpReduce_multithread_simple_mxn(volatile double *sdata, unsigned int sdata_index) {
	if (blockSize >= 64)
		sdata[sdata_index] += sdata[sdata_index + 32 * beh];
	if (blockSize >= 32)
		sdata[sdata_index] += sdata[sdata_index + 16 * beh];
	if (blockSize >= 16)
		sdata[sdata_index] += sdata[sdata_index + 8 * beh];
	if (blockSize >= 8)
		sdata[sdata_index] += sdata[sdata_index + 4 * beh];
	if (blockSize >= 4)
		sdata[sdata_index] += sdata[sdata_index + 2 * beh];
	if (blockSize >= 2) {
		sdata[sdata_index] += sdata[sdata_index + beh];
	}
}

template<unsigned int blockSize, unsigned int beh, unsigned int bew>
__global__ void device_cuda_ellpack_matrixvector_multithread_simple_mxn(double* as, unsigned int* ja, double* x, double*y, unsigned int max_n_blocks) {
	extern __shared__ double sdata[];
	unsigned int sdata_index = threadIdx.y + threadIdx.x * beh;
	unsigned int bes = beh * bew;
	unsigned int as_off = (blockIdx.x * max_n_blocks + threadIdx.x) * bes + threadIdx.y * bew;
	unsigned int tid = threadIdx.x;
	sdata[sdata_index] = 0;
	unsigned int ja_off, ja_end;
	unsigned int bid = blockIdx.x;
	ja_end = (bid + 1) * max_n_blocks;

	unsigned int j;

//	printf("0\tblock=%d\tthread=%d\tthreadY=%d\tvalue=%f\n", blockIdx.x, threadIdx.x, threadIdx.y, sdata[sdata_index]);
	for (ja_off = bid * max_n_blocks + tid; ja_off < ja_end; ja_off += blockSize) {
		for (j = 0; j < bew; ++j) {
			sdata[sdata_index] += x[ja[ja_off] + j] * as[as_off + j];
		}
		as_off += bes * blockSize;
	}

	__syncthreads();
	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[sdata_index] += sdata[sdata_index + 256 * beh];
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128)
			sdata[sdata_index] += sdata[sdata_index + 128 * beh];
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64)
			sdata[sdata_index] += sdata[sdata_index + 64 * beh];
		__syncthreads();
	}
	if (tid < 32)
		warpReduce_multithread_simple_mxn<blockSize, beh>(sdata, sdata_index);

	if (tid == 0) {
		y[threadIdx.y + bid * beh] = sdata[threadIdx.y];
	}
//	printf("Y[%d]=%f\n", threadIdx.y + bid * beh, y[threadIdx.y + bid * beh]);

}

template<unsigned int blockSize, unsigned int beh, unsigned int bew>
__global__ void device_cuda_ellpack_matrixvector_simple_mxn(double* as, unsigned int* ja, double* x, double* y, unsigned int max_n_blocks) {
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int tid_0 = tid * beh;
	unsigned int bid = blockIdx.x;
	unsigned int bes = beh * bew;
	unsigned int as_off = (bid * max_n_blocks + tid) * bes;
	unsigned int ja_end = (bid + 1) * max_n_blocks;
	unsigned int ja_off;
	for (ja_off = 0; ja_off < beh; ++ja_off)
		sdata[tid_0 + ja_off] = 0;
	unsigned int i, j, a;

#pragma unroll
	for (ja_off = bid * max_n_blocks + tid; ja_off < ja_end; ja_off += blockSize) {
		a = 0;
#pragma unroll
		for (i = 0; i < beh; ++i) {
#pragma unroll
			for (j = 0; j < bew; ++j) {
				sdata[tid_0 + i] += x[ja[ja_off] + j] * as[as_off + a];
				a++;
			}
		}
		as_off += bes * blockSize;
	}
	__syncthreads();
	if (blockSize >= 512) {
		if (tid < 256)
			SUM_POSITIONS_H(beh, 256)
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128)
			SUM_POSITIONS_H(beh, 128)
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64)
			SUM_POSITIONS_H(beh, 64)
		__syncthreads();
	}
	if (tid < 32)
		warpReduce_mxn<blockSize, beh>(sdata, tid_0);

	if (tid == 0) {
		for (i = 0; i < beh; i++)
			y[blockIdx.x * beh + i] = sdata[i];
	}

}

__host__ double cuda_ellpack_matrixvector(unsigned int* h_ja, unsigned int ja_size, double* h_as, unsigned int as_size, unsigned int cols, unsigned int rows, unsigned int beh, unsigned int bew, unsigned int blockRows, unsigned int max_n_blocks, double* h_x, double* h_y, unsigned int blockSize) {
	if (blockSize * beh > 1024)
		return 0;
	cudaError_t error;
	unsigned int* d_ja;
	double *d_as, *d_x, *d_y;

	double* x_off = (double*) calloc(cols + bew, sizeof(double));
	memcpy(x_off, h_x, cols * sizeof(double));

	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_ja, ja_size * sizeof(unsigned int)))
	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_as, as_size * sizeof(double)))
	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_y, (rows + beh - rows % beh) * sizeof(double)))
	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_x, (cols + bew) * sizeof(double)))

	CHECK_CUDA_ERROR(cudaMemcpy(d_ja, h_ja, ja_size * sizeof(unsigned int), cudaMemcpyHostToDevice))
	CHECK_CUDA_ERROR(cudaMemcpy(d_as, h_as, as_size * sizeof(double), cudaMemcpyHostToDevice))
	CHECK_CUDA_ERROR(cudaMemcpy(d_x, x_off, (cols + bew) * sizeof(double), cudaMemcpyHostToDevice))
	CHECK_CUDA_ERROR(cudaMemcpy(d_y, h_y, rows * sizeof(double), cudaMemcpyHostToDevice))
	free(x_off);

	float temp, milliseconds = 0;
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaEventCreate(&start))
	CHECK_CUDA_ERROR(cudaEventCreate(&stop))
	dim3 BS(blockSize, beh, 1);
	for (int run = 0; run < TRIES; run++) {

		CHECK_CUDA_ERROR(cudaEventRecord(start));

		SWITCH_BLOCKENTRY_SIZE_AND_CUDA_BLOCK_SIZE(blockSize, beh, bew, device_cuda_ellpack_matrixvector_multithread_simple_mxn, blockRows, BS, 2 * blockSize * beh * sizeof(double), (d_as, d_ja, d_x, d_y, max_n_blocks))

		error = cudaGetLastError();
		if (error != cudaSuccess) {
			exit(1);
		}
		CHECK_CUDA_ERROR(cudaEventRecord(stop))
		CHECK_CUDA_ERROR(cudaEventSynchronize(stop))
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&temp, start, stop));
		milliseconds += temp;
	}
	CHECK_CUDA_ERROR(cudaMemcpy(h_y, d_y, rows * sizeof(double), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaFree(d_ja))
	CHECK_CUDA_ERROR(cudaFree(d_as))
	CHECK_CUDA_ERROR(cudaFree(d_x))
	CHECK_CUDA_ERROR(cudaFree(d_y))

	return milliseconds * 1000.0 / (float) TRIES;
}
