#include "csr.cuh"
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
__device__ void warpReduce_multithread_simple_mxn(volatile double *sdata, unsigned int sdata_index, unsigned int tid) {
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
	if (blockSize >= 2)
		sdata[sdata_index] += sdata[sdata_index + beh];
}

template<unsigned int blockSize, unsigned int beh, unsigned int bew>
__global__ void device_cuda_csr_matrixvector_multithread_simple_mxn(double* as, unsigned int *irp, unsigned int* ja, double* x, double* y) {
	extern __shared__ double sdata[];
//	printf("block size: x%d\ty%d\n", blockDim.x, blockDim.y);
	unsigned int sdata_index = threadIdx.y + threadIdx.x * beh;
	sdata[sdata_index] = 0;
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int irp_off = irp[bid] + tid;
	unsigned int irp_end = irp[bid + 1];

	for (; irp_off < irp_end; irp_off += blockSize) {
		for (int i = 0; i < bew; ++i) {
			sdata[sdata_index] += x[ja[irp_off] + i] * as[irp_off * beh * bew + threadIdx.y * bew + i];
//			if (x[ja[irp_off] + i] * as[irp_off * beh * bew + threadIdx.y * bew + i] > 0)
//				printf("bid:%d\ttidX:%d\ttidY:%d\tvalue was: %f\n",bid,tid,threadIdx.y, x[ja[irp_off] + i] * as[irp_off * beh * bew + threadIdx.y * bew + i]);
		}
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
		warpReduce_multithread_simple_mxn<blockSize, beh>(sdata, sdata_index, threadIdx.x);

	if (tid == 0) {
		y[threadIdx.y + bid * beh] = sdata[threadIdx.y];
	}
}

template<unsigned int BlockSize, unsigned int beh, unsigned int bew>
__global__ void device_cuda_csr_matrixvector_simple_mxn(double* as, unsigned int *irp, unsigned int* ja, double* x, double* y) {

	extern __shared__ double sdata[];
	unsigned int i, j, irp_off;
	unsigned int bes = beh * bew;
	unsigned int bid = blockIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int tid_0 = threadIdx.x * beh;
	for (i = 0; i < beh; ++i) {
		sdata[tid_0 + i] = 0;
	}
	unsigned a;
	unsigned int as_off;
	unsigned int temp = 0;
	for (irp_off = irp[bid] + tid; irp_off < irp[bid + 1]; irp_off += BlockSize) {
		a = 0;
		as_off = (irp_off) * bes;
		for (i = 0; i < beh; ++i) {
			for (j = 0; j < bew; ++j) {
				sdata[tid_0 + i] += x[ja[irp_off] + j] * as[as_off + a++];
			}
		}
	}
	__syncthreads();
	if (BlockSize >= 512) {
		if (tid < 256)
			SUM_POSITIONS_H(beh, 256)
		__syncthreads();
	}
	if (BlockSize >= 256) {
		if (tid < 128)
			SUM_POSITIONS_H(beh, 128)
		__syncthreads();
	}
	if (BlockSize >= 128) {
		if (tid < 64)
			SUM_POSITIONS_H(beh, 64)
		__syncthreads();
	}
	if (tid < 32)
		warpReduce_mxn<BlockSize, beh>(sdata, tid_0);
	if (tid == 0) {
		for (int var = 0; var < beh; ++var) {
			y[blockIdx.x * beh + var] = sdata[var];
		}
	}
}

__host__ double cuda_csr_matrixvector(unsigned int *h_irp, unsigned int irp_size, unsigned int* h_ja, unsigned int ja_size, double* h_as, unsigned int as_size, unsigned int cols, unsigned int rows, unsigned int beh, unsigned int bew, unsigned int blockRows, double* h_x, double* h_y, unsigned int blockSize) {

	double* x_off = (double*) calloc(cols + bew, sizeof(double));
	memcpy(x_off, h_x, cols * sizeof(double));

	unsigned int* d_irp;
	unsigned int* d_ja;
	double* d_as;
	double* d_x;
	double* d_y;
	cudaError_t error;
	cudaSetDevice(0);
	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_irp, irp_size * sizeof(unsigned int)))
	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_ja, ja_size * sizeof(unsigned int)))
	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_as, as_size * sizeof(double)))
	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_y, (rows + beh - rows % beh) * sizeof(double)))
	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_x, (cols + bew) * sizeof(double)))

	CHECK_CUDA_ERROR(cudaMemcpy(d_irp, h_irp, irp_size * sizeof(unsigned int), cudaMemcpyHostToDevice))
	CHECK_CUDA_ERROR(cudaMemcpy(d_ja, h_ja, ja_size * sizeof(unsigned int), cudaMemcpyHostToDevice))
	CHECK_CUDA_ERROR(cudaMemcpy(d_as, h_as, as_size * sizeof(double), cudaMemcpyHostToDevice))
	CHECK_CUDA_ERROR(cudaMemcpy(d_x, x_off, (cols + bew) * sizeof(double), cudaMemcpyHostToDevice))
	CHECK_CUDA_ERROR(cudaMemcpy(d_y, h_y, rows * sizeof(double), cudaMemcpyHostToDevice))

	cudaEvent_t start, stop;
	float temp, milliseconds = 0;
	CHECK_CUDA_ERROR(cudaEventCreate(&start))
	CHECK_CUDA_ERROR(cudaEventCreate(&stop))
	dim3 BS(blockSize, beh, 1);

	for (int run = 0; run < TRIES; run++) {
		CHECK_CUDA_ERROR(cudaEventRecord(start))

	SWITCH_BLOCKENTRY_SIZE_AND_CUDA_BLOCK_SIZE(blockSize, beh, bew, device_cuda_csr_matrixvector_multithread_simple_mxn, blockRows, BS, 2 * blockSize * beh * sizeof(double), (d_as, d_irp, d_ja, d_x, d_y))
		CHECK_CUDA_ERROR(cudaEventRecord(stop))
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			exit(1);
		}
		CHECK_CUDA_ERROR(cudaEventRecord(stop))
		CHECK_CUDA_ERROR(cudaEventSynchronize(stop))
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&temp, start, stop));
		milliseconds += temp;
	}
	CHECK_CUDA_ERROR(cudaMemcpy(h_y, d_y, rows * sizeof(double), cudaMemcpyDeviceToHost))

	CHECK_CUDA_ERROR(cudaEventSynchronize(stop))
	CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop))

	CHECK_CUDA_ERROR(cudaFree(d_ja))
	CHECK_CUDA_ERROR(cudaFree(d_as))
	CHECK_CUDA_ERROR(cudaFree(d_irp))
	CHECK_CUDA_ERROR(cudaFree(d_y))
	CHECK_CUDA_ERROR(cudaFree(d_x))

	return milliseconds * 1000.0 / (float) TRIES;
}

