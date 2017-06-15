#include "csr.cuh"
#include "cudaMacros.cuh"

template<unsigned int blockSize, unsigned int beh>
__device__ void warpReduce_mxn(volatile double *sdata, unsigned int tid) {
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
	for (irp_off = irp[bid] + tid; irp_off < irp[bid + 1]; irp_off += BlockSize) {
		a = 0;
		as_off = (irp[bid] + tid) * bes;
		for (i = 0; i < beh; ++i) {
			for (j = 0; j < bew; ++j) {
				sdata[tid_0 + i] += x[ja[irp_off] + j] * as[as_off + a++];
			}
		}
		as += BlockSize * bes;
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
		memcpy(y + blockIdx.x * beh, sdata, sizeof(double) * beh);
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
	float milliseconds;
	cudaError_t error;

	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_irp, irp_size * sizeof(unsigned int)))
	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_ja, ja_size * sizeof(unsigned int)))
	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_as, as_size* sizeof(double)))
	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_y, (rows+ beh - rows % beh) * sizeof(double)))
	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_x, (cols + bew) * sizeof(double)))

	CHECK_CUDA_ERROR(cudaMemcpy(d_irp, h_irp, irp_size * sizeof(unsigned int), cudaMemcpyHostToDevice))
	CHECK_CUDA_ERROR(cudaMemcpy(d_ja, h_ja, ja_size * sizeof(unsigned int), cudaMemcpyHostToDevice))
	CHECK_CUDA_ERROR(cudaMemcpy(d_as, h_as, as_size * sizeof(double), cudaMemcpyHostToDevice))
	CHECK_CUDA_ERROR(cudaMemcpy(d_x, x_off, (cols+ bew) * sizeof(double), cudaMemcpyHostToDevice))

	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaEventCreate(&start))
	CHECK_CUDA_ERROR(cudaEventCreate(&stop))
	CHECK_CUDA_ERROR(cudaEventRecord(start))
	SWITCH_BLOCKENTRY_SIZE_AND_CUDA_BLOCK_SIZE(
			blockSize, beh, bew,
			device_cuda_csr_matrixvector_simple_mxn,
			blockRows, blockSize, 2 * blockSize * beh* sizeof(double),
			(d_as, d_irp, d_ja, d_x, d_y))
	CHECK_CUDA_ERROR(cudaEventRecord(stop))

	CHECK_CUDA_ERROR(cudaMemcpy(h_y, d_y,rows * sizeof(double), cudaMemcpyDeviceToHost))

	CHECK_CUDA_ERROR(cudaEventSynchronize(stop))
	CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop))

	CHECK_CUDA_ERROR(cudaFree(d_ja))
	CHECK_CUDA_ERROR(cudaFree(d_as))
	CHECK_CUDA_ERROR(cudaFree(d_irp))
	CHECK_CUDA_ERROR(cudaFree(d_y))
	CHECK_CUDA_ERROR(cudaFree(d_x))

	return milliseconds * 1000;
}

