#include "ell.cuh"
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
	for (ja_off = bid * max_n_blocks + tid; ja_off < ja_end; ja_off += blockSize) {
		a = 0;
		for (i = 0; i < beh; ++i) {
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

__host__ double cuda_ellpack_matrixvector(unsigned int* h_ja, unsigned int ja_size, double* h_as, unsigned int as_size, unsigned int cols, unsigned int rows, unsigned int beh, unsigned int bew, unsigned int blockRows,unsigned int max_n_blocks, double* h_x, double* h_y, unsigned int blockSize) {
	cudaError_t error;

	unsigned int* d_ja;
	double *d_as, *d_x, *d_y;

	double* x_off = (double*) calloc(cols + bew, sizeof(double));
	memcpy(x_off, h_x, cols * sizeof(double));

	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_ja, ja_size * sizeof(unsigned int)))
	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_as, as_size * sizeof(double)))
	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_y, (rows + beh- rows%beh) * sizeof(double)))
	CHECK_CUDA_ERROR(cudaMalloc((void** ) &d_x, (cols+bew) * sizeof(double)))

	CHECK_CUDA_ERROR(cudaMemcpy(d_ja, h_ja,ja_size * sizeof(unsigned int), cudaMemcpyHostToDevice))
	CHECK_CUDA_ERROR(cudaMemcpy(d_as, h_as, as_size* sizeof(double), cudaMemcpyHostToDevice))
	CHECK_CUDA_ERROR(cudaMemcpy(d_x, x_off, (cols +bew) * sizeof(double), cudaMemcpyHostToDevice))
	CHECK_CUDA_ERROR(cudaMemcpy(d_y, h_y, rows * sizeof(double), cudaMemcpyHostToDevice))
	free(x_off);


	float milliseconds;
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaEventCreate(&start))
	CHECK_CUDA_ERROR(cudaEventCreate(&stop))
	CHECK_CUDA_ERROR(cudaEventRecord(start));

	SWITCH_BLOCKENTRY_SIZE_AND_CUDA_BLOCK_SIZE(
			blockSize,beh, bew,
			device_cuda_ellpack_matrixvector_simple_mxn,
			blockRows, blockSize, 2 * blockSize *beh* sizeof(double),
			(d_as, d_ja, d_x, d_y, max_n_blocks)
		)
	//	device_cuda_ellpack_matrixvector_simple_mxn<BLOCK_SIZE_X> <<<matrix.getBlockRows(), BLOCK_SIZE_X, 2 * BLOCK_SIZE_X * matrix.getBlockHeight() * sizeof(double)>>>(d_as, d_ja, d_x, d_y, matrix.getMaxBlocks(), matrix.getBlockHeight(), matrix.getBlockWidth());
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		exit(1);
	}
	CHECK_CUDA_ERROR(cudaEventRecord(stop))
	CHECK_CUDA_ERROR(cudaEventSynchronize(stop))
	CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

	CHECK_CUDA_ERROR(cudaMemcpy(h_y, d_y,rows* sizeof(double), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaFree(d_ja))
	CHECK_CUDA_ERROR(cudaFree(d_as))
	CHECK_CUDA_ERROR(cudaFree(d_x))
	CHECK_CUDA_ERROR(cudaFree(d_y))

	return milliseconds * 1000;
}
