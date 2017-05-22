#include "ell.cuh"
#include <stdio.h>
#define SUM_POSITIONS_3(offset) \
	{\
		sdata[tid_0] += sdata[tid_0 + offset*3];\
		sdata[tid_1] += sdata[tid_1 + offset*3];\
		sdata[tid_2] += sdata[tid_2 + offset*3];\
	}
#define SUM_POSITIONS_H(beh,offset) \
	{\
		for(i=0;i<beh;i++){\
			sdata[tid_0+i]+=sdata[tid_0+i+offset*beh];\
		}\
	}

template<unsigned int BlockSize>
__device__ void warpReduce_3x3(volatile double *sdata, unsigned int tid_0, unsigned int tid_1, unsigned int tid_2) {
	if (BlockSize >= 64)
		SUM_POSITIONS_3(32)
	if (BlockSize >= 32)
		SUM_POSITIONS_3(16)
	if (BlockSize >= 16)
		SUM_POSITIONS_3(8)
	if (BlockSize >= 8)
		SUM_POSITIONS_3(4)
	if (BlockSize >= 4)
		SUM_POSITIONS_3(2)
	if (BlockSize >= 2)
		SUM_POSITIONS_3(1)
}

template<unsigned int BlockSize>
__device__ void warpReduce_mxn(volatile double *sdata, unsigned int tid_0, unsigned int beh) {
	int i;
	if (BlockSize >= 64)
		SUM_POSITIONS_H(beh, 32)
	if (BlockSize >= 32)
		SUM_POSITIONS_H(beh, 16)
	if (BlockSize >= 16)
		SUM_POSITIONS_H(beh, 8)
	if (BlockSize >= 8)
		SUM_POSITIONS_H(beh, 4)
	if (BlockSize >= 4)
		SUM_POSITIONS_H(beh, 2)
	if (BlockSize >= 2){
		SUM_POSITIONS_H(beh, 1)
	}
}

template<unsigned int BlockSize>
__global__ void device_cuda_ellpack_matrixvector_simple_3x3(double* as, unsigned int* ja, double* x, double* y, unsigned int max_n_blocks) {
	__shared__ double sdata[BlockSize * 3];
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int as_off = (bid * max_n_blocks + tid) * 9;
	unsigned int ja_end = (bid + 1) * max_n_blocks;
	unsigned int tid_0 = tid * 3;
	unsigned int tid_1 = tid_0 + 1;
	unsigned int tid_2 = tid_0 + 2;

	sdata[tid_0] = 0;
	sdata[tid_1] = 0;
	sdata[tid_2] = 0;
	for (unsigned int ja_off = bid * max_n_blocks + tid; ja_off < ja_end; ja_off += BlockSize) {
		sdata[tid_0] += x[ja[ja_off]] * as[as_off] + x[ja[ja_off] + 1] * as[as_off + 1] + x[ja[ja_off] + 2] * as[as_off + 2];
		sdata[tid_1] += x[ja[ja_off]] * as[as_off + 3] + x[ja[ja_off] + 1] * as[as_off + 4] + x[ja[ja_off] + 2] * as[as_off + 5];
		sdata[tid_2] += x[ja[ja_off]] * as[as_off + 6] + x[ja[ja_off] + 1] * as[as_off + 7] + x[ja[ja_off] + 2] * as[as_off + 8];
		as_off += 9;
	}
	__syncthreads();
	if (BlockSize >= 512) {
		if (tid < 256)
			SUM_POSITIONS_3(256)
		__syncthreads();
	}
	if (BlockSize >= 256) {
		if (tid < 128)
			SUM_POSITIONS_3(128)
		__syncthreads();
	}
	if (BlockSize >= 128) {
		if (tid < 64)
			SUM_POSITIONS_3(64)
		__syncthreads();
	}
	if (tid < 32)
		warpReduce_3x3<BlockSize>(sdata, tid_0, tid_1, tid_2);

	if (tid == 0) {
		y[blockIdx.x * 3 + 0] = sdata[0];
		y[blockIdx.x * 3 + 1] = sdata[1];
		y[blockIdx.x * 3 + 2] = sdata[2];
	}
}

template<unsigned int BlockSize>
__global__ void device_cuda_ellpack_matrixvector_simple_mxn(double* as, unsigned int* ja, double* x, double* y, unsigned int max_n_blocks, unsigned int beh, unsigned int bew) {
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int tid_0=tid*beh;
	unsigned int bid = blockIdx.x;
	unsigned int bes = beh * bew;
	unsigned int as_off = (bid * max_n_blocks + tid) * bes;
	unsigned int ja_end = (bid + 1) * max_n_blocks;
	unsigned int ja_off;
	for (ja_off = 0; ja_off < beh; ++ja_off)
		sdata[tid_0 + ja_off] = 0;
	unsigned int i, j, a;
	for (ja_off = bid * max_n_blocks + tid; ja_off < ja_end; ja_off += BlockSize) {
		a = 0;
		for (i = 0; i < beh; ++i) {
			for (j = 0; j < bew; ++j) {
				sdata[tid_0 + i] += x[ja[ja_off] + j] * as[as_off + a];
				a++;
			}
		}
		as_off += bes*BlockSize;
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
		warpReduce_mxn<BlockSize>(sdata, tid_0, beh);

	if (tid == 0) {
		for (i = 0; i < beh; i++)
			y[blockIdx.x * beh + i] = sdata[i];
	}

}

__host__ void cuda_ellpack_matrixvector(block_ell &matrix, double* x, double* y) {
	unsigned int* d_ja;
	double* d_as;
	double* d_x;
	double* d_y;

	cudaMalloc((void**) &d_ja, matrix.getSizeJa() * sizeof(unsigned int));
	cudaMalloc((void**) &d_as, matrix.getSizeAs() * sizeof(double));

	cudaMalloc((void**) &d_y, (matrix.getRows() + matrix.getBlockHeight() - matrix.getRows() % matrix.getBlockHeight()) * sizeof(double));
	cudaMalloc((void**) &d_x, matrix.getCols() * sizeof(double));

	cudaMemcpy(d_ja, matrix.getCpuJa(), matrix.getSizeJa() * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_as, matrix.getCpuAs(), matrix.getSizeAs() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, matrix.getCols() * sizeof(double), cudaMemcpyHostToDevice);
	//device_cuda_ellpack_matrixvector_simple_3x3<BLOCK_SIZE_X> <<<matrix.getBlockRows(), BLOCK_SIZE_X>>>(d_as, d_ja, d_x, d_y, matrix.getMaxBlocks());
	printf("calling with<%d> <<<%d,%d,%xSize>>> (a,j,x,y,%d,%d,%d)\n",BLOCK_SIZE_X,matrix.getBlockRows(), BLOCK_SIZE_X, BLOCK_SIZE_X * matrix.getBlockHeight(),matrix.getMaxBlocks(), matrix.getBlockHeight(), matrix.getBlockWidth());
	device_cuda_ellpack_matrixvector_simple_mxn<BLOCK_SIZE_X> <<<matrix.getBlockRows(), BLOCK_SIZE_X, BLOCK_SIZE_X * matrix.getBlockHeight() * sizeof(double)>>>(d_as, d_ja, d_x, d_y,
		matrix.getMaxBlocks(), matrix.getBlockHeight(), matrix.getBlockWidth());

	cudaMemcpy(y, d_y, matrix.getRows() * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_ja);
	cudaFree(d_as);
	cudaFree(d_y);
	cudaFree(d_x);

	for (int var = 0; var < matrix.getRows(); ++var) {
		printf("%f,", y[var]);
	}
}
