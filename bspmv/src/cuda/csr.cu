#include "csr.cuh"
#define SUM_POSITIONS_H(beh,offset) \
	{\
		for(i=0;i<beh;i++){\
			sdata[tid_0+i]+=sdata[tid_0+i+offset*beh];\
		}\
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
__global__ void device_cuda_csr_matrixvector_simple_mxn(double* as, unsigned int *irp, unsigned int* ja, double* x, double* y, unsigned int beh, unsigned int bew) {

	extern __shared__ double sdata[];
	unsigned int i, j, irp_off;
	unsigned int bes=beh*bew;
	unsigned int bid = blockIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int tid_0 = threadIdx.x * beh;
	for (i = 0; i < beh; ++i) {
		sdata[tid_0 + i] = 0;
	}
	unsigned a;
	unsigned int as_off;
	for (irp_off = irp[bid] + tid; irp_off < irp[bid + 1]; irp_off +=BlockSize) {
		a=0;
		as_off = (irp[bid] + tid) * bes;
		printf("\nirp_off:%d\tmax=%d\n" ,irp[bid] + tid, irp[bid+1]);

		for (i = 0; i < beh; ++i) {
			for (j = 0; j < bew; ++j) {
				sdata[tid_0 + i] += x[ja[irp_off]+j]*as[as_off+a++];
			}
		}
		as+=BlockSize*bes;
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

__host__ void cuda_csr_matrixvector(block_csr &matrix, double* x, double* y) {
	unsigned int* d_irp;
	unsigned int* d_ja;
	double* d_as;
	double* d_x;
	double* d_y;

	cudaMalloc((void**) &d_irp, matrix.getSizeIrp() * sizeof(unsigned int));
	cudaMalloc((void**) &d_ja, matrix.getSizeJa() * sizeof(unsigned int));
	cudaMalloc((void**) &d_as, matrix.getSizeAs() * sizeof(double));

	cudaMalloc((void**) &d_y, (matrix.getRows() + matrix.getBlockHeight() - matrix.getRows() % matrix.getBlockHeight()) * sizeof(double));
	cudaMalloc((void**) &d_x, matrix.getCols() * sizeof(double));

	cudaMemcpy(d_irp, matrix.getCpuIrp(), matrix.getSizeIrp() * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ja, matrix.getCpuJa(), matrix.getSizeJa() * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_as, matrix.getCpuAs(), matrix.getSizeAs() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, matrix.getCols() * sizeof(double), cudaMemcpyHostToDevice);
	//device_cuda_ellpack_matrixvector_simple_3x3<BLOCK_SIZE_X> <<<matrix.getBlockRows(), BLOCK_SIZE_X>>>(d_as, d_ja, d_x, d_y, matrix.getMaxBlocks());
	//printf("calling with<%d> <<<%d,%d,%xSize>>> (a,j,x,y,%d,%d,%d)\n",BLOCK_SIZE_X,matrix.getBlockRows(), BLOCK_SIZE_X, BLOCK_SIZE_X * matrix.getBlockHeight(),matrix.getMaxBlocks(), matrix.getBlockHeight(), matrix.getBlockWidth());

	device_cuda_csr_matrixvector_simple_mxn<BLOCK_SIZE_X> <<<matrix.getBlockRows(), BLOCK_SIZE_X, BLOCK_SIZE_X * matrix.getBlockHeight() * sizeof(double)>>>(d_as,d_irp,d_ja,d_x,d_y,matrix.getBlockHeight(),matrix.getBlockWidth());
	cudaMemcpy(y, d_y, matrix.getRows() * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_ja);
	cudaFree(d_as);
	cudaFree(d_irp);
	cudaFree(d_y);
	cudaFree(d_x);
	for (int var = 0; var < matrix.getRows(); ++var) {
		printf("%f,", y[var]);
	}
}

