#include "ell.cuh"

template<unsigned int blockSize, unsigned int blockEntryHeight>
__global__ void device_cuda_ellpack_matrixvector_simple(double* as, unsigned int* ja, double* x, unsigned int bew, unsigned int max_n_blocks) {

	__shared__ double sdata[blockSize * blockEntryHeight];
	unsigned int tid = threadIdx.x;
	unsigned int bid = threadIdx.x;

	int as_off = (bid * max_n_blocks +tid )* blockSize;
	int ja_end=(bid+1) * max_n_blocks;
	for (int ja_off = bid * max_n_blocks+tid; ja_off < max_n_blocks; ja_off += blockSize) {
		for (int i = 0; i < blockSize; ++i) {
			sdata[i / bew] += x[ja_off + i % bew] * as[as_off + i];
		}
		as_off+=blockSize;
	}
	/*
	 in
	 */
}

__host__ void cuda_ellpack_matrixvector(block_ell matrix, double* x, double* y) {
	unsigned int* d_ja;
	double* d_as;
	double* d_x;
	double* d_y;

	cudaMalloc((void**) &d_ja, matrix.getSizeJa() * sizeof(unsigned int));
	cudaMalloc((void**) &d_as, matrix.getSizeAs() * sizeof(double));

	cudaMalloc((void**) &d_y, matrix.getRows() * sizeof(double));
	cudaMalloc((void**) &d_x, matrix.getCols() * sizeof(double));

	cudaMemcpy(d_ja, matrix.getCpuJa(), matrix.getSizeJa() * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_as, matrix.getCpuAs(), matrix.getSizeAs() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, matrix.getCols() * sizeof(double), cudaMemcpyHostToDevice);
	device_cuda_ellpack_matrixvector_simple<BLOCK_SIZE_X, BLOCK_ENTRY_H> <<<GRID_DIM, BLOCK_DIM>>>(d_as, d_ja, d_x, matrix.getBlockWidth(), matrix.getMaxBlocks());

	cudaMemcpy(y, d_y, matrix.getRows() * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_ja);
	cudaFree(d_as);
	cudaFree(d_y);
	cudaFree(d_x);

}
