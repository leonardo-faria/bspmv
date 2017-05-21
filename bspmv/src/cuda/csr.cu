#include "csr.cuh"


__host__ void cuda_csrpack_matrixvector(block_csr &matrix, double* x, double* y) {
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


	cudaMemcpy(y, d_y, matrix.getRows() * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_ja);
	cudaFree(d_as);
	cudaFree(d_y);
	cudaFree(d_x);

	for (int var = 0; var < matrix.getRows(); ++var) {
		printf("%f,", y[var]);
	}
}

