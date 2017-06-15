#include <stdio.h>
#include <cstdlib>

#include "benchmarkBlock.h"
#include "../matrixFormats/blockcsr.h"
#include "../matrixFormats/blockell.h"
#include "../cuda/csr.cuh"
#include "../cuda/ell.cuh"

double bcsr_spmv(coo_sparse_matrix *coo,unsigned int blockSize) {
	double flops;
	block_csr csr(*coo);
	double* x = (double*) malloc(sizeof(double) * (csr.getCols()));
	double* y = (double*) malloc(sizeof(double) * (csr.getRows()));

	flops = 2.0 * (double) csr.getNonz();
	flops /= 	cuda_csr_matrixvector(csr.getCpuIrp(),csr.getSizeIrp(), csr.getCpuJa(),csr.getSizeJa(), csr.getCpuAs(),csr.getSizeAs(),csr.getCols(),csr.getRows(),csr.getBlockHeight(),csr.getBlockWidth(),csr.getBlockRows(), x, y, blockSize);


	free(x);
	free(y);
return flops;
}

double bell_spmv(coo_sparse_matrix *coo ,unsigned int blockSize) {
	double flops;
	block_ell ell(*coo);
	double* x = (double*) malloc(sizeof(double) * (ell.getCols()));
	double* y = (double*) malloc(sizeof(double) * (ell.getRows()));

	flops = 2.0 * (double) ell.getNonz();
	flops /= cuda_ellpack_matrixvector(ell.getCpuJa(),ell.getSizeJa(),ell.getCpuAs(),ell.getSizeAs(),ell.getCols(),ell.getRows(),ell.getBlockHeight(),ell.getBlockWidth(),ell.getBlockRows(),ell.getMaxBlocks(), x, y,blockSize);


	free(x);
	free(y);

	return flops;
}
void b_spmv(char* filename,unsigned int blockSize){
	coo_sparse_matrix *coo = new coo_sparse_matrix(filename);

	printf("CSR-FLOPS for %s:\n%f\n\n", filename, bcsr_spmv(coo,blockSize));
	printf("ELL-FLOPS for %s:\n%f\n\n", filename, bell_spmv(coo,blockSize));

	delete coo;
}
