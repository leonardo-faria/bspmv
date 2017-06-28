#include <stdio.h>
#include <cstdlib>

#include "benchmarkBlock.h"
#include "../matrixFormats/blockcsr.h"
#include "../matrixFormats/blockell.h"
#include "../cuda/csr.cuh"
#include "../cuda/ell.cuh"

double bcsr_spmv(coo_sparse_matrix *coo, unsigned int beh, unsigned int bew, unsigned int blockSize) {
	double flops;
	block_csr csr(*coo, beh, bew);
	double* x = (double*) malloc(sizeof(double) * (csr.getCols()));
	double* y = (double*) malloc(sizeof(double) * (csr.getRows()));

	flops = 2.0 * (double) csr.getNonz();
	flops /= cuda_csr_matrixvector(csr.getCpuIrp(), csr.getSizeIrp(), csr.getCpuJa(), csr.getSizeJa(), csr.getCpuAs(), csr.getSizeAs(), csr.getCols(), csr.getRows(), csr.getBlockHeight(), csr.getBlockWidth(), csr.getBlockRows(), x, y, blockSize);

	free(x);
	free(y);
	return flops;
}
double bell_spmv(coo_sparse_matrix *coo, unsigned int beh, unsigned int bew, unsigned int blockSize) {
	double flops;
	block_ell ell(*coo, beh, bew);
	double* x = (double*) malloc(sizeof(double) * (ell.getCols()));
	double* y = (double*) malloc(sizeof(double) * (ell.getRows()));

	flops = 2.0 * (double) ell.getNonz();
	flops /= cuda_ellpack_matrixvector(ell.getCpuJa(), ell.getSizeJa(), ell.getCpuAs(), ell.getSizeAs(), ell.getCols(), ell.getRows(), ell.getBlockHeight(), ell.getBlockWidth(), ell.getBlockRows(), ell.getMaxBlocks(), x, y, blockSize);

	free(x);
	free(y);

	return flops;
}

void b_spmv_csr(char* filename, unsigned int blockSize) {
	printf("CSR bs=%d(beh/bew),", blockSize);
	for (int i = 0; i < 6; ++i) {
		printf("%d,", i + 1);
	}
	printf("\n");
	coo_sparse_matrix *coo = new coo_sparse_matrix(filename);
	for (int beh = 1; beh <= 6; ++beh) {
		printf("%d,",beh);
		for (int bew = 1; bew <= 6; ++bew) {
			printf("%f,", bcsr_spmv(coo, beh, bew, blockSize));
		}
		printf("\n");
	}

	delete coo;
}

void b_spmv_ell(char* filename, unsigned int blockSize) {
	printf("ELL bs=%d(beh/bew),", blockSize);
	for (int i = 0; i < 6; ++i) {
		printf("%d,", i + 1);
	}
	printf("\n");
	coo_sparse_matrix *coo = new coo_sparse_matrix(filename);
	for (int beh = 1; beh <= 6; ++beh) {
		printf("%d,",beh);
		for (int bew = 1; bew <= 6; ++bew) {
			printf("%f,", bell_spmv(coo, beh, bew, blockSize));
		}
		printf("\n");
	}

	delete coo;
}

void b_spmv_ell_ratio(char* filename) {
	printf("Ratios,\n");
	for (int i = 0; i < 6; ++i) {
		printf("%d,", i + 1);
	}
	printf("\n");
	coo_sparse_matrix *coo = new coo_sparse_matrix(filename);
	for (int beh = 1; beh <= 6; ++beh) {
		printf("%d,",beh);
		for (int bew = 1; bew <= 6; ++bew) {
			block_ell ell(*coo, beh, bew);
			printf("%f,",((double) coo->getNonz())/((double)(ell.getSizeJa()*beh*bew)));
		}
		printf("\n");
	}

	delete coo;
}

void b_spmv_csr_ratio(char* filename) {
	printf("Ratios,\n");
	for (int i = 0; i < 6; ++i) {
		printf("%d,", i + 1);
	}
	printf("\n");
	coo_sparse_matrix *coo = new coo_sparse_matrix(filename);
	for (int beh = 1; beh <= 6; ++beh) {
		printf("%d,",beh);
		for (int bew = 1; bew <= 6; ++bew) {
			block_csr csr(*coo, beh, bew);
			printf("%f,",((double) coo->getNonz())/((double)(csr.getSizeJa()*beh*bew)));
		}
		printf("\n");
	}

	delete coo;
}
