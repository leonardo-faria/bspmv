#include <stdio.h>
#include "testCsr.h"
#include "../cuda/csr.cuh"
#include "cpu_coo.h"
#include "../matrixFormats/blockcsr.h"
#include <cstdlib>
void testCsrforMatrix(char* filename, unsigned int blockSize, unsigned int beh, unsigned int bew) {
	coo_sparse_matrix coo(filename);
	block_csr csr(coo, beh, bew);
	double* xcoo = (double*) malloc(sizeof(double) * (csr.getCols()));
	double* xcsr = (double*) malloc(sizeof(double) * (csr.getCols()));
	for (int i = 0; i < csr.getCols(); ++i) {
		xcoo[i] = 1;
		xcsr[i] = 1;
	}
	double* ycoo = (double*) malloc(sizeof(double) * csr.getRows());
	double* ycsr = (double*) malloc(sizeof(double) * csr.getRows());
	matrixvector(coo, xcoo, ycoo);
	cuda_csr_matrixvector(csr.getCpuIrp(), csr.getSizeIrp(), csr.getCpuJa(), csr.getSizeJa(), csr.getCpuAs(), csr.getSizeAs(), csr.getCols(), csr.getRows(), csr.getBlockHeight(), csr.getBlockWidth(), csr.getBlockRows(), xcsr, ycsr, blockSize);
	int good = 0;
	for (int i = 0; i < csr.getRows(); ++i) {
		if (ycsr[i] == ycoo[i])
			good++;
	}
	printf("good cudacpu in %dx%d:%d of %d\n", beh, bew, good, csr.getRows());
	free(xcoo);
	free(xcsr);
	free(ycoo);
	free(ycsr);
}
void testCsrforMatrix(coo_sparse_matrix *coo, unsigned int blockSize, unsigned int beh, unsigned int bew) {
	block_csr csr(*coo, beh, bew);
	double* xcoo = (double*) malloc(sizeof(double) * (csr.getCols()));
	double* xcsr = (double*) malloc(sizeof(double) * (csr.getCols()));
	for (int i = 0; i < csr.getCols(); ++i) {
		xcoo[i] = 1;
		xcsr[i] = 1;
	}
	double* ycoo = (double*) malloc(sizeof(double) * csr.getRows());
	double* ycsr = (double*) malloc(sizeof(double) * csr.getRows());

	for (int i = 0; i < csr.getRows(); ++i) {
		ycoo[i] = 0;
		ycsr[i] = 0;
	}
	matrixvector(*coo, xcoo, ycoo);
	cuda_csr_matrixvector(csr.getCpuIrp(), csr.getSizeIrp(), csr.getCpuJa(), csr.getSizeJa(), csr.getCpuAs(), csr.getSizeAs(), csr.getCols(), csr.getRows(), csr.getBlockHeight(), csr.getBlockWidth(), csr.getBlockRows(), xcsr, ycsr, blockSize);
	int good = 0;
	for (int i = 0; i < csr.getRows(); ++i) {
		if (ycsr[i] == ycoo[i])
			good++;
		else
			printf("ell %d:coo %f\tell %f\n", i, ycoo[i], ycsr[i]);
	}
	printf("good cudacpu in %dx%d:%d of %d\n", beh, bew, good, csr.getRows());
	free(xcoo);
	free(xcsr);
	free(ycoo);
	free(ycsr);
	if(good<4)
		exit(1);
}

void testCsrforMatrix(char* filename, unsigned int blockSize) {
	coo_sparse_matrix* coo = new coo_sparse_matrix(filename);
	for (int i = 1; i < 7; ++i) {
		for (int j = 1; j < 7; ++j) {

			testCsrforMatrix(coo, blockSize, i, j);
		}
	}

	delete coo;

}

