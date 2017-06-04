#include <stdio.h>
#include "testCsr.h"
#include "../cuda/csr.cuh"
#include "cpu_coo.h"

void testCsrforMatrix(char* filename) {
	coo_sparse_matrix coo(filename);
	block_csr csr(coo);
	double* xcoo = (double*) malloc(sizeof(double) * (coo.getCols()));
	double* xcsr = (double*) malloc(sizeof(double) * (coo.getCols()));
	for (int i = 0; i < coo.getCols(); ++i) {
		xcoo[i] = 1;
		xcsr[i] = 1;
	}
	double* ycoo = (double*) malloc(sizeof(double) * coo.getRows());
	double* ycsr = (double*) malloc(sizeof(double) * coo.getRows());
	matrixvector(coo, xcoo, ycoo);
	cuda_csr_matrixvector(csr, xcsr, ycsr);
	int good = 0;
	for (int i = 0; i < coo.getRows(); ++i) {
		if (ycsr[i] != ycoo[i]) {
//			printf("%d:coo %f\tcsr %f\n",i,ycoo[i],ycsr[i]);
		} else
			good++;
	}
	printf("good cudacpu:%d of %d\n", good, coo.getRows());
	free(xcoo);
	free(xcsr);
	free(ycoo);
	free(ycsr);
}

