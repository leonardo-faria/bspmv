#include <stdio.h>
#include "testCsr.h"
#include "../cuda/ell.cuh"
#include "cpu_coo.h"

void testEllforMatrix(char* filename) {
	coo_sparse_matrix coo(filename);
	block_ell ell(coo);
	double* xcoo = (double*) malloc(sizeof(double) * (coo.getCols()));
	double* xell = (double*) malloc(sizeof(double) * (coo.getCols()));
	for (int i = 0; i < coo.getCols(); ++i) {
		xcoo[i] = 1;
		xell[i] = 1;
	}
	double* ycoo = (double*) malloc(sizeof(double) * coo.getRows());
	double* yell = (double*) malloc(sizeof(double) * coo.getRows());
	matrixvector(coo, xcoo, ycoo);
	cuda_ellpack_matrixvector(ell, xell, yell);
	int good = 0;
	for (int i = 0; i < coo.getRows(); ++i) {
		if (yell[i] != ycoo[i]) {
//			printf("%d:coo %f\tcsr %f\n",i,ycoo[i],ycsr[i]);
		} else
			good++;
	}
	printf("good cudacpu:%d of %d\n", good, coo.getRows());
	free(xcoo);
	free(xell);
	free(ycoo);
	free(yell);
}

