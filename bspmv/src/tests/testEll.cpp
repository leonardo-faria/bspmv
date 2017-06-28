#include <stdio.h>
#include <cstdlib>

#include "cpu_coo.h"
#include "../matrixFormats/blockell.h"
#include "../cuda/ell.cuh"
void testEllforMatrix(char* filename,unsigned int blockSize) {
	coo_sparse_matrix coo(filename);
	block_ell ell(coo,5,1);
	double* xcoo = (double*) malloc(sizeof(double) * coo.getCols());
	double* xell = (double*) malloc(sizeof(double) * coo.getCols());
	double* ycoo = (double*) malloc(sizeof(double) * coo.getRows());
	double* yell = (double*) malloc(sizeof(double) * coo.getRows());

	for (int i = 0; i < coo.getCols(); ++i) {
		xcoo[i] = 1;
		xell[i] = 1;
	}

	matrixvector(coo, xcoo, ycoo);


	cuda_ellpack_matrixvector(ell.getCpuJa(),ell.getSizeJa(),ell.getCpuAs(),ell.getSizeAs(),ell.getCols(),ell.getRows(),ell.getBlockHeight(),ell.getBlockWidth(),ell.getBlockRows(),ell.getMaxBlocks(), xell, yell,256);


	int good = 0;
	for (int i = 0; i < coo.getRows(); ++i) {
		if (yell[i] != ycoo[i]) {
			printf("ell %d:coo %f\tell %f\n", i, ycoo[i], yell[i]);
		} else
			good++;
	}


	printf("good ell-coo:%d of %d\n", good, coo.getRows());

	free(xcoo);
	free(xell);
	free(ycoo);
	free(yell);
}

