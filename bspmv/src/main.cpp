#include <stdio.h>
#include <stdlib.h>
#include "matrixFormats/sparsematrix.h"
#include "matrixFormats/blockell.h"
#include "cuda/ell.cuh"
#include "matrixFormats/blockcsr.h"
#include "cuda/csr.cuh"

int main(int argc, char **argv) {
	coo_sparse_matrix coo("matrices/test.mtx");
	coo_sparse_matrix* c = coo.to_coo();
	block_ell ell(coo);
	block_csr csr(coo);
	double* x1 = (double*) malloc(sizeof(double) * (ell.getCols()));
	double* x2 = (double*) malloc(sizeof(double) * (ell.getCols()));
	for (int i = 0; i < ell.getCols(); ++i) {
		x1[i] = 1;
		x2[i] = 1;
	}
	double* y1 = (double*) malloc(sizeof(double) * ell.getRows());
	double* y2 = (double*) malloc(sizeof(double) * ell.getRows());
	cuda_ellpack_matrixvector(ell, x1, y1);
	cuda_csr_matrixvector(csr, x2, y2);
	free(x1);
	free(x2);
	free(y1);
	free(y2);

}
