
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include "sparsematrix.h"
#include "blockell.h"
#include "blockcsr.h"
#include <cuda.h>
#include "ell.cuh"
int main(int argc, char **argv)
{
	coo_sparse_matrix coo("matrices/test.mtx");
	coo_sparse_matrix* c = coo.to_coo();
	block_ell b(coo);
	double* x=(double*) malloc(sizeof(double)*(b.getCols()));
	for (int i = 0; i < b.getCols(); ++i) {
		x[i]=1;
	}
	double* y=(double*) malloc(sizeof(double)*b.getRows());
	cuda_ellpack_matrixvector(b,x,y);
	free(x);
	free(y);


}
