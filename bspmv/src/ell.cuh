/*
 * ell.cuh
 *
 *  Created on: 16/05/2017
 *      Author: leonardo
 */

#ifndef ELL_CUH_
#define ELL_CUH_

#define BLOCK_SIZE_X 1
#define BLOCK_SIZE_Y 1
#define BLOCK_SIZE_Z 1
#define GRID_SIZE_X 1
#define GRID_SIZE_Y 1
#define GRID_SIZE_Z 1

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>
#include "blockell.h"

__host__ void cuda_ellpack_matrixvector(block_ell &matrix, double* x, double* y);

const dim3 BLOCK_DIM(BLOCK_SIZE_X,BLOCK_SIZE_Y,BLOCK_SIZE_Z);
const dim3 GRID_DIM( GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);

#endif /* ELL_CUH_ */
