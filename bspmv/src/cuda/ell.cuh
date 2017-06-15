/*
 * ell.cuh
 *
 *  Created on: 16/05/2017
 *      Author: leonardo
 */

#ifndef ELL_CUH_
#define ELL_CUH_
#include <cuda_runtime.h>

__host__ double cuda_ellpack_matrixvector(unsigned int* h_ja, unsigned int ja_size, double* h_as, unsigned int as_size, unsigned int cols, unsigned int rows, unsigned int beh, unsigned int bew, unsigned int blockRows, unsigned int max_n_blocks, double* h_x, double* h_y, unsigned int blockSize);

#endif /* ELL_CUH_ */
