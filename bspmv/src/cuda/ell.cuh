/*
 * ell.cuh
 *
 *  Created on: 16/05/2017
 *      Author: leonardo
 */

#ifndef ELL_CUH_
#define ELL_CUH_

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../matrixFormats/blockell.h"

__host__ void cuda_ellpack_matrixvector(block_ell &matrix, double* x, double* y);

#endif /* ELL_CUH_ */
