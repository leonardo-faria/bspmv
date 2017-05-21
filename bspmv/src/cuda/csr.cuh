/*
 * csr.cuh
 *
 *  Created on: 22/05/2017
 *      Author: leonardo
 */

#ifndef CSR_CUH_
#define CSR_CUH_

#include "../matrixFormats/blockcsr.h"

__host__ void cuda_csr_matrixvector(block_csr &matrix, double* x, double* y);



#endif /* CSR_CUH_ */
