/*
 * csr.cuh
 *
 *  Created on: 22/05/2017
 *      Author: leonardo
 */

#ifndef CSR_CUH_
#define CSR_CUH_
#include <cuda_runtime.h>


__host__ double cuda_csr_matrixvector(
		unsigned int *h_irp, 	unsigned int irp_size,
		unsigned int* h_ja, unsigned int ja_size,
		double* h_as, unsigned int as_size,
		unsigned int cols, unsigned int rows,
		unsigned int beh, unsigned int bew,
		unsigned int blockRows,
		double* h_x,
		double* h_y,
		unsigned int blockSize);

#endif /* CSR_CUH_ */
