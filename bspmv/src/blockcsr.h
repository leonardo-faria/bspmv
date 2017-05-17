/*
 * blockcsr.h
 *
 *  Created on: 05/05/2017
 *      Author: leonardo
 */
#include "sparsematrix.h"
#ifndef BLOCKCSR_H_
#define BLOCKCSR_H_

class block_csr : public sparse_matrix {
	unsigned int block_width=BLOCK_ENTRY_H;
	unsigned int block_height=BLOCK_ENTRY_W;
	unsigned int block_size=BLOCK_ENTRY_W*BLOCK_ENTRY_H;
	unsigned int size_irp;
	unsigned int size_ja;
	unsigned int size_as;
	unsigned int* cpu_irp;
	unsigned int* cpu_ja;
	double* cpu_as;
public:
	block_csr(sparse_matrix &s);
	virtual ~block_csr();
	virtual coo_sparse_matrix* to_coo(){return 0;};
	virtual void from_coo(coo_sparse_matrix coo){};
};

#endif /* BLOCKCSR_H_ */
