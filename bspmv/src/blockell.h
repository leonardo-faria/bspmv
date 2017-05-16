/*
 * blockell.h
 *
 *  Created on: 05/05/2017
 *      Author: leonardo
 */

#include "sparsematrix.h"
#ifndef BLOCKELL_H_
#define BLOCKELL_H_

class block_ell : public sparse_matrix {
	unsigned int block_width=3;
	unsigned int block_height=3;
	unsigned int block_size=block_width*block_height;
	unsigned int max_blocks;
	unsigned int size_ja;
	unsigned int size_as;
	unsigned int * cpu_ja;
	double* cpu_as;
public:
	block_ell(sparse_matrix &s);
	virtual ~block_ell();
	virtual coo_sparse_matrix* to_coo(){return 0;};
	virtual void from_coo(coo_sparse_matrix coo){};
};

#endif /* BLOCKELL_H_ */
