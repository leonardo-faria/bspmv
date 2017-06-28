/*
 * benchmarkBlock.h
 *
 *  Created on: 05/06/2017
 *      Author: leonardo
 */

#ifndef BENCHMARKBLOCK_H_
#define BENCHMARKBLOCK_H_
void b_spmv(char* filename,unsigned int blockSize);
void b_spmv_csr(char* filename, unsigned int blockSize) ;
void b_spmv_ell(char* filename, unsigned int blockSize);
void b_spmv(char* filename);
void b_spmv_csr(char* filename) ;
void b_spmv_ell(char* filename);

void b_spmv_ell_ratio(char* filename);
void b_spmv_csr_ratio(char* filename);

#endif /* BENCHMARKBLOCK_H_ */
