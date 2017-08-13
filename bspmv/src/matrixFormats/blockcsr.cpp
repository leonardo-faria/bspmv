/*
 * blockcsr.cpp
 *
 *  Created on: 05/05/2017
 *      Author: leonardo
 */

#include "blockcsr.h"
#include <vector>
#include <algorithm>
#include <stdio.h>

#include <iostream>

unsigned int get_csr_block_index(unsigned int* ja_row, unsigned int row_size, unsigned int block_width, unsigned int coo_ja) {
	for (unsigned int i = 0; i < row_size; ++i) {
		if (ja_row[i] + block_width > coo_ja && ja_row[i] <= coo_ja)
			return i;
	}
	printf("ERROR with construction.");
	std::exit(1);
}

void v_redux_csr(std::vector<unsigned int> &v, unsigned int block_w) {
	unsigned int ja = v[0];
	for (unsigned int i = 1; i < v.size(); ++i) {
		if (v[i] < ja + block_w) {
			v.erase(v.begin() + i);
			i--;
		} else
			ja = v[i];
	}
}
block_csr::block_csr(sparse_matrix &s, unsigned int beh, unsigned int bew) {
	collumns = s.getCols();
	rows = s.getRows();
	nonzeros = s.getNonz();
	block_width = bew;
	block_height = beh;
	block_size = block_width * block_height;
	coo_sparse_matrix* coo = s.to_coo();
	block_rows = coo->getRows() / block_height;
	if (coo->getRows() % block_height != 0)
		block_rows++;
	size_irp = block_rows + 1;
	cpu_irp = (unsigned int*) malloc(sizeof(unsigned int) * size_irp);
	cpu_irp[0] = 0;
	std::vector<std::vector<unsigned int> > temp_ja;
	unsigned int b_row = 0;
	for (unsigned int i = 0; i < block_rows; ++i) {
		temp_ja.push_back(std::vector<unsigned int>());
	}
	for (unsigned int i = 0; i < coo->getNonz(); ++i) {
		if (b_row == coo->getCpuIrp()[i] / block_height) {
			temp_ja[b_row].push_back(coo->getCpuJa()[i]);
		} else {
			std::sort(temp_ja[b_row].begin(), temp_ja[b_row].end());
			v_redux_csr(temp_ja[b_row], block_width);
			cpu_irp[b_row + 1] = cpu_irp[b_row] + temp_ja[b_row].size();
			b_row++;
			--i;
		}
	}
	std::sort(temp_ja[b_row].begin(), temp_ja[b_row].end());
	v_redux_csr(temp_ja[b_row], block_width);
	cpu_irp[b_row + 1] = cpu_irp[b_row] + temp_ja[b_row].size();
	size_ja = cpu_irp[b_row + 1];
	size_as = size_ja * block_size;
	cpu_ja = (unsigned int*) calloc(size_ja, sizeof(unsigned int));
	cpu_as = (double*) calloc(size_as, sizeof(double));
//	cpu_ja = (unsigned int*) malloc(sizeof(unsigned int) * size_ja);
//	cpu_as = (double*) malloc(sizeof(double) * size_as);
	unsigned int ja_index = 0;
	for (int i = 0; i < temp_ja.size(); ++i) {
		std::copy(temp_ja[i].begin(), temp_ja[i].end(), cpu_ja + ja_index);
		ja_index += temp_ja[i].size();
	}

	for (unsigned int i = 0; i < coo->getNonz(); ++i) {
		unsigned int block_h = coo->getCpuIrp()[i] / block_height;
		unsigned int block_index = get_csr_block_index(cpu_ja + cpu_irp[block_h], cpu_irp[block_h + 1], block_width, coo->getCpuJa()[i]);
		cpu_as[(cpu_irp[block_h] + block_index) * block_size + coo->getCpuJa()[i] - cpu_ja[cpu_irp[block_h] + block_index] + (coo->getCpuIrp()[i] - (block_h * block_height)) * block_width] = coo->getCpuAs()[i];
	}
	printf("Size: %dx%d\t\tas size:%d\n",beh,bew,size_as);
//	for (int ir = 0; ir < size_ja; ++ir) {
//		printf("ja:%d\n",cpu_ja[ir]);
//		for (int i = 0; i < beh; ++i) {
//			for (int j = 0; j < bew; ++j) {
//				printf("%f\t", cpu_as[i * bew + j + ir * beh * bew]);
//			}
//			printf("\n");
//		}
//		printf("\n");
//		printf("\n");
//	}
//	for (int i = 0; i < 6; ++i) {
//		for (int j = 0; j < 6; ++j) {
//			printf("%f\t",cpu_as[i*6+j]);
//		}
//		printf("\n");
//	}
//	printf("as size:%d\n",size_ja*beh*bew);
//	printf("nonz:%d\n",nonzeros);
//	printf("csr fill in ratio: %f:\n",((double) coo->getNonz())/((double)(size_ja*block_size)));

}

block_csr::~block_csr() {
	free(cpu_as);
	free(cpu_irp);
	free(cpu_ja);
	// TODO Auto-generated destructor stub
}

