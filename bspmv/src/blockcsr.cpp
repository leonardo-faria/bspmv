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
	exit(1);
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
block_csr::block_csr(sparse_matrix &s) {
	block_width = BLOCK_ENTRY_H;
	block_height = BLOCK_ENTRY_W;
	block_size = BLOCK_ENTRY_W * BLOCK_ENTRY_H;
	coo_sparse_matrix* coo = s.to_coo();
	unsigned int n_blocks_rows = coo->getRows() / block_height;
	if (coo->getRows() % block_height != 0)
		n_blocks_rows++;
	size_irp = n_blocks_rows + 1;
	cpu_irp = (unsigned int*) malloc(sizeof(unsigned int) * size_irp);
	cpu_irp[0] = 0;
	std::vector<std::vector<unsigned int> > temp_ja;
	unsigned int b_row = 0;
	for (unsigned int i = 0; i < n_blocks_rows; ++i) {
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
	cpu_ja = (unsigned int*) malloc(sizeof(unsigned int) * size_ja);
	size_as = size_ja * block_size;
	cpu_as = (double*) malloc(sizeof(double) * size_as);
	unsigned int ja_index = 0;
	for (int i = 0; i < temp_ja.size(); ++i) {
		std::copy(temp_ja[i].begin(), temp_ja[i].end(), cpu_ja + ja_index);
		ja_index += temp_ja[i].size();
	}

	for (unsigned int i = 0; i < coo->getNonz(); ++i) {
		unsigned int block_h = coo->getCpuIrp()[i] / block_height;
		unsigned int block_index = get_csr_block_index(cpu_ja + cpu_irp[block_h], cpu_irp[block_h + 1], block_width, coo->getCpuJa()[i]);
		cpu_as[(cpu_irp[block_h] + block_index) * block_size + coo->getCpuJa()[i] - cpu_ja[cpu_irp[block_h] + block_index] + (coo->getCpuIrp()[i] - (block_h * block_height)) * block_width] =
				coo->getCpuAs()[i];
	}
	std::cout << "END" << std::endl;
}

block_csr::~block_csr() {
	free(cpu_as);
	free(cpu_irp);
	free(cpu_ja);
	// TODO Auto-generated destructor stub
}

