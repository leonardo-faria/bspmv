/*
 * blockell.cpp
 *
 *  Created on: 05/05/2017
 *      Author: leonardo
 */

#include <blockell.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <limits.h>

unsigned int get_bloc_index(unsigned int* ja_row, unsigned int row_size, unsigned int block_width, unsigned int coo_ja) {
	for (unsigned int i = 0; i < row_size; ++i) {
		if (ja_row[i] + block_width > coo_ja && ja_row[i] <= coo_ja)
			return i;
	}
	printf("ERROR with construction.");
	exit(1);
}

void v_redux_ell(std::vector<unsigned int> &v, unsigned int block_w) {
	unsigned int ja = v[0];
	for (unsigned int i = 1; i < v.size(); ++i) {
		if (v[i] < ja + block_w) {
			v.erase(v.begin() + i);
			i--;
		} else
			ja = v[i];
	}
}
block_ell::block_ell(sparse_matrix & s) {

	coo_sparse_matrix* coo = s.to_coo();
	if (!coo) {
		printf("Invalid argument (to_coo not implemented?).\n");
		exit(1);
	}
	max_blocks = 0;
	unsigned int b_row = 0;

	unsigned int n_blocks_rows = coo->getRows() / block_height;
	if (coo->getRows() % block_height != 0)
		n_blocks_rows++;
	std::vector<std::vector<unsigned int> > temp_ja;
	for (unsigned int i = 0; i < n_blocks_rows; ++i) {
		temp_ja.push_back(std::vector<unsigned int>());
	}
	for (unsigned int i = 0; i < coo->getNonz(); ++i) {
		if (b_row == coo->getCpuIrp()[i] / block_height) {
			temp_ja[b_row].push_back(coo->getCpuJa()[i]);
		} else {
			std::sort(temp_ja[b_row].begin(), temp_ja[b_row].end());
			v_redux_ell(temp_ja[b_row], block_width);
			if (max_blocks < temp_ja[b_row].size())
				max_blocks = temp_ja[b_row].size();
			b_row++;
			--i;
		}
	}

	std::sort(temp_ja[b_row].begin(), temp_ja[b_row].end());
	v_redux_ell(temp_ja[b_row], block_width);
	if (max_blocks < temp_ja[b_row].size())
		max_blocks = temp_ja[b_row].size();

	size_ja = max_blocks * n_blocks_rows;
	size_as = max_blocks * n_blocks_rows * block_height * block_width;
	std::cout << "n_blocks_rows " << n_blocks_rows << std::endl;
	std::cout << "max_blocks " << max_blocks << std::endl;
	std::cout << "size_ja " << max_blocks << "*" << n_blocks_rows << "=" << size_ja << std::endl;
	std::cout << "size_as " << max_blocks << "*" << n_blocks_rows << "*" << block_height << "*" << block_width << "=" << size_as << std::endl;
	std::cout << "MAX INT: " << INT_MAX << std::endl;
	cpu_ja = (unsigned int*) calloc(size_ja, sizeof(unsigned int));
	cpu_as = (double*) calloc(size_as, sizeof(double));
//
	std::cout << "asd1" << std::endl;
	for (unsigned int i = 0; i < n_blocks_rows; ++i) {
		std::copy(temp_ja[i].begin(), temp_ja[i].end(), cpu_ja + i * max_blocks);
//		for (int j = temp_ja[i].size(); j < max_blocks; ++j) {
//			cpu_ja[i*max_blocks+j]=cpu_ja[i*max_blocks];
//		}
	}
	std::cout << "nonzeros:" << coo->getNonz() << std::endl;
	for (unsigned int i = 0; i < coo->getNonz(); ++i) {
		unsigned int block_h = coo->getCpuIrp()[i] / block_height;
		unsigned int block_index = get_bloc_index(cpu_ja + block_h * max_blocks, max_blocks, block_width, coo->getCpuJa()[i]);
		cpu_as[block_h * block_size * max_blocks + block_index * block_size + coo->getCpuJa()[i] - cpu_ja[block_index + max_blocks * block_h]
				+ (coo->getCpuIrp()[i] - (block_h * block_height)) * block_width] = coo->getCpuAs()[i];
	}
	for (int i = 0; i < size_ja; ++i) {
		printf("%d-%d\n", i / max_blocks, cpu_ja[i]);
		for (int k = 0; k < block_height; ++k) {
			for (int j = 0; j < block_width; ++j) {
				printf("%f\t", cpu_as[i * block_height * block_width + k * block_width + j]);
			}
			printf("\n");
		}
	}
	std::cout << "END" << std::endl;
}

block_ell::~block_ell() {
	free(cpu_as);
	free(cpu_ja);
// TODO Auto-generated destructor stub
}

