/*
 * sparsematrix.h
 *
 *  Created on: 30/04/2017
 *      Author: leonardo
 */

#ifndef SPARSEMATRIX_H_
#define SPARSEMATRIX_H_
class coo_sparse_matrix;

class sparse_matrix {
protected:
	int nonzeros;
	int rows;
	int collumns;

public:
	sparse_matrix();
	virtual ~sparse_matrix() {
	}
	;
	virtual coo_sparse_matrix* to_coo()=0;
	virtual void from_coo(coo_sparse_matrix coo)=0;
	virtual void transp() {
		int temp;
		temp = rows;
		rows = collumns;
		collumns = temp;
	}
	;

	int getCols() const {
		return collumns;
	}

	int getNonz() const {
		return nonzeros;
	}

	int getRows() const {
		return rows;
	}


};

class coo_sparse_matrix: public sparse_matrix {
private:
	int* work_irp;
	int* work_ja;
	double* work_as;

	int* cpu_irp;
	int* cpu_ja;
	double* cpu_as;


	void BottomUpMerge(int iLeft, int iRight, int iEnd);
	void BottomUpMergeSort(int n);
	void CopyArray(int n);
public:
	virtual coo_sparse_matrix* to_coo();
	virtual void from_coo(coo_sparse_matrix coo);
	coo_sparse_matrix(char* filename);
	~coo_sparse_matrix();

	void parse_matrix(char* filename);

	double* getCpuAs() const {
		return cpu_as;
	}

	int* getCpuIrp() const {
		return cpu_irp;
	}

	int* getCpuJa() const {
		return cpu_ja;
	}
};

#endif /* SPARSEMATRIX_H_ */
