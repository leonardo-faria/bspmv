/*
 * globals.h
 *
 *  Created on: 17/05/2017
 *      Author: leonardo
 */

#ifndef GLOBALS_H_
#define GLOBALS_H_

#include <cstdlib>

#define BLOCK_SIZE_X 2
#define BLOCK_SIZE_Y 1
#define BLOCK_SIZE_Z 1
#define GRID_SIZE_X 1
#define GRID_SIZE_Y 1
#define GRID_SIZE_Z 1

const dim3 BLOCK_DIM(BLOCK_SIZE_X,BLOCK_SIZE_Y,BLOCK_SIZE_Z);
const dim3 GRID_DIM( GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);


#define BLOCK_ENTRY_H 3
#define BLOCK_ENTRY_W 3
#define BLOCK_ENTRY_S BLOCK_ENTRY_H*BLOCK_ENTRY_W


#endif /* GLOBALS_H_ */
