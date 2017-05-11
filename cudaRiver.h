#ifndef _CUDA_RIVER_H
#define _CUDA_RIVER_H

#include "rank.h"

typedef struct GlobalConstants {
    //must be in cuda
    int *oopRanks;
    int oopSize;

    //must be in cuda
    int *ipRanks;
    int ipSize;
    int potSize;
    int betSize;
    int afterBetSize;
} GlobalConstants;

#define CHECK_CALL 0
#define CHECK_FOLD 1
#define BET 2
#define OOP_MOVES 3

#define IP_MOVES 4

#define NUM_STRATEGIES_PER_ITERATION 100

GlobalConstants *calcGlobalConsts(board_t board, hand_t *oopRange,
        int oopSize, hand_t *ipRange, int ipSize, int potSize, int betSize);

void calcMaxStrategy(char *bestStrat, int *stratVal, GlobalConstants *params);

#endif /* _CUDA_RIVER_H */
