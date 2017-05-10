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


GlobalConstants *calcGlobalConsts(card_t *board, card_t *oopRange,
        int oopSize, card_t*ipRange, int ipSize, int potSize, int betSize);

void calcMaxStrategy(char *bestStrat, int *stratVal, GlobalConstants *params);
