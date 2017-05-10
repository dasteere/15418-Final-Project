#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cudaRiver.h"

int main() {
    int oopSize, ipSize, potSize, betSize;
    char rank1, suit1, rank2, suit2;
    scanf("%d %d %d %d\n", oopSize, ipSize, potSize, betSize);
    card_t *oopRange = (card_t *) malloc(2 * oopSize * sizeof(card_t));
    card_t *ipRange = (card_t *) malloc(2 * ipSize * sizeof(card_t));
    card_t board[HAND_SIZE];
    for (int i = 0; i < HAND_SIZE; i++) {
        if (i > 0) scanf(" ");
        scanf("%c%c", rank1, rank1);
        board[i] = str_to_card(rank1, suit1);
    }
    scanf("\n");
    for (int i = 0; i < oopSize; i++) {
        scanf("%c%c %c%c\n", rank1, suit1, rank2, suit2);
        oopRange[i*2] = str_to_card(rank1, suit1);
        oopRange[(i*2) + 1] = str_to_card(rank2, suit2);
    }

    for (int i = 0; i < ipSize; i++) {
        scanf("%c%c %c%c\n", rank1, suit1, rank2, suit2);
        ipRange[i*2] = str_to_card(rank1, suit1);
        ipRange[(i*2) + 1] = str_to_card(rank2, suit2);
    }

    GlobalConstants *params = calcGlobalConsts(board, oopRange, oopSize, ipRange, ipSize, potSize, betSize);

    char *bestOopStrategy = (char *) malloc(oopSize * sizeof(char));
    int *stratVal;
    calcMaxStrategy(bestOopStrategy, stratVal, params);

    printf("OOP best strategy has a value of %d", *stratVal);
}
