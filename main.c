#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cudaRiver.h"

int main() {
    int oopSize, ipSize, potSize, betSize;
    char card1[2], card2[2];
    scanf("%d %d %d %d\n", &oopSize, &ipSize, &potSize, &betSize);
    card_t *oopRange = (card_t *) malloc(2 * oopSize * sizeof(card_t));
    card_t *ipRange = (card_t *) malloc(2 * ipSize * sizeof(card_t));
    card_t board[BOARD_SIZE];
    for (int i = 0; i < BOARD_SIZE; i++) {
        scanf("%c%c,", card1, card1 +1);
        if (str_to_card(card1, &board[i]) < 0) {
            printf("Invalid input: %c%c\n", card1[0], card1[1]);
            return 0;
        }
    }
    scanf("\n---OOP RANGE---\n");
    printf("Read in board\n");
    for (int i = 0; i < oopSize; i++) {
        scanf("%c%c,%c%c,\n", card1, card1 + 1, card2, card2 + 1);
        if (str_to_card(card1, &oopRange[i*2]) < 0) {
            printf("Invalid input: %c%c\n", card1[0], card1[1]);
            return 0;
        }
        if (str_to_card(card2, &oopRange[(i*2) + 1]) < 0) {
            printf("Invalid input: %c%c\n", card2[0], card2[1]);
            return 0;
        }
    }
    scanf("---IP RANGE---\n");
    printf("Read in OOP\n");
    for (int i = 0; i < ipSize; i++) {
        scanf("%c%c,%c%c,\n", card1, card1 + 1, card2, card2 + 1);
        if (str_to_card(card1, &ipRange[i*2]) < 0) {
            printf("Invalid input: %c%c\n", card1[0], card1[1]);
            return 0;
        }
        if (str_to_card(card2, &ipRange[(i*2) + 1]) < 0) {
            printf("Invalid input: %c%c\n", card2[0], card2[1]);
            return 0;
        }
    }
    printf("Read in IP\n");
    GlobalConstants *params = calcGlobalConsts(board, oopRange, oopSize, ipRange, ipSize, potSize, betSize);

    char *bestOopStrategy = (char *) malloc(oopSize * sizeof(char));
    int *stratVal;
    calcMaxStrategy(bestOopStrategy, stratVal, params);

    printf("OOP best strategy has a value of %d", *stratVal);
}
