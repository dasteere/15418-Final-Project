#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cudaRiver.h"

int main() {
    int oopSize, ipSize, potSize, betSize;
    char card1[2], card2[2];
    scanf("%d %d %d %d\n", &oopSize, &ipSize, &potSize, &betSize);
    hand_t *oopRange = (hand_t *) malloc(oopSize * sizeof(hand_t));
    hand_t *ipRange = (hand_t *) malloc(ipSize * sizeof(hand_t));
    board_t board;
    for (int i = 0; i < BOARD_SIZE; i++) {
        scanf("%c%c,", card1, card1 +1);
        if (str_to_card(card1, &board.cards[i]) < 0) {
            printf("Invalid input: %c%c\n", card1[0], card1[1]);
            return 0;
        }
    }
    scanf("\n---OOP RANGE---\n");
    printf("Read in board\n");
    for (int i = 0; i < oopSize; i++) {
        scanf("%c%c,%c%c,\n", card1, card1 + 1, card2, card2 + 1);
        if (str_to_card(card1, &oopRange[i].cards[0]) < 0) {
            printf("Invalid input: %c%c\n", card1[0], card1[1]);
            return 0;
        }
        if (str_to_card(card2, &oopRange[i].cards[1]) < 0) {
            printf("Invalid input: %c%c\n", card2[0], card2[1]);
            return 0;
        }
    }
    scanf("---IP RANGE---\n");
    printf("Read in OOP\n");
    for (int i = 0; i < ipSize; i++) {
        scanf("%c%c,%c%c,\n", card1, card1 + 1, card2, card2 + 1);
        if (str_to_card(card1, &ipRange[i].cards[0]) < 0) {
            printf("Invalid input: %c%c\n", card1[0], card1[1]);
            return 0;
        }
        if (str_to_card(card2, &ipRange[i].cards[1]) < 0) {
            printf("Invalid input: %c%c\n", card2[0], card2[1]);
            return 0;
        }
    }
    printf("Read in IP\n");
    GlobalConstants *params = calcGlobalConsts(board, oopRange, oopSize, ipRange, ipSize, potSize, betSize);

    char *bestOopStrategy = (char *) malloc(oopSize * sizeof(char));
    int stratVal;
    calcMaxStrategy(bestOopStrategy, &stratVal, params);

    printf("OOP best strategy has a value of %d", stratVal);
}
