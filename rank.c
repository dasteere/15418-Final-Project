#include <stdio.h>
#include <stdbool.h>
#include "rank.h"

int hand_to_int(enum hand_type t, int qual) {
    return (1 << (QUAL_BITS  + t)) | qual;
}

int rank_of(card_t board[BOARD_SIZE], card_t hand[HAND_SIZE]) {
    char rank_suits[NUM_RANKS];
    char ranks[NUM_RANKS];
    char suits[MAX_SUIT_VAL];

    for (int i = 0; i < BOARD_SIZE; i++) {
        char suit = board[i].suit;
        char rank = board[i].rank;

        rank_suits[rank] |= suit;
        ranks[rank]++;
        suits[suit]++;
    }
    for (int i = 0; i < HAND_SIZE; i++) {
        char suit = board[i].suit;
        char rank = board[i].rank;

        rank_suits[rank] |= suit;
        ranks[rank]++;
        suits[suit]++;
    }

    char last_rank = 0;
    char num_consec = 0;

    int straight_flush = -1;
    int flush = -1;
    int straight = -1;
    int quads = -1;
    int boat = -1;
    int trips = -1;
    int pair = -1;
    int two_pair = -1;

    /* Check for straight flush */
    for (int i = 0; i < NUM_SUITS; i++) {
        char suit = 1 << i;
        num_consec = 0;
        for (int j = NUM_RANKS - 1; j >= 0; j--) {
            if (rank_suits[j] & suit) {
                num_consec++;
                if (num_consec == CARDS_TO_STRAIGHT) {
                    straight_flush = j; 
                }
            } else {
                num_consec = 0;
            }
        }
    }

    for (int i = 0; i < NUM_SUITS; i++) {
        char suit = 1 << i;
        /* Find highest rank in flush */
        if (suits[suit] == CARDS_TO_FLUSH) {
            for (int j = NUM_RANKS - 1; j >= 0; j++) {
                if (rank_suits[j] & suit) {
                    flush = j;
                    break;
                }
            }
        } 
    }

    num_consec = 0;
    for (int i = NUM_RANKS - 1; i >= 0; i--) {
        if (ranks[i] == CARDS_TO_QUADS) quads = i;
        else if (ranks[i] == CARDS_TO_TRIPS) {
            if (trips == -1) {
                trips = i;
            }
            if (pair >= 0 && boat == -1) {
                boat = trips << RANK_SHIFT | i;
            }
        } else if (ranks[i] == CARDS_TO_PAIR) {
            if (pair >= 0 && two_pair == -1) {
                two_pair = pair << RANK_SHIFT | i;
            } else if (pair == -1) {
                pair = i;
            }
            if (trips >= 0 && boat == -1) boat = trips << RANK_SHIFT | i;
        }

        if (ranks[i]) {
            num_consec++;
            if (num_consec == CARDS_TO_STRAIGHT) straight = i;
        } else {
            num_consec = 0;
        }
    }

    int kicker;
    /* Add kicker to two pair, quads, trips */
    for (int i = NUM_RANKS - 1; i >= 0; i--) {
        if (ranks[i] == 1) {
            if (quads >= 0) quads = (quads << RANK_SHIFT) | i; 
            if (trips >= 0) trips = (trips << RANK_SHIFT) | i;
            if (two_pair >= 0) two_pair = (two_pair << RANK_SHIFT) | i;
            kicker = i;
            break;
        }
    }

    if (straight_flush >= 0) return hand_to_int(STRAIGHT_FLUSH, straight_flush);
    if (quads >= 0) return hand_to_int(QUADS, quads);
    if (boat >= 0) return hand_to_int(BOAT, boat);
    if (flush >= 0) return hand_to_int(FLUSH, flush);
    if (straight >= 0) return hand_to_int(STRAIGHT, straight);
    if (trips >= 0) return hand_to_int(TRIPS, trips);
    if (two_pair >= 0) return hand_to_int(TWO_PAIR, two_pair);
    if (pair >= 0) return hand_to_int(PAIR, pair);
    return hand_to_int(HIGH_CARD, kicker);
}

int main() {
    printf("%d", sizeof(card_t));
}

