#include <stdio.h>
#include <stdbool.h>
#include "rank.h"

int hand_to_int(enum hand_type t, int qual) {
    return (1 << (QUAL_BITS  + t)) | qual;
}

int str_to_card(char *str, card_t *card) {
    char rank = str[0];
    char suit = str[1];

    unsigned char rank_num;
    switch (rank) {
        case 'A':
            rank_num = A_RANK;
            break;
        case 'K':
            rank_num = K_RANK;
            break;
        case 'Q':
            rank_num = Q_RANK;
            break;
        case 'J':
            rank_num = J_RANK;
            break;
        case 'T':
            rank_num = T_RANK;
        default:
            rank_num = rank - ASCII_NUM_OFFSET;
    } 
    if (rank_num >= NUM_RANKS) return -1;

    unsigned char suit_num;
    switch (suit) {
        case 'd':
            suit_num = DIAMOND;
            break;
        case 'h':
            suit_num = HEART;
            break;
        case 'c':
            suit_num = CLUB;
            break;
        case 's':
            suit_num = SPADE;
            break;
        default:
            return -1;
    }

    card->rank = rank_num;
    card->suit = suit_num;
    return 0;
}

void int_to_hand(int hand, char *buf) {
    int type_bits = hand >> QUAL_BITS;
    enum hand_type type = 0;

    while (type_bits != 1) {
        type_bits >>= 1;
        type++;
    }

    int num_cards;
    int n = 0;

    switch (type) {
        case HIGH_CARD:
            n += sprintf(buf + n, "High card: ");
            num_cards = 5;
            break;
        case PAIR:
            n += sprintf(buf + n, "Pair: ");
            num_cards = 4;
            break;
        case TWO_PAIR:
            n += sprintf(buf + n, "Two pair: ");
            num_cards = 3;
            break;
        case TRIPS:
            n += sprintf(buf + n, "Trips: ");
            num_cards = 3;
            break;
        case STRAIGHT:
            n += sprintf(buf + n, "Straight: ");
            num_cards = 1;
            break;
        case FLUSH:
            n += sprintf(buf + n, "Flush: ");
            num_cards = 1;
            break;
        case QUADS:
            n += sprintf(buf + n, "Quads: ");
            num_cards = 2;
            break;
        case STRAIGHT_FLUSH:
            n += sprintf(buf + n, "Straight flush: ");
            num_cards = 1;
            break;
        default:
            n += sprintf(buf + n, "Unrecognized");
            num_cards = 0;
    }

    for (int i = 0; i < num_cards; i++) {
        unsigned char rank = (unsigned char)hand & 0xF;
        n += sprintf(buf + n, "\nCard %d: Rank %d", i, rank);

        hand >>= RANK_SHIFT;
    }
}

int rank_of(card_t *board, card_t *hand) {
    char rank_suits[NUM_RANKS];
    char ranks[NUM_RANKS];
    char suits[MAX_SUIT_VAL];

    for (int i = 0; i < NUM_RANKS; i++) {
        rank_suits[i] = 0;
        ranks[i] = 0;
    }
    for (int i = 0; i < MAX_SUIT_VAL; i++) {
        suits[i] = 0;
    }

    for (int i = 0; i < BOARD_SIZE; i++) {
        unsigned char suit = board[i].suit;
        unsigned char rank = board[i].rank;

        rank_suits[rank] |= suit;
        ranks[rank]++;
        suits[suit]++;
    }
    for (int i = 0; i < HAND_SIZE; i++) {
        unsigned char suit = hand[i].suit;
        unsigned char rank = hand[i].rank;

        rank_suits[rank] |= suit;
        ranks[rank]++;
        suits[suit]++;
    }

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
        unsigned char suit = 1 << i;
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
        unsigned char suit = 1 << i;
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

    int num_kickers = 0;
    int high_card = 0;
    /* Add kicker to two pair, quads, trips, high card */
    for (int i = NUM_RANKS - 1; i >= 0; i--) {
        if (ranks[i] == 1) {
            if (num_kickers < QUADS_KICKERS && quads >= 0)
                quads = (quads << RANK_SHIFT) | i;
            if (num_kickers < TRIPS_KICKERS && trips >= 0)
                trips = (trips << RANK_SHIFT) | i;
            if (num_kickers < TWO_PAIR_KICKERS && two_pair >= 0)
                two_pair = (two_pair << RANK_SHIFT) | i;
            if (num_kickers < HIGH_CARD_KICKERS)
                high_card = (high_card << RANK_SHIFT) | i;

            num_kickers++;
            if (num_kickers >= HIGH_CARD_KICKERS) break;
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
    return hand_to_int(HIGH_CARD, high_card);
}

/*
int main() {
    card_t board[BOARD_SIZE];
    card_t hand[HAND_SIZE];
    char buf[256];

    card_t card1 = { .suit = 1, .rank = 5 };
    card_t card2 = { .suit = 2, .rank = 7 };
    card_t card3 = { .suit = 4, .rank = 8 };
    card_t card4 = { .suit = 2, .rank = 10 };
    card_t card5 = { .suit = 1, .rank = 11 };

    card_t hand1 = { .suit = 1, .rank = 5 };
    card_t hand2 = { .suit = 2, .rank = 7 };

    card_t hand3 = { .suit = 2, .rank = 8 };
    card_t hand4 = { .suit = 2, .rank = 7 };

    board[0] = card1;
    board[1] = card2;
    board[2] = card3;
    board[3] = card4;
    board[4] = card5;

    for (int i = 0; i < BOARD_SIZE; i++) {
        printf("suit %d rank %d\n", board[i].suit, board[i].rank);
    }

    hand[0] = hand1;
    hand[1] = hand2;
    int rank = rank_of(board, hand);

    printf("rank: %x\n", rank);
    int_to_hand(rank, buf);
    printf("%s\n", buf);



    hand[0] = hand3;
    hand[1] = hand4;
    rank = rank_of(board, hand);

    printf("rank: %x\n", rank);
    int_to_hand(rank, buf);
    printf("%s\n", buf);

    /*
    for (int i = 0; i < BOARD_SIZE; i++) board[i] = card;
    for (int i = 0; i < HAND_SIZE; i++) hand[i] = card;

    for (int i = 0; i < 2000; i++) {
        printf("%d\n", rank_of(board, hand));
    }*/
//}

