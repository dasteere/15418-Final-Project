#ifndef _RANK_H
#define _RANK_H

#define DIAMOND (1 << 0)
#define HEART   (1 << 1)
#define SPADE   (1 << 2)
#define CLUB    (1 << 3)

#define MAX_SUIT_VAL CLUB+1

#define A_RANK 12
#define K_RANK 11
#define Q_RANK 10
#define J_RANK 9
#define T_RANK 8

#define ASCII_NUM_OFFSET 50

#define BOARD_SIZE 5
#define HAND_SIZE 2

#define NUM_RANKS 13
#define NUM_SUITS 4

#define SUIT_BITS 4
#define RANK_BITS 4

#define RANK_SHIFT RANK_BITS

#define CARDS_TO_FLUSH 5
#define CARDS_TO_STRAIGHT 5
#define CARDS_TO_QUADS 4
#define CARDS_TO_TRIPS 3
#define CARDS_TO_PAIR 2

#define QUADS_KICKERS 1
#define FLUSH_KICKERS 5
#define TRIPS_KICKERS 2
#define TWO_PAIR_KICKERS 1
#define PAIR_KICKERS 1
#define HIGH_CARD_KICKERS 5

#define QUAL_BITS (5 * RANK_SHIFT)

#define CARD_STR_LEN 3


typedef struct __attribute__((packed)) card {
    unsigned char suit:4;
    unsigned char rank:4;
} card_t;

typedef struct __attribute__((packed)) {
    card_t cards[HAND_SIZE];
} hand_t;

typedef struct __attribute__((packed)) {
    card_t cards[BOARD_SIZE];
} board_t;

enum hand_type { STRAIGHT_FLUSH = 8, QUADS = 7, BOAT = 6, FLUSH = 5,
                    STRAIGHT = 4, TRIPS = 3, TWO_PAIR = 2, PAIR = 1,
                    HIGH_CARD = 0 };

int rank_of(board_t *board, hand_t *hand);

int suit_to_uchar(char suit, unsigned char *uchar);

int rank_to_uchar(char rank, unsigned char *uchar);

int uchar_to_suit(unsigned char uchar, char *suit);

int uchar_to_rank(unsigned char uchar, char *rank);

void int_to_hand(int hand, char *buf);

int card_to_str(card_t card, char *buf);

int str_to_card(char *str, card_t *card);

#endif /* _RANK_H */
