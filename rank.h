#define DIAMOND (1 << 0)
#define HEART   (1 << 1)
#define SPADE   (1 << 2)
#define CLUB    (1 << 3)

#define MAX_SUIT_VAL CLUB+1

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
#define TRIPS_KICKERS 2
#define TWO_PAIR_KICKERS 1
#define PAIR_KICKERS 1
#define HIGH_CARD_KICKERS 5

#define QUAL_BITS (5 * RANK_SHIFT)

#define MAX(x,y) ((x) > (y) ? (x) : (y))

typedef struct __attribute__((packed)) card {
    unsigned char suit:4;
    unsigned char rank:4;  
} card_t;

enum hand_type { STRAIGHT_FLUSH = 8, QUADS = 7, BOAT = 6, FLUSH = 5, 
                    STRAIGHT = 4, TRIPS = 3, TWO_PAIR = 2, PAIR = 1, 
                    HIGH_CARD = 0 };

int rank_of(card_t board[BOARD_SIZE], card_t hand[HAND_SIZE]); 

void int_to_hand(int hand, char *buf);
