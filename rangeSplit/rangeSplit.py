import fileinput
import sys

ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
suits = ['d', 'c', 'h', 's']

def printHand(hand, offsuit):
    r1 = hand[0]
    r2 = hand[1]
    if offsuit:
        for i in suits:
            for j in suits:
                if suits.index(i) <= suits.index(j):
                    continue
                print(r1 + i + "," + r2 + j + ',')
    else:
        if r1 == r2:
            return
        for i in suits:
            print(r1 + i + ',' + r2 + i + ',')

for grp in sys.stdin.readlines()[0].split(','):
    if "-" in grp:
        l = grp.split("-")[0]
        r = grp.split("-")[1]
        if (len(l) == 3): #has an o or s
            offsuit = "o" in grp
            rnge = ranks[ranks.index(l[1]):ranks.index(r[1])+1]
            for rnk in rnge:
                hand = [l[0],rnk]
                printHand(hand, offsuit)
        else: # pocket pairs
            rnge = ranks[ranks.index(l[1]):ranks.index(r[1])+1]
            for rnk in rnge:
                hand = [rnk,rnk]
                printHand(hand, True)
                printHand(hand, False)
    else:
        print(grp + ',')
