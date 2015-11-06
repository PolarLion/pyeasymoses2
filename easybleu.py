#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import math
import os
import sys

def corrects (translated, reference) :
    corrects = 0
    for k, v in translated.items () :
        # print k, v, reference.get(k, 0)
        corrects += min (v, reference.get (k, 0))
    return corrects

def score (translated, reference, n) :
    tl = translated
    t = {}
    for i in xrange (len (tl) - n + 1) :
        k = tuple (tl [i:i+n])
        t [k] = t.get (k, 0) + 1
    # print t
    rl = reference
    r = {}
    for i in xrange (len (rl) - n + 1) :
        k = tuple (rl [i:i+n])
        r [k] = r.get (k, 0) + 1
    # print r
    # print "corrects ", corrects (t, r)
    # print n, (len (tl) - n + 1) 
    precision = corrects (t, r) * 1.0 / (len (tl) - n + 1) 
    # print "precision ", precision
    # print "recall ", corrects (t, r) * 1.0 / (len (rl) - n + 1)
    return precision#, recall

def bleu_n (translated, reference, n) :
    t = translated.lower ().rstrip ().split (" ")
    r = reference.lower ().rstrip ().split (" ")
    b = {}
    bleu = 1.0
    if len (r) <= n : n -= 1
    for i in range (0, n) :
        if i + 1 > len (t) : 
            # print "$$$$$$$$$$$$$$$$$$$$$$"
            break
        p = score (t, r, i+1)
        bleu *= p
        # print min (1, len (t) * 1.0 / len (r))
        # print "bleu", bleu * min (1, len (t) * 1.0 / len (r))
        b [i+1] = bleu * min (1, len (t) * 1.0 / len (r))
    return b

def bleu (translated, reference) :
    b = bleu_n (translated, reference, 4)
    score = 0
    w = 0
    for k, v in b.items () :
        score += k * v
        w += k
    return score / w


if __name__ == "__main__":
    # t = "airport security Israeli officials are responsible"
    # r = "Israeli officials are responsible for airport security"
    t = "oh , so good !"
    r = "oh , nice !"

    print bleu_n (t, r, 4)
    print bleu (t, r)