# -*- coding: utf-8 -*-
__author__ = 'pangbochen'

import random

def cumulative(arr):
    p = [0.0 for _ in range(len(arr))]
    p = [k for k in arr]

    for i in range(1, len(arr)):
        p[i] += p[i-1]

    u = random.random() * p[-1]
    index = 0
    for index in range(len(p)-1):
        if p[index]>u:
            break
    return index