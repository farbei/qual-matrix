# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:14:28 2019

@author: ifarber
"""

from datetime import datetime as dt
from time import time

t = time()  

def param_dict(param):
    return dict(tuple(p.split('=',1)) for p in param.split(';') if '=' in p)


def wild_cards(col):
    return col.str.replace('*','.*').str.replace('?','.')


def print_error(txt):
    with open('checkSum.txt', 'a') as myfile: 
            myfile.write(txt + str(dt.now()) + '\n') 
            

def timeit():
    print(time() - t)          