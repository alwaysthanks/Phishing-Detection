#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
import re
from nltk.tokenize import WordPunctTokenizer

def get_TF(words):
    tf_dic = {}
    for w in words:
        tf_dic[w] = tf_dic.get(w,0)+1
    return sorted(tf_dic.items(), key = lambda x:x[1], reverse=True)

def read():
    url = []
    with open("./data/PhishingURL.txt",'r',encoding='utf-8') as f:
        for line in f:
            url.append(line.strip().split(',')[3])
        return url

if __name__ == "__main__":
    wordslist = []
    url = read()
    for aURL in url:
        words = WordPunctTokenizer().tokenize(aURL)
        words = [i for i in words if i.isalnum() and not i.isdigit()]
        wordslist.extend(words)
    result = get_TF(wordslist)
    with open('./product/URL_keyword.txt','w',encoding='utf-8') as f:
        for j in result:
            f.write(j[0]+','+str(j[1])+'\n')

