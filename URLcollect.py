#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou

in_path = "./data/"
out_path = "./product/"
filename = 'file_list_20170430_new的副本.txt'
filename2='file_list的副本.txt'

writter = open(out_path + 'NormalURL.txt', 'w', encoding='utf-8')
def collect():
    with open(in_path+filename,'r',encoding='utf-8') as f:
        count = 0
        for line in f:
            if line.split(',')[1] == 'n' and count < 250:
                count += 1
                writter.write(line)

def collect2():
    with open(in_path+filename2,'r',encoding='utf-8') as f:
        for line in f:
            if line.split(',')[1] == 'p':
                writter.write(line)

if __name__ == "__main__":
    collect()
    writter.close()