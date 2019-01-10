#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou


in_path = "./data/"
out_path = "./product/"
filename = 'file_list_20170430_new的副本.txt'
filename2='file_list.txt'
phishing_list = []
normal_list = []
dark_list = []
def samp_count():
    normal, dark, phishing = 0, 0, 0
    with open(in_path+filename,'r',encoding='utf-8') as f:
        for sample in f:
            if sample.split(',')[1] == 'n':
                normal += 1
                normal_list.append(sample.split(',')[2])
            elif sample.split(',')[1] == 'd':
                dark += 1
                dark_list.append(sample.split(',')[2])
            else:
                phishing += 1
                phishing_list.append(sample.split(',')[2])
        return normal, dark, phishing

def samp_count2():
    normal, dark, phishing = 0, 0, 0
    with open(in_path+filename2,'r',encoding='utf-8') as f:
        for sample in f:
            if sample.split(',')[1] == 'n':
                normal += 1
                normal_list.append(sample.split(',')[2])
            elif sample.split(',')[1] == 'd':
                dark += 1
                dark_list.append(sample.split(',')[2])
            else:
                phishing += 1
                phishing_list.append(sample.split(',')[2])
        return normal, dark, phishing


if __name__ == "__main__":
    normal, dark, phishing = samp_count()
    print(normal, dark, phishing)
    normal, dark, phishing = samp_count2()
    print(normal, dark, phishing)
    print(len(normal_list),len(dark_list),len(phishing_list))
    normal_list = list(set(normal_list))
    dark_list = list(set(dark_list))
    phishing_list = list(set(phishing_list))
    print(len(normal_list), len(dark_list), len(phishing_list))