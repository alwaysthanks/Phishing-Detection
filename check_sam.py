#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
import codecs
import os


mainfile = './data/file_list.txt'
WebDirectory = './data/file1/'

def check_sam(check_list):
    with open("file_list_10000.txt",'w',encoding='utf-8') as f1:
        with open(mainfile, 'r', encoding='utf-8') as f:
            for i in f:
                md5 = i.split(',')[2]
                if md5 not in check_list:
                    f1.write(i)



def read_file(filename):
    try:
        f=codecs.open(filename,'r',encoding='utf-8')
        Web_data=f.readlines()
        # print(filename)
        # print('******')
        Web_data = '\n'.join(Web_data)
        f.close()
    except:
        f=codecs.open(filename,'r',encoding='gb18030')
        Web_data = f.readlines()
        # print(Web_data)
        # print(filename)
        # print('******')
        Web_data = '\n'.join(Web_data)
        f.close()
    return Web_data

def traverse_directory(WebDirectory,mainfile):
    MD5_list = list()
    flag_list = list()
    URL_list = list()
    with open(mainfile,'r') as f:
        for i in f:
            flag=i.split(',',4)[1]
            if flag=='n':
                flag_list.append(1)
                MD5 = i.split(',', 4)[2]
                # each_file = os.path.join(WebDirectory, MD5)
                MD5_list.append(MD5)
                URL_list.append(i.strip().split(',',4)[3])
            elif flag=='p':
                flag_list.append(0)
                MD5 = i.split(',', 4)[2]
                MD5_list.append(MD5)
                URL_list.append(i.strip().split(',', 4)[3])
    return MD5_list,flag_list,URL_list


if __name__=="__main__":
    MD5_list, flag_list, URL_list = traverse_directory(WebDirectory, mainfile)
    check_list=[]
    for i in range(len(MD5_list)):
        try:
            Web_data = read_file(os.path.join(WebDirectory,MD5_list[i]))
        except:
            check_list.append(MD5_list[i])

    check_sam(check_list)
