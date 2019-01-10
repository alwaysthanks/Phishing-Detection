#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou

from lxml import etree
import urllib.parse
import urllib
import os
import codecs



def traverse_directory(WebDirectory,mainfile):
    Dark_MD5=list()
    with open(mainfile,'r') as f:
        for i in f:
            flag=i.split(',',4)[1]
            MD5=i.split(',',4)[2]
            if flag=='p':
                each_file=os.path.join(WebDirectory,MD5)
                Dark_MD5.append(each_file)
    return Dark_MD5

def read_file(filename):
    try:
        f=codecs.open(filename,'r',encoding='utf-8')
        Web_data=f.readlines()
        # print(filename)
        print('******')
        f.close()
        Web_data = '\n'.join(Web_data)
    except:
        f=codecs.open(filename,'r','gb18030')
        Web_data = f.readlines()
        # print(Web_data)
        # print(filename)
        print('******')
        f.close()
        Web_data = '\n'.join(Web_data)
    return Web_data

def parse_html(Web_data):
    html = bytes(bytearray(Web_data, encoding='utf-8'))
    html = etree.HTML(html)
    title = html.xpath('//title/text()')
    action = html.xpath('//form/@action')
    return title, action


if __name__ == "__main__":
    # try:
        title_list, action_list = [], []
        p_title = open("title.txt",'a',encoding='utf-8')
        p_action = open("action.txt",'a',encoding='utf-8')
        mainfile = './data/file_list_10000.txt'
        WebDirectory = './data/file1/'
        phish_list = traverse_directory(WebDirectory,mainfile)
        print(len(phish_list))
        for aline in phish_list:
            Web_data = read_file(aline)
            title, action = parse_html(Web_data)
            title_list.extend(title)
            action_list.extend(action)
        title_list = list(set(title_list))
        action_list = list(set(action_list))
        for a_title in title_list:
            p_title.write(a_title.strip()+'\n')
        for a_action in action_list:
            p_action.write(a_action.strip()+'\n')
    # except:
    #     print(aline)
