#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
import jieba


def read_data(filename):
    text = []
    with open(filename,'r') as f:
        for line in f:
            text.append(line.strip())
        return text
def split_word(text):
    title_list = []
    for i in text:
        title_list.extend(jieba.cut(i))
    return title_list


def unique_word(text):
    text = list(set(text))
    return text


if __name__ == "__main__":
    filename1 = './data/title.txt'
    filename2 = './data/action.txt'
    title = read_data(filename1)
    action = read_data(filename2)
    action = unique_word(action)
    title = unique_word(split_word(title))
    with open('unique_title.txt','w') as f:
        for i in title:
            f.write(i+'\n')
    with open("unique_action.txt",'w') as f2:
        for j in action:
            f2.write(j+'\n')