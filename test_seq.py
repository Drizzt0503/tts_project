import os
from text import cleaned_text_to_sequence
import torch

def get_text(text):
    text_norm = cleaned_text_to_sequence(text)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


ff= open('./temp/filelists/all_t.txt','rt')

abc=ff.readlines()
ff.close()

for any in abc:
    bbb=any.strip().split('|')
    ccc=get_text(bbb[2].strip())
    print(bbb)
    print(ccc)
    print(ccc.size())
    print('===========')
