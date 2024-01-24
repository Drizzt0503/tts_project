import os
import torch
import numpy as np
import statistics as sts


anaf='./analyse_result'

josh_emb = torch.load("./register/ally.pt")
#vm2_emb = torch.load("./register/vm2_inside.pt")
#ai3_wav = os.listdir('/data2/tts_dataset/AISHELL-mix/ai3_waves_total/')

sdir='/data2/small_finetune/ally/embed/'

for any in josh_emb:
    print(any[:-4]+'.pt',josh_emb[any][0].shape)
    torch.save(josh_emb[any][0],sdir+any[:-4]+'.pt')

