# TTS infer library
install method:

import way:
import tts_service
#import tts_service as vits
from tts_service.infer_service import TTS_Vits as vits
from tts_service.infer_service import TTS_VitsM as vitsm

usage:
class TTS_Vits,tts_VitsM
initial: 
	tts_vits("cuda",modelpath)    #model pace do not use yet
infer:
	ifer(fileName,input_text, pace)

functions:
return 
0, 'ok'
1, 'not in dic'

text_normalization still from Evan's code
use tw2s from OpenCC
error source: 	1. not in dic
		2. not chinese

model path
/home/yuhang/tts_model/b99084aeaa1411eeaf7b0011328a21bb(commnuity_id)/122(user_id)/44.pth(user_model_id)
/home/yuhang/tts_model/b99084aeaa1411eeaf7b0011328a21bb(commnuity_id)/122(user_id)/44.pth(user_model_id)+44_emb.pt()

db structure|user_model
id | send_key | community_id | user_id | template_id | model_id | create_time | msg 


need to chage?:
add volumn?


sample_code:

import os
#import sys
#import numpy as np

#import torch
#import infer_lib.utils as utils
#import argparse
import time

import tts_service
#import tts_service as vits
from tts_service.infer_service import TTS_Vits as vits
from tts_service.infer_service import TTS_VitsM as vitsm

#from scipy.io import wavfile
#from infer_lib.text.symbols import symbols
#from infer_lib.text import cleaned_text_to_sequence
#from infer_lib.vits_pinyin import VITS_PinYin
#from infer_lib.models import SynthesizerEval


if __name__ == '__main__':
    text="零件費用產生，服務人員將另行報價，請問您接受嗎?"
    print(text)
    #tts_serv=vits("cuda",None)
    aaa=vitsm("cuda",None)    #model pace do not use yet
    aaa.infer('./hello.wav',text,1.0)
    #tts_serv.infer('./hello.wav',text,0.8)



## TTS_VoiceClone_Training_Service

flow:
init & prepare>find job(db)>collect data for pre-processing(task)> pre&train(task)
> genereate example(task)> move files(task)>update db(db)>return msg> reset env(task)

1. interact with db
a. find job
b. find sentence
c. create user_model
d. update train_request
e. get id from user_model

class DB_interact():
methods:{
init(db_settings): set db_settings, connection, cursor
rollback():connection rollback
search_job(): find if job exist(generated = 0) and fetch fist one by time order. if no job return None.
}





2. create and reset temp environment
./temp/train_set for dataset
./models/ for pre-model
./temp/gen for generated files in train_service
make .txt and waves
for vitsm pick up speaker embedding .pt


class TTS_Task():
method:{
init:
prepare_dataset(job,zipfile_path,template):
select_sp_emb(): return to_memo, longest text file's embedding  (xxx.pt)
train1(): for vits
train2(): for vitsm
generate1(): generate example for vits
generate2(): generate example for vitsm
}
3. pre-processing and train
for pre-processing: should be in the py
for train: need pre-model path and filelists path(in config)


4. generate 24 example wave files

5. update db and return and move file to corespond dir

6. reset environment


example_text.txt for example
to_unzip=f'/home/dataFile/uploadzip/{ajob[2]}_{str(ajob[1])}.zip'
example_path = f'/home/dataFile/tts_train_return/{ajob[2]}_{str(ajob[1])}'
mod_dir='/home/yuhang/tts_model/'+db_job[2]+'/'+str(db_job[3])+'/'
url = 'http://192.168.77.4:8080/index.php/PostAiServer/CompleteVoiceModel/'


# db section
    db_settings={
                'host': 'localhost',
                'user': 'root',
                'password': '',
                'db':'tts'
                }
# train_request table



save model name id.pth+id_emb.pt


示範因黨?
