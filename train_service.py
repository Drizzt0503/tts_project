import os
import time
import pymysql
import datetime
import json
import sys
from opencc import OpenCC
from torch.cuda import is_available

def getReturn(sendMsg,url,status,msg):
    # 發送完成訊息
    import urllib.request as req
    print(url+sendMsg)
    with req.urlopen(url+sendMsg) as res:
        returnData = json.load(res)
        print(returnData)
    if (returnData['Status'] == 0) :
        # key error
        pass
    elif (returnData['Status'] == 1) :
        # db update error
        pass
    elif (returnData['Status'] == 2) :
        # srt content == null
        pass
    else :
        # other error
        pass



class TTS_Task():
    def __init__(self,main_dir):
        self.main_dir=main_dir
        self.data_dir = self.main_dir+'/temp/train_set/'
        self.model_dir = self.main_dir+'/models/'
        self.gen_dir = self.main_dir+'/temp/gen/'
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(self.data_dir+'owaves/', exist_ok=True)
        os.makedirs(self.data_dir+'waves/', exist_ok=True)
        self.cc = OpenCC('tw2s')

    def prepare_dataset(self,db_job,to_unzip,template):
        to_unzip=to_unzip+f'{db_job[2]}_{str(db_job[1])}.zip'
        os.system(f'unzip {to_unzip} -d ./temp/train_set/owaves')
        with open(f'./temp/train_set/owaves/{str(ajob[1])}.txt') as f:
            udata = json.load(f)
        vdata=[]
        for any in udata:                
            user_data = any['VoiceFile'].split('/')
            vdata.append([user_data[4][:-4],int(any['TextModelId'])])
        temp_text_dic={}
        temp_phoneme_dic={}
        for any in template:
            temp_text_dic[any[0]]=any[4]
            temp_phoneme_dic[any[0]]=any[5]
        with open(self.data_dir+'/text_info.txt','wt') as tfile:
            for any in vdata:  #need decide vdata
                tocn= self.cc.convert(temp_text_dic[any[1]])
                tfile.write(f"{any[0]}|{tocn}|{temp_phoneme_dic[any[1]]}\n")
        #not using abs path here, need to be modified
        cwd = os.getcwd()
        os.chdir(task.data_dir+'/owaves')
        cmd ='for f in *.wav; do ffmpeg -loglevel error -y -i "$f" -acodec pcm_s16le -ac 1 -ar 16000 "../waves/$f"; done'
        os.system(cmd)
        os.chdir(cwd)
        print('finished make dataset')

    def select_sp_emb(self):
        with open(self.data_dir+'/text_info.txt','rt') as tfile:
            atext=tfile.readlines()
        len_sort=[]
        for any in atext:
            texts=any.strip().split('|')
            len_sort.append([texts[0],len(texts[1])])
        len_sort.sort(key = lambda i: i[1], reverse = True)
        for any in len_sort:
            print(any)
        print('===')
        return len_sort[0][0]+'.pt'


    def train1(self):
        cwd = os.getcwd()
        os.chdir(self.main_dir)
        cmd='python3 pre_processing.py -c ./config/train_service_vits.json  -m vits'
        print('start pre-process')
        os.system(cmd)
        print('finish pre-process')
        #start train
        #os.system('cp ../model/*_BZN.pth ../model/Vits')
        cmd='python3 vits_train_edge.py --config ./config/train_service_vits.json -m vits'
        print('start train')
        os.system(cmd)
        os.chdir(cwd)

    def train2(self):
        cwd = os.getcwd()
        os.chdir(self.main_dir)
        cmd='python3 pre_processing.py -c ./config/train_service_vitsm.json  -m vitsm'
        print('start pre-process')
        os.system(cmd)
        print('finish pre-process')
        #start train
        #os.system('cp ../model/*_BZN.pth ../model/Vits')
        cmd='python3 vitsm_train_edge.py --config ./config/train_service_vitsm.json -m vits'
        print('start train')
        os.system(cmd)
        os.chdir(cwd)

    def generate1(self,ex_path):
        cwd = os.getcwd()
        from infer_service import TTS_Vits as vits
        device = "cuda" if is_available() else "cpu"
        tts = vits(device,None)
        fo = open("./example_text.txt", "r+", encoding='utf-8')
        texts = fo.readlines()
        fo.close()
        pace_list=[8,10,12]
        print('start generate example')
        for i,txt in enumerate(texts):
            for speed in pace_list:
                tts.infer(f'./temp/gen/{str(speed)}_{str(i+1)}.wav',txt, speed/10.0)
        os.chdir(self.gen_dir)
        cmd ='for f in *.wav; do ffmpeg -loglevel error -y -i "$f" -vn -ar 16000 -ac 1  "'+ex_path+'/${f%.wav}.mp3"; done'
        os.system(cmd)
        os.chdir(cwd)

    def generate2(self,ex_path):
        cwd = os.getcwd()
        from infer_service import TTS_VitsM as vitsm
        device = "cuda" if is_available() else "cpu"
        tts = vitsm(device,None)
        fo = open("./example_text.txt", "r+", encoding='utf-8')
        texts = fo.readlines()
        fo.close()
        pace_list=[8,10,12]
        print('start generate example')
        for i,txt in enumerate(texts):
            for speed in pace_list:
                tts.infer(f'./temp/gen/{str(speed)}_{str(i+1)}.wav',txt, speed/10.0)
        os.chdir(self.gen_dir)
        cmd ='for f in *.wav; do ffmpeg -loglevel error -y -i "$f" -vn -ar 16000 -ac 1  "'+ex_path+'/${f%.wav}.mp3"; done'
        os.system(cmd)
        os.chdir(cwd)

    def finish1(self,db_job,model_key,model_path):
        mod_dir=model_path+f'{db_job[2]}/{str(db_job[3])}/'
        os.makedirs(mod_dir, exist_ok=True)
        os.system(f'mv {self.gen_dir}/vits.pth {mod_dir}{str(model_key)}.pth')
        os.system(f'rm {self.gen_dir} -r')
        os.system(f'rm {self.data_dir} -r')
        os.system('rm ./temp/vits -r')
        os.system('rm ./temp/filelists -r')
        print('finish')
    def finish2(self,db_job,model_key,model_path):
        mod_dir=model_path+f'{db_job[2]}/{str(db_job[3])}/'
        os.makedirs(mod_dir, exist_ok=True)
        os.system(f'mv {self.gen_dir}/vits.pth {mod_dir}{str(model_key)}.pth')
        os.system(f'mv {self.gen_dir}/key.pt {mod_dir}{str(model_key)}_emb.pt')
        os.system(f'rm {self.gen_dir} -r')
        os.system(f'rm {self.data_dir} -r')
        os.system('rm ./temp/vits -r')
        os.system('rm ./temp/filelists -r')
        print('finish')


#off function, get template every time needed
#get template
#tts_template= []
#cursor = conn.cursor()
#command = "SELECT * FROM t_ai_featuremodels"
#cursor.execute(command)
#result= cursor.fetchall()
#for any in result:
#    tts_template.append(any)
#print(tts_template)
#for template in tts_template:
#    print(template[0])
#    command = "SELECT * FROM t_ai_textmodels WHERE FeatureModelId = %s"
#    cursor.execute(command,(template[0]))
#    result= cursor.fetchall()
#    print(result)
#cursor.close()

class DB_interact():
    def __init__(self,db_settings):
        self.settings = db_settings
        self.connect=pymysql.connect(**self.settings)
        self.cursor = self.connect.cursor()

    def rollback(self):
        self.connect.rollback()
        
    def search_job(self):
        command = "SELECT * FROM tts.train_request WHERE train_request.generated = false AND train_request.msg = '' ORDER BY create_time ASC"
        self.cursor.execute(command)
        ajob= self.cursor.fetchone()
        return ajob

    def make_template(self,temp_no):
        command = "SELECT * FROM tts.data_collection_sentence  \
                WHERE data_collection_sentence.template_id = %s" 
        self.cursor.execute(command,(temp_no))
        template= self.cursor.fetchall()
        return template
    def update_db(self,db_job):
        now = datetime.datetime.now()
        command = "INSERT INTO user_model (send_key,community_id,user_id,template_id,model_id,create_time,msg) VALUES (%s,%s,%s,%s,%s,%s,%s)"
        self.cursor.execute(command,(db_job[1],db_job[2],db_job[3],db_job[4],db_job[5],str(now),''))
        command = "UPDATE train_request SET train_request.generated=%s, \
                   update_time = %s WHERE train_request.id=%s"
        self.cursor.execute(command,(True,str(now),db_job[0]))
        self.connect.commit()
    def get_usermodel_id(self,db_job):
        command = "SELECT * FROM tts.user_model  \
                WHERE send_key=%s AND community_id=%s AND user_id=%s AND model_id=%s \
                ORDER BY create_time DESC" 
        self.cursor.execute(command,(db_job[1],db_job[2],db_job[3],db_job[5]))
        model_data = self.cursor.fetchone()
        return model_data[0]
    def error_report(self,db_job,text):
        now = datetime.datetime.now()
        command = "UPDATE train_request SET train_request.msg=%s, train_request.generated=%s,\
                   update_time = %s WHERE train_request.id=%s"
        self.cursor.execute(command,(text,True,str(now),db_job[0]))
        self.connect.commit()

        


if __name__ == '__main__':
    #===initialize the train script
    #---settings
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.realpath(__file__))
    #rurl = 'http://106.104.151.145:22026/index.php/PostAiServer/CompleteVoiceModel/'
    #rurl = 'http://192.168.77.4:8080/index.php/PostAiServer/CompleteVoiceModel/'
    example_path = '/var/www/html/webuploader/tts_train_return/'
    model_path = '/data2/yuhang/tts_model/'
    to_unzip = '/var/www/html/webuploader/uploadzip/'
    db_settings={
                'host': '172.17.0.1',
                'user': 'root',
                'password': '',
                'db':'tts'
                }

    #---make connection to db
    DB=DB_interact(db_settings)
    #---set task function & prepare initial environment for training
    task=TTS_Task(main_dir)
    
    #===search jobs
    ajob = DB.search_job()
    if ajob == None:
        print('no job')
        sys.exit(0)
    else:
        print(ajob)
        mtype = ajob[5]
        print('model_type=',mtype)
        os.system(f'rm {main_dir}/temp/* -r')
        send_key=ajob[1]
        rurl = f'{ajob[11]}/index.php/PostAiServer/CompleteVoiceModel/'

    #===prepare dataset
    template = DB.make_template(ajob[4])
    task.prepare_dataset(ajob,to_unzip,template)
    if mtype == 2:
        to_memo = task.select_sp_emb()

    #===start pre-processing and train
    if mtype == 1:
        task.train1()
    if mtype == 2:
        task.train2()
        os.system(f'cp ./temp/train_set/embed/{to_memo} ./temp/gen/key.pt')
    path =main_dir+'/temp/gen/train_fin'
    if os.path.exists(path):
        pass
    else:
        DB.error_report(ajob,'training not correctly finished')
        sendMsg = f"{send_key}/1/train_not_finished"
        getReturn(sendMsg,rurl)
        sys.exit(0)


    #===generate example wave files
    example_path = example_path+f'{ajob[2]}_{str(ajob[1])}/'
    os.makedirs(example_path, exist_ok=True)
    try:
        if mtype == 1:
            task.generate1(example_path)
        if mtype == 2:
            task.generate2(example_path)
    except:
        DB.error_report(ajob,'generate example waves failed')
        sendMsg = f"{send_key}/1/generate_example_failed"
        getReturn(sendMsg,rurl)

    #=== update user_model and train_request in DB
    DB.rollback()
    DB.update_db(ajob)

    #===get model id and return to php
    model_key = DB.get_usermodel_id(ajob)
    sendMsg = f"{send_key}/{model_key}/0/ok"
    getReturn(sendMsg,rurl)

    #===move files and reset environment
    if mtype == 1:
        task.finish1(ajob,model_key,model_path)
    if mtype == 2:
        task.finish2(ajob,model_key,model_path)
    time.sleep(1)
