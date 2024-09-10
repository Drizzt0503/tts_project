# TTS Service
這是一個基於VITS的中文text to speech(TTS)模型，附加一個bert模型來優化韻律的部分。
就像原始VITS，支援多人語音，但另外引用Speaker Verification任務的Ecapa模型產生的
speaker embedding，因此可以做zero shot的語音克隆。

分兩部分，一是TTS模型的推理調用，用python import package的形式。
二是用finetune模型方式來語音克隆的train_service.py檔。

### 安裝(ubuntu)


1. 拉下預設docker image 並啟動(或任意支援GPU的python環境)

(pytorch docker image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

example:
```
sudo docker run --gpus all -it --name tts_test --shm-size 63G -v /data2/:/data2/ -v /var/www/html/webuploader/:/var/www/html/webuploader/ -v /mnt/share/AI_service/tts_project/:/mnt/share/ nvcr.io/nvidia/pytorch:23.10-py3
```

2. git clone這個repo並

```
git clone https://gitlab.com/drizztmon/tts_project.git
```

3. 進入tts_project的資料夾,執行install

```
cd tts_project
sh install.sh
```

## TTS infer

單人語音模型: class TTS_Vits  
多人語音模型: class TTS_VitsM  

**initialize(device,model_path):** 
 
device: "cpu" or "cuda"  
model_path:預設路徑是/home/yuhang/tts_model/+輸入路徑  
&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;
可以輸入None，用於生成訓練服務example

**infer(fileName,input_text, pace):** 

fileName: 存檔路徑  
input_text: 要生成文檔  
pace: 語速，浮點數。(1.0是正常 0.8較快 1.2較慢) 

輸入文本要繁中，由OpenCC轉為簡中輸入(tw2s)

 
會return生成狀況{ 0, 'ok' }表示成功  
{1, 'not in dic'}表示有文檔不再字典裡




### sample code

```python
import tts_project
from tts_project.infer_service import TTS_Vits as vits
from tts_project.infer_service import TTS_VitsM as vitsm

if __name__ == '__main__':
    text="零件費用產生，服務人員將另行報價，請問您接受嗎?"
    tts=vitsm("cuda",'/vits_model.pth') 
    rmsg = tts.infer('./hello.wav',text,1.0)
    print(rmsg)
```


## TTS VoiceClone Training Service

下面內容，因為要跟前端跟DB合作，所以定義了許多環境參數，如果只是要語音克隆，
請忽略下面說明，finetune原本的推理模型就好。

### 主程式流程:

* init
* search for job
* construct dataset for training
* start pre-processing and training
* generate example wave files
* update DB
* get model id and return to php
* move files and reset environment

### 環境參數

示範音檔文本: example_text.txt

傳入壓縮檔路徑: f'/var/www/html/webuploader/uploadzip/{社區id}_{send_key}.zip'

回傳示範音檔路徑: f'/var/www/html/webuploader/tts_train_return/{社區id}_{send_key}'

訓練完成模型路徑: f'/data2/yuhang/tts_model/{社區id}/{user_id}/'

回傳url: http://192.168.77.4:8080/index.php/PostAiServer/CompleteVoiceModel/


### Functions

---
**class DB_interact():**

methods:{

**init(db_settings):**  
set db_settings, connection, cursor

**rollback():**  
connection rollback

**search_job():**  
find if job exist(generated = 0), fetch fist one by time order and return it. If there is no job, return None.

**make_template(self,temp_no):**  
return template from DB.  
temp_no: template number.

**update_db(self,db_job):**  
Update table train_request that the job has finished(generate=1).  
Insert an item into user_model.

**get_usermodel_id(self,db_job):**  
Return the model_id for current job.  

}

---

**class TTS_Task():**  
method:{  

**init:**  
*./temp/train_set* for dataset  
*./models/* for pre-trained model  
*./temp/gen* for generated files in train_service  

**prepare_dataset(job,zipfile_path,template):**  
From zipfile_path and template, create dataset for finetune

**select_sp_emb():**  
return name of longest text file's embedding  (xxx.pt)

**train1():**  
for vits

**train2():**  
for vitsm

**generate1(ex_path):**  
generate example for vits  
ex_path: path to put example wave files.

**generate2(ex_path):**  
generate example for vitsm  
ex_path: path to put example wave files.

**finish1(self,db_job,model_key,model_path):**  
Move files to corrsponding path and clear environment.(Vits)

**finish2(self,db_job,model_key,model_path):**  
Move files to corrsponding path and clear environment.(VitsM)

}

---

**getReturn(sendMsg,url):**  
return sendMsg to PHP server(url).

---


### 備註

訓練前處理會用os執行pre-processing.py  
訓練vits用os執行vits_train_edge.py  
訓練vits用os執行vitsm_train_edge.py  

示範音檔由example_text.txt裡的8個句子  

	謝謝大家的配合，有問題歡迎提出喔！
	總之人工整理作業還是會繼續，目前的計畫內容是暫時的，還會再有調整，再請留意後續的更新。
	我雖然喜歡畫畫，但卻不會看畫。
	哈囉各位大家好！今天又是美好的一天。
	產官學十幾年來的研究成果，都已經濃縮在這一份十幾頁的報告內了。
	根據網路行銷及大數據，本計畫在後續執行中，仍有更改推廣方針的可能。
	若參觀人潮未提升就要減少人力配置。
	一看就知道我拍照的技術比較好。
	
搭配三種語速0.8, 1.0, 1.2 生成24個音檔

Vits模型名字為{user_model_id}.pth  
VitsM模型名字為{user_model_id}.pth加上speaker embedding:{user_model_id}_emb.pt

外部template示範語句path:  
/var/www/html/webuploader/audio_temp

本地(AI server)template示範語句path:  
/home/yuhang/tts_dataset



# DB section
    db_settings={
                'host': 'localhost',
                'user': 'root',
                'password': '',
                'db':'tts'
                }





