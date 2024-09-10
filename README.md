# VITS_Taiwanese
這個分支要開發多語言TTS，目前可以同時產生中文，英文，台語三種語言。
bert韻律模型僅支援中文，所以拿掉，英文dataset來自VCTK，台語dataset僅使用單人資料集-
台灣媠聲。



## infer

多人多語言語音模型: class: vits_multilingual


**initialize(device,model_path,speaker_embedding):** 
 
device: "cpu" or "cuda"  

model_path: 使用模型路徑

speaker_embedding: 要使用的語者embedding路徑

**infer(fileName,input_text, pace, main_lang, token_lang):** 

fileName: 存檔路徑  

input_text: 要生成文檔  

pace: 語速，浮點數。(1.0是正常 0.8較快 1.2較慢) 

main_lang(python list): 0-英文，1-中文，2-台語 e.g. 中文[1]

token_lang(python list)：每一個輸入token的語言編號 e.g. [0,0,0,1,1,2,0]

<br>
<br>

台語輸入請用台羅並夾於符號|tw>中間 e.g |tw>gu5 le5 pe7 bo5 - puann3 - hang7 e7|tw>

先將text先送入handle_text function，可以得到輸入用textf跟token_lang

```
textf,token_lang=handle_text(text,main_lang)
```

### sample code

```python
import tts_project
from tts_project.infer_no_bert import vits_multilingual,handle_text

if __name__ == '__main__':
	main_lang = 1
	t_text="gu5 le5 pe7 bo5 - puann3 - hang7 e7"
	text=f"零件費用產生Trump,|tw>{t_text}|tw>服務人員hello kitty將另行報價,請問您遠東A棟B棟C棟接受嗎?"

	textf,lid_list=handle_text(text,main_lang)
	m_path='./tts_dev/temp/G_500000.pth'
	emb_path='./tts_dev/temp/SSB04070479.pt'

	tts=vits_multilingual("cuda",m_path,emb_path)    
	rmsg,b = tts.infer('./hello.wav',textf,1.0,[main_lang],lid_list)


```


## TTS VoiceClone

一樣可以用finetune的方式做多語言語音克隆

