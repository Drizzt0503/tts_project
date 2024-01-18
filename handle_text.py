# coding=utf-8
import os, argparse, torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class ttsInferService():
	def __init__(self, **kwargs):
		parser = argparse.ArgumentParser(description = "ttsInfer")
		parser.add_argument("--ram_path",  		default="/dev/shm/", 		type=str, help="")
		parser.add_argument("--save_path",  	default="./audio_path/", 	type=str, help="")
		parser.add_argument("--device", 		default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")

		self.args = parser.parse_args()
		self.arabic_num = ['0','1','2','3','4','5','6','7','8','9']
		self.chinese_num = ['零','一','二','三','四','五','六','七','八','九']
		self.splits = [',','.','!','?',' ','，','。','！','？','　','\n']
		self.para_passing(**kwargs)

		import re
		self.chinese_pattern = re.compile(r'[^\u4e00-\u9fa5^,^，^.^．^。^:^：]')

	def para_passing(self,  **kwargs):
		self.args = vars(self.args)
		for arg in kwargs :
			self.args[arg] = kwargs[arg]
		self.args = argparse.Namespace(**self.args)

	def arabic_transcrip(self, sentence):
		opt_sentence = ''
		for char in sentence :
			if char not in self.arabic_num :
				opt_sentence += char;
			else :
				index_char = self.arabic_num.index(char)
				opt_sentence += self.chinese_num[index_char]
		return opt_sentence
		
	def rm_En_Alphabet(self, sentence):
		sentence = chinese_pattern.sub('', sentence)

	def split_sentence(self, sentence, step) :
		start_val = 0
		end_val = start_val + step
		split_index = -1
		end_check = False
		sentences = []
		while True :
			if end_val > len(sentence) :
				# 判斷是否超出索引
				split_index = len(sentence)-1
				end_check = True
			else :
				# 找最接近段落結尾的標點符號
				for val in self.splits :
					final_val = sentence.find(val, start_val, end_val)
					if (split_index == -1 and final_val != -1) :
						split_index = final_val
					elif (split_index != -1 and final_val != -1) :
						if final_val > split_index :
							split_index = final_val
			sentences.append(sentence[start_val: split_index+1])
			start_val = split_index+1
			end_val = start_val + step

			if end_check == True :
				# 是否結束
				break
		return sentences

	def generate(self, audio_file, model_type, model_file, pace, sentences):
		# 判斷使用的模型種類
		if model_type == 1 :
			from infer_lib import TTS_Vits
			tts_serv=TTS_Vits(self.args.device, model_file)
		elif model_type == 2 :
			from VitsM import TTS_VitsM
			tts_serv=TTS_VitsM(self.args.device, model_file)

		# 數字轉錄
		sentences = self.arabic_transcrip(sentences)

		# 短文本直接生成
		if len(sentences) < 400 :
			tts_serv.infer(self.args.ram_path+audio_file+'.wav', sentences, float(pace)*0.1)
			os.system(f"ffmpeg -i {self.args.ram_path+audio_file}.wav -vn -ar 16000 -ac 1 {self.args.save_path+audio_file}.mp3 -y -loglevel quiet")

		# 長文本分段生成
		else :
			import datetime
			date = datetime.datetime.now().strftime("%y%m%d%H%M%S%f")
			
			split_sentences = self.split_sentence(sentences, 400)
			count = 0
			for sentence in split_sentences :
				# 生成
				fileName = self.args.ram_path+date+str(count)+'.wav'
				tts_serv.infer(fileName, sentence, float(pace)*0.1)

				# 生成列表
				with open(self.args.ram_path+date+'.txt', 'a+') as f:
					f.write("file '"+fileName+"'"+'\r\n')
				count += 1

			# 多音檔合併
			os.system(f'ffmpeg -f concat -safe 0 -i {self.args.ram_path+date}.txt {self.args.ram_path+date}.wav -y -loglevel quiet')
			os.system(f'ffmpeg -f concat -safe 0 -i {self.args.ram_path+date}.txt {self.args.ram_path+date}.wav -y -loglevel quiet')
			os.system(f"ffmpeg -i {self.args.ram_path+date}.wav -vn -ar 16000 -ac 1 {self.args.save_path+audio_file}.mp3 -y -loglevel quiet")
			
			# 清除緩存
			os.system(f"rm {self.args.ram_path+date}.wav")
			os.system(f"rm {self.args.ram_path+date}.txt")
			for i in range(0, count):
				os.system(f'rm {self.args.ram_path+date+str(i)}.wav')

if __name__ == "__main__":
	ttsInfer = ttsInferService()
	sen = '''當你搞懂後就會明白了。我們要從本質思考，從根本解決問題。施企巴喬夫在過去曾經講過，愛情是一本永恆的書，有人只是信手拈來瀏覽過幾個片斷。有人卻流連忘返，為它灑下熱淚斑斑。但願諸位理解後能從中有所成長。現在，正視沒吧的問題，是非常非常重要的。因為，儘管如此，我們仍然需要對沒吧保持懷疑的態度。話雖如此，我們卻也不能夠這麼篤定。沒吧必定會成為未來世界的新標準。李四光說過一句著名的話，科學尊重事實，不能胡亂編造理由來附會一部學說。這段話看似複雜，其中的邏輯思路卻清晰可見。其實，若思緒夠清晰，那麼沒吧也就不那麼複雜了。列寧曾提出，誰不會休息，誰就不會工作。這句話幾乎解讀出了問題的根本。華羅庚在不經意間這樣說過，科學是老老實實的學問，搞科學研究工作就要採取老老實實、實事求是的態度，不能有半點虛假浮誇。不知就不知，不懂就不懂，不懂的不要裝懂，而且還要追下去，不懂，不懂在什麼地方; 懂，懂在什麼地方。老老實實的態度，首先就是要紮紮實實地打好基礎。科學是踏實的學問，連貫性和系統性都很強，前面的東西沒有學好，後面的東西就上不去; 基礎沒有打好。搞尖端就比較困難。我們在工作中經常遇到一些問題解決不了，其中不少是由於基礎未打好所致。一個人在科學研究和其他工作上進步的快慢，往往和他的基礎有關。這句話把我們帶到了一個新的維度去思考這個問題。巴爾扎克說過一句經典的名言，人生有些關口非狠狠地鬥一下不可，不能為了混口飯吃而蹉跎了幸福。他會這麼說是有理由的。如果仔細思考沒吧，會發現其中蘊含的深遠意義。我以為我了解沒吧，但我真的了解沒吧嗎？仔細想想，我對沒吧的理解只是皮毛而已。我們都有個共識，若問題很困難，那就勢必不好解決。說到沒吧，你會想到什麼呢？博賓斯卡講過一段深奧的話，我們應該有恆心，尤其是要有自信心，必須相信自己是有能力的，而且要不惜任何代價把這種能力發揮出來。這把視野帶到了全新的高度。毛澤東說過一句富有哲理的話，江山如此多嬌，引無數英雄競折腰。這讓我對於看待這個問題的方法有了巨大的改變。沒吧的存在，令我無法停止對他的思考。我們都很清楚，這是個嚴謹的議題。'''
	ttsInfer.generate('tts_test', 2, '3f602da8acb143fabccc330e239c8f8b/122/22.pth', 10, sen)