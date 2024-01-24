'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN

class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, **kwargs):
		super(ECAPAModel, self).__init__()

		## ECAPA-TDNN
		# ECAPA_TDNN_mulit = nn.DataParallel(ECAPA_TDNN(C = C))
		# self.speaker_encoder = ECAPA_TDNN_mulit.cuda()

		self.speaker_encoder = ECAPA_TDNN(C = C).cuda()

		## Classifier
		# AAMsoftmax_mulit = nn.DataParallel(AAMsoftmax(n_class = n_class, m = m, s = s))
		# self.speaker_loss    = AAMsoftmax_mulit.cuda()

		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda()

		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		for num, (data, labels) in enumerate(loader, start = 1):
			self.zero_grad()
			labels            = torch.LongTensor(labels).cuda()
			speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug = True)
			nloss, prec       = self.speaker_loss.forward(speaker_embedding, labels)	

			# nloss.backward()
			nloss.sum().backward()

			self.optim.step()
			index += len(labels)
			top1 += prec

			# loss += nloss.detach().cpu().numpy()
			
			loss += nloss.detach().cpu().numpy().sum()
			print('================')
			print('index:',index,'\n')
			print('prec:',prec.shape,'\n')
			print('top1:',top1.shape,'\n')
			print('loss:',loss,'\n')
			print('================')

			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
			sys.stderr.flush()
		sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)

	def eval_network(self, eval_list, eval_path):
		self.eval()
		files = []
		embeddings = {}
		lines = open(eval_list).read().splitlines()
		for line in lines:
			files.append(line.split()[1])
			files.append(line.split()[2])
		setfiles = list(set(files))
		setfiles.sort()
		print('setfiles[0]',setfiles[0])
		print('setfiles[1]',setfiles[1])

		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _  = soundfile.read(os.path.join(eval_path, file))
			print('os.path.join(eval_path, file)',os.path.join(eval_path, file))
			print('audio.shape',audio.shape)
			# Full utterance
			data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()
			print('data_1',data_1)
			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			# 少於max_audio時補0
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			# 
			print('audio.shape[0]',audio.shape[0])
			print('max_audio', max_audio)
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			print('startframe',startframe.shape)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf)+max_audio])
			feats = numpy.stack(feats, axis = 0).astype(float)
			data_2 = torch.FloatTensor(feats).cuda()
			# Speaker embeddings
			with torch.no_grad():
				embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]
		scores, labels  = [], []

		for line in lines:
			embedding_11, embedding_12 = embeddings[line.split()[1]]
			embedding_21, embedding_22 = embeddings[line.split()[2]]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			score = (score_1 + score_2) / 2
			score = score.detach().cpu().numpy()
			scores.append(score)
			labels.append(int(line.split()[0]))
			
		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
		return EER, minDCF

	def spk_register(self, spk_name, spkIDs, files, spkidPath, audioPath) :
		zipped = zip(spkIDs,files)
		self.eval()

		# find register file
		if os.path.isfile(spkidPath+"spkID.pth"):
			print("got speakerID File")
			embeddings = torch.load(spkidPath+"spkID.pth")
		else:
			print("speakerID File not found")
			embeddings = {}

		# sound to embedding
		for spkID, file in zipped :

			audio, _  = soundfile.read(os.path.join(audioPath, file))
			data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()

			# Spliited utterance matrix vvvvv
			max_audio = 300 * 160 + 240 # ~= 16k*3+240 = 3sec
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf)+max_audio])
			feats = numpy.stack(feats, axis = 0).astype(float)
			data_2 = torch.FloatTensor(feats).cuda()
			# Spliited utterance matrix ^^^^^

			print(data_1.shape)
			# 多聲道檢查
			if (len(data_1.shape) > 2) :
				data_1 = data_1[:,:,1]
			if (len(data_2.shape) > 2) :
				data_2 = data_2[:,:,1]
			print(data_1.shape)
			# Speaker embeddings
			with torch.no_grad():
				embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			if embeddings.get(spkID) == None :
				embeddings[spkID] = [embedding_1, embedding_2]
			else :
				embeddings[spkID] = embeddings[spkID]+[embedding_1, embedding_2]
		# 儲存
		torch.save(embeddings, spkidPath+spk_name)

	def spk_eval(self, spkIDs, files, spkidPath, audioPath) :
		self.eval()

		start_val = False 
		# find register file
		if os.path.isfile(spkidPath+"spkID.pth"):
			print("got speakerID File")
			embeddings_spk = torch.load(spkidPath+"spkID.pth")
			start_val = True
		else:
			print("speakerID File not found")

		if start_val == True :
			score_total = 0
			count_total = 0
			# speaker embedding
			for file in files :
				audio, _  = soundfile.read(os.path.join(audioPath, file))
				data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()

				# Spliited utterance matrix vvvvv
				max_audio = 300 * 160 + 240
				if audio.shape[0] <= max_audio:
					shortage = max_audio - audio.shape[0]
					audio = numpy.pad(audio, (0, shortage), 'wrap')
				feats = []
				startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
				for asf in startframe:
					feats.append(audio[int(asf):int(asf)+max_audio])
				feats = numpy.stack(feats, axis = 0).astype(float)
				data_2 = torch.FloatTensor(feats).cuda()
				# Spliited utterance matrix ^^^^^

				# Speaker embeddings
				with torch.no_grad():
					embedding = self.speaker_encoder.forward(data_1, aug = False)
					embedding = F.normalize(embedding, p=2, dim=1)
					embedding_spl = self.speaker_encoder.forward(data_2, aug = False)
					embedding_spl = F.normalize(embedding_spl, p=2, dim=1)

				# Compute the scores
				count = 0
				score = 0
				for spkID in embeddings_spk[spkIDs[0]] :
					if count%2 == 0 :
						score_val = torch.mean(torch.matmul(spkID, embedding.T)) # higher is positive
					elif count%2 == 1 :
						score_val =  torch.mean(torch.matmul(spkID, embedding_spl.T))

					score = score + score_val
					count = count + 1
				print(file,'file score : ',(score / count).detach().cpu().numpy())
				score_total = score_total + ((score / count))
				count_total = count_total + 1

			similarity = score_total / count_total
			similarity = similarity.detach().cpu().numpy()
			print('avg score : ',similarity)
			return similarity

	def spk_meeting(self, spkIDs, file, spkidPath, audioPath) :
		self.eval()

		start_val = False 
		# find register file
		if os.path.isfile(spkidPath+"spkID.pth"):
			print("got speakerID File")
			embeddings_spk = torch.load(spkidPath+"spkID.pth")
			start_val = True
		else:
			print("speakerID File not found")

		if start_val == True :
			score_total = 0
			count_total = 0

			# speaker embedding
			audio, _  = soundfile.read(os.path.join(audioPath, file))
			print(len(audio))
			# 多通道防呆
			if (numpy.array(audio).shape[1] > 1):
				audio = audio[:,0]
			if (numpy.array(audio).shape[0] < 16000*4) :
				print('Audio less than four seconds')
			else :
				path = './opt.txt'
				f = open(path, 'w')
				fs = 16000
				split_sec = 0.1
				split_step = fs*split_sec
				split_win = 3
				split_val = int((len(audio)-(split_win*fs))/split_step)
				print(split_val)
				for i in range(0, split_val):
					spk_list = []
					audio_sp = audio[int(i*0.1*16000):int(i*0.1*16000+split_win*16000)]
					data_1 = torch.FloatTensor(numpy.stack([audio_sp],axis=0)).cuda()
					# Spliited utterance matrix vvvvv
					max_audio = 300 * 160 + 240
					if audio_sp.shape[0] <= max_audio:
						shortage = max_audio - audio_sp.shape[0]
						audio_sp = numpy.pad(audio_sp, (0, shortage), 'wrap')
					feats = []
					startframe = numpy.linspace(0, audio_sp.shape[0]-max_audio, num=5)
					for asf in startframe:
						feats.append(audio_sp[int(asf):int(asf)+max_audio])
					feats = numpy.stack(feats, axis = 0).astype(float)
					data_2 = torch.FloatTensor(feats).cuda()
					# Spliited utterance matrix ^^^^^

					# Speaker embeddings
					with torch.no_grad():
						embedding = self.speaker_encoder.forward(data_1, aug = False)
						embedding = F.normalize(embedding, p=2, dim=1)
						embedding_spl = self.speaker_encoder.forward(data_2, aug = False)
						embedding_spl = F.normalize(embedding_spl, p=2, dim=1)

					# Compute the scores
					for spkID in spkIDs :
						score_val = torch.mean(torch.matmul(embeddings_spk[spkID][0], embedding.T)) # higher is positive
						score_val = score_val + torch.mean(torch.matmul(embeddings_spk[spkID][1], embedding_spl.T))
						# print('time:',round(1.5+split_sec*i, 1),'|spkID:',spkID,'|score:',score_val/2)
						spk_list.append(round((score_val/2).detach().cpu().tolist(), 2))
					# print('time:',round(1.5+split_sec*i, 1),spkIDs[spk_list.index(max(spk_list))])
					if (i == 0):
						# print('time:',round(1.5+split_sec*i, 1),spkIDs[spk_list.index(max(spk_list))])
						# print('simi:',spk_list)
						f.write(str(round(1.5+split_sec*i, 1))+','+spkIDs[spk_list.index(max(spk_list))]+'\n')
						# f.write('['+str(int((1.5+split_sec*i)/60))+':'+str(int((1.5+split_sec*i)%60))+':'+str(int(((1.5+split_sec*i)%1)*100))+']'+\
						# 	spkIDs[spk_list.index(max(spk_list))]+'\n')
						# print('['+str(int((1.5+split_sec*i)/60))+':'+str(int((1.5+split_sec*i)%60))+':'+str(int(((1.5+split_sec*i)%1)*100))+']'+\
						# 	spkIDs[spk_list.index(max(spk_list))]+'\n')
						now_spk = spkIDs[spk_list.index(max(spk_list))]
					else :
						if (now_spk != spkIDs[spk_list.index(max(spk_list))]) :
							# print('time:',round(1.5+split_sec*i, 1),spkIDs[spk_list.index(max(spk_list))])
							# print('simi:',spk_list)
							f.write(str(round(1.5+split_sec*i, 1))+','+spkIDs[spk_list.index(max(spk_list))]+'\n')
							# f.write('['+str(int((1.5+split_sec*i)/60))+':'+str(int((1.5+split_sec*i)%60))+':'+str(int(((1.5+split_sec*i)%1)*100))+']'+\
							# 	spkIDs[spk_list.index(max(spk_list))]+'\n')
							# print('['+str(int((1.5+split_sec*i)/60))+':'+str(int((1.5+split_sec*i)%60))+':'+str(int(((1.5+split_sec*i)%1)*100))+']'+\
							# 	spkIDs[spk_list.index(max(spk_list))]+'\n')
							now_spk = spkIDs[spk_list.index(max(spk_list))]
						else :
							pass


	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)
