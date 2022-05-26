import os
import wave
import pickle
import pyaudio
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture 
from listen import *
from time import time
import speech_recognition as sr  
import playsound
from gtts import gTTS


def extract_features(audio,rate):
       
	mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
	mfcc_feature = preprocessing.scale(mfcc_feature)
	#print(mfcc_feature)
	delta = calculate_delta(mfcc_feature)
	combined = np.hstack((mfcc_feature,delta)) 
	return combined

def calculate_delta(array):
   
    rows,cols = array.shape
    #print(rows)
    #print(cols)
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
              first =0
            else:
              first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def record_audio_test():

	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 44100
	CHUNK = 512
	RECORD_SECONDS = 10
	device_index = 2
	audio = pyaudio.PyAudio()
	# print("----------------------record device list---------------------")
	# info = audio.get_host_api_info_by_index(0)
	# numdevices = info.get('deviceCount')
	# for i in range(0, numdevices):
	# 	if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0: print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
	# # print("-------------------------------------------------------------")
	index = 0#int(input())		
	#print("recording via index "+str(index))
	stream = audio.open(format=FORMAT, channels=CHANNELS,
	                rate=RATE, input=True,input_device_index = index,
	                frames_per_buffer=CHUNK)
	print ("recording started")
	Recordframes = []
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		Recordframes.append(data)
	print ("recording stopped")
	stream.stop_stream()
	stream.close()
	audio.terminate()
	OUTPUT_FILENAME="sample.wav"
	WAVE_OUTPUT_FILENAME=os.path.join("testing_set",OUTPUT_FILENAME)
	trainedfilelist = open("testing_set_addition.txt", 'a')
	trainedfilelist.write(OUTPUT_FILENAME+"\n")
	waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	waveFile.setnchannels(CHANNELS)
	waveFile.setsampwidth(audio.get_sample_size(FORMAT))
	waveFile.setframerate(RATE)
	waveFile.writeframes(b''.join(Recordframes))
	waveFile.close()

def test_model():

	source = "C:\\Users\\marka\\Desktop\\Hackathon project\\Speaker-Identification\\testing_set\\"  
	modelpath = "C:\\Users\\marka\\Desktop\\Hackathon project\\Speaker-Identification\\"
	test_file = "C:\\Users\\marka\\Desktop\\Hackathon project\\Speaker-Identification\\testing_set_addition.txt"       
	file_paths = open(test_file,'r')
	

	gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
	#gmm_files = ["C:\\Users\\marka\\Desktop\\Programming\\Hackathon project\\trained_modelsMark.gmm","C:\\Users\\marka\\Desktop\\Programming\\Hackathon project\\trained_modelsNeha.gmm","C:\\Users\\marka\\Desktop\\Programming\\Hackathon project\\trained_modelsCrazy.gmm"]
	
	#Load the Gaussian gender Models
	models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
	speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]
	 
	# Read the test directory and get the list of test audio files 
	for path in file_paths:  
		path = path.strip()   
		# print(path)
		sr,audio = read(source + path)
		vector   = extract_features(audio,sr)
	     
		log_likelihood = np.zeros(len(models)) 
	    
		for i in range(len(models)):
			gmm    = models[i]  #checking with each model one by one
			scores = np.array(gmm.score(vector))
			log_likelihood[i] = scores.sum()
	     
		winner = np.argmax(log_likelihood)
		print("\tdetected as - ", speakers[winner])
		return(speakers[winner])
		# time.sleep(1.0)  
#choice=int(input("\n1.Record audio for training \n 2.Train Model \n 3.Record audio for testing \n 4.Test Model\n"))
# record_audio_test()
# test_model()