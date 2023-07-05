# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 23:21:54 2023

@author: hp
"""

import librosa
import pickle
import soundfile
import numpy as np
from pydub import AudioSegment
import streamlit as st

#loading the model
loaded_model=pickle.load(open('C:/Users/hp/Documents/ML_DEPLOY/trained_model-2.sav','rb'))



#Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    X = file_name.read(dtype="float32")
    sample_rate=file_name.samplerate
    if chroma:
        stft=np.abs(librosa.stft(X))
        result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft,sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel:
         mel=np.mean(librosa.feature.melspectrogram(y=X,sr=sample_rate).T,axis=0)
         result=np.hstack((result, mel))
    return result

#Emotions in the dataset
emotions={'01':'neutral','02':'calm','03':'happy','04':'sad','05':'angry','06':'fearful','07':'disgust','08':'surprised'}
#print("Emotions in the data set are : " , emotions)
#Observing emotions
observed_emotions=['angry', 'happy', 'neutral', 'sad']
#print("Emotions being observed are : " , observed_emotions)
#Load the data and extract features for each sound file
def load_data(name):
  t = []
  x = AudioSegment.from_file("C:/Users/hp/Downloads/archive/Actor_"+name.split("-")[6].split(".")[0]+"/"+name, format = "wav")
  audio = open("C:/Users/hp/Downloads/archive/Actor_"+name.split("-")[6].split(".")[0]+"/"+name,'rb')
  st.audio(audio, format='wav')
  mono_audios = x.split_to_mono()
  mono =mono_audios[0].export("C:/Users/hp/Downloads/archive/test1.wav",format="wav")
  h = soundfile.SoundFile(mono)
  feature=extract_feature(h, mfcc=True, chroma=True, mel=True)
  t.append(feature)
  return t


def main():
    st.title("Speech Emotion Recognition")
    file_upload = st.file_uploader("Please upload the audio file of 'WAV' format")
    #st.success(file_upload)
    #code for prediction
    
    #creating a button
    if st.button('Get Emotion'):
        value_x=load_data(file_upload.name)
        value_x=np.array(value_x)
        res=value_x.reshape(1, -1)
        ans=loaded_model.predict(res)
        st.subheader("emotion of the given audio is")
        if(ans[0]=='happy'):
            st.success(ans[0],icon="😄")
            st.success("Always be Happy")
        elif(ans[0]=='sad'):
            st.success(ans[0],icon="😥")
            st.success("Don't be sad, be Strong")
        elif(ans[0]=='angry'):
            st.success(ans[0],icon="😠")
            st.success("Don't get angry,Everything will be alright")
        else:
            st.success(ans[0],icon="😐")
            st.success("Don't get overthinked")
        
       
    
      
if __name__=='__main__':
    main()