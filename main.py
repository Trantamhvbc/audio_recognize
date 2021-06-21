import librosa
import numpy as np
import pandas as pd

import librosa
import os
from MFCC import *
import  pickle

Mfccer = Mfcc()
forcast = pickle.load(open("KNNer.bin","rb"))



def read_audio(path):
    if os.path.isfile(path) == False:
        return {"code":"path is not file","data":None,"sr":None}
    signal,sr = librosa.load(path=path, sr=44100)
    if len(signal) / sr > 3:
        signal = signal[:3*sr]
    return {"code":"done","data":signal,"sr":sr}





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    while True:
        print("vui lòng nhập path của file âm thanh:")
        path = input()
        res = read_audio(path)
        if res["code"] != "done":
            print("path is not file")
            print("--------------->^<-------------")
            print('\n\n')
        else:
            signal = res["data"]
            sr = res["sr"]
            x = np.array(Mfccer.mfcc(signal,sr))
            x = np.array([np.mean(x, axis=1)])
            res = forcast.predict(x)[0]
            print(f"file vừa đưa vào là nhạc cụ : {res}")
            print("--------------->^<-------------")
            print('\n\n')
