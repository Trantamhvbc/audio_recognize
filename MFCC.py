
from scipy.io import wavfile
import numpy as np
import librosa

class Mfcc():
	def __int__(self):
		pass
	def pad(seft, signal,w = 2):
		res = []
		for i in range(w):
			res.append(signal[0])
		for i in signal:
			res.append(i)
		for i in range(w):
			res.append(signal[-1])
		return np.array(res)

	def create_paramater(self,w = 2):
		res = []
		sum_square = 0
		for i in range(-w,0):
			res.append(i*1.0)
			sum_square += i*i
		res.append(0.0)
		for i in range(1,w+1):
			res.append(i*1.0)
			sum_square += i*i
		return sum_square, np.array(res)

	def pad_zero(self, sig ,sr = 44100):
		thresh = int(0.7 * sr)
		length = sig.shape[0]
		res = sig.copy()
		while (length < thresh):
			res = np.append(res,0)
			length += 1
		return res

	def delta(self,  signal,w = 2):
		signal = self.pad(signal,w)
		sum_square, kernel = self.create_paramater(w = 2)
		i = w
		length = signal.shape[0]
		res = []
		while i + w < length :
			res.append( (signal[i-w: i + w + 1] * kernel).sum()/sum_square )
			i += 1
		return np.array(res)

	def read_audio(self,path):
		sig,sr = librosa.load(path, sr = 44100)
		return np.array(sig),sr

	def mfcc_10mms(self,signal, sr ):
		res = []
		mfcc = librosa.feature.mfcc(y=signal, sr=sr,n_mfcc = 12)
		for i in mfcc:
			res.append(i[0])
		energy = np.sum( signal.astype(float)**2 )
		res.append(energy)
		deltas_signal = self.delta(signal= signal , w = 2)
		delta_energy = np.sum( deltas_signal.astype(float)**2 )
		res.append(delta_energy)
		double_deltas_signal = self.delta(signal= deltas_signal , w = 2)
		double_delta_energy = np.sum( double_deltas_signal.astype(float)**2 )
		res.append(double_delta_energy)
		return res

	def mfcc(self,signal,sr):
		signal = self.pad_zero(signal)
		res = []
		length = signal.shape[0]
		step = sr//100
		i = step
		while i < length:
			res.append( self.mfcc_10mms(signal[i- step : i] , sr = sr)  )
			i += step
		return np.array(res).T