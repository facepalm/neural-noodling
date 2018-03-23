from pydub import AudioSegment
import scipy.io.wavfile
from scipy.signal import spectrogram, stft, istft
from matplotlib import pyplot as plt

import numpy as np

def f2m(freq): #frequency to melody scale
    return 1127*np.log(1+freq/700.)

song = AudioSegment.from_mp3('Stellardrone - Between The Rings - 01 To The Great Beyond.mp3')
song.export('temp.wav',format='wav')

rate, audio = scipy.io.wavfile.read('temp.wav')


print rate, audio.shape
print audio.shape[0]/rate

left_ = audio[:,0]

#f, t, Sxx = scipy.signal.spectrogram(left_,rate/40.,nperseg=1024,noverlap=512,return_onesided=True)
f, t, Zxx = scipy.signal.stft(left_, rate, nperseg=14400)

print np.abs(Zxx)

#Zxx = np.where(np.abs(Zxx) >= .001, Zxx, 0)

t2, x = scipy.signal.istft(Zxx,rate)#,nperseg=2048)

scipy.io.wavfile.write('out.wav',rate,left_)

print Zxx.shape

print f, Zxx
print 'Max mel:',f2m(Zxx.shape[1])

plt.pcolormesh(t,f,np.abs(Zxx))
#plt.yscale('log')

plt.show()
