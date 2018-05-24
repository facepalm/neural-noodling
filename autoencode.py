
from pydub import AudioSegment
import scipy.io.wavfile
from scipy.signal import spectrogram, stft, istft
from matplotlib import pyplot as plt
import python_speech_features as psf
import keras
from keras.layers import Dense, Flatten, Input, Dropout
import numpy as np

def f2m(freq): #frequency to melody scale
    return 1127*np.log(1+freq/700.)

mp3 = 'Stellardrone - Between The Rings - 01 To The Great Beyond.mp3'
#mp3 = '../13 - Laika.mp3'

def make_mfcc_training_data(mp3):
    AudioSegment.from_mp3(mp3).export('temp.wav',format='wav')
    rate, audio = scipy.io.wavfile.read('temp.wav')
    stride = int(0.01*rate)
    left_ = audio[:,0]
    fbank = psf.base.fbank(left_,samplerate=rate,nfft=1103)
    mfcc = psf.base.mfcc(left_,samplerate=rate,numcep=15,nfft=1103)
    train_x = np.append(mfcc, np.expand_dims(fbank[1],axis=1),axis=1)
    train_y = np.zeros((mfcc.shape[0],stride),dtype=np.float32)

    for a in range(mfcc.shape[0]):
        #print(rate*a,int((a+ .01)*rate))
        train_y[a,:] = left_[stride*a:stride*(a+1)]

    #normalize data for return
    train_x = train_x + abs(train_x.min(axis=0))
    train_x = train_x / train_x.max(axis=0)
    train_y = train_y + abs(train_y.min())
    train_y = train_y / train_y.max()
    print(train_x[10000],train_y[10000])
    return train_x, train_y

def data_gen(train_x,train_y, batchsize=8):
    while True:
        batch_x = np.zeros((batchsize,3,train_x.shape[1]))
        batch_y = np.zeros((batchsize,train_y.shape[1]))
        for es in range(train_x.shape[0]):


            for i in range(3):
                batch_x[es % batchsize,i,:] = train_x[es - i]
            batch_y[es % batchsize,:] = train_y[es,:]
            if es % batchsize == batchsize-1:
                yield batch_x, batch_y
                batch_x = np.zeros((batchsize,3,train_x.shape[1]))
                batch_y = np.zeros((batchsize,train_y.shape[1]))


train_x, train_y = make_mfcc_training_data(mp3)
gen = data_gen(train_x,train_y,batchsize=8)
for bx,by in gen:
    print (bx.shape,by.shape)

input = Input( shape = (16,) )
print (input.shape)
#x = Flatten()(input)
x = Dense( 1000, activation='relu' )(input)
x = Dropout(0.25)(x)
out = Dense(441,activation='sigmoid')(x)

model = keras.models.Model( inputs = input, outputs = out)

opt = keras.optimizers.SGD(lr=0.05)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

'''his = model.fit(x=train_x,
        y = train_y_all[:,0], class_weight=class_weight,
        batch_size=BATCHSIZE, epochs=20, validation_split=0.1,
        verbose=2, callbacks=[stop])'''

BATCHSIZE = 16
his = model.fit(x=train_x, y = train_y,
        batch_size=BATCHSIZE, epochs=100, validation_split=0.1,
        verbose=1, callbacks=[])#stop])


#plt.pcolormesh(mfcc.T)
#plt.yscale('log')

#plt.show()
