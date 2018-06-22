
from pydub import AudioSegment
import scipy.io.wavfile
from scipy.signal import spectrogram, stft, istft
from matplotlib import pyplot as plt
import python_speech_features as psf
import keras
from keras.layers import Dense, Flatten, Input, Dropout, Conv1D, MaxPooling1D
import numpy as np

def f2m(freq): #frequency to melody scale
    return 1127*np.log(1+freq/700.)

XWINDOW = 3

mp3 = 'Stellardrone - Between The Rings - 01 To The Great Beyond.mp3'
#mp3 = '../13 - Laika.mp3'
rate = 0

def make_fft_training_data(mp3):
    global rate
    AudioSegment.from_mp3(mp3).export('temp.wav',format='wav')
    rate, audio = scipy.io.wavfile.read('temp.wav')
    stride = 256#int(0.01*rate)
    left_ = audio[:,0]
    print (max(left_), min(left_), np.mean(left_), len(left_))
    raw = np.array(left_/max(abs(left_)))
    f,t,Zxx = scipy.signal.stft(raw,nperseg=stride*2,fs=rate,padded=True)#nperseg=rate/100,padded=True
    #t2,x2 = scipy.signal.istft(Zxx.real + 1j*Zxx.imag,fs=rate)
    Zr,Zi = np.real(Zxx.T),np.imag(Zxx.T)
    z = np.append(Zr,Zi,axis=1)
    print(len(raw)/z.shape[0])

    yraw = np.append(raw,np.zeros(z.shape[0]*stride-raw.shape[0]))
    train_y = np.reshape(yraw,(z.shape[0],stride))/2 + 0.5

    train_x = np.expand_dims(z,axis=1)/np.abs(np.unwrap(z)).max()/1 + 0.5
    print(train_x.min(),np.min(train_y))
    print(train_x,train_y)
    print('Alright, here')
    print(np.mean(train_x),np.std(train_x))
    print(np.mean(train_y),np.std(train_y))
    return train_x, train_y



def make_mfcc_training_data(mp3):
    AudioSegment.from_mp3(mp3).export('temp.wav',format='wav')
    rate, audio = scipy.io.wavfile.read('temp.wav')
    stride = int(0.01*rate)
    left_ = audio[:,0]
    print (max(left_), min(left_), np.mean(left_))
    print('Running filters')
    #f,t,Zxx = scipy.signal.stft(left_/max(abs(left_)),fs=rate,padded=True)#nperseg=rate/100,padded=True
    #t2,x2 = scipy.signal.istft(Zxx.real + 1j*Zxx.imag,fs=rate)
    #scaled = np.int16((x2) * 65535)
    #scipy.io.wavfile.write('testfft.wav', 44100, scaled)

    #Zr,Zi = np.real(Zxx),np.imag(Zxx)
    #print( Zr.max(),Zi.max(),Zr.min(),Zi.min())
    #print( f,t,Zxx)
    #print(f.shape,t.shape,Zxx.shape)
    fbank = psf.base.fbank(left_,samplerate=rate,nfft=1103)
    #mfcc = psf.base.mfcc(left_,samplerate=rate,numcep=15,nfft=1103)
    all_x = np.append(fbank[0], np.expand_dims(fbank[1],axis=1),axis=1)
    train_y = np.zeros((fbank[0].shape[0],stride),dtype=np.float32)
    train_x = None

    #normalize data for return
    all_x = all_x + abs(all_x.min(axis=0))
    all_x = all_x / all_x.max(axis=0)
    #print (left_)
    left_ = (left_ + 32767.0)/65535.0
    print(left_)

    for a in range(fbank[0].shape[0]):
        #print(rate*a,int((a+ .01)*rate))
        train_y[a,:] = left_[stride*a:stride*(a+1)]
        this_x = np.zeros((1,XWINDOW,all_x.shape[1]))
        #print(a, this_x[0,:min(XWINDOW,a+1),:].shape, np.expand_dims(all_x[max(0,a-XWINDOW+1):a+1,:],axis=0).shape)
        this_x[0,:min(XWINDOW,a+1),:] = np.expand_dims(all_x[max(0,a-XWINDOW+1):a+1,:],axis=0)
        if train_x is None:
            train_x = this_x
        else:
            train_x = np.append(train_x, this_x,axis=0)
        #print (this_x.shape)
    print (train_x.shape)

    #train_y = train_y + abs(train_y.min())
    #train_y = train_y / train_y.max()
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


load_data = False

train_x, train_y = None, None
if load_data:
    train_x = np.load('train_x.npy')
    train_y = np.load('train_y.npy')
else:
    train_x, train_y = make_fft_training_data(mp3)
    print('training data generated!')

    np.save('train_x',train_x)
    np.save('train_y',train_y)
#gen = data_gen(train_x,train_y,batchsize=8)
#for bx,by in gen:
#    print (bx.shape,by.shape)

for x in range(train_x.shape[0]):
    z = train_x[x,:]
    t2,x2 = scipy.signal.istft(np.expand_dims(z[:257] + 1j*z[257:],axis=0),fs=rate,noverlap=0)
    print(t2,x2)


input = Input( shape = (1,514) )
print (input.shape)
print (train_y.shape)
#print(train_x[10000],train_y[10000])
#print(train_x[0],train_y[0])
#x = Flatten()(input)

#x = Conv1D(16, (3,), activation='relu')(input)
#x = MaxPooling1D((1,))(x)
x = Flatten()(input)
#x = Dense( 500, activation='relu' )(x)
#x = Dropout(0.25)(x)
x = Dense( 500, activation='relu' )(x)
#x = Dropout(0.25)(x)
out = Dense(256,activation='sigmoid', name='output')(x)
print(out.shape)

model = keras.models.Model( inputs = input, outputs = out)

opt = keras.optimizers.SGD(lr=0.01)
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

'''his = model.fit(x=train_x,
        y = train_y_all[:,0], class_weight=class_weight,
        batch_size=BATCHSIZE, epochs=20, validation_split=0.1,
        verbose=2, callbacks=[stop])'''

ind = np.array(range(train_x.shape[0]))
np.random.shuffle(ind)

BATCHSIZE = 2
his = model.fit(x=train_x[ind], y = train_y[ind],
        batch_size=BATCHSIZE, epochs=10, validation_split=0.1,
        verbose=1, callbacks=[])#stop])

out_y = model.predict(train_x)

out_y_flat = out_y.flatten()

print(np.mean(out_y),np.std(out_y))
#print(out_y_flat)
print(out_y[10000])

scaled = np.int16((out_y_flat-0.5) * 65535)
scipy.io.wavfile.write('test.wav', 44100, scaled)

#plt.pcolormesh(mfcc.T)
#plt.yscale('log')

#plt.show()
