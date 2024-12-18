# %% [markdown]
# # Signal Processing Project: real-time sound localisation

# %% [markdown]
# ## 1 Offline system
# ### 1.1 Data generation and dataset

# %%
import numpy as np
import matplotlib.pyplot as plt

#%%
def create_sine_wave(f, A, fs, N, temps=False):
    #A amplitud enormal
    #temps si veut récuprer le temps pour plot plus proprement
    tps = np.arange(1, N)/fs
    if temps:
        return A*np.sin(2*np.pi*f*tps), tps #converti bien radiant
    else:
        return A*np.sin(2*np.pi*f*tps)

# call and test your function here #
fs =44100
N = 8000
freq = 20
amplitude = 4

SinTest, temps = create_sine_wave(freq, amplitude, fs, N, temps=True)
plt.figure('Sinwave test')
plt.title('Sinus à 20Hz échantillonné sur 44100Hz')
plt.xlabel('temps (s)')
plt.ylabel('amplitude')
plt.plot(temps, SinTest)


# %%
from glob import glob
import scipy.io.wavfile as wf

def read_wavefile(path):
    
    return wf.read(path)

# call and test your function here #
LocateClaps = "LocateClaps/"
files = glob(f"{LocateClaps}/*.wav")

signal = read_wavefile(files[0])[1]

#test lecture
plt.figure('fichier M1_0.wav')
plt.title('fichier M1_0.wav')
plt.xlabel('échantillons')
plt.plot(signal)



# %% [markdown]
# ### 1.2 Buffering

# %%
from collections import deque

def create_ringbuffer(maxlen): #cree un deque de taille maxlen
    return deque(maxlen=maxlen)

# call and test your function here #
stride = 1000 #pour décaller la fenettre qu'on prend les données
maxlen = 750

buffer = create_ringbuffer(maxlen)

fs =44100
N = 8000
freq = 20
amplitude = 4
signal= create_sine_wave(freq, amplitude, fs, N)

# reading your signal as a stream:
for i, sample in enumerate(signal):
    buffer.append(sample)
    
    #biens avoir un i qui complte le nombre de sample fait car ainsi peut décaller exactement de combien on veut
    if i == maxlen:
        plt.figure(f'buffer à {maxlen}')
        plt.title(f'buffer pour {maxlen}')
        plt.xlabel('échantillons')
        plt.plot(buffer)
    # print décallé.
    if i == maxlen+stride:
        plt.figure(f'buffer à {maxlen+stride}')
        plt.title(f'buffer pour {maxlen+ stride}')
        plt.xlabel('échantillons')
        plt.plot(buffer)


# %% [markdown]
# ### 1.3 Pre-processing
# #### 1.3.1 Normalisation

# %%
def normalise(s):
    #s signal
    ValeurExtreme = np.abs(s).max()
    
    return s/ValeurExtreme

#verificaiton normalisation
# plt.figure(1)

# son = read_wavefile(files[0])[1]
# plt.plot(son)

# norm = normalise(son)
# plt.figure(2)
# plt.plot(norm)


# %% [markdown]
# #### 1.3.2 Downsampling

# %%
## 1 - spectral analysis via spectrogram
from scipy.signal import *
from zplane import zplane

sample, data = read_wavefile(files[0])
plt.specgram(data, Fs=sample )
plt.title("Spectrogram")
plt.show()

## 2 - Anti-aliasing filter synthesis
def create_filter_cheby(wp, ws, gpass, gstop, fs):
    
    N, omega = cheb1ord(wp, ws, gpass=gpass, gstop=gstop, fs=fs)
    B, A = cheby1(N,gpass, omega, fs=fs)

    return B, A

def create_filter_cauer(wp, ws, gpass, gstop, fs):

    N, omega = ellipord(wp, ws, gpass=gpass, gstop= gstop, fs=fs)
    
    
    B,A = ellip(N, gpass, gstop, omega, fs=fs)

    return B, A

## 3 - Decimation
def downsampling(sig, B, A, M):
    return lfilter(B.copy(), A.copy(), sig)[::M] #prend le filtre calcule et renvoie 1 itéré tous les M

fs=44100 #commence à 44100Hz c'est aprèq que réduit
N=8000
SignalATraite = create_sine_wave(8500, 1000, fs, N)+create_sine_wave(7500, 20, fs, N)
petitsin =create_sine_wave(7500, 20, fs, N)

plt.figure(' freqz signal originel')
e,f=freqz(SignalATraite[:900:3], worN=2048, fs=fs)
plt.plot(e,20*np.log10(f))
exit()
plt.figure('signal originel')

plt.plot(SignalATraite)



numCaueur, denCaueur = create_filter_cauer(8000,8500, 0.1, 70, fs=fs) #valeru encodé en F
numCheby, denCheby = create_filter_cheby(8000,8500, 0.1, 70, fs=fs)


M=1
fs=fs/M


plt.figure("caueur")
plt.title("caueur", fontsize=30)
plt.plot(downsampling(SignalATraite,numCaueur , denCaueur, M))


plt.figure('freqz caueur')
plt.title('spectre du filtre de Cauer', fontsize=20)
a,b =freqz(numCaueur.copy(),denCaueur.copy(), worN=8096, fs=fs)
plt.ylabel('amplitude (dB)')
plt.xlabel('fréquence (Hz)')
plt.plot(a, 20*np.log10(np.abs(b)))
plt.show()


plt.figure('freqz signal traité par caueur')
plt.title('spectre de fréquence du signal traité par Cauer' , fontsize=20)
e,f =freqz(downsampling(SignalATraite,numCaueur , denCaueur, M), worN=2048, fs=fs)
plt.plot(e, 20*np.log10(f))
plt.ylabel('amplitude (dB)')
plt.xlabel('fréquence (Hz)')

plt.figure('freqz signal downsamplé')
plt.title('spectre du signal downsamplé' , fontsize=20)
e,f =freqz(downsampling(SignalATraite,[1] , [1], M), worN=2048, fs=fs)
plt.plot(e, 20*np.log10(f))
plt.ylabel('amplitude (dB)')
plt.xlabel('fréquence (Hz)')

plt.figure('signal traité par Cauer')
plt.title('signal traité par Cauer', fontsize=20)
plt.plot(downsampling(SignalATraite,numCaueur , denCaueur, M))
plt.xlabel('échantillons')
plt.ylabel('amplitude')

plt.figure('freqz cheby')
plt.title('spectre du filtre de Chebychev', fontsize=20)
a,b =freqz(numCheby.copy(),denCheby.copy(), worN=8096, fs=fs)
plt.ylabel('amplitude (dB)')
plt.xlabel('fréquence (Hz)')
plt.plot(a, 20*np.log10(np.abs(b)))

plt.figure('freqz signal traité par cheby1')
plt.title('freqz signal traité par cheby1', fontsize=30)
e,f =freqz(downsampling(SignalATraite,numCheby , denCheby, M), worN=2048, fs=fs)
plt.plot(e, 20*np.log10(f))

plt.figure('signal traité par cheby1')
plt.title('signal traité par chebychev', fontsize=20)
plt.plot(downsampling(SignalATraite,numCheby , denCheby, M))
plt.xlabel('échantillons')
plt.ylabel('amplitude')

plt.figure('delai')
plt.title('délai en échantillons des filtres', fontsize=20)
t = np.zeros(100)
t[0]=1

plt.plot(lfilter(numCheby, denCheby, t), label="Chebychev")
plt.plot(lfilter(numCaueur , denCaueur, t), label="Cauer")
plt.xlabel('échantillons')
plt.legend(loc=1)




'''
normal que début pas OK car recursif et pas encore en régime
'''



# %% [markdown]
# ### 1.4 Cross-correlation

# %%
## 1.4
import scipy.signal as sc
import numpy as np

def fftxcorr(in1, in2):
    n1 = len(in1)
    n2 = len(in2)
    N=n1+n2-1
    expo =  int(np.ceil(np.log(N)/np.log(2)))
    N=2**expo
    signal1 = in1.copy()
    signal2 = in2.copy()
    
    b1 = np.fft.fft(signal1, N)
    b2 = np.fft.fft(signal2, N)
    b3=b1*np.conjugate(b2)

    fft =np.fft.ifft(b3, N) #calcule fft
    return np.fft.fftshift(fft) #décalle pour afficahge correcte
    
# call and test your function here #

sig = read_wavefile(files[0])[1]
sig2= read_wavefile(files[12])[1]

test_correlation = fftxcorr(sig, sig2)
plt.figure('test correlation')
plt.title('fonction fftxcorr')
plt.xlabel('échantillons')
plt.plot((abs(test_correlation)))   

plt.figure('test conv')
xcorr_fftconv = sc.fftconvolve(sig, sig2[::-1], 'full') # [::-1] flips the signal but you can also use np.flip()
plt.plot((abs(xcorr_fftconv)))
plt.title('fonction fftconvolve')
plt.xlabel('échantillons')
#fonctionne correctement
# %% [markdown]
# ### 1.5 Localisation
# #### 1.5.1 TDOA

# %%
def TDOA(xcorr, fs):
    indmax = np.argmax(xcorr) -len(xcorr)//2 #prend position du max
    
    temps = indmax/fs #converti en temps
    
    return temps

#test du code

sig = read_wavefile(files[0])[1]
sig2= read_wavefile(files[12])[1]

test_correlation = fftxcorr(sig, sig2)

print(TDOA(test_correlation, 44100)*44100)


# %% [markdown]
# #### 1.5.2 Equation system

# %%
from scipy.optimize import root

# mic coordinates in meters
MICS = [{'x': 0, 'y': 0.0487}, {'x': 0.0425, 'y': -0.025}, {'x': -0.0425, 'y': -0.025}] 

def equations(p, deltas):
    v = 343 #vitesse du son
    x, y = p
    alpha = np.arctan2((MICS[1]['y'] - MICS[0]['y']), (MICS[1]['x'] - MICS[0]['x']))
    beta = np.arctan2((MICS[2]['y'] - MICS[0]['y']), (MICS[2]['x'] - MICS[0]['x']))
    
    eq1 = v*deltas[0] - (np.sqrt((MICS[1]['x'] - MICS[0]['x'])**2 + (MICS[1]['y'] - MICS[0]['y'])**2) * np.sqrt((x)**2 + (y)**2) * np.cos(alpha-np.arctan2(y, x)))
    eq2 = v*deltas[1] - (np.sqrt((MICS[2]['x'] - MICS[0]['x'])**2 + (MICS[2]['y'] - MICS[0]['y'])**2) * np.sqrt((x)**2 + (y)**2) * np.cos(beta-np.arctan2(y, x)))
    return (eq1, eq2)
    
def localize_sound(deltas):
    sol = root(equations, [0, 0], (deltas), tol=10)
    return sol.x

def source_angle(coordinates):
    #trouver angle entre coord et axe x
    #renvoie angle en degré
    
    alpha = np.arccos(coordinates[0]/np.linalg.norm(coordinates))
    if coordinates[1]<0:
        alpha = 2*np.pi-alpha #pour prendre autre côté
    alpha = 180*alpha/np.pi
    return alpha

# call and test your function here #
def GetAngle(signal1, signal2, signal3, fs):
    corr12 = fftxcorr(signal1, signal2)
    corr13 = fftxcorr(signal1, signal3)
    
    delta = np.array([TDOA(corr12, fs), TDOA(corr13, fs)])
    
    pos = localize_sound(delta)
    
    
    
    alpha = source_angle(pos.copy())
    plt.plot(pos[0], pos[1], '*', label=f"a={alpha}", markersize=20)
    return alpha

# # plt.figure('test pts')
possible_angle = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
for angle in possible_angle:
    url1 = files[0].split('/')[-1].split('_')[0] +'_'+ str(angle)+'.wav' #pour avoir début url
    url2 = files[12].split('/')[-1].split('_')[0] +'_'+str(angle)+'.wav' #pour avoir début url
    url3 = files[24].split('/')[-1].split('_')[0] +'_'+str(angle)+'.wav' #pour avoir début url
    indexMic1 = files.index(url1)
    indexMic2 = files.index(url2)
    indexMic3 = files.index(url3)
    
    signal1 = read_wavefile(files[indexMic1])[1]
    signal2 = read_wavefile(files[indexMic2])[1]
    signal3 = read_wavefile(files[indexMic3])[1]
    fs=44100
    
    plt.figure("position des points")
    plt.title('position estimés des sources de bruit', fontsize=20)
    print(f"pour {angle}, on prédit {GetAngle(signal1, signal2, signal3, fs)}")
    plt.legend(loc=1, fontsize=20)
    plt.plot
    

# %% [markdown]
# ### 1.6 System accuracy and speed

# %%
## 1.6.1
def accuracy(pred_angle, gt_angle, threshold): #mets 15° car led tous les 30° donc peut se permettre ça max comme erreur pour que se trompe pas de led
    # your code here #
    diff = np.abs(pred_angle-gt_angle)
    if diff>=270:
        return diff<= 360-threshold
    return np.abs(pred_angle-gt_angle)<=threshold #renvoie si au seil près on est bon

# ## 1.6.2
possible_angle = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
Mic1=files[0].split('/')[-1].split('_')[0] #début url pour chaque
Mic2=files[12].split('/')[-1].split('_')[0]
Mic3=files[24].split('/')[-1].split('_')[0]
erreur=0



for angle in possible_angle:
    url1 = Mic1+'_'+ str(angle)+'.wav' #pour avoir début url
    url2 = Mic2+'_'+str(angle)+'.wav' #pour avoir début url
    url3 = Mic3 +'_'+str(angle)+'.wav' #pour avoir début url
    
    indexMic1 = files.index(url1)
    indexMic2 = files.index(url2)
    indexMic3 = files.index(url3)
    
    signal1 = read_wavefile(files[indexMic1])[1]
    signal2 = read_wavefile(files[indexMic2])[1]
    signal3 = read_wavefile(files[indexMic3])[1]
    fs=44100 #pas de downsampling
    
    #downsampling
    normaliser1 =normalise(signal1)
    normaliser2 =normalise(signal2)
    normaliser3 =normalise(signal3)
    
    M=3
    numCaueur, denCaueur = create_filter_cauer(8000,8500, 0.1, 70, fs=fs) #encore  44100 car pas encore fait down sampling
    fs=fs/3
    down1 = downsampling(normaliser1, numCaueur, denCaueur, M)
    down2 = downsampling(normaliser2, numCaueur, denCaueur, M)
    down3 = downsampling(normaliser3, numCaueur, denCaueur, M)
    
    
    
    AnglePredit = GetAngle(down1, down2, down3, fs)
    precision = accuracy(AnglePredit, angle, 15) #prend préicision de 15° car on a 12 led pour 360 ° donc 15° d'erreur d'affichage
    diff = np.abs(AnglePredit- angle)
    if diff>=270: #si 330 et trouvé genre 362
        diff = 360 - diff
    erreur = erreur + np.abs(AnglePredit- angle)
    print(f"pour {angle}, angle prédit {AnglePredit} précision : {precision}")
print(f"erreur moyenne {erreur/len(possible_angle)}")
            
# call and test your function here #

## 1.6.3
from time import time_ns, sleep

def fun_complet(debug):
    signaltemps1 = read_wavefile(files[0])[1]
    signaltemps2 = read_wavefile(files[12])[1]
    signaltemps3 = read_wavefile(files[24])[1]

    fs=44100

    normaliser1 =normalise(signaltemps1)
    normaliser2 =normalise(signaltemps2)
    normaliser3 =normalise(signaltemps3)


    numCaueur, denCaueur = create_filter_cauer(8000,8500, 0.1, 70, fs=fs) #encore  44100 car pas encore fait down sampling

    M=3
    down1 = downsampling(normaliser1, numCaueur, denCaueur, M)
    down2 = downsampling(normaliser2, numCaueur, denCaueur, M)
    down3 = downsampling(normaliser3, numCaueur, denCaueur, M)
    fs=44100//M #pour divier par combien d'élément on a garder

    #calcule corrélation
    corr12 = fftxcorr(down1, down2)
    corr13 = fftxcorr(down1, down3)

    tdoa12 = TDOA(corr12, fs)
    tdoa13 = TDOA(corr12, fs)

    delta = np.array([tdoa12, tdoa13])

    pos = localize_sound(delta)

    angle = source_angle(pos)
    print('fct tout en un')

def time_delay(func, args):
    start_time = time_ns()
    out = func(*args)    
    end_time = time_ns()
    print(f"{func.__name__} in {end_time - start_time} ns")
    return out

print('temps pour chaque fonction')
#prend pour l'angle 0
signaltemps1 = read_wavefile(files[0])[1]
signaltemps2 = read_wavefile(files[12])[1]
signaltemps3 = read_wavefile(files[24])[1]

fs=44100

normaliser1 =time_delay(normalise, [np.array(signaltemps1)])
normaliser2 =time_delay(normalise, [np.array(signaltemps2)])
normaliser3 =time_delay(normalise, [np.array(signaltemps3)])


numCaueur, denCaueur = create_filter_cauer(8000,8500, 0.1, 70, fs=fs) #encore  44100 car pas encore fait down sampling

M=3
down1 = time_delay(downsampling, [normaliser1, numCaueur, denCaueur, M])
down2 = time_delay(downsampling, [normaliser2, numCaueur, denCaueur, M])
down3 = time_delay(downsampling, [normaliser3, numCaueur, denCaueur, M])
fs=44100//M #pour divier par combien d'élément on a garder

#calcule corrélation
corr12 = time_delay(fftxcorr, [down1, down2])
corr13 = time_delay(fftxcorr, [down1, down3])

tdoa12 = time_delay(TDOA, [corr12, fs])
tdoa13 = time_delay(TDOA, [corr12, fs])

delta = np.array([tdoa12, tdoa13])

pos = time_delay(localize_sound,[delta])

angle = time_delay(source_angle, [pos])

print(f'angle = {angle}')

time_delay(fun_complet, [np.array([5])])


# call and test your previous functions here #

# %% [markdown]
# ## 2 Real-time localisation

# %% [markdown]
# ### 2.1 Research one Raspberry Pi application

# %% [markdown]
# ### 2.2 Data acquisition and processing

