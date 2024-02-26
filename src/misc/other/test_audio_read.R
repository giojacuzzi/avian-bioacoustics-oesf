library(seewave)
library(av)
library(sound)
library(audio)
library(signal)

library(microbenchmark)

path = normalizePath('~/Desktop/oesf-examples/03.wav', mustWork=T) # 1 hour long file

# Read metadata with av
audio_info = av_media_info(path)$audio
duration = av_media_info(path)$duration
metadata = list(
  channels    = audio_info$channels,
  sample_rate = audio_info$sample_rate,
  codec       = audio_info$codec,
  bitrate     = audio_info$bitrate,
  duration    = duration,
  samples     = duration * audio_info$sample_rate
)

# Read time domain

results <- microbenchmark(
  wav_tuneR = readWave(path),
  wav_av = read_audio_bin(path),
  wav_sound = loadSample(path),
  times = 10
)
plot(results)

# Read frequency domain
wl = 512

# results <- microbenchmark(
#   fft_av = read_audio_fft(
#     audio = path,
#     window = hanning(wl),
#     overlap = 0.5
#   ),
#   fft_seewave = spectro(
#     wave = readWave(path),
#     wl = wl,
#     wn = 'hanning',
#     ovlp = 0.5,
#     plot = F
#   ),
#   times = 10
# )
# plot(results)

path = normalizePath('~/Desktop/bewicks-wren.wav', mustWork=T) # 2 sec file

fft_av = read_audio_fft(
  audio = path,
  window = hanning(wl),
  overlap = 0.5
)

fft_seewave = spectro(
  wave = readWave(path),
  wl = wl,
  wn = 'hanning',
  ovlp = 0.5,
  plot = F
)

dim(fft_av)
dim(fft_seewave$amp)

head(fft_av)
head(fft_seewave$amp)

## STFT
wave = readWave(path)
n <- nrow(wave)
step <- seq(1,n+1-wl,wl-(ovlp*wl/100)) # +1 added @ 2017-04-20
z <- stdft(wave=wave,f=f,wl=wl,zp=zp,step=step,wn=wn,fftw=fftw,scale=norm,complex=complex,correction=correction)

plot(fft_av)
