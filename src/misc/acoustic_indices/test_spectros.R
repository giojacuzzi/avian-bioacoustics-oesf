# ## first layer
# data(tico)
# v <- ggspectro(tico, ovlp=50)
# summary(v)
# ## using geom_tile ##
# v + geom_tile(aes(fill = amplitude)) + stat_contour()
# ## coordinates flip (interest?)
# v + geom_tile(aes(fill = amplitude)) + stat_contour() + coord_flip()
# ## using stat_contour ##
# # default (not nice at all)
# v + stat_contour(geom="polygon", aes(fill=..level..))
# # set up to 30 color levels with the argument bins
# (vv <- v + stat_contour(geom="polygon", aes(fill=..level..), bins=30))
# # change the limits of amplitude and NA values as transparent
# vv + scale_fill_continuous(name="Amplitude\n(dB)\n", limits=c(-30,0), na.value="transparent")
# # Black-and-white theme
# (vv + scale_fill_continuous(name="Amplitude\n(dB)\n", limits=c(-30,0),
#                             na.value="transparent", low="white", high="black") + theme_bw())
# # Other colour scale (close to spectro() default output)
# v + stat_contour(geom="polygon", aes(fill=..level..), bins=30) + scale_fill_gradientn(name="Amplitude\n(dB)\n", limits=c(-30,0),
#                        na.value="transparent", colours = spectro.colors(30))

library(ggplot2)
library(seewave)
library(soundecology)
library(tuneR)

# Load test data
file = '~/Desktop/oesf-examples/short/04_short.wav'

wav = readWave(file)
dur = length(wav)/wav@samp.rate
fs  = wav@samp.rate

has_dc_offset = function(wav) {
  return(all.equal(mean(wav@left), 0) != T)
}

if (has_dc_offset(wav)) {
  wav@left = wav@left - mean(wav@left)
}

nfft = 1024   # number of points to use for the fft
overlap = 128 # overlap (in points)

library(signal) # signal processing functions
library(oce) # image plotting functions and nice color maps
library(viridis)


window   = 512  # FFT window size

# Create spectrogram
spec = specgram(x = wav@left,
                n = nfft,
                Fs = fs,
                window = window,
                overlap = overlap
)

P = abs(spec$S) # Discard phase information
P = P/max(P)    # Normalize
P = 10*log10(P) # Convert to dB
t = spec$t      # Config time axis

# Plot spectrogram
layout(matrix(c(1,2), ncol=1), heights=c(3,1))
imagep(x = t,
       y = spec$f,
       z = t(P),
       ylim = c(0, 12000),
       zlim = c(-24, 0),
       col = magma,
       ylab = 'Frequency (Hz)',
       xlab = 'Time (s)',
       drawPalette = T,
       decimate = F
)


# # ggplot2 spectrogram
# wav2 = wav
# wav2@left = snd
# ggspectro(wav2, ovlp=50) +
#   geom_raster(aes(fill = amplitude), interpolate = T) +
#   scale_fill_viridis_c(limits = c(-50, 0), option = 'magma', na.value = 'black')
# 
# ggspectro(wav2, ovlp=50) +
#   geom_raster(aes(fill = amplitude), interpolate = T) +
#   scale_fill_gradientn(colors=spectro.colors(30))


# Custom functions ##############################################################
# wav - a tuneR Wave object
# timelim - time limits (sec)
# freqlim - frequency limits (kHz)
# amplim  - amplitude limits (dB)
spectrogram = function(wav, tlim, flim, alim, color, rem_dc_offset=T, interpolate=T, ...) {
  # Remove DC offset
  if (rem_dc_offset) wav@left = wav@left - mean(wav@left)
  # STFT with seewave `spectro`
  message('Computing spectrogram...')
  s = spectro(wav, plot=F, ...)
  data = data.frame(
    time = rep(s$time, each =nrow(s$amp)),
    freq = rep(s$freq, times=ncol(s$amp)),
    amp  = as.vector(s$amp)
  )
  # Apply limits
  if (missing(tlim)) tlim = c(min(s$time), max(s$time))
  data = data[data$time>=tlim[1] & data$time<=tlim[2],]
  
  if (missing(flim)) flim = c(min(s$freq), max(s$freq))
  data = data[data$freq>=flim[1] & data$freq<=flim[2],]
  
  if (missing(alim)) alim = c()
  
  if (missing(color)) color = 'viridis'
  
  # Plot
  message('Plotting spectrogram...')
  plot = ggplot(data, aes(x=time, y=freq, z=amp)) +
    geom_raster(aes(fill=amp), interpolate=interpolate) +
    scale_fill_viridis_c('Amplitude (dB)', limits=alim, option=color, na.value='black') +
    xlab('Time (s)') + ylab('Frequency (kHz)')
  return(plot)
}

oscillogram = function(wav, rem_dc_offset=T) {
  # Remove DC offset
  if (rem_dc_offset) wav@left = wav@left - mean(wav@left)
  # Prepare data
  data = data.frame(
    time=seq(1, length(wav@left)) / wav@samp.rate,
    amplitude=wav@left
  )
  # Plot
  plot = ggplot(data, aes(x=time, y=amplitude)) +
    geom_line() +
    scale_x_continuous() +
    xlab('Time (s)') + ylab('Amplitude')
  return(plot)
}

spectrogram(wav, flim=c(0,12), alim=c(-40,0), color='magma') + theme_minimal()

library(microbenchmark)
