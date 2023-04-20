library(tuneR)
library(seewave)
library(ggplot2)

dc_offset = function(s) {
  return(all.equal(mean(s), 0) != T)
}

dc_correct = function(s) {
  if (typeof(s) == 'integer') {
    return(s - round(mean(s)))
  } else {
    return(s - mean(s))
  }
}

clipping = function(wave) {
  sample.data = sound@left
  if (wave@pcm) {
    if (wave@bit == 8)              
      return( (max(sample.data) >= 255) || (min(sample.data) <= 0) )
    if (wave@bit == 16)
      return( (max(sample.data) >= 32767) || (min(sample.data) <= -32768) )
    if (wave@bit == 24)
      return( (max(sample.data) >= 8388607) || (min(sample.data) <= -8388608) )
    if (wave@bit == 32)
      return( (max(sample.data) >= 2147483647) || (min(sample.data) <= -2147483648) )
  } else {                                                                                    
    return( (max(sample.data) >= 1.0) || (min(sample.data) <= -1.0) )
  }
}

spectrogram = function(wav, tlim, flim, alim, color, rem_dc_offset=T, interpolate=T, ...) {
  # STFT with seewave `spectro`
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
  plot = ggplot(data, aes(x=time, y=freq, z=amp)) +
    geom_raster(aes(fill=amp), interpolate=interpolate) +
    scale_fill_viridis_c('Amplitude (dB)', limits=alim, option=color, na.value='black') +
    xlab('Time (s)') + ylab('Frequency (kHz)')
  return(plot)
}

oscillogram = function(wav, rem_dc_offset=T) {
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

# TODO: 3D waterfall plots with rayshader?

## DEMO
file = '~/Desktop/oesf-examples/short/04_short.wav'
sound = readWave(file)
if (dc_offset(sound@left)) sound@left = dc_correct(sound@left)
spectrogram(sound, flim=c(0,8), alim=c(-40,0)) + theme_minimal()
oscillogram(sound) + theme_minimal()
