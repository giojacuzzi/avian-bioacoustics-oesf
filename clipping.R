# Check for clipping
file = '~/../../Volumes/SAFS Work/DNR/test/MATURE/SMA00351_20210502_130002.wav'

library(tuneR)
sound = readWave(file)

if (typeof(sound@left) == 'integer') {
  message('int!')
} else {
  message('not!')
}


source('helpers.R')
if (dc_offset(sound@left)) sound@left = dc_correct(sound@left) # remove DC offset
# 
# max(abs(sound@left))


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
