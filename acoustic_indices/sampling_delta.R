library(soundecology)
library(tuneR)
library(seewave)

# Load test data
path = '~/../../Volumes/SAFS Work/DNR/2020/June30/SMA00310_20200630'
files = list.files(path=path, pattern='SMA00310_20200702.*.wav', full.names=T, recursive=T)

intervals = c(1,5,15,30,60*1,60*15,60*30,60*60) # index measurement interval (sec)

delta = intervals[5]

results = c()
for (file in files) {
  message('Loading ', basename(file))
  wav = readWave(file)
  
  dur = length(wav@left) / wav@samp.rate
  
  i = 0.0
  while (i < dur) {
    j = min(i + delta, dur)
    test = cutw(wav, f=wav@samp.rate, from=i, to=j, output='Wave')
    
    # Compute bioacoustic index
    bio = bioacoustic_index(test)
    message(i, ' ', bio$left_area)
    
    results = append(results, bio$left_area)
    i = j
  }
}