library(soundecology)
library(tuneR)
library(seewave)
source('helpers.R')

# Load test data
# path = '~/../../Volumes/SAFS Work/DNR/2020/June30/SMA00310_20200630'
# files = list.files(path=path, pattern='SMA00310_20200702.*.wav', full.names=T, recursive=T)
file = '~/Desktop/oesf-examples/04.wav'

intervals = c(15,60,60*2,60*10) # index measurement interval (sec)
# c(1,5,15,30,60*1,60*2,60*5,60*15,60*30,60*60)

# NOTE: ACI values will scale with length of measurement interval, so ensure they are normalized by the duration of the interval (most commonly in minutes)

results = data.frame()
for (delta in intervals) {
  message('Delta ', delta)
  sound = readWave(file)
  if (dc_offset(sound@left)) sound@left = dc_correct(sound@left)
  
  dur = length(sound@left) / sound@samp.rate
  
  i = 0.0
  while (i < dur) {
    j = min(i + delta, dur)
    test = cutw(sound, f=sound@samp.rate, from=i, to=j, output='Wave')
    
    # # Compute bioacoustic index
    bio = bioacoustic_index(test)
    index = bio$left_area
    
    # # Compute acoustic complexity index
    # aci = acoustic_complexity(test)
    # index = aci$AciTotAll_left / (delta/60) # normalize
    
    
    message(i, ' to ', j, ' sec:', index)
    
    results = rbind(results, data.frame(
      delta = delta,
      start = i,
      end   = j,
      index = index
    ))
    i = j
  }
}
results$delta = factor(results$delta)

# Add a duplicate data point for the last observation
# of each delta to complete geom_step lines below 
last_rows = aggregate(. ~ delta, results, tail, n = 1)
last_rows$start = last_rows$end
results = rbind(results, last_rows)

# Plot results per delta period
ggplot(results, aes(x=start, y=index, color=delta)) +
  geom_step(linewidth=1) +
  scale_color_viridis_d() +
  labs(title = 'Effect of temporal sampling on acoustic index',
       subtitle = 'Calculated from a recording of the onset of the dawn chorus',
       x = 'Time (sec)', y = 'Acoustic Index', color = 'Sample\nLength (sec)') +
  theme_minimal()
