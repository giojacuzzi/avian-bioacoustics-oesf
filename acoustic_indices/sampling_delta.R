library(soundecology)
library(tuneR)
library(seewave)

# Load test data
# path = '~/../../Volumes/SAFS Work/DNR/2020/June30/SMA00310_20200630'
# files = list.files(path=path, pattern='SMA00310_20200702.*.wav', full.names=T, recursive=T)
file = '~/Desktop/oesf-examples/04.wav'

intervals = c(1,5,15,30,60*1,60*2,60*5,60*15,60*30,60*60) # index measurement interval (sec)

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
    
    # Compute bioacoustic index
    bio = bioacoustic_index(test)
    message(i, ' to ', j, ' sec:', bio$left_area)
    
    results = rbind(results, data.frame(
      delta = delta,
      start = i,
      end   = j,
      BIO   = bio$left_area
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
ggplot(results, aes(x=start, y=BIO, color=delta)) +
  geom_step(size=1) +
  scale_color_viridis_d() +
  labs(title = 'Effect of temporal sampling on bioacoustic index',
       subtitle = 'Calculated from a recording of the onset of the dawn chorus',
       x = 'Time (sec)', y = 'Bioacoustic Index (BIO)', color = 'Sample\nLength (sec)') +
  theme_minimal()
