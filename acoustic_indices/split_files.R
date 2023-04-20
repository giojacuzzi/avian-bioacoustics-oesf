library(tuneR)
source('helpers.R')
source('global.R')

path = '~/../../Volumes/SAFS Work/DNR/test/MATURE/'
files = list.files(path=path, pattern='.*.wav', full.names=T, recursive=F)

interval = 60*60  # new file length (sec)

for (file in files) {
  message('Splitting ', basename(file), '...')
  sound = readWave(file)
  if (dc_offset(sound@left)) sound@left = dc_correct(sound@left) # remove DC offset
  
  dur = length(sound@left) / sound@samp.rate
  
  # Split files by 'interval'
  start = 0.0
  while (start < dur) {
    end = min(start + interval, dur)
    message(start, ' to ', end)
    subsample = cutw(sound, f=sound@samp.rate, from=start, to=end, output='Wave')
    
    writeWave(subsample, filename = paste0(path,'_split/',
                                           get_serial_from_file_name(file), '_',
                                           get_hour_from_file_name(file),'_',
                                           start,'.wav'))
    
    start = end
  }
}

# Compute ACI for split files
multiple_sounds(directory = paste0(path, '_split/'), 
                resultfile = paste0(path, '_aci.csv'), 
                soundindex = 'acoustic_complexity',
                no_cores = 8,
                min_freq = 2000,
                max_freq = 8000,
                j = 5,
                fft_w = 512
)

# Plot ACI results
data = read.csv(paste0(path, '_aci.csv'))
ggplot(data, aes(x=(2*seq(1,nrow(data))), y=LEFT_CHANNEL)) + geom_line() + theme_minimal()
