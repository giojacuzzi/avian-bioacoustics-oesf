## Batch process a list of .wav files, based on `multiple_sounds` from soundecology
library(parallel)
library(tuneR)
library(soundecology)

# Params
path = '~/../../Volumes/SAFS Work/DNR/test/subset'
wav_files = list.files(path=path, pattern='*.wav', full.names=T, recursive=F) # dir(path = directory, pattern = 'wav$', ignore.case = TRUE)
resultfile = '~/../../Volumes/SAFS Work/DNR/test/subset/output/results.csv'
soundindex = 'bioacoustic_index' #c('ndsi', 'acoustic_complexity', 'acoustic_diversity', 'acoustic_evenness', 'bioacoustic_index', 'H')

# NOTE: only mono .wav support
batch_process = function(
    wav_files, resultfile,
    soundindex = c('ndsi', 'acoustic_complexity', 'acoustic_diversity', 'acoustic_evenness', 'bioacoustic_index', 'H'),
    no_cores = 1, from = NA, to = NA, units = NA, ...) {

  if (soundindex == 'bioacoustic_index'){ # Bioacoustic index
    
    # fileheader = c('FILENAME,SAMPLINGRATE,BIT,DURATION,CHANNELS,INDEX,FFT_W,MIN_FREQ,MAX_FREQ,LEFT_CHANNEL,RIGHT_CHANNEL')
    fileheader <- c('File,SamplingRate,BitDepth,Duration,DcBias,Clipping,BIO')
    
    getindex = function(soundfile, inCluster = FALSE, ...) {
      source('helpers.R')
      
      if (inCluster == TRUE){ # if launched in cluster, require package for each node
        require(soundecology)
        require(tuneR)
        # require(parallel)
      }
      
      # Get args and set params
      args = list(...)
      if (!is.null(args[['min_freq']])) {
        min_freq = args[['min_freq']]
      } else { min_freq = formals(bioacoustic_index)$min_freq }
      if(!is.null(args[['max_freq']])) {
        max_freq = args[['max_freq']]
      } else { max_freq = formals(bioacoustic_index)$max_freq }
      if (!is.null(args[['fft_w']])) {
        fft_w = args[['fft_w']]
      } else { fft_w = formals(bioacoustic_index)$fft_w }
      
      soundfile_path = soundfile
      
      if (is.na(from) == FALSE){
        this_soundfile = readWave(soundfile_path, from = from, to = to, units = units)
      } else {
        this_soundfile = readWave(soundfile_path)
      }
      
      cat('BIO')
      return_list = bioacoustic_index(this_soundfile, ...)
      
      if (this_soundfile@pcm) {
        dc_bias = round(mean(this_soundfile@left))
      } else {                                                                                    
        dc_bias = mean(this_soundfile@left)
      }
      clipping = clipping(this_soundfile)
      
      return(paste0('\n', soundfile, ',',
                    this_soundfile@samp.rate, ',', # SamplingRate
                    this_soundfile@bit, ',',       # BitDepth
                    round(length(this_soundfile@left)/this_soundfile@samp.rate, 2), ',', # Duration (sec)
                    dc_bias, ',',         # DcBias (mean value)
                    clipping, ',',        # Clipping (boolean)
                    return_list$left_area # BIO (bioacoustic index value)
      ))
    }
  }
  
  ###############
  time0 <- proc.time() # start timer
  cat(fileheader, file = resultfile, append = FALSE) # open results file
  
  # Use parallel processing
  if (no_cores > 1){
    
    no_files = length(wav_files)
    if (no_cores > no_files) {
      no_cores = no_files
      message(' Number of cores limited to number of files')
    }
    message(paste0(' Processing ', no_files, ' files in parallel using ', no_cores, ' cores'))
    
    cl = makeCluster(no_cores, type = 'PSOCK')
    res = parLapply(cl, wav_files, getindex, inCluster = TRUE, ...)
    write.table(res, file = resultfile, append = TRUE, quote = FALSE, col.names = FALSE, row.names = FALSE)
    Sys.sleep(1) # pause to allow all to end
    
    stopCluster(cl)
  } else {
    
    message(paste0(' Processing ', length(wav_files), ' files in series using 1 core'))
    for (soundfile in wav_files){
      this_res <- getindex(soundfile, ...)
      cat(this_res, file = resultfile, append = TRUE)
    }
  }
  
  time1 <- proc.time() - time0 # stop timer
  message(paste0(' Created ', resultfile, '\n Total time: ', round(time1['elapsed'], 2), ' sec'))
}

# TEST
# Serial
batch_process(wav_files = wav_files, resultfile = resultfile, soundindex = soundindex, no_cores = 1)
# Parallel
batch_process(wav_files = wav_files, resultfile = resultfile, soundindex = soundindex, no_cores = 3)
#########

