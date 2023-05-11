## Batch process acoustic indices from a list of .wav files with optimized acidx_ functions
library(parallel)
library(tuneR)
library(soundecology)
library(seewave)

# input_files     - a list of paths to .wav files
# output_path     - path to the output directory
# output_file     - name of the generated output file (without .csv extension)
# alpha_indices   - indices to calculate, e.g. c('BIO','ADI')
# time_interval   - sampling interval length per calculation (sec)
# ncores          - number of cores to use
# dc_correct      - correct dc bias offset prior to index calculation
# digits          - number of decimal places for results
# diagnostics     - generate an additional file with information about the input file(s)
#
# This process will generate <output_file>.csv with the following fields:
# File      - the path and name of the input file
# Start     - the start time of the time interval processed
# End       - the end time of the time interval processed (if no interval specified, then this is the entire length of the file)
# ACI, etc. - the calculated values for each associated acoustic index
#
# It will also generate <output_file>_diagnostics.csv, if specified:
# SamplingRate - the sampling rate of the file (Hz)
# BitDepth     - the bit depth of the file
# Duration     - the duration of the file (sec)
# DcBias       - the mean DC bias amplitude of the file (before applying any DC correction)
# Clipping     - whether any clipping was detected
# 
# NOTE: only mono .wav support
# NOTE: ACI will return NA for long duration files (perhaps >= 10 min + @ 32kHz),
# this is an issue with the 'soundecology' package
batch_process_acidx = function(
    input_files, output_path, output_file = 'batch', alpha_indices = c(), time_interval = NA, ncores = 1, dc_correct = T, digits = 6, diagnostics = T, ...) {
  
  # force(db_threshold)
  alpha_indices = sort(alpha_indices)
  if (length(alpha_indices) == 0)
    stop('Specify at least one alpha acoustic index')
  if (!all(alpha_indices %in% c('ACI','ADI','AEI','BIO','H','M','NDSI')))
    stop('Unsupported index specified')

  message(' Starting batch process (', paste(alpha_indices, collapse=','),')')
    
  output_header = paste(c('File','Start','End',alpha_indices), collapse = ',')
  diagnostics_header = c('File,SamplingRate,BitDepth,Duration,DcBias,Clipping')
  
  ## Internal function to calculate indices
  # Returns a list where element 1 is the acoustic index result data in csv form,
  # and element 2 is the diagnostic data (if requested)
  calculate_alpha_indices = function(file, clustered = F, ...) {
    source('helpers.R')
    
    if (clustered) { # if launched in cluster, require package for each node
      require(soundecology)
      require(seewave)
      require(tuneR)
    }
    
    # Get args and set params
    args = list(...)
    if (!is.null(args[['min_freq']])) {
      min_freq = args[['min_freq']]
    } else {
      min_freq = formals(bioacoustic_index)$min_freq # BIO
      # min_freq = formals(acoustic_complexity)$max_freq # ACI
    }
    if(!is.null(args[['max_freq']])) {
      max_freq = args[['max_freq']]
    } else {
      max_freq = formals(bioacoustic_index)$max_freq # BIO
      # max_freq = formals(acoustic_diversity)$max_freq # ADI
      # max_freq = formals(acoustic_evenness)$max_freq # AEI
      # max_freq = formals(acoustic_complexity)$max_freq # ACI
    }
    if (!is.null(args[['fft_w']])) {
      fft_w = args[['fft_w']]
    } else {
      fft_w = formals(bioacoustic_index)$fft_w # BIO
      # fft_w = formals(acoustic_complexity)$fft_w # ACI
      # fft_w = formals(ndsi)$fft_w # NDSI
    }
    if (!is.null(args[['wl']])) { # TODO: combine with fft_w above?
      wl = args[['wl']]
    } else {
      wl = formals(H)$wl # H
    }
    if (!is.null(args[['db_threshold']])) {
      db_threshold = args[['db_threshold']]
    } else {
      db_threshold = formals(acoustic_diversity)$db_threshold # ADI
      # db_threshold = formals(acoustic_evenness)$db_threshold # AEI
      db_threshold = as.numeric(paste(db_threshold, collapse = ''))
    }
    if(!is.null(args[['freq_step']])) {
      freq_step = args[['freq_step']]
    }else{
      freq_step = formals(acoustic_diversity)$freq_step # ADI
      # freq_step = formals(acoustic_evenness)$freq_step # AEI
    }
    if(!is.null(args[['j']])) {
      j = args[['j']]
    } else {
      j = formals(acoustic_complexity)$j # ACI
    }
    if(!is.null(args[['anthro_min']])) {
      anthro_min = args[['anthro_min']]
    }else{
      anthro_min = formals(ndsi)$anthro_min # NDSI
    }
    if(!is.null(args[['anthro_max']])) {
      anthro_max = args[['anthro_max']]
    }else{
      anthro_max = formals(ndsi)$anthro_max # NDSI
    }
    if(!is.null(args[['bio_min']])) {
      bio_min = args[['bio_min']]
    }else{
      bio_min = formals(ndsi)$bio_min # NDSI
    }
    if(!is.null(args[['bio_max']])) {
      bio_max = args[['bio_max']]
    }else{
      bio_max = formals(ndsi)$bio_max # NDSI
    }
    
    # Read data from file
    wav = readWave(file)
    duration = length(wav@left)/wav@samp.rate
    if (is.na(time_interval)) {
      time_interval = duration
    }
    
    # Calculate file diagnostics
    if (diagnostics) {
      results_diagnostics = paste0(
        '\n', file,',',                   # File (including path)
        wav@samp.rate,',',                # SamplingRate (Hz)
        wav@bit,',',                      # BitDepth
        duration,',',                     # Duration (sec)
        round(mean(wav@left),digits),',', # DcBias (mean amplitude value)
        clipping(wav)                     # Clipping (boolean)
      )
    } else {
      results_diagnostics = NA
    }
    
    # Correct DC offset bias
    if (dc_correct) wav@left = dc_correct(wav@left)
    
    # Calculate requested indices at each time interval
    start = 0.0
    results = ''
    while (start < duration) {
      
      end = min(start + time_interval, duration)
      if (time_interval != duration) { # process interval
        wav_interval = cutw(wav, f=wav@samp.rate, from=start, to=end, output='Wave')
      } else { # process entire file
        wav_interval = wav
      }
      
      if ('ACI' %in% alpha_indices) { # Acoustic complexity index
        ACI = acoustic_complexity( # soundecology
          wav_interval,
          min_freq,
          max_freq,
          j,
          fft_w
        )
        ACI = ACI$AciTotAll_left
        # ACI = ACI( # seewave
        #   wave = wav_interval,
        #   wl = fft_w,
        #   flim = c(min_freq/1000, max_freq/1000),
        #   nbwindows = (duration/j) # number of windows
        # )
      } else { ACI = NA }
      if ('ADI' %in% alpha_indices) { # Acoustic diversity index
        ADI = acoustic_diversity(
          wav_interval,
          max_freq,
          db_threshold,
          freq_step
        )
        ADI = ADI$adi_left
      } else { ADI = NA }
      if ('AEI' %in% alpha_indices) { # Acoustic evenness index
        AEI = acoustic_evenness(
          wav_interval,
          max_freq,
          db_threshold,
          freq_step
        )
        AEI = AEI$aei_left
      } else { AEI = NA }
      if ('BIO' %in% alpha_indices) { # Bioacoustic index
        BIO = bioacoustic_index(
          wav_interval,
          min_freq,
          max_freq,
          fft_w
        )
        BIO = BIO$left_area
      } else { BIO = NA }
      if ('H' %in% alpha_indices) { # Acoustic entropy index
        H = H(
          wav_interval,
          wl
        )
      } else { H = NA }
      if ('M' %in% alpha_indices) { # Amplitude index
        M = M(
          wav_interval
        )
      } else { M = NA }
      if ('NDSI' %in% alpha_indices) { # Normalized difference soundscape index
        NDSI = ndsi(
          wav_interval,
          fft_w,
          anthro_min,
          anthro_max,
          bio_min,
          bio_max
        )
        NDSI = NDSI$ndsi_left
      } else { NDSI = NA }
      
      # Write results to file
      # TODO: only include columns for acidx that were requested
      results = paste0(
        paste0(results,'\n'),
        paste(
          c(
            file,              # File (including path)
            start,             # Start time (sec)
            end,               # End time (sec)
            na.omit(c(
              round(ACI,digits), # acoustic complexity index
              round(ADI,digits), # acoustic diversity index
              round(AEI,digits), # acoustic evenness index
              round(BIO,digits), # bioacoustic index
              round(H,digits),   # acoustic entropy index
              round(M,digits),   # amplitude index
              round(NDSI,digits) # normalized difference soundscape index
            ))
          ),
          collapse = ','
        )
      )
      start = end # move to next time interval
    }
    return(list('results' = results, 'diagnostics' = results_diagnostics))
  }
  
  no_input_files = length(input_files)
  if (ncores > no_input_files) {
    ncores = no_input_files
    warning(' Number of cores limited to number of files')
  }
  
  ## Start processing
  time_start = proc.time() # start timer
  output = paste0(output_path, output_file, '.csv', collapse = '')
  cat(output_header, file = output, append = F) # open results file
  if (diagnostics) {
    output_diagnostics = paste0(output_path, output_file, '_diagnostics.csv', collapse = '')
    cat(diagnostics_header, file = output_diagnostics, append = F) # open diagnostics file
  }
  
  if (ncores > 1) { # Process in parallel using clusters
    
    message(paste0(' Processing ', no_input_files, ' files in parallel using ', ncores, ' cores'))
    
    cluster = makeCluster(ncores, type = 'PSOCK')
    results = parLapply(cluster, input_files, calculate_alpha_indices, clustered = T, time_interval, ...)

    # Write results to file(s)
    write.table(paste(unlist(sapply(results,"[[",1)), collapse = ''),
                file = output, append = T, quote = F, col.names = F, row.names = F)
    if (diagnostics) {
      write.table(paste(unlist(sapply(results,"[[",2)), collapse = ''),
                  file = output_diagnostics, append = T, quote = F, col.names = F, row.names = F)
    }
    Sys.sleep(1) # pause to allow all to end
    
    stopCluster(cluster)
    
  } else { # Process in series on the main thread
    
    message(paste0(' Processing ', no_input_files, ' file(s) in series using 1 core'))
    for (file in input_files) {
      results = calculate_alpha_indices(file, ...)

      # Write results to file(s)
      cat(results$results, file = output, append = T)
      if (diagnostics) cat(results$diagnostics, file = output_diagnostics, append = T)
    }
  }
  
  time_end = proc.time() - time_start # stop timer
  message(paste0(' Created ', output))
  if (diagnostics) message(paste0(' Created ', output_diagnostics))
  message(paste0(' Total time: ', round(time_end['elapsed'], 2), ' sec'))
}

# EXAMPLE #########
# Params
path = '~/../../Volumes/SAFS Work/DNR/test/subset'
input_files = list.files(path=path, pattern='*.wav', full.names=T, recursive=F)
output_path = '~/../../Volumes/SAFS Work/DNR/test/subset/output/'
alpha_indices = c('ADI','BIO', 'ACI')
# Series
# batch_process_alpha_indices(input_files, output_path, alpha_indices = alpha_indices, time_interval = 60*2, ncores = 1)
# Parallel
batch_process_acidx(input_files, output_path, alpha_indices = alpha_indices, time_interval = 60*2, ncores = 2, min_freq = 2000, max_freq = 8000, db_threshold = -40)
# Other
# batch_process_alpha_indices(input_files, output_path, alpha_indices = c('BIO','AEI'), ncores = 3, min_freq = 200, max_freq = 2000, db_threshold = -45)
