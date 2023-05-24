## Batch process acoustic indices from a list of .wav files with optimized acidx_ functions
library(parallel)
library(tuneR)
library(soundecology)
library(seewave)
source('acoustic_indices/acidx.R')

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
# It will also generate (or append to) '_files_processed.csv', a table that lists
# all files that have successfully been processed. This table can be referenced
# to resume the batch processing of large datasets in chunks.
#
# It will also generate <output_file>_diagnostics.csv, if specified:
# SamplingRate - the sampling rate of the file (Hz)
# BitDepth     - the bit depth of the file
# Duration     - the duration of the file (sec)
# DcBias       - the mean DC bias amplitude of the file (before applying any DC correction)
# Clipping     - whether any clipping was detected
# 
# NOTE: only mono .wav support
#
# Returns the number of files that were successfully processed
batch_process_acidx = function(
    input_files, output_path, output_file = 'batch', alpha_indices = c(), wl = 512, time_interval = NA, ncores = 1, dc_correct = T, digits = 6, ...) {
  
  alpha_indices = sort(alpha_indices)
  if (length(alpha_indices) == 0)
    stop('Specify at least one alpha acoustic index')
  if (!all(alpha_indices %in% c('ACI','ADI','AEI','BIO','H','M','NDSI')))
    stop('Unsupported index specified')
  
  output_header = paste(c('File','Start','End',alpha_indices), collapse = ',')
  diagnostics_header = c('File,SamplingRate,BitDepth,Duration,DcBias,Clipping')
  
  ## Internal function to calculate indices
  # Returns a list where element 1 is the acoustic index result data in csv form,
  # and element 2 is the diagnostic data (if requested)
  calculate_alpha_indices = function(file, clustered = F, ...) {
    source('helpers.R')
    source('acoustic_indices/acidx.R')
    
    results = results_diagnostics = file_processed = NULL
    tryCatch({
      
      # If launched in a cluster, require package for each node
      if (clustered) {
        require(soundecology)
        require(seewave)
        require(tuneR)
      }
      
      # Get arguments and set parameters
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
      # if (!is.null(args[['fft_w']])) {
      #   fft_w = args[['fft_w']]
      # } else {
      #   fft_w = formals(bioacoustic_index)$fft_w # BIO
      #   # fft_w = formals(acoustic_complexity)$fft_w # ACI
      #   # fft_w = formals(ndsi)$fft_w # NDSI
      # }
      # if (!is.null(args[['wl']])) { # TODO: combine with fft_w above?
      #   wl = args[['wl']]
      # } else {
      #   wl = formals(H)$wl # H
      # }
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
      results_diagnostics = data.frame(
        File = file,
        SamplingRate = wav@samp.rate,
        BitDepth = wav@bit,
        Duration = duration,
        DcBias = round(mean(wav@left),digits),
        Clipping = clipping(wav)
      )
      
      # Correct DC offset bias
      if (dc_correct) wav@left = dc_correct(wav@left)
      
      # Calculate requested indices at each time interval
      start = 0.0
      results = data.frame()
      while (start < duration) {
        
        end = min(start + time_interval, duration)
        if (time_interval != duration) { # process interval
          wav_interval = cutw(wav, f=wav@samp.rate, from=start, to=end, output='Wave')
        } else { # process entire file
          wav_interval = wav
        }
        
        ACI = ADI = AEI = BIO = H = M = NDSI = NA
        if ('ACI' %in% alpha_indices | 'BIO' %in% alpha_indices) {
          spectrum = spectro(wav_interval, f=wav_interval@samp.rate, wl=wl, plot=F, dB=NULL)
          if ('ACI' %in% alpha_indices) { # Acoustic complexity index
            ACI = acidx_aci(
              spectrum,
              fs = wav_interval@samp.rate,
              fmin = min_freq,
              fmax = max_freq,
              j=j
            )
            # ACI = acoustic_complexity( # soundecology
            #   wav_interval,
            #   min_freq,
            #   max_freq,
            #   j,
            #   wl
            # )
            # ACI = ACI$AciTotAll_left
            # ACI = ACI( # seewave
            #   wave = wav_interval,
            #   wl = fft_w,
            #   flim = c(min_freq/1000, max_freq/1000),
            #   nbwindows = (duration/j) # number of windows
            # )
            if (is.na(ACI)) ACI = 0
          }
          if ('BIO' %in% alpha_indices) { # Bioacoustic index
            BIO = acidx_bio(
              spectrum,
              fs = wav_interval@samp.rate,
              fmin = min_freq,
              fmax = max_freq
            )
            # BIO = bioacoustic_index( # soundecology
            #   wav_interval,
            #   min_freq,
            #   max_freq,
            #   wl
            # )
            # BIO = BIO$left_area
          }
        }
        if ('ADI' %in% alpha_indices) { # Acoustic diversity index
          ADI = acoustic_diversity(
            wav_interval,
            max_freq,
            db_threshold,
            freq_step
          )
          ADI = ADI$adi_left
        }
        if ('AEI' %in% alpha_indices) { # Acoustic evenness index
          AEI = acoustic_evenness(
            wav_interval,
            max_freq,
            db_threshold,
            freq_step
          )
          AEI = AEI$aei_left
        }
        if ('H' %in% alpha_indices) { # Acoustic entropy index
          H = H(
            wav_interval,
            wl
          )
        }
        if ('M' %in% alpha_indices) { # Amplitude index
          M = M(
            wav_interval
          )
        }
        if ('NDSI' %in% alpha_indices) { # Normalized difference soundscape index
          NDSI = ndsi(
            wav_interval,
            wl,
            anthro_min,
            anthro_max,
            bio_min,
            bio_max
          )
          NDSI = NDSI$ndsi_left
        }
        
        # Store results
        result = data.frame(
          File = file,
          TimeStart = start,
          TimeEnd = end
        )
        if (!is.na(ACI))  result$ACI  = round(ACI,digits)
        if (!is.na(ADI))  result$ADI  = round(ADI,digits)
        if (!is.na(AEI))  result$AEI  = round(AEI,digits)
        if (!is.na(BIO))  result$BIO  = round(BIO,digits)
        if (!is.na(H))    result$H    = round(H,digits)
        if (!is.na(M))    result$M    = round(M,digits)
        if (!is.na(NDSI)) result$NDSI = round(NDSI,digits)
        
        results = rbind(results, result)
        
        start = end # move to next time interval
      }
    }, error = function(err) {
      warning(err$message)
    }, warning = function(wrn) {
      warning(wrn$message)
    })
    
    if (!is.null(results)) {
      return(list(
        'results' = results,
        'diagnostics' = results_diagnostics,
        'file_processed' = file
      ))
    } else {
      return(NULL)
    }
  } # calculate_alpha_indices
  
  no_input_files = length(input_files)
  if (ncores > no_input_files) {
    ncores = no_input_files
    message(' Number of cores limited to number of files (', ncores, ')')
  }
  
  ## Start processing
  time_start = proc.time() # start timer
  
  if (ncores > 1) { # Process in parallel with a cluster
    
    message(' Batch processing ', no_input_files,
            ' files in parallel using ', ncores, ' cores (',
            paste(alpha_indices, collapse=','), ')')
    
    cluster = makeCluster(ncores, type = 'PSOCK')
    cluster_results = parLapply(cluster, input_files, calculate_alpha_indices, clustered = T, time_interval, ...)
    
    if (length(which(sapply(cluster_results, is.null))) > 0)
      cluster_results = cluster_results[-which(sapply(cluster_results, is.null))]
    results = suppressMessages(Reduce(full_join, lapply(cluster_results,"[[",1))) #as.data.frame(t(sapply(cluster_results,"[[",1)))
    diagnostics = suppressMessages(Reduce(full_join, lapply(cluster_results,"[[",2))) #as.data.frame(t(sapply(cluster_results,"[[",2)))
    files_processed = unlist(sapply(cluster_results,"[[",3))
    
  } else { # Process in series on the main thread
    
    message(' Processing ', no_input_files, ' file(s) in series using 1 core')
    
    results = data.frame()
    diagnostics = data.frame()
    files_processed = c()
    for (file in input_files) {
      file_results = calculate_alpha_indices(file, ...)
      results = rbind(results, file_results$results)
      diagnostics = rbind(diagnostics, file_results$diagnostics)
      files_processed = append(files_processed, file_results$file_processed)
    }
  }
  
  if (exists('cluster')) stopCluster(cluster)
  
  num_files_processed = length(files_processed)
  if (num_files_processed != no_input_files)
    warning('The following files were not processed (', num_files_processed, ' of ', no_input_files, '):\n ', paste(input_files[which(!input_files %in% files_processed)], collapse='\n '))
  
  if (ncol(results) == 0) results = NULL
  if (ncol(diagnostics) == 0) diagnostics = NULL
  
  # Stop timer
  time_end = proc.time() - time_start
  message(' Total time: ', round(time_end['elapsed']/60, 2),' min (',
          round(time_end['elapsed'], 2), ' sec)')
  
  return(list(
    results = results,
    diagnostics = diagnostics,
    files_processed = files_processed
  ))
}

# # EXAMPLES #########
# # Params
# path = '~/../../Volumes/SAFS Work/DNR/test/subset'
# input_files = list.files(path=path, pattern='*.wav', full.names=T, recursive=F)
# output_path = '~/../../Volumes/SAFS Work/DNR/test/subset/output/'
# alpha_indices = c('BIO', 'ACI')
# # Series
# # batch_process_alpha_indices(input_files, output_path, alpha_indices = alpha_indices, time_interval = 60*2, ncores = 1)
# # Parallel
# batch_process_acidx(input_files, output_path, alpha_indices = alpha_indices, time_interval = 60*2, ncores = 2, min_freq = 2000, max_freq = 8000, db_threshold = -40)
# # Other
# # batch_process_alpha_indices(input_files, output_path, alpha_indices = c('BIO','AEI'), ncores = 3, min_freq = 200, max_freq = 2000, db_threshold = -45)
# 
# 

#
# out = batch_process_acidx(c('~/Desktop/oesf-examples/short/03_short.wav', '~/Desktop/oesf-examples/short/04_short.wav'),
#                           c('~/Desktop/oesf-examples/output/'),
#                           output_file = 'testing',
#                           alpha_indices = c('BIO', 'ACI'), time_interval = 10,
#                           ncores = 2, min_freq = 2000, max_freq = 8000, db_threshold = -40)
