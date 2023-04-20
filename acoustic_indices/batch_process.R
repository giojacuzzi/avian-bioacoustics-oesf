## Batch process alpha acoustic indices from a list of .wav files
library(parallel)
library(tuneR)
library(soundecology)
library(seewave)

# NOTE: only mono .wav support
# NOTE: ACI will return NA for long duration files (perhaps 1 hr + @ 32kHz),
# this is an issue with the 'soundecology' package
batch_process = function(
    input_files, output_file, alpha_indices, ncores = 1, digits = 4, ...) {

  message(' Starting batch process (', paste(alpha_indices,collapse=' '),')')
    
  file_header = c('File,SamplingRate,BitDepth,Duration,DcBias,Clipping,ACI,ADI,AEI,BIO,H,M,NDSI')
  
  ## Internal function to calculate indices
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
    
    # Calculate requested indices
    if ('ACI' %in% alpha_indices) { # Acoustic complexity index
      message('ACI')
      ACI = acoustic_complexity(
        wav,
        min_freq,
        max_freq,
        j,
        fft_w
      )
      ACI = ACI$AciTotAll_left
    } else { ACI = NA }
    if ('ADI' %in% alpha_indices) { # Acoustic diversity index
      message('ADI')
      ADI = acoustic_diversity(
        wav,
        max_freq,
        db_threshold,
        freq_step
      )
      ADI = ADI$adi_left
    } else { ADI = NA }
    if ('AEI' %in% alpha_indices) { # Acoustic evenness index
      message('AEI')
      AEI = acoustic_evenness(
        wav,
        max_freq,
        db_threshold,
        freq_step
      )
      AEI = AEI$aei_left
    } else { AEI = NA }
    if ('BIO' %in% alpha_indices) { # Bioacoustic index
      message('BIO')
      BIO = bioacoustic_index(
        wav,
        min_freq,
        max_freq,
        fft_w
      )
      BIO = BIO$left_area
    } else { BIO = NA }
    if ('H' %in% alpha_indices) { # Acoustic entropy index
      message('H')
      H = H(
        wav,
        wl
      )
    } else { H = NA }
    if ('M' %in% alpha_indices) { # Amplitude index
      message('M')
      M = M(
        wav
      )
    } else { M = NA }
    if ('NDSI' %in% alpha_indices) { # Normalized difference soundscape index
      message('NDSI')
      NDSI = ndsi(
        wav,
        fft_w,
        anthro_min,
        anthro_max,
        bio_min,
        bio_max
      )
      NDSI = NDSI$ndsi_left
    } else { NDSI = NA }
    
    # Calculate file diagnostics
    if (wav@pcm) {
      dc_bias = round(mean(wav@left))
    } else {                                                                                    
      dc_bias = mean(wav@left)
    }
    clipping = clipping(wav)
    duration = round(length(wav@left)/wav@samp.rate, 2)
    
    # Write results to file
    return(paste0(
      '\n', file,',',         # File (including path)
      wav@samp.rate,',',      # SamplingRate (Hz)
      wav@bit,',',            # BitDepth
      duration,',',           # Duration (sec)
      dc_bias,',',            # DcBias (mean amplitude value)
      clipping,',',           # Clipping (boolean)
      round(ACI,digits),',',  # acoustic complexity index
      round(ADI,digits),',',  # acoustic diversity index
      round(AEI,digits),',',  # acoustic evenness index
      round(BIO,digits),',',  # bioacoustic index
      round(H,digits),',',    # acoustic entropy index
      round(M,digits),',',    # amplitude index
      round(NDSI,digits)      # normalized difference soundscape index
    ))
  }
  
  ## Start processing
  time_start = proc.time() # start timer
  cat(file_header, file = output_file, append = F) # open results file
  
  if (ncores > 1) { # Process in parallel using clusters
    
    no_input_files = length(input_files)
    if (ncores > no_input_files) {
      ncores = no_input_files
      message(' Number of cores limited to number of files')
    }
    message(paste0(' Processing ', no_input_files, ' files in parallel using ', ncores, ' cores'))
    
    cluster = makeCluster(ncores, type = 'PSOCK')
    results = parLapply(cluster, input_files, calculate_alpha_indices, clustered = T, ...)
    write.table(results, file = output_file, append = T, quote = F, col.names = F, row.names = F)
    Sys.sleep(1) # pause to allow all to end
    
    stopCluster(cluster)
    
  } else { # Process in series on the main thread
    
    message(paste0(' Processing ', length(input_files), ' files in series using 1 core'))
    for (file in input_files) {
      results = calculate_alpha_indices(file, ...)
      cat(results, file = output_file, append = T)
    }
  }
  
  time_end = proc.time() - time_start # stop timer
  message(paste0(' Created ', output_file, '\n Total time: ', round(time_end['elapsed'], 2), ' sec'))
}

# TEST #########
# Params
path = '~/../../Volumes/SAFS Work/DNR/test/subset'
input_files = list.files(path=path, pattern='*.wav', full.names=T, recursive=F)
output_file = '~/../../Volumes/SAFS Work/DNR/test/subset/output/results.csv'
alpha_indices = 'ACI'
# Serial
batch_process(input_files, output_file, alpha_indices, ncores = 1)
# Parallel
batch_process(input_files, output_file, alpha_indices, ncores = 3)
# Other
batch_process(input_files, output_file, alpha_indices = c('BIO','H'), ncores = 3, min_freq = 200, max_freq = 2000, db_threshold = -45)
##############
