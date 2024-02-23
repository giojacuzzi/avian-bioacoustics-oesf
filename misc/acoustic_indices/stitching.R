library(tuneR)
source('helpers.R')

# TLDR; spectro falls apart once the number of samples is greater than the 32-bit integer limit 2147483647
# At 32kHz, this occurs at ~18 hours of recording, which may not be enough to cover dawn/dusk buffers on the summer solstice. So, instead of stitching we will average time intervals. Note that this finer temporal resolution of index values can shed light on diel patterns in such a way that full-day single index calculation cannot.

hr1 = readWave('~/Desktop/oesf-examples/03.wav')
hr1@left = dc_correct(hr1@left)
hr1 = prepComb(hr1, where='end')

hr2 = readWave('~/Desktop/oesf-examples/04.wav')
hr2@left = dc_correct(hr2@left)
hrt2 = prepComb(hr2, where='start')

stitched = bind(hr1, hr2)
writeWave(stitched, '~/Desktop/oesf-examples/stitched.wav') # 2 hrs

source('acoustic_indices/acidx.R')
aci = acidx_aci(stitched, fs = stitched@samp.rate)
aci_hr1 = acidx_aci(hr1, fs = hr1@samp.rate)
aci_hr2 = acidx_aci(hr2, fs = hr1@samp.rate)

stitch4 = bind(stitched, stitched)
aci4 = acidx_aci(stitch4, stitch4@samp.rate)

stitch8 = bind(stitch4, stitch4)
stitch16 = bind(stitch8, stitch8)
stitch24 = bind(stitch16, stitch8)

##########
time_start = proc.time() # start timer
specy = spectro(channel(stitch24, which = "left"), stitch24@samp.rate, wl = 512, dB = NULL, 
                plot = F)

# aci24 = acidx_aci(stitch24, stitch24@samp.rate)
time_end = proc.time() - time_start # stop timer
message(paste0('ACI ', aci24, ', total time: ', round(time_end['elapsed'], 2), ' sec'))
time_start = proc.time() # start timer
bio24 = acidx_bio(stitch24, stitch24@samp.rate)
time_end = proc.time() - time_start # stop timer
message(paste0('BIO ', bio24, ', total time: ', round(time_end['elapsed'], 2), ' sec'))