library(soundecology)

in_path = '~/../../Volumes/SAFS Work/DNR/test_day/subset/'
output = '~/../../Volumes/SAFS Work/DNR/test_day/output/'
no_cores = 8

multiple_sounds(directory = in_path, 
                resultfile = paste0(output, 'aci.csv'), 
                soundindex = 'acoustic_complexity',
                no_cores = no_cores,
                min_freq = 2000,
                max_freq = 8000,
                j = 5,
                fft_w = 512
)

multiple_sounds(directory = in_path, 
                resultfile = paste0(output, 'bio.csv'), 
                soundindex = 'bioacoustic_index',
                no_cores = no_cores,
                min_freq = 2000,
                max_freq = 8000,
                fft_w = 512
)

multiple_sounds(directory = in_path, 
                resultfile = paste0(output, 'adi.csv'), 
                soundindex = 'acoustic_diversity',
                no_cores = no_cores,
                max_freq = 8000,
                db_threshold = -40,
                freq_step = 1000
)

multiple_sounds(directory = in_path, 
                resultfile = paste0(output, 'aei.csv'), 
                soundindex = 'acoustic_evenness',
                no_cores = no_cores,
                max_freq = 8000,
                db_threshold = -40,
                freq_step = 1000
)