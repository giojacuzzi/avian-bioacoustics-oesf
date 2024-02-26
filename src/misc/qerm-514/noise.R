source('global.R')
library(suncalc)

data = get_recording_date_serial_data()

# Only look at 2021
data = data[data$DataYear==2021, ]

# 13 is hour of solar noon
# 05 and 06 are the hours of sunrise

noise_prompt = 'NOISE: (r)ain, (w)ind, (a)irplane, (o)ther, [enter] = none (q = quit): '

outpath = normalizePath('qerm-514/output/annotations.csv')
if (file.exists(outpath)) {
  annotations = read.csv(outpath)
} else {
  annotations = data.frame()
}

charmap = function(char) {
  if (char == '' | char == 'q') return(char)
  mapping = c('a' = 'airplane', 'r' = 'rain', 'w' = 'wind', 'o' = 'other')
  
  if (char %in% names(mapping)) {
    return(mapping[char])
  } else {
    return(charmap(readline(paste('Invalid input', noise_prompt))))
  }
}

for (date in as.character(sort(unique(data$SurveyDate)))) {
  message(date)
  
  data_date_6 = data[data$SurveyDate==date & data$NearHour==6, ]
  
  if (nrow(data_date_6) == 0) next

  for (row in 1:nrow(data_date_6)) {

    file = data_date_6[row, 'File']
    
    if (file %in% annotations$File) {
      message(basename(file), ' already annotated, skipping.')
      next
    }
    
    serialno = data_date_6[row, 'SerialNo']
    
    utils::browseURL(file) # open in default .wav app (i.e. RX)
    
    noise_code = charmap(readline(paste(serialno, noise_prompt)))
    if (noise_code == 'q') stop()
    
    note = readline('NOTES: ')
    if (note == 'q') stop()

    annotations = rbind(annotations, data.frame(
      File = file,
      SurveyDate = date,
      SerialNo = serialno,
      Noise = noise_code,
      Note = note
    ))
    write.csv(annotations, outpath, row.names = F)
  }
}
