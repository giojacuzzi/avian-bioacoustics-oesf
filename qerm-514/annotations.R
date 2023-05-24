source('global.R')
library(suncalc)

data = get_recording_date_serial_data()

# Only look at 2021
data = data[data$DataYear==2021, ]

# 13 is hour of solar noon
# 05 and 06 are the hours of sunrise

prompt = 'r = rain, w = wind, a = airplane, o = other, enter = no noise events (q to quit): '

outpath = normalizePath('qerm-514/output/annotations.csv')
if (file.exists(outpath)) {
  annotations = read.csv(outpath)
} else {
  annotations = data.frame()
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
    
    annotation = readline(paste(serialno, prompt))
    if (annotation == 'q') stop()
    
    note = readline('Notes?')

    annotations = rbind(annotations, data.frame(
      File = file,
      SurveyDate = date,
      SerialNo = serialno,
      Annotation = annotation,
      Note = note
    ))
    write.csv(annotations, outpath, row.names = F)
  }
}
