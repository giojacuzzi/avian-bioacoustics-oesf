## Generate a csv mapping raw audio data files to metadata and inspect it

# Paths to directories comprising the DNR database
database_paths = c(
  '~/../../Volumes/GIOJ Backup/DNR',
  '~/../../Volumes/SAFS Backup/DNR'
)
database_path_working = '~/../../Volumes/SAFS Work/DNR'

## Generate filemap ------------------------------------------------------------
# In:  The DNR database drives
# Out: output/filemap.csv
library(stringr)

output_path = 'data/output/filemap.csv'
tz = 'America/Los_Angeles'

filemap = data.frame()
for (database_path in database_paths) {
  message('Scanning ', database_path)
  files = list.files(path=paste0(database_path), pattern="*.wav", full.names=T, recursive=T)
  deployment = str_locate_all(pattern='Deployment', files)
  deployment = substring(files, sapply(deployment, '[[', 2)+1, sapply(deployment, '[[', 2)+1)
  substrings = str_split(str_sub(basename(files), start = 1, end = -5), '_')
  temp = data.frame(
    serial = sapply(substrings, '[[', 1),
    deploy = deployment,
    year   = substring(sapply(substrings, '[[', 2), 1, 4),
    date   = as.POSIXct(sapply(substrings, '[[', 2),
                        tz = tz, format='%Y%m%d'),
    time   = as.POSIXct(
      paste(sapply(substrings, '[[', 2), sapply(substrings, '[[', 3)),
      tz = tz, format='%Y%m%d %H%M%S'),
    file   = files
  )
  filemap = rbind(filemap, temp)
}
if (nrow(filemap) != 75658) {
  stop('filemap length is unexpected!')
}
write.csv(filemap, file=output_path, row.names = FALSE)
message('Created ', output_path)

## Inspect filemap -------------------------------------------------------------
filemap = read.csv(output_path)
filemap$serial = factor(filemap$serial)
filemap$deploy = factor(filemap$deploy)
filemap$year   = factor(filemap$year)
filemap$date   = as.POSIXlt(filemap$date, tz = tz)

# Explore
unique(filemap$serial)
unique(filemap$deploy)
tapply(filemap$date, filemap$serial, unique)

# Show calendars of recorded dates
library(calendR)
for (year in unique(filemap$year)) {
  message('Creating calendar for ', year)
  filemap_year = filemap[filemap$year==year,]
  print(calendR(year = year,
          start_date = as.Date(filemap_year$date)[1],
          end_date   = as.Date(filemap_year$date[nrow(filemap_year)]),
          start = 'M'))
}
