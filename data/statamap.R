## Read date, site, strata, serial table from file
library(xlsx)

file = 'data/ARUdates-site-strata_v2.xlsx'

message('Reading ', file)
data = read.xlsx(file, sheetName = 'ARUdates-site-strata')

## Metadata:
# SurveyID
# SiteID
# DataYear
# DeployNo
# StationName
# StationName_AGG
# SurveyType
# SerialNo
# SurveyDate
# DataHours
# UnitType
# Strata
# UTM_E
# UTM_N