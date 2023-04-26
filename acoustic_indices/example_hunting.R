# All serial nos and strata active on a given date
# See also 2020-04-12/20 and 2020-05-30/31
date = '2020-04-19'
distinct(site_data[site_data$SurveyDate==date, c('SerialNo', 'Strata')])
