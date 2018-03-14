* Extra day of week

  `as.numeric(strftime( date_data, format='%u'))`

* Character to Date. if it is numeric date, needs origin to define yearcutoff date

  `as.Date(chr_date,format='%Y%m%d')`

  `as.Date(chr_date,format='%Y-%m-%d')`

  `as.Date(format(strptime(df$date,format="%Y%m%d")))`

* date to string

  `as.character(Sys.time(),'%Y%m%d')`

* Create time value. output will be 'YYYY-MM-DD HHMMSS CST'

  `strptime('11:56:50','%H:%M:%S')`

* Compare diff time

  `difftime(sys.time(), time_data)`

  `sys.time()-time_data`

  `Sys.time()-10**6`

* Excel origin date

  `as.Date(excel_date_number,origin='1899-12-30')`

* SAS origin date

  `as.Date(sas_date_number,origin='1960-01-01')`

* system date/time

  `Sys.time()``Sys.Date()`

* list of dates

  `seq(start.date,end.date,by='days')`