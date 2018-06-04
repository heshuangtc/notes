#### url

* read from url

  `library(RCurl)`
  `raw_txt = getURL(raw_url)`

  * read img

  `rasterGrob(readJPEG(getURLContent(img_url)),width=unit(1,"npc"), height=unit(1,"npc"))`

  * read zip file

  `download.file(file_url,temp)`

  `read.table(gzfile(temp, "037720-99999-2014.op"),skip=1)`

  `unlink(temp)`

* read weather data from web

  `require(weatherData)`

  `getWeatherForDate('EGSS',num_days_before_today,end_date = today, opt_detailed = TRUE,opt_all_columns=TRUE)`