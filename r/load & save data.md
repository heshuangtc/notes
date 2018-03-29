#### load/read

* From http and get text contents

  `library(XML)`
	
  `library(RCurl)`
	
  `keywork<-'ipad'`

  `link<-paste0('http://www.amazon.com/s/ref=nb_sb_noss_2?url=search-alias%3Daps&field-keywords=',keywork)`
	
  `raw.url<-getURL(link)`
	
  `raw.html.tree<-htmlTreeParse(raw.url)`
	
  `raw.body<-raw.html.tree$children$html$children$body`

* Load saved RData

  `load('file_name.RData')`

* Read zip file data. if want read RData/csv direcly by unzip need zip instead of gzip.

  `read.csv(unzip('path/filename.zip'))`

  `load(unzip('path/filename.zip'))`
  
  a little slower than directly load a non-zip file

  `load(unzip) 0.58 0.08 0.69 46MB`

  `load()  0.49  0.008  0.5  46MB`

  `load(unzip) 6.55 0.95 7.67 454MB`

  `load() 5.34 0.12 5.45 454MB`
  
* Load sas data/spss data/excel data/etc data [link](https://www.datacamp.com/community/tutorials/importing-data-r-part-two#gs.9cAlaqw)

* When read csv file with large number and don't let it be scientific number 1E5

  `options(scipen=999)` or `format(dfr$x, scientific = FALSE)`

* read csv
  - define column data type when read csv 
  `colClasses=c(rep('character',5))` or `colClass=rep('character',1)`
  - basic package `read.csv()`

* `load(file.RData)`

* read excel 
  - `openxlsx::read.xlsx('./path/file.xlsx',sheet=i,colName = TRUE,skipEmptyRows = TRUE,cols = 1:12))`

* list excel sheets
  - openxlsx package 
  `openxlsx::getSheetNames('./Data_Files/file.xlsx')`
  - readxl pacakge
  `readxl::excel_sheets('./Data_Files/file.xlsx')`

  
#### save

* Only save one table as RData. 
  
  `save(df1,df2,...., file='file_name.RData')`

* export xlsx

  `openxlsx::write.xlsx(df, './output/file.xlsx)`
  
* `write.csv(df,'path/filename.csv',,row.names = FALSE,fileEncoding = "UTF-8")`

* `save.image("path/filename.RData")`

* `openxlsx::write.xlsx(df,'./path/file.xlsx')`

* output a doc word

  `library( ReporteRs )`

  `mydoc = docx() each time will refresh the report`

  `mydoc= './path/file.docx'`

  `mydoc = addTitle(mydoc, value = 'string title', level = 1)` add title, could be wrote multiple times in order

  `mydoc = addParagraph(mydoc, value ='some words')`

  `docpath<-'./path/filename.docx'` `writeDoc( mydoc, file = docpath)`  save final doc