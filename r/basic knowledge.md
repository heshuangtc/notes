#### basic
* assign

  `<-` is assign to object. 

  `<<-` absolutely assign wont lose on the fly such as in shiny ui

  `assign('name',value)`

* `true==0` in r but in sql `false==0`

* `unique()`

* Zip files in R.use OS command to zip.

  `system('gzip path/filename.csv')`

  `system('zip path/filename.zip path/filename.csv')`

  `system('zip path/filename.zip path/filename.RData')`

* remove element from environment
  `rm(obj1,obj2)`

#### run R in command line

* windows
  
    `cd C:/Program Files/R/R-3.2.0/bin/` -> `Rscript /filepath/filename.R`

* UNIX/Linux

    `/path/bin/Rscript  /path/file_name.R`

* crontab

  `Crontab -e` to create a new cron

  `i` insert mode 

  `* * * * * /usr/bin/Rscript /path/filename.R >/path/logfile.log 2>&1`

  `Esc`  exit insert

  `ZZ`  exit cron editor

* Call function in external script `source( 'path/rscript_name.R' )` `the_function()`

* send email

* Install package

  * Install download package `install.package('package_name')` `install('path/file_name')`

  * List all my installed packages `installed.packages()`


#### flow control

  * if condition

    `if (a=b) {c=1}else{c=2}`

    `if (a=b) {}else if{}else{}`

  * for loop

    `for ( i in 1:7){c=i}`

  * user function

    `obj <- function(par1,par2){... return(obj)}`

#### send email

* `library(sendmailR)`

  `from <- 'myemail@gmail.com'`

  `to <- c('myemail1@gmail.com','myemail2@gmail.com')`

  `subject <- paset('Email Subject:TESTING email from server','hahaha')`

  `body <- 'Email body:testing'            `

  `mailControl=list(smtpServer='server.url.com')`

  `attachmentObject <- mime_part(x='./path/file.csv',name='file.csv')`

  `bodyWithAttachment = list(body,attachmentObject)`
  
  `sendmail(from=from,to=to,subject=subject,msg=body,control=mailControl) or sendmail(  from=from,to=to,subject=subject,msg=bodyWithAttachment,control=mailControl)`

#### basic func

* is.na(df$col)

* `c(1,2,3)`

* `setwd("./path/")`

* `length()` `levels()`

* datatype

  * turn to classification `as.factor(c('a','b','a'))` [output: 1,2,1]

  * `as.data.frame(df)`

* repeated times `replicate(12,df$col,simplify = T)` `rep(1,10)`

* compare

  * `library(compare)`

    `setequal(list1, list2)`

    `setdiff(list1, list2)`

    `comparison <- compare(list1, list2, allowAll=TRUE, ignoreNames = TRUE)`

    `comparison$tM`  #same elements

    `comparison$tC`

    `comparison$transform`




### flow control
* error control
    ```
    tryCatch(
        {testing_expr},
        warning = function(w){},
        error = function(e){},
        finally = {}
    )
    ```
    if has warning, the process will be stopped by warning.
    if has error, the process will be stopped by error.
    every case, finally will run

### RStudio
* convert r script to doc report with print results/graphs
  - in script window, click `compile report`=`Ctrl+Shift+K` choose `MS Word` or `HTML` or `PDF` (this requires `pdflatex` package installed. But didn't work for me)
  - use markdown: create a Rmarkdown file(output can be Word or PDF or HTML) and copy those codes to the right place. this requires `knitr` package
* markdown word
  - papaja package