#### basic dataframe func

* `colnames(df)`

* row names
  - list row names `rownames(df)`
  - remove row name `rownames(data) <- c()` = `rownames(data) <- NULL`

* duplicates
  - `df2$dup <- duplicated(df)`
  - `duplicated(df,incomparables = FALSE,na.rm=TRUE)`
  - remove duplicates from data frame`df <- df[duplicated(df)==FALSE,]`

* shape of a data frame
  - `dim(df)` output: #rows #cols
  - `nrow(df)`
  - `ncol(df)`

*

* `ftable(df$col)`

* missing values
  - fill missing values with previous non na values `zoo::na.locf(object, na.rm = TRUE, ...)`

* data frame overview statistic
  - `summary(df)`
  - `psych::describe(df)`
  - `pastecs::stat.desc(df)`


#### transpose dataframe

* col to row or row to col `t(df)`
* `pivot()`
* 
* wide to long
  - `melt(df, id.var='idx_combination')`
* long to wide
  - `reshape2::dcast(df, rowidx~colidx, value.var = 'value')`

#### combine data frames

* Rbind data frames, append by rows
  - same number of columns `rbind()`
  - same number of columns and rbind many times need avoid new data as factors `rbind(alist,stringsAsFactors=F)`
  - different number of columns `rbind.fill` 

* Cbind data frames, concatenated by columns

  `cbind` or `cbind.data.frame`

* Inner join:[key] match both A and B
  * `merge(A,B)`
  * `sqldf::sqldf('select * from A JOIN B using()')`
  * `dplyr::inner_join(A,B,by=NULL,copy=FALSE)` if 2 diff source only keep A. if B copy or not
  * `plyr::join(A, B, by=NULL, type='inner', match='all')` by=NULL means key is all column. match:if duplicated ids, use 'all' columns to match. OR 'first', use one and applied to some id

* Left join:[key] match A and B + extra in A

  * `merge(A,B, all.x=TRUE)`
  * `sqldf::sqldf('select * from A LEFT JOIN B using()')`
  * `dplyr::left_join(A,B,by=NULL,copy=FALSE)`
  * `plyr::join(A, B, by=NULL, type='left', match='first')`

* Right join:[key] match A and B + extra in B

  * `merge(A,B, all.y=TRUE)`
  * `sqldf::sqldf("select * from A RIGHT JOIN B using()")`
  * `dplyr::right_join(A,B,by=NULL,copy=FALSE)`
  * `plyr::join(A, B, by=NULL, type='right', match='first')`

* Full join:[key] all in both A and B

  * `merge(A,B, all=TRUE)`
  
  * `sqldf::sqldf("select * from A UNION B using()")`
  
  * `dplyr::full_join(A,B,by=NULL,copy=FALSE)`
  
  * `plyr::join(A, B, by=NULL, type='full')`

* Semi join:[key] match both A and B [col] only in A

  * `dplyr::semi_join(A,B,by=NULL,copy=FALSE)`

* Anti join:[key] in A but not in B [col] only in A

  * `sqldf::sqldf("select * from A EXCEPT select * from B")`

  * `dplyr::anti_join(A,B,by=NULL,copy=FALSE)`

#### subset data frames

* Filter dataframe based on list of values (keep) `Df[which(df$col %in% c()), ]` or `df[which(df$col1 == 'xxx' | df$col2 == 'yyy'), ]`

* Filter dataframe based on list of values (exclude) `Df[!which(df$col %in% c()), ]` `df[which(!df$col %in% c(6,7)),]`

* count number of cells when match a condition
  - `length(which(df$col1== 1))`

*

* Select a single col 

  * `df$col` will be an array instead of 1 col dataframe

  * `Df[, 'col']` or `df[['col']]` will still be data frame

* Not include some col based on colname

  * `Df[ -grep('pattern', colnames(df))]`

  * `Df[ -which(colnames(df) %in% 'pattern')]`   faster

* Select col2 based on filter col1 `subset(df, col1=value)$col2`

* delete one column `df$col1 <- NULL`

* delete one row (e.g. 3rd row) `df[-3,]`

#### Compare data frame

* compare all columns and allow all transformation `compare::compare(A,B, allowAll=TRUE)`  

  find different elements(better for 1 column) `Data.frame(lapply(1:col(A),function(i) setdiff(A[,i],$™[,i])))` 

* compare with `daff` pkg

  * way 1
     `x<-daff::differs_from(A,B)` `y<-as.data.frame( x$get_matrix())` `y[which(y[,1] %in% c('---','+++'))]`

  * way 2
    `x<-daff::diff_data(A,B)` `y<-as.data.frame( x$get_matrix())` `y[which(y[,1] %in% c('---','+++'))]`



#### aggregation functions

* sapply 

  `sapply(list_of_tables, function(i) dbGetQuery(con1,paste('select * from ',i)))`

* mapply

  * pull from multiple db tables

  `df = mapply(c,
    min.date_create=sapply(1:5,function(r) dbGetQuery(con1,paste("select min(date) from table1 where id=",r))),
    max.date_create=sapply(1:5,function(r) dbGetQuery(con1,paste("select max(date) from table2 where id=",r)))
  )`

* lapply

  - `lapply(1:7,function(i) mean(df[df$col1==i,'col2'],na.rm = T))`
  - output a list of string. each string is from a data frame row `unlist(lapply(x, function(i) paste(unlist(i),collapse = '')))`

* aggregate

  col1 is aggregation values, col2 is level

  - `aggregate(df$col1,list(df$col2),function(i) length(unique(i)))`

  - `aggregate(df$col1,list(df$col2,df$col3),function(i) max(i))`

  - `aggregate(df$col1,list(df$col2),sum)`

  - `aggregate(df$col1,list(name=df$col2),sum,na.rm=TRUE)`

  - aggregate one column `aggregate(col1~col2,data=df,FUN=sum)`
  - aggregate multiple columns `x <- aggregate(.~col2,data=df,FUN=sum)`

  - `aggregate(col1~col2,data=df,FUN=sum,na.rm=T,na.action=NULL)` remove na and also on those na rows no actions keep keys




#### modify
* sort data frame

  * `df[order(df$col1,-df$col2),]`

  * `df[order(df$`1`,rev(df$`2`)),]` column name with number start cannot use '-' to descending

  * `df[order(df[,col1], -df[,col2]),]`

* change multiple columns types
  - change from chr to int with column index `x[,2:6] <- as.integer(unlist(x[,2:6]))`
  - change from int to chr with column names `x[,c('NA','SA','EU')] <- as.character(unlist(x[,c('NA','SA','EU')]))`

* remove a column
  `x <- x[!colnames(x) %in% c('date')]`

* select few columns
  - `x <- x[[c('a','b')]]` `x <- x[[c(1,2,3)]]`
  - `x <- subset(x,select=c('a','b'))`

* numeric data to category/ categorize into groups
  `cut(df$col, c(0,10,20))`

*

#### other
* Create empty data frame 

  * `data.frame()`

  * `data.frame(matrix(data=NA, nrow=0, ncol=10))` with 10 columns

  * `read.table(text='',col.names = tableview_colnames,colClasses='character')` with defining column names and col data type

  * `data.frame("PRCSS_NM"=NA,"DATA_TYPE"=NA,"LOCN_ID"=NA,"ITEM_ID"=NA,"date"=NA,"dow"=NA,"value"=NA,"source"=NA)` with column names and data type. this has 1 row.

    `temp_df_template[0,]` remove all rows.




### LIST

* find all combination by given list
  - pick with order `combn(c(1,2),c(3),2)` from c(1,2,3) pick 2 elements
  - like a union join `expand.grid(var1,var2,var3)`
  - pick with order `gtools::permutations(n=#choices,r=#pick,v=choices_pool)`

*

* Remove elements from list 

  `a_list<-a_list[-1]` 

  or `a_list$elment_name<-NULL`

* Give names to elements `names(a_list)<-c( 'name_a', 'name_b', 'name_c'.....)`

* Add elements into list, at same level 

  `a_list['new_element_name'] <- new_element` 

  or `a_list$new_element_name <- new_element`

* List to data frame

  `do.call(rbind.data.frame, your_list)`

  or `df <- data.frame(matrix(unlist(l), nrow=132, byrow=T),stringsAsFactors=FALSE)`

* List to multiple data frames `list2env(the_list, envir=.GlobalEnv)`

* `unlist()`

* make a list 

  * `list()`

  * `split(df,f=df$col1)`




