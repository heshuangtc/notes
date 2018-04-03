* Split/cut a string when meet 'pattern'

  `strsplit(x, split='pattern', fixed = FALSE, perl = FALSE, useBytes = FALSE)`

* Concatenate strings to one

  `paste('abc','def',collapse=' ')`   [output: abc def]

  `paste0('str1','str2')` [output:str1str2]

* Mark a col/array

  `paste(df$col , collapse=',')`   [output a,b,c]

  `paste( sQuote ( df$col ) , collapse=',')`  [output 'a','b','c']

  `paste( shQuote ( df$col ), collapse=',')`   [output "a","b","c"]

  `paste( dQuote(df$col), collapse=',')`    [output 'a','b','c']

  `paste( shQuote( df$col, type='sh'), collapse=',')`

  `paste( 'str1','str2', collapse=',', sep='')`

* Mark elements in SQL

  `paste("'", as.character( unlist( (dbGetquery()))), "'",collapse=",",sep=" ")`

* Multiple times of one array/vector

  `replicate( n, array)`

* Cut string by certain chr

  - `strsplit( string, split='term')`

  - `sub('term', '', string)`

  - replace 'term' by '' `gsub('term','',string)`

  - replace 's' by '' `gsub('s','',df$col)`

* match certain words

  `agrep('loc',names(x))`

  `grep()`

  `gerpl()`

* number of character in string `nchar('a_string')`

* case tranformation

  * `tolower('WORD')`