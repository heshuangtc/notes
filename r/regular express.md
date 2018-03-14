* Find matched string, will return value

  `regmatches(dirlist$num[20:30], regexpr("[0-9]{9}", dirlist$num[20:30]))`

* Replace all numeric with ‘’, in other words, remove numbers, return value

  `grep('[0-9]{9}', '', dirlist$num, value=T)`