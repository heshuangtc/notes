* Find largest files
  `du -a /path | sort -rn | head -n 10`
* Find large files (over 10M) modified in last 24 hours in current path(.)
  ```
  Find . -size +10000k -mtime -1
  Find . -size +10000k -mtime -30 (in last 30 days)
  Find . -size +10000k -mtime +30 (before 30 days)
  Find . -size +10000k -mtime 30 (just at 30 days ago)
  ```
  [link](http://www.cyberciti.biz/faq/howto-finding-files-by-date/)

* Get file size
  `Du -g filename`
* from largest to smaller and just for top 5 
  `Du -m filename | sort -rn | head` 
* Find top 10 oldest files
  `ls -alt | grep event* | tail -10`
* Do something on files older than 30 days
  ```
  Find . -name ‘xxx_pattarn’ -mtime -2 -print
  Find . -name ‘xxx_pattarn’ -mtime +30 -exec gzip {} \;
  Find . -name ‘xxx_pattarn’ -mtime +30 -exec rm {} \;
  ```

* Find top CPU cost process
  `Nmon` open the function
  `M`  get real summary
  `T` get top list

Choose which mode want to check 1,2,3,4. Descr will on screen
[link](https://www.ibm.com/developerworks/community/blogs/aixpert/entry/aix_memory_usage_or_who_is_using_the_memory_and_how20?lang=en)

* copy 
  * Copy files
    `cp from_path/file to_path/file`
  * Copy directory
    `cp -R` from_directory to_directory

* Compress files
  `Gzip`
  `Compress`
  `Zip`

* find number of files in directory
`ls | wc -l`
`ls | grep pad | wc -l`

* HOTKEY
Clean command line: `Ctrl+L`

* Basic code

| command  | description  |
|----------|--------------|
|-h|Get code history|
|!20|Rerun history code|
|Ps -ef|what is running right now|
|Ps -ef | grep xxx|filter by keywords xxx|
|ls|list contents in this directory|
|Cd /path/path|go to this root path|
|Cd path|go to this sub-folder|
|Pwd|show current directory|
|Dt xxx.txt|Open a file|
|Cp path/xxx.txt another.path/xxx.new.txt|Copy a file|
|Cp file1 file2|Copy 1 to 2|
|Cp path1/* path2/|Copy everying in path1 to path2|
|Rm xxx.txt|Remove a file|
|Touch xxx.txt|Create an empty file|
|Find -name filename \| xargs grep xxxx|Find file name filtered by xxx|
|Find path -name ‘filename’|Find file in path|
|Kill -9#|Kill a process with id #|
|df|View size of files in current path|
|Du -sk \| sort -rn|Check space and sort from large-->small|
|du|Check space for everything|
|Mv oldname newname|Rename a file|
|Mv filename path/|Move a file to a new path|
|Mkdir folder_name|Create a new path or folder|
|Rmdir folder_name|Delete a path or folder|
|Gzip filename||
|Gzip -d filename||
|Zip file||
|Unzip file||
|Compress filename||
|Uncompress filename||
|Zip filename.zip filename||
|Unzip filename.zip||
|Bzip2 filename||
|Bzip2 -d filename||

* Crons

|min|hour|dayofmonth|month|dayofweek|
|---|----|----------|-----|---------|
|*|*|*|*|*|
|0-59|0-23(12am-)|1-31|1-12|0-6(sun-sat)|

* crons examples
  * Yearly `0 0 1 1 *`
  * monthly `0 0 1 * *`
  * weekly(sun)`0 0 * * 0`
  * daily`0 0 * * *`
  * Every 3 month`0 0 1 */3 *`