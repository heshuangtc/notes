* Run script
  * By use absolute path (with shebang in it)
    `/home/lzeng/script_name`
    `#!bin/sh` or `#!bin/bash`  to locate which shell to use
  * Execute script with spec typing interpreter
    `Sh` filename or `bash` filename
  * Use current shell without sub shell
    `.[space]./filename`

* Permission
  * Change permission on shell script file(owner permission)
    `Chmod u+x filename.sh`
  * Give permission to a group or other
    `Chmod 755 filename.sh`

* Assign value
  `A= date[space]+%H`
  `B= expr[space]$a[space]\*[space]60` 
  need exact ``, expr, [space]


* Output a string
  `Echo 'string $variable'`
  `Echo string $variable`

* Get system info
  * Get current shell info
  `echo $SHELL`
  * Get system info
  `Uname -a`
  * Get system usage CPU/user etc
  `/usr/bin/sar`
  * Find top cost/usage process
  `top`
  * Find where is this cmd/library
  `Which libxml2`
  `Find -name 'file.name'`
  * Get history ran  command
  `history`
  * List all files include ./files (not visible usually)
  `ls -all`

* Edit a file
`Vi`   path/filename.txt
`-i`   insert code
`esc`  end coding
`ZZ`   quit

* AWS
  * List files in a bucket
  `aws s3 ls s3://xxx-yyy-demand-planning/../`
  * Upload a file to S3
  `aws s3 cp` local/path/file s3://xxx-yyy-demand-planning/../`
  * Rm a file from S3
  `aws s3 rm s3://xxx-yyy-demand-planning/../`