
* Store a list of value into a macro
  ```
  Data step
  Data _null_;
  Set table;
  symput(macro_name,col_name);
  run;
  ```
* Proc sql
  ```
  Proc sql noprint;
  Select … into :macro_name separated by ‘ ‘
  From table;
  quit;
  ```
* %let macro_name= 1 2 3 4;

* Execute system command in SAS
  - `x ls -l;`
  - `data _null_;``call system ('cd /users/smith/report');``run;`

* `%let macro.name=%qsysfunc(putn(1736208777.1,datetime 18.))`

* customize table name with macro variable
  ```
  %let var_name = value_string;
  data table&var_name;
  set tb;
  run;
  ```
* customize path/string with macro variable. note: need to use a double quote and if another string after macro variable adds`.`
  ```
  proc export data=work.tb
  outfile="a_path/&var_name..csv"
  dbms=csv
  replace;
  run;
  ```

* do loop [link](http://support.sas.com/documentation/cdl/en/mcrolref/62978/HTML/default/viewer.htm#p0ri72c3ud2fdtn1qzs2q9vvdiwk.htm)
  ```
  %macro create(howmany);
     %do i=1 %to &howmany;
        data month&i;
           infile in&i;
           input product cost date;
        run;
     %end;
  %mend create;
  %create(3)
  ```