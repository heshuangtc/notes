## Proc import and export
* Import csv file
  ```
  proc import datafile="a_path/data/temp/xxx.csv"
  out=baseline_history
  dbms=csv
  replace;
  getnames=no;
  run;
  ```
* import csv file
  ```
  proc import datafile = '/folders/myfolders/SASCrunch/cars.csv'
   out = work.cars
   dbms = CSV ;
  run;
  ```
* Output csv file
  - save as csv
    ```
    proc export data=baseline_history
    outfile='/janus_pilot/janus_us/janus/data/temp/baseline_all20160412.csv'
    dbms=csv
    replace;
    Run;
    ```
  - save as csv with filter
    ```
    proc export data=sashelp.class (where=(sex='F'))
       outfile='c:\myfiles\Femalelist.csv'
       dbms=csv
       replace;
    run;
    ```
  - If want to use &macro part of name
    ```
    Proc export data=xxx
    outfile=”/janus_pilot/janus_us/janus/data/temp/baseline_&macro..csv”
    dbms=csv
    replace;
    Run;
    ```
* Output csv file with long header or too many columns
  - Get header (if not too long over 32767)
    ```
    PROC EXPORT DATA= input_dataset
    OUTFILE= "path/filename.csv"
    DBMS=CSV REPLACE;
    PUTNAMES=YES;
    RUN;
    ```



## statistic
* sum every column
  `proc means data=work.lz_tmp_count;output out=work.lz_tmp_count sum=;run;`
* calculate all columns freq
  ```
  ods output onewayfreqs=lz_tmp.tb_out;
  proc freq data=lz_tmp.tb_input;
      tables _all_;    
  run;
  ods output close; 
  data lz_tmp.tb_out(keep=Table Frequency Percent value);
    set lz_tmp.tb_out;
    value = left(coalescec(of f_:));
  run;
  ```

## datasets info
* remove duplicates
  `PROC SORT DATA=libname.table NODUPKEY;BY varname;RUN;`

* save columans of a table
  `proc contents data=table1 out=table2 (keep=NAME); run ;`

* Compare 2 datasets
  - With different name in 2 tables
    ```
    proc compare base=proclib.one compare=proclib.two nosummary;
    var gr1;
    with gr2;
    Run;
    ```
  - With same name in 2 tables
    ```
    proc compare base=test.baseline_history_120000002 compare=main_i.baseline_history_120000002 nosummary;
    var HIST_TOTAL_UNITS19475 - hist_total_units20575;
    run;
    ```

* Get header (if too long over 32767)
  ```
  Proc contents data=input_dateset out=header(keep=NAME);
  Run;
  Data _null_;
  Set header;
  File ‘path/file.header.name.csv’ mod dsd dlm=’,’ lrecl=9999999;
  put(_all_)(+0);
  run;
  ```
* Get contents
  ```
  Data _null_;
  Set input_dataset;
  File ‘path/filename.csv’ mod dsd dlm=’,’ lrecl=9999999;
  put(_all_)(+0);
  Run;
  ```
  mod : if append header no need if dont need header


## transpose
* Convert data table structure
  - When convert character variable needs var anyway!!
    ```
    proc sort data=baseline_history;by dpia_key loc_key;run;
    proc transpose data=baseline_history out=output;by dpia_key loc_key;run;
    ```
  - ...
  `Proc transpose data= out=;id ; var _all_;run;`
* Append
  `Proc append base=xxx data=yyy;run`
* transpose wide to long[link](https://stats.idre.ucla.edu/sas/modules/how-to-reshape-data-wide-to-long-using-proc-transpose/)
  `proc transpose data=work.tb1 out=work.tb2;run;`


## PROC SQL
* One proc sql to create table and update value
  ```
  Proc sql noprint;
  Create table A as
  Select …. From ….;
  Update A
  Set col=value
  Where col2=xx and col3=xxx;
  quit;
  ```
* Update DB base
  ```
  proc sql noprint;
  update fdppad_i.f_dp_pad
  set attribute_value = '1'
  where pad_key in (24351631,24468899)
  and information_type = 247
  and attribute_value ne '1';
  quit; 
  ```
* Delete rows in DB base
* count with conditions
  ```
  proc sql;
  create table table2 as
  select
       sum(var3=1) as a_count,
       sum(var4=0) as d_count
  from table1;
  quit;
  ```
  >>note: this cannot work in hadoop as hadoop cannot sum boolean.
* delete dataset from library
  ```
  proc datasets library=usclim;
     delete rain;
  run;
  ```


*