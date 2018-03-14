* HOTKEY
  * UNIX world, F2 in editor will call out hotkey setup window
  * Ctrl+F mark lines in editor and Ctrl+H unmark
  * Ctrl+Y is to type command in table window
  * Autoexec can run sas code when open sas session
  * R in AIX UNIX editor can copy the line r\ will copy many lines
  * D in AIX UNIX editor delete line

* DATA STEP
  * Subset
  `Data newtable; set oldtable (where=(colname=’dateformat’D));run;`

  `Data newtable; set oldtable (obs=100);run;`

  * Space issue
    * `Data newtable(compress=binary); set oldtable ;run;`
    * `Proc sql` to create table cannot use compress to reduce size
    * Binary for big data table, for small one dont need to as it will increase the size and also for few columns table, char type is better than binary.
  * Combine tables
    * vertical/cbind
      ```
      Data z;
        Set x;
        Set y;
      run;
      ```
    * horizontal/rbind
      ```
      Data z;
        Set x,y;
      Run;
      ```
  * List first row in all tables in one library
    ```
    data candidate;
      set usercand.pad:(keep=process transaction_time useridvar);
    if _n_= 1 then output;
    Run;
    ```
    symbol will find all tables with name pad* in library usercand.

* FORMAT
  * DateTime
    ```
    Data newtable; 
    colname=1736208777.1;
    format=datetime 18.;
    Run;
    Output will be 07JAN15:00:12:57
    ```
  * `%let macro.name=%qsysfunc(putn(1736208777.1,datetime 18.))`
  * Convert from sas date to yyyymmdd `%sysfunc(putn(&planning_date,yymmddn8.))`
  * as a is 5 digits no matter what is its length need 5 here
    ```
    data x; set xseasonality;
    a='20456';
    date=input(a,5.);     
    date2=put(date,yymmddn8.);
    Run;
    ```

* LIBRARY
  * Copy from one to one and only copy dataset no other marco things
    `Proc copy in=libraryfrom out=libraryout memtype=data;run;`
  * Delete all datasets in one library
    `Proc datasets lib=libraryname kill;run;`
  * List all filese in a directory/library over 360 days
    ```
    Filename dirlist pipe ‘find . -mtime +360’;
    Data dirlist;
    Infile dirlist length=reclen;
    Input filename $varying100. reclen;
    Dpia1 = scan(filename,-2,’.’);
    Dpia2 = scan(filename,-1,’_’);
    run;
    ```

* JANUS
  * Open janus log
  `F11` to open the log window, and open editor.

  Run these code to get log for WorkFlowTool
  `Proc Printto log=log;run;`
  `Options symbolgen macrogen mprint;`

* PROC
  * Compare 2 datasets
    * With different name in 2 tables
      ```
      proc compare base=proclib.one compare=proclib.two nosummary;
      var gr1;
      with gr2;
      Run;
      ```
    * With same name in 2 tables
      ```
      proc compare base=test.baseline_history_120000002 compare=main_i.baseline_history_120000002 nosummary;
      var HIST_TOTAL_UNITS19475 - hist_total_units20575;
      run;
      ```
  * Import csv file
    ```
    proc import datafile="/janus_pilot/janus_us/janus/data/temp/xxx.csv"
    out=baseline_history
    dbms=csv
    replace;
    getnames=no;
    run;
    ```
  * Output csv file
    ```
    proc export data=baseline_history
    outfile='/janus_pilot/janus_us/janus/data/temp/baseline_all20160412.csv'
    dbms=csv
    replace;
    Run;
    ```
    If want to use &macro part of name
    ```
    Proc export data=xxx
    outfile=”/janus_pilot/janus_us/janus/data/temp/baseline_&macro..csv”
    dbms=csv
    replace;
    Run;
    ```


  * Output csv file with long header or too many columns
    * Get header (if not too long over 32767)
      ```
      PROC EXPORT DATA= input_dataset
      OUTFILE= "path/filename.csv"
      DBMS=CSV REPLACE;
      PUTNAMES=YES;
      RUN;
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
  * Convert data table structure
    `proc sort data=baseline_history;by dpia_key loc_key;run;`

    `proc transpose data=baseline_history out=output;by dpia_key loc_key;run;`

    When convert character variable needs var anyway!!
    
    `Proc transpose data= out=;id ; var _all_;run;`
* Append
  `Proc append base=xxx data=yyy;run`

* PROC SQL
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

* MACRO
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
  * `x ls -l;`
  * `data _null_;``call system ('cd /users/smith/report');``run;`