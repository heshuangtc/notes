## DATA STEP
* Subset
`Data newtable; set oldtable (where=(colname=’dateformat’D));run;`

`Data newtable; set oldtable (obs=100);run;`

* Space issue
  - `Data newtable(compress=binary); set oldtable ;run;`
  - `Proc sql` to create table cannot use compress to reduce size
  - Binary for big data table, for small one dont need to as it will increase the size and also for few columns table, char type is better than binary.
* Combine tables
  - vertical/cbind
    ```
    Data z;
      Set x;
      Set y;
    run;
    ```
  - horizontal/rbind
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
* add all columns/variables by row [link](https://communities.sas.com/t5/SAS-Enterprise-Guide/How-to-sum-along-row/td-p/1576)
  ```
  data sums;
   set test;

   /* method 1: array of all numerics */
   array n {*} _numeric_;
   sum_allnum=sum(of n[*]);

   /*method 2: variable range in data set */
   sum_range=sum(of a1-a5);
  run;
  ```
* create multiple datasets
  ```
  data tb1 tb2;
  set tb;
  if var1=1 then output tb1;
  if var2=3 then output tb2;
  run;
  ```
* merge tables
  ```
  proc sort data=tb1;by var1;run;
  proc sort data=tb2;by var1;run;

  data tb
  merge tb1 tb2;
  by var1;
  run;
  ```
* append all similar names datasets
  ```
  data work.tmp ;
    set work.lz_output_summ: ;
  run;
  ```
* nested if statement
  ```
  data tb_out;
  set tb_in;
  if var1=1 then
      if var2=2 then newvar = 0;
      else newvar=1;
  else
      if var3=3 then newvar=2;
      else newvar=3;
  run;
  ```
* merge with the indicator in which source
  
  proc sort both tb1&tb2 first
  ```
  data work.tb_out;
      merge tb1(in=a obs=1000)
            tb2(in=b obs=1000);
      by key_id;
      if a=1 then flag_tb1=1;
      if b=1 then flag_tb2=1;
  run;
  ```





## FORMAT
* DateTime
  ```
  Data newtable; 
  colname=1736208777.1;
  format=datetime 18.;
  Run;
  Output will be 07JAN15:00:12:57
  ```
* change variable/column data type
  - chr to num[link](http://support.sas.com/kb/24/590.html)
    `new_var = input(chr_var, 8.);`

* Convert from sas date to yyyymmdd `%sysfunc(putn(&planning_date,yymmddn8.))`
* as a is 5 digits no matter what is its length need 5 here
  ```
  data x; set xseasonality;
  a='20456';
  date=input(a,5.);     
  date2=put(date,yymmddn8.);
  Run;
  ```
* change to uppercase
  ```
  data tb1;
  set tb2;
  var1 = upcase(var1);
  run;
  ```