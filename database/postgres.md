* doc reference [tutorial 1](http://www.postgresqltutorial.com/) [tutorial 2](https://www.tutorialspoint.com/postgresql/)

* comments
  - single line `--select * from t`
  - multiple lines `/* select... select...  */`

### select statement
* comparison operation
[link](https://www.postgresql.org/docs/9.1/static/functions-comparison.html)

* not equal
  ```
  select * from t1
  where col1 != value
  ```
* calculate calculated column

  event use twice formula, postgre wont process twice [link](https://stackoverflow.com/questions/5067745/postgresql-use-previously-computed-value-from-same-query)

* select nest

  - `select p.col1, p.col2 from (select * from table) as p`
  - select top 1 as nest select
    ```
    select * 
    from deals
    inner join (
      select count(startup_id) as numofsub, startup_id
      from deals
      group by startup_id
      order by numofsub desc
      limit 1) newtable
    on deals.startup_id = newtable.startup_id
    ```

* distinct on all columns

  `select distinct col1, col2 from table`

* distinct on single column

  `select distinct on (col1),col1,col2 from table`

* select with if statement and group by
  
  `select sum(if(col1=='value',1,0))/sum(if(col2=='value',1,0)) from table group by col3`

*

### with (common to expression)
* 
  ```
  with tmp1 as (select * from t1),
          tmp2 as (select col1 from tmp2),
          tmp3 as (select * from tmp1 inner join tmp2 on tmp1.col1=tmp2.col1)
  select col2, col4 from tmp3
  ```

* same temporary tables in with clause
those temporary tables can be used in same with and immediate next select clause [link](https://www.postgresql.org/docs/9.1/static/queries-with.html)



### join
* inner join
  `select * from t1 inner join t2 on t1.key = t2.key and t1.key2=t2.key2`
* left join
  - left join = left outer join: include every key from left. key only in left, right columns will generate null value
  - `select * from T1 left join T2 on T1.id = T2.id`
* only keep value in T1 but not in T2
  - null 
    ```
    select *
    from t1
    left join t2
    on t1.id = t2.id
    where t1 is null ;
    ```
  - exist
    ```
    SELECT table2.id FROM table2 
    WHERE NOT EXISTS 
      (SELECT * 
       FROM table1 
       WHERE table2.id = table1.id)
    ```
* join multiple tables
  - 
  ```
  select *
  from table1
  inner join locations l1
  on l1.locatable_id = table1.table1_id1
  inner join locations l2
  on l2.locatable_id = table1.table1_id2
  ```



### UNION
* `UNION` removes duplicate records (where all columns in the results are the same), `UNION ALL` does not.

### data format
* add few days to a date

  `select time_key + interval '7'  from t1`
  output is timestamp with time zone

* add a integer to a date

  `select time_key + integer '7' from t1`
  output is date if time_key is date format

* extract from date [link](https://www.postgresql.org/docs/9.4/static/functions-datetime.html)


*

### table information
* if has constraints

  `select * from information_schema.table_constraints where table_schema = 'public' and table_name='tablenamehere'`

* if has index
```
    select
        t.relname as table_name,
        i.relname as index_name,
        a.attname as column_name
    from
        pg_class t,
        pg_class i,
        pg_index ix,
        pg_attribute a
    where
        t.oid = ix.indrelid
        and i.oid = ix.indexrelid
        and a.attrelid = t.oid
        and a.attnum = ANY(ix.indkey)
        and t.relkind = 'r'
        and t.relname like 'geographyitems'
    order by
        t.relname,
        i.relname
```

### rename
* rename table
* rename table constraints
* rename columns

### create table
* create an empty table
  ```
    CREATE TABLE public.deals (
        id integer,
        startup_id integer,
        investor_group_id integer
    );
  ```
* insert values
  - single row `INSERT INTO deals VALUES (1,51,2);`
  - multiple rows `INSERT INTO deals VALUES (1,51,2),(2,33,7);`
