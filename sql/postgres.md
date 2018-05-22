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

  `select p.col1, p.col2 from (select * from table) as p`

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