### create/delete table
* create index and drop index
    ```
    drop index index_name on [db].[schema].[table1]
    create index index_name ON[db].[schema].[table1] (col1, col2)
    ```

### aggregate
* cumulative sum
    ```
    select sum(quantity) over (partition by item_id order by date_registered) as cumulative
    from table;
    ```


## select/where
* extract date part from date variable [link](https://www.w3schools.com/sql/func_sqlserver_datepart.asp)
    ```
    select datepart(yy, ClickDate), datepart(mm, ClickDate),count(col1)
    from tb1
    group by datepart(yy, ClickDate), datepart(mm, ClickDate)
    ```
* nest select always give allies
    ```
    select count(distinct a.var) as cnt
    from (select ...from...) as a
    ```

* calculate subtotal and percentage for each group [link](http://support.sas.com/documentation/cdl/en/sqlproc/63043/HTML/default/viewer.htm#n112pviu616u5rn1uileli1b03zb.htm)
    ```
    proc sql;
       select survey.groupid, cntid, count(cntid) as Count,
              calculated Count/Subtotal as Percent 
       from survey,
            (select groupid, count(*) as Subtotal from survey
                group by groupid) as survey2
       where survey.groupid=survey2.groupid
       group by survey.groupid, cntid;
    quit;
    ```

* cumulative count
    ```
    select groupid,
        rank() over (partition by groupid order by orderid) as [rank]
    from tb
    ```

* full join
    ```
    select *
    from tb1 full join tb2 on tb1.key=tb2.key
    ```

*