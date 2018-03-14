#### basic
* nan is not allowed in postgres, need to change to null
  ```
  df.where((pd.notnull(df)), None)
  s = 'insert into tablename values %s'.replace('None','null')
  ```
* create index on single col no dup allowed (https://www.tutorialspoint.com/postgresql/postgresql_indexes.htm)

  `CREATE UNIQUE INDEX index_name ON table_name (column_name)`

* create index on single/multiple columns (https://www.tutorialspoint.com/postgresql/postgresql_indexes.htm)
  ```
  CREATE INDEX index_nameON table_name (column1_name, column2_name)
  CREATE INDEX index_nameON table_name (column1_name)
  ```

* drop/delete a table (https://www.postgresql.org/docs/8.2/static/sql-droptable.html)

  `DROP TABLE films, distributors`

* delete all rows from a table (https://www.postgresql.org/docs/8.2/static/sql-truncate.html)

  `TRUNCATE TABLE bigtable, fattable`



* postgres 

  * datatype (https://www.postgresql.org/docs/9.5/static/datatype.html)

  * postgres list all tables `select * from pg_catalog.pg_tables`

  * rename table will wipe out its index. be careful with a big table as need index to pull data faster.

* oracle
  * To list all tables accessible to the current user, type `select tablespace_name, table_name from all_tables`
  * list all tables in DB`select tablespace_name, table_name from dba_tables`
  * To list all tables owned by the current user, type `select tablespace_name, table_name from user_tables`

#### sqlalchemy
* [tutorial](http://docs.sqlalchemy.org/en/latest/orm/tutorial.html)

* column types(http://docs.sqlalchemy.org/en/latest/core/type_basics.html#generic-types)

* build connection with postgres
  ```
  from sqlalchemy import create_engine
  pgurl = 'postgres://sequoiadb:5OjQuetVigUc'
  engine = create_engine(pgurl)
  ```

* delete multiple rows (https://www.postgresql.org/docs/8.2/static/sql-delete.html)
  ```
  s="DELETE FROM films USING producers WHERE producer_id = producers.id AND producers.name = 'foo'"
  engine.execute(s)
  ```

* rename/alter table
  ```
  from sqlalchemy import *
  engine = create_engine(pgurl1)
  s = 'ALTER TABLE karina_test RENAME TO karina_test2'
  engine.execute(s)
  ```

* create/drop a table
  * use library functions
    ```
    engine = create_engine('sqlite:///:memory:')
    meta = MetaData()

    employees = Table('employees', meta,
        Column('employee_id', Integer, primary_key=True))

    employees.create(engine)
    employees.drop(engine)
    ```

  * use SQL query to create simple table
    `s = 'create table if not exists table_name (col1 int null, col2 date not null, col3 varchar(50) null)'`
    `engine.execute(s)`

  * use SQL query to create table with constraints
    `s = 'create table if not exists table_name (col1 int not null, col2 date not null, col3 varchar(50) null, constraint table_name UNIQUE (col1,col2))'`

  * use SQL query to drop table
    `s = 'drop table if exists table_name'`

* read data
  ```
  from sqlalchemy import create_engine
  pgurl = 'postgres://sequoiadb:5OjQuetVigUc'
  engine = create_engine(pgurl)
  df_tmp1 = pd.read_sql_query('select * from public.modelinput where geography=7 and year=2013', engine)
  ```

#### pandas
* create a table in DB
  (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html)

  `df.to_sql(name, con, schema=None, if_exists='replace', index=False, chunksize=None)`
  the `con` is SQLAlchemy engine or DBAPI2 connection

* insert multiple rows
  `df.to_sql(name, con, schema=None, if_exists='append', index=False)`

* put a data frame to DB (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html)

  `df.to_sql('db_table2', engine, if_exists='replace',index=False)`

#### psycopg2
* postgres install and use data as local database

* connect local database(https://www.a2hosting.com/kb/developer-corner/postgresql/connecting-to-postgresql-using-python)

  `myConnection = psycopg2.connect( host='localhost', user=username, password=password, dbname=database )`
  these configuration information can be found in pgadmin database properties

* connect postgres local database
  ```
  import psycopg2
  myConn = psycopg2.connect( host=hostname, user=username, password=password, dbname=database )
  cur = myConn .cursor()
  ...
  myConn .close()
  ```
  
* insert rows do nothing if conflict
  ```
  s = 'insert into {0} values {1} on conflict on constraint {2}_const do nothing'.format(
    table_name,
    ','.join([str(tuple(i)) for i in df_tmp1.values]),
    table_name
  )
  cur.execute(s)
  myConn.commit()
  ```

* insert/update rows into table
  ```
  s = 'insert into tablename values {0} on conflict on constraint tablename _const do update set {1}'.format(
    ','.join([str(tuple(i)) for i in df_tmp1.values]),
    ','.join([i+'=excluded.'+i for i in df_tmp1.columns.values])
  )
  cur.execute(s)
  myConn.commit()
  ```

* drop a table
  ```
  s = 'drop table if exists public.oil'
  cur.execute(s)

  * create table
  ls_col_name = ['date','dcoilwtico']
      ls_col_type = ['date'] + ['numeric']
      ls_col_null = ['not null'] + ['null']
      s = ' '.join(['create table if not exists public.oil (',
          ','.join([' '.join([i1,i2,i3]) for i1,i2,i3 in zip(ls_col_name,ls_col_type,ls_col_null)]),
          ',constraint oil_const UNIQUE (date)'
          ')'])
  cur.execute(s)
  ```

### cx_oracle
