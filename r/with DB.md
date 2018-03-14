### connect to database

  `require(RPostgreSQL)`

  `require(RJDBC)`

* redshift

  `drv1 = dbDriver("PostgreSQL")`

  `con1 = dbConnect(drv1, 'xxx.redshift.amazonaws.com', prot='', dbname='', user='', password='')`

  `df = dbGetQuery(con1,'select * from table')`


* mysql
  
  `drv2 = JDBC('com.mysql.jdbc.Driver','/fullpath/mysql-connector-java-5.1.jar', identifier.quote='`')`

  `con2 = dbConnect(drv2,'jdbc:mysql://xxx.amazonaws.com',prot='', dbname='', user='', password='')`

  `df = dbGetQuery(con2,'select * from table')`

* oracle


### database functions

* view data

  `dbListTables(con1)`

  `dbListFields(con1,'table_name')`

* `dbDisconnect(con1)`