## basic
* terms
  - HDFS: Hadoop Distributed Files System
  - Data Node: contains the blocks for HDHS and has the actual data
  - Name Node: knows which DataNode each block exist on
  - three block redundancy for fault tolerance
  - YARN: Yet Another Resource Negotiator (maybe next generation of MapReduce)
  - Tez: execution engine
  - Hive: Hadoop SQL engine
* command
  - `distcp` launches a MapReduce job that will stream data from one cluster to another
  - `hadoop fs ls /hadoop/data/path` list items in path folder
  - `hadoop fs -put /a/local/path/file /hadoop/data/path/` upload file into a new folder
  - `head =15 /hadoop/data/path/file` read first few lines in a file
  - `hadoop fs -mkdir /hadoop/data/path` make a folder/path
  - `hadoop fs - cp /hadoop/old/path/file /hadoop/new/path/` copy file into new folder
  - `hadoop fs -rm /hadoop/file` romve a file
  - `hadoop fs -mv /hadoop/old/file /hadoop/new/path` move file to a new folder
  - `hadoop fs -rmdir /hadoop/path` remove a folder when folder is empty
  - `hadoop fs -rm -R /hadoop/path` remove a folder and also files in folder
  - 

|function|features|
|-------|-------|
|storage|HDFS or DBFS or Cloud-based file (AWS S3， GCP cloud storage, Azure blob)|
|compute|MapReduce or Spark |
|management|YARN or Mesos|

* Hadoop distributions
    - open source: apache hadoop
    - commercial: cloudera, databricks(DBFS)
    - public cloud: Google GCD(Cloud Dataproc), AWS EMR(Elastic MapReduce), Azure (HDInsight)
* Hadoop libraries
    - MapReduce
    - Hive
    - Pig
    - Apache Spark
    - Apache Storm
* Hadoop pipeline
    - streaming ingest services

    |product|type|
    |-------|-------|
    |apache kafka|open source|
    |amazon kinesis|cloud|
    |google cloud pub/sub|cloud|
    |azure service bus|cloud|
* Kafka
    - server cluster stores record streams into topics
    - each record has a key, a value, and a timestamp
    - 4 core API
        + producer API (publish to topics)
        + consumer API (subscribe to topics)
        + streams API (input/output streams)
        + connector API (connects to existing system)

## databricks
*layout
  - Home
  - workspaces : create scripts/notebooks (notebook display data frame can easily convert it to graph)
  - recent
  - tables: create/load/manipulate data tables
  - clusters: computing cluster (free version is 6g with 1 node)
  - jobs: spark jobs (will also to notify in notebook when run a cube)
  - search
* packages
  - import packages from workspaces
  - python(pyspark), r(sparkr), sql(sparksql),ml(mllib) visualization(graphX) packages can be used on top of spark
* 
