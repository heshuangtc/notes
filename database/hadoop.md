## basic
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
