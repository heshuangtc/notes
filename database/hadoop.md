## basic
|-----|-----|
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
    |-----|-----|
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
        