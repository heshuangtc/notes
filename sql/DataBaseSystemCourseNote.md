## Data Base System Course Note
### class 1 Intro
* database
  - share data (cross applications/organization/time)
  - archive data (keep it safe)
  - control data (security/access control)
* benefit
  - reuse data:no redundant data entry, keep everyone informed, reduce confusion by master model, improve decision making, keep history
  - reduce programming costs: CASE/ management tool
* DBMS
  - a set of tools for creating and managing database
  - DBMS Qualities： availability/reliability/accuracy
  - DBMS technologies: ER diagrams/normalization/B tree/SQL/query optimization/concurrency control
* Network Hierarchical(DBMS past)
  - minimize disk access, intricate programmer control
* Relational DBS(present)
  - quality metrics for DB design
  - client server computing
  - concurrency control: short transaction/lots of users/ACID(atomicity/consistency/isolation/duration)
* DBMS(future)
  - new kinds of applications: design/engineering/enterprise/medical
  - new technologies: object oriented/massive amount of media/parallel processors/triple store
  - more intelligence: semantic db/ knowledge db
* DB data model
  - a set of data definition: value based ones/structural ones(attributes/ fields)
  - a set of data instances: conforms to data model/stared in DB/managed by DB/used by application
* Relational DB
  - data is defined by relations
  - each relation is a set of attributes
  - each attribute is defined by a domain
  - each attribute has a key
* Observation
  - RDBS: when to stop (ER diagram) / definition good or bad (normalization theory)
  - key in RI often in other than RI
  - key is compound values/synthetic


### class 2 Relational Algebra
* intension `R=(A1, A2, ...)` 
* extension `R  A1 A2 A3` R is relation, A1 A2 A3 are attributes/tuple
* string is domain of major and name, tuples are a set.
* Relational DB
  - a relation is a table
  - an attribute is a column of a table
  - a domain defines the values allowed in an attribute
  - a tuple is a row of table
  - a relation is a set of tuples
* key
  - is a set of attribute values that will uniquely identify any tuple
  - primary key is main key of a relation
  - a foreign key is key of another relation, very important and much used in relational queries
* terms
  - select: satisfy a condition (`δ`delta/ `θ`theta)
  - project: a subset of attributes (`π`pi)
  - join: two relations by comparing the values in their attributes
  - Cartesian project: multiply tow relations together(`×`)
  - union: (`∪`)
  - intersection: (`∩`)
  - difference
* example
  - `δLetter = 'C' Grade` (Letter='C' SELECT Grade)
  - `πss# Grade` (SS#) PROJECT Grade
* Natural join
  - compare two relations and select those tuple that have the same values in their common attributes
  - student JOIN grade
* Equi-join
  - names of the attributes to be joined
  - student JOINss# grade are given as arguments 
*outer join: dont loose data from the arguments that is not joined


### class 3 Normal Forms
* tips"
  - dont put 2 pieces of information into one
  - dont record the same information twice
  - minimize the volume of a database
* normal forms
  - mathematical technique
  - are a way of saving that every tuple in every relation should only hold information about one thing
* functional dependencies
  - to prove 'the relations are correct or incorrect'
  - are a way to identify the information in a database
* good relation design: all of attributes (and only) depend on keys of that relation
* superkey
  - a set of one or more attributes that determine all of the other attributes in a relation
  - every relation has at least one superkey
* key: a superkey without any redundant attributes
* if a relation has bad fds. this relation must be divided without lossing information(eg. fds attributes)
* 1NF
  - First Normal Form
  - all attributes are atomic
  - no nested attributes
  - no hidden data structures
* 2NF
  - second normal form
  - no partial fds
* 3NF
  - third normal form
  - no transitive fds
  - no attributes on the left side
* BCNF
  - boyce codd normal form
  - every fds is a key fds
  - no keys on the right side
* divide attributes if not meet NF
  - lossless: get same result if put divided relations back together again
  - dependency preserving: all of the dependencies belong to at least one relation after decomposition. can be enforced by the relation keys
  ```
    U = (A,B,C,D,E,F)  A -> B,C,D   D -> E   B -> F
    become
    U1 = (A,B,C,D)  A -> B,C,D
    U2 = (D,E)      D -> E
    U3 = (B,F)      B -> F
  ```

### class 4 More Normal Form
* terms
  - always fix 2NF and 3NF
  - sometimes cannot fix BCNF without data loss
  - unfix-able BCNF is rare in practice
* inference rules
  - Y is a subset of X ---> X->Y
  - X->Y  ---> XZ->YZ
  - X->Y Y->Z  ---> X->Z
  - X->YZ  ---> X->Y X->Z
  - X->Y X->Z  ---> X->YZ
  - X->Y WY->Z  ---> WX->Z
* if fd can be inferred from other fds then drop it from the minimal set
* 4NF
  - fourth normal form
  - based on multi-value ds
  - A -> B: each A determines a set of B (e.g. parent -> children / person -> address)
* 5NF
  - consider joins
  - advanced NF let any pattern be identified
  - pattern is predetermined -> make sure that instances of the pattern are stored exactly once

### class 5 E-R diagrams
* divide（A/B）
  - attributes of B must be in A
  - result is A attributes minus B attributes
  - example 1: `A={X,Y,Z} B={X,Z}` result `Y`
  - example 2: `A= a c b  B=a d` result `{ac match ad}`
  - example 3: `A= 1 2 3  B=1 2` result `{3}` first match -> minus
* entity relationship models（E-R）
  - information modeling
  - E-R diagram: create database from ER diagrams
  - create database from ER diagrams
* information modeling
  - information level: logical organization/ what is information in the data/ no system constraints
  - data level: data definition in a specific system/ relational network
  - internal level: organization of bits and bytes on media
* info model
  - capture the information that will be in DB exactly and precisely without concern for the practicality of mapping
  - good one should be extensible as it will never need to change its exiting facts only add new ones
* info diagram
  - relate to it easily
  - examples, ER diagrams, NIAM diagrams, IDEF diagrams, EXPRESS-G diagrams
  - object oriented diagramming techniques/ OMT/ GOOCH/ unified model
* terms
  - entities: things in the real world
  - attributes: properties of things or relationships
  - relationships: dependencies between things in the real world
  - IS-A Hierarchies/Cardinalities/Dependencies
* ER diagram [link](https://www.lucidchart.com/pages/ER-diagram-symbols-and-meaning)
* cardinalities describe constraints between instances how many instances participate in a relationship
* total participation: every instance must be in the relationship
* weak entity
  - its instances cannot be identified using its own key
  - use another entity's key for additional identification
  - must be total participation with the another entity
* map to SQL
  - each entity becomes a table
  - each relationship becomes a table
  - borrow key from parent if dependent(weak)


### class 6 More ER diagrams
* attributes
  - simple attribute
  - key
  - attribute is derived
  - multi-value attribute
  - composite attribute has its won attributes
* relationship: binary/ ternary/ connect or attribute box
* cardinalities
  - give upper and lower limit
  - upper limit --> fancy arrow
  - lower limt=1 --> double line
  - 1->N
* weak entity
  - must be in total participation
  - key borrow from that participation
* ER -> relations
  - entity: include all attributes
  - attribute: entity key + own attributes
  - weak entity: own attributes + borrow key
  - relationship: n->m 2 borrow key/ own attribute
  - never 1-N to be written


### class 7 SQL
* select
  `SELECT attribute 1 2 3 FROM relation 1 2 3 WHERE condition 1 AND 2 AND 3`
* create table 
  ```CREATE TABLE employee (FNAME VARCHAR(5) 
                            SSNUM CHAR(9) 
                            SALARY INTEGER)
  ```
* drop table: delete table and all its data `DROP TABLE employee`
* alter table: null values put into exiting tuples `ALTER TABLE employee ADD age INTERGER`
* create index: `CREATE INDEX`

### class 8


### class 9



### class 10