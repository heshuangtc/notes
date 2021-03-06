## library
* [pandas library link](http://pandas.pydata.org/pandas-docs/stable/)


## basic info
* Try using `df.loc[row_indexer,col_indexer] = value` instead
* `.copy()` before subset
* frequency table
  - `pandas.crosstab(index=df.col_label,columns=df.col_count)` [link](http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.crosstab.html)
* set up pandas display
  - width of table col by num of char
    `pd.set_option('display.max_colwidth',100)`
  - 
* summary data frame/overview of data frame
  - describe all numeric columns `df.describe()`
  - describe all columns `df.describe(include='all')`
  - get type/names/nullable `df.info()`



## row
* find min/max in each row
  - return value `df.max(axis=1)`
  - return column name `df.idxmax(axis=1)`

## column
* col has same values or not
  ```
  x = df.describe()
  x.columns[x.ix['max'] != x.ix['min']]
  if max==min or not, if yes, then only 1 values
  ```

* limit num of digits/round [link](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.round.html)

  `df.col.round(1)`

* count unique values in a col/series [link](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html)

  `df.col.value_counts()`

* if is duplicates `df.duplicated([col1,col2])`
* convert categorical col into dummy cols
  - `pd.get_dummies(df.col)`
  - rename dummy column names `pd.get_dummies(df.col,prefix='add_part')`
* convert continuous col into category [link](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html)
  - by number of bins `pd.cut(df.col, bins=4)` if want index instead of label, `labels=False`, default `labels=True`
  - by list of values `pd.cut(df.col, bins = [0,1,2]` # two bins actually
* col type
  * change col type

    `df['col1'] = pd.to_datetime(df.col1)`

  * category col
    ```
    df['col'] = df['col'].astype('category')

    df.col1 = pd.Categorical(df.col1).codes

    df.index = pd.CategoricalIndex(df.col1)
    ```
  * convert multiple col to categorical col
    ```
    df[df.select_dtypes(include=['object']).columns.values] = df[df.select_dtypes(include=['object']).columns.values].apply(lambda x:pd.Categorical(x).codes)
    ```
  * select certain datatype columns [link](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.select_dtypes.html)

    `df.select_dtypes(include=['float64'])`

* missing values
  - check which col has missing in df 
    `df_out.isnull().any().values`
  - check number of rows on each have missing
    `train.isnull().sum()`

  - fill missing values[link](https://pandas.pydata.org/pandas-docs/stable/missing_data.html#cleaning-filling-missing-data)

    `fillna()`
    `interpolate()`

  - fillna
    ```
    df['col1'] = fillna(replacement)
    df['col1'] = fillna(method='bfill')
    df['col1'] = fillna(method='ffill')

    different replacement for different columns by using dictionary
    values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    df.fillna(value=values)

    ```
  - limit values in an array
    `numpy.clip(a, a_min, a_max, out=None)`



  - drop na if row has

    `df.dropna(axis=0,how='any')`

  - drop na if col has

    `df.dropna(axis=1,how='any')`

  - drop na if entire row is na

    `df.dropna(axis=0,how='all')`

  - drop na if entire col is na

    `df.dropna(axis=1,how='all')`

  - combine 2 text/str columns with nan
    ```
    df = pd.DataFrame({'col1':['a','b','c',np.nan], 'col2':['1','2','3','4']})
    df['col3'] = df.col1.fillna('') + df.col2.fillna('')
    ```

  - sum 2 columns ignore nan
    ```
    df['source'] = df[['source1','source2']].sum(axis=1)
    df['source'] = df.source1.fillna(0)+df.source2.fillna(0)
    ```

* string column
  - remove spaces `df['col1'] = df.col1.str.strip()`
  - split a string 
    ```
    df.col.str.split('delimeter')

    x['FName'] = [i[1] for i in x.Name.str.split(',')]
    x['LName'] = [i[0] for i in x.Name.str.split(',')]
    ```
  - string as col pointer 
    `df['col1'] = getattr(df, 'col1').fillna(1)`
  - change column to lowercase
    `df.col.str.lower()`
  - select certain col based on string pattern[link](https://pandas.pydata.org/pandas-docs/stable/text.html)
  ```
  prod1.columns.str.startswith('geog')
  df.columns[df.columns.str.contains('^hol[0-9]+$')]
  ```

  - substring in col `df.col = df.col.str[:3]`

  - extract values from string col. warning current return list but later expeand=True return df
    + number `df['B'].str.extract('(\d+)').astype(int)`
    + character `df['B'].str.extract('(\D+)')
  - replace string
    ```
    df.col.str.replace(be_replaced, replacement)
    df.col.str.replace(be_replaced, replacement).str.replace(be_replaced, replacement)
    ```

* find max between col and isolated value
  ```
  a = date(2010,1,10)
  df['col'] = [max(i,a) for i in df.date1.dt.date ]
  ```

* find min/max in each col `df.max(asxi=0)` `df.min(asxi=0)`


* change values
  - replace values

    `DataFrame.replace(to_replace=None, value=None, inplace=False, limit=None, regex=False, method=’pad’, axis=None)[source]`
  - change existing col value by assign

    `df = df.assign('col1' = value)`

* count number
  * number of columns `len(df[0])`
  * number of rows `len(df)`

* move/shift col values
  * copy col by lag 1 row `df.col.shift(1)`
  * move forward 2 rows `df.col.shift(2)`

* sort dataframe(pandas) 

  `df.sort_values([col1], ascending=True)`

* change order of col
  * original col1-col4. change to 

    `df = df[[col2,col1,col4,col3]]`

  * insert col1 as first left col and copy from_col value

    `df = df.insert(0,'col1', df['from_col'])`

* rename `df = df.rename(columns={'old_col':'new_col'})` or `df.rename(columns={'old_col':'new_col'}, inplace=True)`

* index
  * remove index name `df.index.name = None`
  * reset index by dropping `df = df.reset_index(drop=True)`
  * reset index and rename `df.reset_index(name='name_for_index')`
* 


* 

* 

* 

* 

## subset
* select n rows
  - select first 5 rows
    `df[0:6]`or`df.ix[0:6,:]`
  - select 0,2,4,6...rows
    `df.ix[0::2,:]`
  - select first 10 rows on every 2 rows
    `df.ix[0:11:2,:]`
  - select 0,3,6,9...rows
    `df.ix[0::3,:]`
* dont select last n rows
  - dont select last 1 rows
    `df.ix[:-1,:]`
  - dont select last 2 rows
* remove duplicates[link](http://pandas.pydata.org/pandas-docs/version/0.17.1/generated/pandas.DataFrame.drop_duplicates.html)

  `DataFrame.drop_duplicates(['col1','col2'],keep='first')`


* delete df `del df``del [df1,df2,df2]`

* subset dataframe col between 2 values [link](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.between.html
)

  `df[df['closing_price'].between(99, 101, inclusive=True)]`

* subset dataframe if col1 values in/not in array [link](http://pandas.pydata.org/pandas-docs/stable/indexing.html)

  - in `df = df[df.col1.isin(array)]`
  - not in `df[df.col1.isin(alist)==False]`

* cut data by date
  ```
  df.time_col is datetime type not string
  df[df.time_col>''2016-01-01']
  ```

* multiple subset conditions
  ```
  df[(con1) & (con2)]
  df[ ((con1) & (con2)) | con3 ]
  ```

* check nan

  `df = df.loc[df.col.isnull()==True, 'col2'] = value`

* certain string in column names

  `[i for i in train.columns if 'hol' in i]`

* drop/remove columns
  ```
  del df[name]  #changed on original df
  del df.column_name(NOT working properly)
  df = df.drop([col1],axis=1)  #need assign to an object
  ```






## merge
* basic merge function[link](http://pandas.pydata.org/pandas-docs/stable/merging.html)

  `pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)`

* left/right join
  - if left has additional key than right, keep left. but if right has additions, drop it. vice versa
  - merge `result = pd.merge(left, right, on=['key1', 'key2'], how='left')`
  - join `df.join(df2,on='key',how='left')`

* merge on keys inner join
  - merge `result = pd.merge(left, right, on=['key1', 'key2'], how='inner')`
  - join `df.join(df2,on='key')`

* full join[link](https://stackoverflow.com/questions/35265613/pandas-cross-join-no-columns-in-common)
  add a common col for both table and make single value in it.merge on this new common col.

* merge on different col names

  `pandas.merge(df1, df2, how='left', left_on=['id_key'], right_on=['fk_key'])`

* combine dataframe by row

  `df.append(df2, ignore_index=True)`

* combine dataframe by col

  `pd.concat([df1, df4], axis=1)`

* 

* 

## data type
### numeric
* change col data type to numeric[link](http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.to_numeric.html#pandas.to_numeric)

  `pd.to_numeric(df.col, errors='coerce') return nan if cannot convert`
* astype(if nan will give error)
  `df.col = df.col.astype('int64')`
  `df.col = df.col.astype('int32')`
  `df.col = df.col.astype('int')`
* convert boolean to numeric
  - `df.col = df.col.astype('int32')`

### datetime
* convert timedelta to int
  ```
  import numpy as np
  td = 380 #days
  (td / np.timedelta64(1, 'D')).astype(int)
  ```
* convert timedelta64[ns] type to float64
  ```
  df.track_day = df.date1 - df.date2
  df.track_day.astype('timedelta64[D]')
  ```
* datetime to date
  `df.td_col.dt.days`

* year and day of year back to time key
  * way 1
    `pd.to_datetime(df.year*1000+df.day_of_year, format='%Y%j')`
  * way 2
    `df.apply(lambda x:datetime.datetime(x.year, 1, 1) + datetime.timedelta(x.day_of_year - 1), axis=1)`

* get weekly date from date series
  * number of weekday=6 : Sunday is start date of week
    `df['week start'] = df.time_key-pd.offsets.Week(weekday=5)`
  * number of weekday=0 : Monday is start date of week
    `df['week start'] = df.time_key-pd.offsets.Week(weekday=6)`

* col date to string[link](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.dt.strftime.html)
  `df.time_key.dt.strftime('%Y-%m-%d')`

* a row of dataframe to list
`df.ix[1,:].values.tolist()`

* convert int to timedelta[link](https://pandas.pydata.org/pandas-docs/stable/timedeltas.html)
`pd.to_timedelta(df.col, unit='s')`

* convert timedelta to int

* unix time to date
  ```
  from datetime import datetime
  df['date2']=list(map(datetime.fromtimestamp,df.date))
  ```

* only date from datetime `pd.to_datetime(df.col2, format='%m/%d/%Y').dt.date`

* extract month or day from datetime in dataframe
  ```
  from datetime import datetime
  df_fcst.time_key = pd.to_datetime(df_fcst.time_key)
  df_fcst['month'] = df_fcst.time_key.dt.month
  df_fcst['day'] = df_fcst.time_key.dt.day
  ```

* extract single value from df y cell

  `x[x.time_key>y.time_key[y.v3==y.v3.max()].values[0]]`

* convert col to datetime type 
  * `pd.to_datetime(df.col1)`
  * always provide format as this will MUCH faster `pd.to_datetime(df.col, ,format='%d/%m/%Y')`

* convert string to date
  ```
  from datetime import datetime
  datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
  ```

* add/minus hours from a datetime
  - way1
    ```
    df['format_datetime'] = pd.to_datetime(df.orginaldatetime)
    df.format_datetime - pd.to_timedelta(2, unit='h')
    ```
  - way2
    ```
    df['format_datetime'] = pd.to_datetime(df.orginaldatetime)
    df.format_datetime - pd.DateOffset(hours=2)
    ```
* 





## save
* series to dataframe[link](http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.Series.to_frame.html)
  `sr.to_frame('col_name')`

* save to excel file

  - simple file `df.to_excel('path')`
  - multiple sheets
    ```
    writer = pd.ExcelWriter('output.xlsx')
    df1.to_excel(writer,'Sheet1')
    df2.to_excel(writer,'Sheet2')
    writer.save()
    ```
* save to csv file `df.to_csv('filename.csv', index = False)`
  


## load 
* read csv[link](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)
  - without header
    `pd.read_csv('./data/file.csv',skiprows=i,nrows=n,header=None)`
  - with different delimiter
    `pd.read_csv('./data/file.csv', delimiter=None)`
  - define number of rows
    `pd.read_csv('./data/file.csv',nrows=100)` or `pd.read_csv('./data/file.csv',nrows=100_000)` 
  - skip number of rows
* list all files in directory [glob link](https://docs.python.org/3/library/glob.html)
  ```
  from glob import glob
  glob('./path/file_pattarn.csv')
  ```
* read sas export csv file with Unicode error[link](https://stackoverflow.com/questions/5552555/unicodedecodeerror-invalid-continuation-byte)
  `pd.read_csv('file.csv', sep=',', encoding='latin-1')`


* read partial csv

  `df = pd.read_csv(path,usecols=None,skiprows=None, nrows=None)`

* replace existing row in csv file part2
  ```
  a = {5:[0,0]}
  with open('data00.csv','w') as file:
      f = csv.writer(in_file)
      for line,row in enumerate(ff):
          data = a.get(line,row)
          f.writerow(data)
  file.close()
  ```

* replace existing row in csv file part1
  ```
  import csv
  ff = []
  with open('data0.csv','r') as file:
      f = csv.reader(in_file)
      ff.extend(f)
  file.close()
  ```

* read csv file with different dilimeter [link](https://docs.python.org/3/library/codecs.html#standard-encodings
)

  `pd.read_csv('path', sep=',', encoding='utf-8')`

* read excel
  - read first sheet `df = pd.read_excel('path')`
  - read second sheet `df = pd.read_excel('path', sheet_name=1)`
  - list sheet name `pd.ExcelFile('Academy data scientist - data for case study.xlsx').sheet_names`
  - read certain sheet `pd.ExcelFile('Academy data scientist - data for case study.xlsx').parse('sheetname')`

* remove index from/to csv file
  ```
  df.to_csv(filename ,  index = False, encoding='UTF-8')
  df = pd.read_csv(filename ,  index = False)
  ```
* read tsv file
  - pandas `pd.read_csv('../input/train.tsv',delimiter='\t',encoding='utf-8')`

## create
* create df from list

  `df2=pd.DataFrame([[0,0,1,2]], columns=['deseas_bl','product_key','geog_key','discount_rate'])`

* create zeros df

  `df2=pd.DataFrame(0, index=np.arange(1), columns=['deseas_bl','product_key','geog_key','discount_rate'])`

* create a new dataframe
  ```
  pd.DataFrame(data={'col1':range(5), 'col2':range(5,10)})
  pd.DataFrame(np.random.randn(10,5), columns=list(map(lambda x: 'col'+str(x), range(1,6))))
  ```

* create a calendar
  - `df_out = pd.DataFrame({'date':pd.date_range(start='2000-01-01',end='2018-01-01')})`

* 

* 

* 

* 



## transpose
* wide to long [link](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.melt.html)

  `pd.melt(df,id_vars = ['geography','mitm_key','baseline_discount_rate'], value_vars=[hol1,hol2,hol3])`

* long to wide
  * if no index or catetory col [link](https://stackoverflow.com/questions/32651084/pivot-table-from-a-pandas-dataframe-without-an-apply-function)
  * long to wide [link](https://pandas.pydata.org/pandas-docs/stable/reshaping.html)

    - value are numeric `df = df.pivot_table(index=['geography','mitm_key'], columns='wide_col', values='seas_use')`
    - value are string `pd.pivot_table(df,index=['col1','col2'],columns='col3',values='col4',aggfunc='first')`

* one row to columns rest to values [link](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.pivot.html)

  `df.pivot(index='foo', columns='bar', values='baz')`

* rows to columns `df.T`


## aggregation

* Missing values will be removed from the aggregated groups. 
```
df = pd.DataFrame([[np.nan, 2, np.nan, 0], 
                   [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, 5],
                   [np.nan, 3, np.nan, 4]],
                  columns=list('ABCD'))

df
     A    B   C  D
0  NaN  2.0 NaN  0
1  3.0  4.0 NaN  1
2  NaN  NaN NaN  5
3  NaN  3.0 NaN  4

df[['A','B','D']].groupby(['A','B']).size()
A    B  
3.0  4.0    1
dtype: int64

```

* To include missing values in aggregated groups, replace the NaN value with other values before groupby.
** tips: can use dictionary to assign different values for different variables
```
values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
df.fillna(value=values)
A B C D
0 0.0 2.0 2.0 0
1 3.0 4.0 2.0 1
2 0.0 1.0 2.0 5
3 0.0 3.0 2.0 4
```


* calculate quantile [link](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.quantile.html)
  ```
  df.col.quantile(0.25)

  x=df.col.quantile([0.25,0.5,0.75])
  x[0.25] will be value of 25% on df.col
  ```
* calculate cumulative sum on list
  ```
  import numpy as np
  a = [4,6,12]
  np.cumsum(a) #array([4, 10, 22])
  ```
* groupby built-in functions [link](https://pandas.pydata.org/pandas-docs/stable/api.html#groupby)

* count unique elements in group. cannot user as_index=False when grouping and only can apply to single col

  `df.groupby([col1,col2]).col3.nunique()`

* apply multiple functions in groupby [link](https://stackoverflow.com/questions/14529838/apply-multiple-functions-to-multiple-groupby-columns)
  ```
  f = {'num1':['mean','sum'], 'num2':['median']}
  df.groupby(['cat1'], as_index=False).agg(f)
  ```

* calculate correlation
  * calculate 2 arrays
    `np.corrcoef(x.col1, x.col2)`

  * calculate all cross correlations in df
    `df.corr()`

* groupby with customized def/function
  ```
  def a_fun(x):
      ...

  df.groupby('col1').apply(a_fun)
  ```
* groupby to find most common elements
  - `df.groupby('indexcol').targetcol.agg(lambda i: i.value_counts().index[0])`
  
* about index in groupby
  - keep index `df.groupby('indexcol',as_index=False)['targetcol'].sum()`
  - however nunique() need use reset_index. if use as_index=False, actually wont keep original indexcol `df.groupby('indexcol')['targetcol'].sum().reset_index(drop=False)`
  - customized function as_index=False doesnt work as well, need reset_index

* different between next value and previous value
  - period is current-previous  `df.col.diff()`
  - period is 7 cells `df.col.diff(7)`

* rolling 
  - mean/median [link](http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.rolling_mean.html)
  ```
  df.col1.rolling(window=10).mean()
  df.col1.rolling(window=10).median()

  pd.rolling_mean(df.col1,windows,min_priods)
  ```
  - rolling by datetime [link](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rolling.html)
  ```
  df.col.rolling('2s',on='datetimecol').sum()
  ```
  - rolling attributions/computing operations [link](https://pandas.pydata.org/pandas-docs/stable/computation.html)

* groupby output to dataframe
  ```
  df.groupby(col1).mean().reset_index()
  df.groupby(col1).quantile(.7,interpolation='higher').reset_index() #higher 30% mean
  df.groupby(col1).quantile(.3,interpolation='higher').reset_index() #lower 30% mean
  ```

* groupby [link](http://pandas.pydata.org/pandas-docs/stable/groupby.html)

  `df.groupby(['col1','col2']).mean()`

* apply a function to each row
  - with 1 column `df['col'].apply(lambda x: afunction(x))`
  - with multiple columns 
  `df.apply(lambda i: afunction(i['col1'], i['col2']), axis=1)`

* SAS data step -- if/then logic equivalent [link](https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_sas.html#comparison-with-sas)
```
data tips;
    set tips;
    format bucket $4.;

    if total_bill < 10 then bucket = 'low';
    else bucket = 'high';
run;

In [12]: tips['bucket'] = np.where(tips['total_bill'] < 10, 'low', 'high')

In [13]: tips.head()
Out[13]: 
   total_bill   tip     sex smoker  day    time  size bucket
0       14.99  1.01  Female     No  Sun  Dinner     2   high
1        8.34  1.66    Male     No  Sun  Dinner     3    low
2       19.01  3.50    Male     No  Sun  Dinner     3   high
3       21.68  3.31    Male     No  Sun  Dinner     2   high
4       22.59  3.61  Female     No  Sun  Dinner     4   high
```




### Others
* adding prefix to all column names `df.columns = ["_".join(x) for x in df.columns.ravel()]`
```
import pandas as pd
import pandas.rpy.common as com
import numpy as np

data = com.load_data('Loblolly')
print(data.head())
#     height  age Seed
# 1     4.51    3  301
# 15   10.89    5  301
# 29   28.72   10  301
# 43   41.74   15  301
# 57   52.70   20  301

df = data.groupby('Seed').agg(
    {'age':['sum'],
     'height':['mean', 'std']})
print(df.head())
#       age     height           
#       sum        std       mean
# Seed                           
# 301    78  22.638417  33.246667
# 303    78  23.499706  34.106667
# 305    78  23.927090  35.115000
# 307    78  22.222266  31.328333
# 309    78  23.132574  33.781667

df.columns = df.columns.droplevel(0)
print(df.head())
yields

      sum        std       mean
Seed                           
301    78  22.638417  33.246667
303    78  23.499706  34.106667
305    78  23.927090  35.115000
307    78  22.222266  31.328333
309    78  23.132574  33.781667
Alternatively, to keep the first level of the index:

df = data.groupby('Seed').agg(
    {'age':['sum'],
     'height':['mean', 'std']})
df.columns = ["_".join(x) for x in df.columns.ravel()]
yields

      age_sum   height_std  height_mean
Seed                           
301        78    22.638417    33.246667
303        78    23.499706    34.106667
305        78    23.927090    35.115000
307        78    22.222266    31.328333
309        78    23.132574    33.781667
```

* renaming dataframe 
  - `df.rename(columns={"A": "a", "C": "c"})`
  - `df.columns = ({})`

* Series: One-dimensional ndarray with axis labels (including time series)
  - sorted_series = df.isnull().sum().sort_values(ascending=False)


* count unique combination of Major Minor [link](https://stackoverflow.com/questions/22105452/pandas-what-is-the-equivalent-of-sql-group-by-having)
  ```
  fullfusion_test_skus.drop_duplicates(['group_code','major','minor'])\
                    .groupby(['group_code'])\
                    .filter(lambda x: len(x)>1)\
                    .groupby(['group_code'])\
                    .group_code.count()\
                    .to_frame('uniq_mm')\
  ```
  or
  ```
  df1 = df.groupby('id')['airport'].nunique().reset_index().query('airport > 3')
  ```
  
* transpose dataframe columns into rows: df.melt [link](https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html) 
  ```
  In [41]: cheese = pd.DataFrame({'first': ['John', 'Mary'],
                                'last': ['Doe', 'Bo'],
                                'height': [5.5, 6.0],
                                'weight': [130, 150]})
  In [42]: cheese
  Out[42]: 
    first last  height  weight
  0  John  Doe     5.5     130
  1  Mary   Bo     6.0     150

  In [43]: cheese.melt(id_vars=['first', 'last'])
  Out[43]: 
    first last variable  value
  0  John  Doe   height    5.5
  1  Mary   Bo   height    6.0
  2  John  Doe   weight  130.0
  3  Mary   Bo   weight  150.0

  In [44]: cheese.melt(id_vars=['first', 'last'], var_name='quantity')
  Out[44]: 
    first last quantity  value
  0  John  Doe   height    5.5
  1  Mary   Bo   height    6.0
  2  John  Doe   weight  130.0
  3  Mary   Bo   weight  150.0
  ```
<<<<<<< HEAD
*Creating a new column based on if-elif-else condition
  ```
  df = pd.DataFrame({'A': [2,3,1],
                     'B': [2,1,3]})

  def f(row):
      if row['A'] == row['B']:
          val = 0
      elif row['A'] > row['B']:
          val = 1
      else:
          val = -1
      return val


  df['C'] = df.apply(f, axis=1)

  df
  ```

  - or use .loc
  ```
  df.loc[df['A'] == df['B'], 'C'] = 0
  df.loc[df['A'] > df['B'], 'C'] = 1
  df.loc[df['A'] < df['B'], 'C'] = -1
  ```
* equivalent to SQL group by ... having ...



* dataframe, view V.S. copy [link]:(https://stackoverflow.com/questions/48173980/pandas-knowing-when-an-operation-affects-the-original-dataframe)



=======
* add prefix to rename after aggregation `sum().add_prefix("Sum_of_")`


<<<<<<< HEAD
* add row number `GroupBy.cumcount(ascending=True)[source]` Number each item in each group from 0 to the length of that group - 1.


* dataframe boolean, can not use df.col1 == df.col2   should use df['df1'] == df['col2']

* also when creating new column, can not use df.new_col = ..., should use df['new_col'] = ...



=======
* update existing dataframe. df1.update(df2), which is an inplace operation




>>>>>>> 62e57cb1ea4f24ecda4b209fe2b7045ca7725df4
