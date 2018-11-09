[data structure tutorial](https://docs.python.org/3.5/tutorial/datastructures.html)


### number

* log a number `import numpy as np` `np.log(df.col1)`

* split list by same length chunk `range(0,9,5)` step 5 split 0-8

* round decimals `round(1.244444,2)`

* from string to num `float(str)``int(str)`

* generate random numbers [link](https://docs.scipy.org/doc/numpy/reference/routines.random.html)
  ```
  import numpy as np
  np.random.randint(low, high=None, size=None)
  np.random.randn(dim0,dim2...) #dimension
  ```
* define inf value. cannot directly use inf as it is treated as object name instead of infinite number `float('inf')`

* bool + int `True +0 =1`

* divide
  `int/int = int (py2) but float(py3)``int/float = float`

* fractions
  - `from fractions import Fraction; Fraction(16, -10)`
* check if nan value
  - `import math; math.isnan(x)`
  - `x == np.nan`
  - `x == float('nan')`
* geographic longitude and latitude
  - calculate distance between 2 longitude/latitude
    ```
    from math import radians, sin, cos, acos
    
    def f_earth_distance(plat,plon,dlat,dlon):
    slat = radians(float(plat))
    slon = radians(float(plon))
    elat = radians(float(dlat))
    elon = radians(float(dlon))
    try:
        dist = 6371.01 * acos(sin(slat)*sin(elat) + cos(slat)*cos(elat)*cos(slon - elon))
    except:
        dist = 0
    return round(dist,2)
    ```
  - find zipcode based on longitude/latitude
    ```
    ```



### string
* assign value to string, make string as name/object/variable
  - obj[string] = value `setattr(obj, string, value)`
  - string = value `exec(string + '= value')`
  - df_string = value `'df_'+exec(string + ' = value')`
* find certain str index in string 
  - forewards `string.index('a')`
  - backwards `string.rindex('b')`
* find certain str in list element
  - `[s for s in a_list if "abc" in s]`

* split string into list `string.split(',')`

* number to alphabet `chr(5)`
* special character
  - `import string;string.punctuation`

* library pickle [link](https://docs.python.org/2/library/pickle.html#pickle.HIGHEST_PROTOCOL)

  * save model object as pkl file
    ```
    import pickle

    output = open('model.pkl', 'wb')
    pickle.dump(ml_out, output)
    output.close()
  * read model object from pkl
    output = open('model.pkl', 'rb')
    ml_out = pickle.load(output)
    output.close()
    ```
    
* format string 

  no need to change int to string in additional step

  `'here is a {0} and {1}'.format(4,6)` output is `here is a 4 and 6`

  `"here is a '{}'".format('str')` output is `here is a 'str'`
  
  `s = 'here is a string {0} and rest {1}'.format('lalala',1111)` output is `here is a string lalala and rest 1111`

* to uppercase or lowercase
  
  `str.lower()` `str.upper()`

* extract values from string
  - number from string
  ```
  import re
  re.findall('\d+',string)
  re.findall('[0-9]',string)
  ```
  - character from string
  ```
  import re
  re.findall('\D+',string)
  ```
* add string to every element in a list `[i+'string' for i in a_list]`

* regular expression subset string `df.col.str.extract('(^[0-9]+/[0-9]+)',expand=False)`
[test re](https://regex101.com/#python)

* a-z  chr(97)-chr(122), A-Z chr(65)-chr(90)

* convert from other data type

  * alphabet to number `ord('a')` output is 97

  * bool to string `str(True)`
  
  * category string to number `df.col = pd.Categorical(df.col).codes`

* use `\'` to escape the single quote or other special symbol

* contact string 

  `string1 + '_' + string2`

  `','.join(['a','bb'])` output is `a,bb`

* remove spaces front and tail `string.strip()`


### datetime
* current time
  - `import time; time.time()`

* define a repeated holiday/event in calendar
  * dateutil pkg [github source](https://github.com/dateutil/dateutil)
    ```
    from dateutil import rrule
    rs = rrule.rruleset()
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=start_dt, until=end_dt, bymonth= 1, byweekday= rrule.MO(3)))
    list(rs)
    ```
  * pandas pkg [source code](https://github.com/pandas-dev/pandas/blob/master/pandas/tseries/holiday.py)
      ```
      from pandas.tseries.holiday import Holiday,AbstractHolidayCalendar,MO
      rules = [Holiday('h1', month=1, day=1,offset=pd.DateOffset(weekday=MO(3)))]
      x = AbstractHolidayCalendar(rules=rules)
      x.holidays(start_dt,end_dt)
      ```
    
  * 3rd monday(+2weeks, it is monday). use pandas dateoffset
  
    `Holiday('h1', month=1, day=1,offset=pd.DateOffset(weekday=0,weeks=2))`
    
* us default holiday list [link](https://pandas.pydata.org/pandas-docs/stable/timeseries.html)
```
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
calendar().holidays(start='2016-12-01',end='2017-12-31',return_name=True)
```

* list of dates `pandas.date_range(start=None, end=None, periods=None, freq='D', tz=None, normalize=False, name=None, closed=None, **kwargs)`

* moving date
  * date +1 year 

    ```
    import datetime
    datetime.datetime.now()+datetime.timedelta(years=1)
    ```
  * date +1day
    ```
    import datetime
    datetime.datetime.now() + datetime.timedelta(days=1)
    ```
    
* change date's year
  ```
  adate = pd.to_datetime(df,col1)[0]
  adate = adate.replace(year=2016)
  ```

* string to date `datetime.strptime('2016-1-5','%Y-%m-%d').date()`

* create a new date object
  ```
  from datetime import date
  date(2017,8,2)
  ```
* convert timedelta to int
  `timedelta/np.timedelta64(1, 'D')`
* different between 2 dates
  - datetime.datetime.now() - pd.to_datetime(df.date,format='%Y-%m-%d')



### list
* find index of item in a list `["foo", "bar", "baz"].index("bar")`
* find every 2 elements in first 5 elements of a list `alist[1:5:2]`
* delete element from original list
  ```
  x = [1,2,3]
  x.pop(0) #delete by index
  ```

* is list or not `isinstance(y_col,list)`

* find closest value in a list `min(myList, key=lambda x:abs(x-myNumber))`

* random pick an item from list [link](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html)
```
import numpy as np
np.random.choice(df.col)
np.random.choice(df.col, replace=False) #pick unique values
```

* random pick an item from list. not sure why df.col doesnt work
  ```
  import random
  random.choice(a_list)
  ```
  
* add element to list
  ```
  x = [1, 2, 3]
  x.append([4, 5]) #output is [1, 2, 3, [4, 5]]
  x.extend([4, 5]) #output is [1, 2, 3, 4, 5]
  x.insert(0,99)   #output is [99,1,2,3]
  ```

* subset a list based on same length of another list with condition

  select relative position x>5 in a `a[np.where(x>5)]`
  
* split a number list by same distance [link](https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.arange.html)

  `np.arange(startnumber,endnumber, length)`

* unlist a list
  - `flat_list = [item for sublist in nested_list for item in sublist]`
* regular expression
  - filter elements with match
    ```
    import re
    [i for i in ls_col if re.match('[0-9]+',i)]
    ```
  - 




### function
* use string as function pointer
  ```
  a = 'lower'
  getattr('AAAA',a)()
  ```
  output is 'aaaa'

* `**keywords` a list of  `kwarg=value`. must after `*args`.

* `*arguments` or `*args` will be wrapped up in a tuple, dont need object name.

>> note: The default value is evaluated only once. default value will be changed in def. second time call def, default value is not the same as the first time.


### tuples and sequences
* tuple [link](https://docs.python.org/3.5/library/stdtypes.html#typesseq)
* tuples containing 0 or 1 items
* Empty tuples are constructed by an empty pair of parentheses `empty = ()``len(empty)==0`
* a tuple with one item is constructed by following a value with a comma
`singleton = 'hello',``len(singleton)==1`
* join elements in each tuple in a list
`(' '.join(w) for w in sixgrams)`


### set
* A set is an unordered collection with no duplicate elements
* set operation
  ```
  a = set('abracadabra')
  b = set('alacazam')
  a  # unique letters in a
  a - b  # letters in a but not in b
  a | b   # letters in a or b or both
  a & b # letters in both a and b
  a ^ b # letters in a or b but not both
  ```
  
* difference between 2 sets `set(a)^set(b)` a and b can be list then this is used for comparing 2 lists


### dictionary
* extract key if exist else use default value `dict.get(key, default_value)` [link](https://docs.quantifiedcode.com/python-anti-patterns/correctness/not_using_get_to_return_a_default_value_from_a_dictionary.html)


* `tel = {'jack': 4098, 'sape': 4139}`

* `list(a_dict.keys())`

* The dict() constructor builds dictionaries directly from sequences of key-value pairs
`dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])`


### array
* remove nan from array `array[~np.isnan(array)]`

* find common elements in 2 array [link](https://docs.scipy.org/doc/numpy/reference/generated/numpy.intersect1d.htm)
`numpy.intersect1d(ar1, ar2, assume_unique=False)`

* remove by row or col `scipy.delete(A,1,0)`

* delete items from list `np.delete(array, index)` `array.remove(index)`

* add col to matrix/np.array
  ```
  a = np.array([list(range(1,5)),list(range(3,7))])
  b = np.array(list(range(0,2)))
  b = b.reshape(2,1)
  c = np.append(a,b,1) 
  ```

### class
* define object1 in a class, which can be called by class functions to get global variables
  ```
  class name:
  object1=1
  def f_a(input=name.object1):
  pass
  ```

* once use @staticmethod. means this function is isolated from other parts of this class. cannot use self variables or object2. f_a inputs needed.
  ```
  class name:
  object2=1
  def __init__(self):
  object1=1
  @staticmethod
  def f_a(inpu1,input2):
  pass
  ```

* non-self variable cannot directly be used in function. also this function cannot separated called in another script as missing input
  ```
  class name:
  object2 =2
  def function(self,input=object2)
  pass
  ```

* when use class non-self variable:
name.object2
  ```
  class name:
  object2 = 2
  def `__init__`(self):
  self.object1 = 1
  ```


* when use class self variable:
name().object1
  ```
  class name:
  def `__init__`(self):
  self.object1 = 1
  ```

* init is to define class global objects
new is class output when directly call class
  ```
  class name:
     def `__init__`(self):
         pass
     def `__new__`(self):
         pass
  ```

* only those defined in self, can be used in function. if not defined in `__init__`, then function need all outside inputs
  ```
  class name:
     def `__init__`(self,input1):
     self.object1 = 1
     self.object2 = input1

     def function():
          pass
  ```


### config
* configuration file can be imported directly[link](https://docs.python.org/3/library/configparser.html#configparser.ConfigParser.get)

* 

* 
