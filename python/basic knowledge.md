### basic
* [PEP8](https://www.python.org/dev/peps/pep-0008/)

* execute another script(not fun or class)

  once run this another script will be run
  then don't need to be on top of import
  use it where needed
  `import another_script`

* a variable exist or not `'var' in globals()`

* syntax: wrap too long line
  ```
  long_python_codes \\
  continue_long_python_codes
  ```

* Intra-package References
  ```
  from package.path1.path2.pyfile import module1
  == from . import module1
  from ..pyfile2 import module2
  from ...path22.pyfile3 import module3
  ```

* execute modules in package
  ```
  from package.path1.path2.pyfile import module1
  package.path1.path2.pyfile.module1(...)

  import all modules in pyfile
  from package.path1.path2.pyfile import *
  ```

* 'Compiled' Python files

* Executing modules as scripts by adding this code at the end of your module
  ```
  if __name__ == "__main__":
      run_something
  ```

* external execute [link](https://docs.python.org/3.5/tutorial/modules.html)
`python filename.py inputvalue`

* remove/delete an element or a list
  ```
  del alist[0]
  del alist
  del [df1, df2]
  ```
  
* In interactive mode(IPython), the last printed expression is assigned to the variable `_`

* compress files
  * gzip
    ```
    import gzip
    from subprocess import check_call
    check_call(['gzip', fullFilePath])
    ```
* compress directory
  * shutil
    ```
    import shutil
    shutil.make_archive(output_filename, 'zip', dir_name)
    ```
* ignore warnings
  ```
  import warnings
  warnings.filterwarnings('ignore')
  ```

* combination
  - combination of 2 'import itertools;itertools.combinations(alist, 2)'




## install
* Git clone URL
  
  set up git in command, find git.exe then add to PATH

* list modules in library

  The built-in function `dir()` is used to find out which names a module defines.`dir()` does not list the names of built-in functions and variables.
  `import builtins; dir(builtins)`

* check current python version
  `import sys;print(sys.version)`

* install package
  `pip install package_name`

* install certain version

## control flow
* Comparing Sequences
  ```
  [1, 2, 3]              < [1, 2, 4]
  'ABC' < 'C' < 'Pas' < 'Pyth'
  (1, 2, 3, 4)           < (1, 2, 4)
  (1, 2)                 < (1, 2, -1)
  (1, 2, 3)             == (1.0, 2.0, 3.0)
  (1, 2, ('aa', 'ab'))   < (1, 2, ('abc', 'a'), 4)
  ```

* `not` has the highest priority and `or` the lowest. example: A and not B or C means (A and (not B)) or C.
 
* equal or not
  ```
  in / not in
  is / is not
  a < b == c means a<b and b==c
  ```

* print """...""" contents

* `print(my_function.__doc__)`

* anonymous function `lambda a, b: a+b`

* The `pass` statement does nothing. It can be used when a statement is required syntactically but the program requires no action

* `continues` with the next iteration of the loop

* `breaks` out of the smallest enclosing for or while loop

* loop
  * while loop
  ```
  x = 0
  while x<10:
      print(x)
      x=+1
  ```

  * for loop
    ```
    for i in range(3):
      print(i)
    ```
  
    one line for loop with one paramter
    `[x**2 for x in range(10)]`
    
    one line for loop with 2 paramters
    `[(x, y) for x in [1,2,3] for y in [3,1,4] if x != y]`

* if statement
  ```
  if x<0:
      print()
  elif x == 0:
      print()
  else:
      print()
  ```
  
  one line if statement
  `print('Yes') if fruit == 'Apple' else print('No')`


## load files
* if a file in folder
  * way 1
    ```
    import os.path
    os.path.isfile('./data/filename.csv')
    ```
* list all files in folder
  * way1
    ```
    import glob, os
    os.chdir('./data/train_subset/')
    for i_file in glob.glob('*.csv'):
        print(i_file)
    ```
  * way2
    ```
    import os
    for file in os.listdir('./data/train_subset/'):
        if file.endswith('.csv'):
            print(file)
    ```
  * way3
    ```
    import os
    for root, dirs, files in os.walk('./data/train_subset/'):
        for file in files:
            if file.endswith('.csv'):
                print(file)
    ```
  * file without certain words
    ```
    import glob, os
    os.chdir('./JANUS/')
    for i_file in glob.glob('*'):
        if not i_file.endswith('.7z'):
            print(i_file)
    ```
    
* read a certain row
  ```
  with open('filepath/filename.csv','r') as f:
  for i,iline in enumerate(f):
  if i == 25:
          # 26th line
         print(aline)
  f.closed
  ```

* read file by row [link](https://docs.python.org/3/tutorial/inputoutput.html)
  ```
  with open('filepath/filename.csv','r') as f:
  f.read() #all contents
  f.read(10) #first 10 characters
  f.readline() #read a line
  f.closed
  ```

## save files

* save as excel xlsx
  * xlsxwriter links
  [link tutorial](http://xlsxwriter.readthedocs.io/tutorial01.html)
  [link format](http://xlsxwriter.readthedocs.io/format.html)
  [link example](http://xlsxwriter.readthedocs.io/example_merge1.html)
  
  * xlsxwriter with pandas
  
  need to create a xlsx file and close it. then reopen write mode with pandas and save dataframe to. this can add multiple sheets to one xlsx file.
  ```
    workbook = xlsxwriter.Workbook('./output/{}.xlsx'.format(filename))
    workbook.close()
    workbook = pd.ExcelWriter('./output/{}.xlsx'.format(filename), engine='xlsxwriter')
    df1.to_excel(workbook,sheet_name='final1',index=False)
    df2.to_excel(workbook,sheet_name='final2',index=False)
    workbook.close()
  ```
  
  * merge cells in xlsx
    ```
    import xlsxwriter
    workbook = xlsxwriter.Workbook('merge1.xlsx')
    worksheet = workbook.add_worksheet()
    merge_format = workbook.add_format()
    worksheet.merge_range('B4:D4', 'Merged Range', merge_format)
    ```
