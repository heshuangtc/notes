### plotly
[Plotly](https://plot.ly/) is Python library is free and open source and makes interactive, publication-quality graphs online.

### matloplib
* create a plot

  * simple plot
    ```
    import matplotlib.pylab as plt
    plt.plot(ts_df)
    ```
  * Add a vertical line across the axes [link](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.axvline)

    `axvline(linewidth=4, color='r')`

  * multiple layers plot
    ```
    import matplotlib.pylab as plt
    graph_layer1 = plt.plot(ts_df, color='blue',label='Original')
    graph_layer2 = plt.plot(ts_series, color='red', label='a_label')
    plt.legend(loc='best')
    plt.title('a_title')
    plt.show(block=False)
    ```

  * multiple lines in one graph

    until execute `plt.figure()` again all previous one are in one chart
    ```
    plt.figure()
    plt.plot()
    plt.plot()
    ```
  
* plot options
  * size of graph [link](http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure)

    `plt.figure(figsize=(8, 6),dpi=80)`
  
* clean memory

in sumbline actually i need `plt.close()` to release memory not `plt.clf()`
  ```
  plt.clf() #clean figures
  plt.close()  #close windows
  ```

* save plot
  * save plots to one pdf
    ```
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pp = PdfPages('path/graphs.pdf')
    #creating plots with pp.savefig() in it
    pp.close()
    ```

  * save a plot as png or pdf
    ```
    import matplotlib.pyplot as plt

    plt.plot(df.col)
    plt.savefig('path/file.png')
    plt.savefig('path/file.pdf')
    ```

* show in sublime
  ```
  import matplotlib.pyplot as plt
  plt.plot(df.col)
  plt.show()
  ```



### pandas
* single line

  `df.plot(x='col1',y='col2')`

* multiple lines in one graph

  `df.plot()` df only has those data want to be in graph

* boxplot [link](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.boxplot.html)
  ```
  df.boxplot(column=y_col, by=x_col, grid=False)
  plt.suptitle('') #get rid of default title
  plt.title('new title')
  ```


### graphviz
* installation
  * need to install graphviz not only pip but also local exe programs
  * need to add path to system PATH for dot.exe
* save decision tree graph as png or pdf
  * use graphviz self
  `graph.render("iris")` this render cannot claim a path like this `'./output/file.png'`
  * use pydotplus
  ```
  from sklearn import tree
  import graphviz
  import pydotplus
  from sklearn.externals.six import StringIO

  model = tree.DecisionTreeRegressor()
  model = model.fit(df_train,df_hist.target)
  importance = model.feature_importances_
  graph_data = StringIO()
  tree.export_graphviz(model,feature_names=df_train.columns,out_file=graph_data)
  graph = pydotplus.graph_from_dot_data(graph_data.getvalue())

  graph.write_png('output/tree/tree_num.png')
  graph.write_pdf('output/tree/tree_num.pdf')
  ```
  * use pydot
  ```
  from sklearn import tree
  import graphviz
  import pydotplus
  from sklearn.externals.six import StringIO

  model = tree.DecisionTreeRegressor()
  model = model.fit(df_train,df_hist.target)
  importance = model.feature_importances_
  graph_data = StringIO()
  tree.export_graphviz(model,feature_names=df_train.columns,out_file=graph_data)
  graph = pydotplus.graph_from_dot_data(graph_data.getvalue())

  graph[0].write_png('output/tree/tree_num.png')
  graph[0].write_pdf('output/tree/tree_num.pdf')
  ```