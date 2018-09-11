### plotly
[Plotly](https://plot.ly/) is Python library is free and open source and makes interactive, publication-quality graphs online.

### matloplib
* `%matplotlib inline`
* create a plot simple charts 
    - line chart [plt.plot](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html)
      ```
      import matplotlib.pylab as plt
      plt.plot(ts_df)
      ```
    - hist charts [link](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html)
      `plt.hist(df.col, bins=20)`
    - pyplot.hist
      ```
      bins = np.linspace(startnum,endnum,numofcuts)
      from matplotlib import pyplot
      pyplot.hist(df['col'],bins=bins,alpha=.5,normed=True) #norm if normalized plot, alpha how dark the bar
      ```
    - bar charts
    - scatter plot [plt.scatter](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html)
      `plt.scatter(x,y)`
    - parallel coordinates graph
    - 
  
* Add a vertical line across the axes [link](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.axvline)

  `axvline(linewidth=4, color='r')`

* multiple plot
  - multiple layers plot
    ```
    import matplotlib.pylab as plt
    graph_layer1 = plt.plot(ts_df, color='blue',label='Original')
    graph_layer2 = plt.plot(ts_series, color='red', label='a_label')
    plt.legend(loc='best')
    plt.title('a_title')
    plt.show(block=False)
    ```

  - multiple lines in one graph

    until execute `plt.figure()` again all previous one are in one chart
    ```
    plt.figure()
    plt.plot()
    plt.plot()
    ```
  - multiple hist in one graph - overlap
    ```
    plt.hist(x[x.Pclass==1].Survived, bins=2, alpha=0.5, label='class1')
    plt.hist(x[x.Pclass==2].Survived, bins=2, alpha=0.5, label='class2')
    plt.legend(loc='best')
    ```
  - multiple hist in one graph - side by side
    ```
    plt.hist([x[x.Pclass==1].Survived.values,x[x.Pclass==2].Survived.values], bins=5, alpha=0.4, label=['pclass1','pclass2'])
    plt.legend(loc='best')
    ```
  - multiple hist - overlap
    ```
    bins = np.linspace(startnum,endnum,numofcuts)
    from matplotlib import pyplot
    pyplot.hist(df['col'],bins=bins,alpha=.5,normed=True) #norm if normalized plot, alpha how dark the bar
    pyplot.hist(df['col'],bins=bins,alpha=.5,normed=True)
    pyplot.legend(loc='upper right')
    ```
  - 2 axis plots
    ```
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, y1, 'g-')
    ax2.plot(x, y2, 'b-')
    ```
  - multiple subplots arrange multiple plots
    ```
    # plt.subplot(nrows, ncols, index, **kwargs)
    # nrow and ncols define how many plots in rows/cols
    # index define plot index/order

    plt.figure()
    plt.subplot(221) #left top one in 2*2 net
    plt.plot(x,y)

    plt.subplot(222) #right top one in 2*2 net
    plt.plot(x,y)

    plt.subplot(223) #left bottom one in 2*2 net
    plt.plot(x,y)
    ```
  - groupby to generate plots
    ```
    plt.figure(figsize = (10,8))
    df.groupby('group_col').plot_col.count().plot(kind="bar")
    plt.title("plot_col per group_col")
    plt.show()
    ```
* plot options
  - size of graph [link](http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure)
    `plt.figure(figsize=(8, 6),dpi=80)` size(width,height)
  - pandas hist plot size[link](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.hist.html) `DataFrame.hist(data,column=None, by=None, figsize=None)`
  - add label to x or y axis `plt.xlabel('Smarts',fontsize=18)` `plt.ylabel('Probability',fontsize=18)`
  - add title `plt.title('Histogram of IQ')`
  - figure axis range
    `plt.ylim(-70.999,-71.175);plt.xlim(42.236,42.395)`
  - scatter plot dot size `pd.scatter(y,x,s=value)`
  - plot plot dot size `markersize=3`

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

* 3d graph
  - basic 3d graph
  ```
  import matplotlib.pyplot as plt
  from mpl_toolkits import mplot3d
  ax = plt.axes(projection='3d')
  ax.scatter3D(x,y,z)
  ax.set_xlabel('accommodates')
  ax.set_ylabel('minstay')
  ax.set_zlabel('availability_365')
  ax.view_init(60, 35) #vertical,horizontal
  ```


### pandas
[pandas doc](https://pandas.pydata.org/pandas-docs/stable/visualization.html#visualization-parallel-coordinates)
* single line
`df.plot(x='col1',y='col2')`

* multiple lines in one graph
`df.plot()` df only has those data want to be in graph

* multiple hist graph by group
  - `df['value_col'].hist(by=df['group_col'])`
  - `df.groupby('group_col').hist('value_col')`

* boxplot [link](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.boxplot.html)
  ```
  df.boxplot(column=y_col, by=x_col, grid=False)
  plt.suptitle('') #get rid of default title
  plt.title('new title')
  ```
* parallel coordinates graph
  ```
  from pandas.plotting import parallel_coordinates
  import matplotlib.pyplot as plt
  plt.figure()
  parallel_coordinates(df,'class_label')
  ```
* lag plot
* pie chart
```
df.col.plot.pie(figsize=(5,5))
```
* 

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

### folium geog graph
* doc
  - quick start guide[link](http://folium.readthedocs.io/en/latest/quickstart.html)
  - sample [link](https://github.com/python-visualization/folium/blob/master/examples/MarkerCluster.ipynb) [link](http://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/MarkerCluster.ipynb) [national wide](https://python-visualization.github.io/folium/quickstart.html)
* simple geog graph
  ```
  import folium
  map_nyc = folium.Map(location=(40.7, -74), zoom_start=10, tiles='Stamen Toner')
  map_nyc
  map_nyc.save('./map.html')
  ```
* add marker/circle marker
  ```
  import folium
  map_nyc = folium.Map(location=(40.7, -74), zoom_start=10, tiles='Stamen Toner')
  folium.Marker(location = [40.7, -74], popup = 'picked here').add_to(map_nyc)
  folium.CircleMarker(location = [40.7, -74],popup = 'a string',
    color='#3186cc', fill_color='#3186cc',radius=50).add_to(map_nyc)
  map_nyc
  map_nyc.save('./map.html')
  ```
* marker type
  - cloud `folium.Marker(location = [40.7, -74], popup = 'picked here',icon=folium.Icon(icon='cloud')).add_to(map_nyc)`
  - color `icon=folium.Icon(color='green')`
  - 
* clustering marker
  - normal way
  ```
  import folium
  from folium.plugins import MarkerCluster
  map_nyc = folium.Map(location=(40.7, -74), zoom_start=10, tiles='Stamen Toner')
  marker_cluster = MarkerCluster().add_to(map_nyc)
  folium.Marker(location = [a_latitude,a_longitude],
  popup='picked here').add_to(marker_cluster)
   map_nyc
  map_nyc.save('./map.html') 
  ```
  - faster way
  ```
  import folium
  from folium.plugins import FastMarkerCluster
  map_nyc = folium.Map(location=(40.7, -74), zoom_start=10, tiles='Stamen Toner')
  FastMarkerCluster(data=list(zip(ls_latitude, ls_longitude))).add_to(map_nyc)
  map_nyc
  map_nyc.save('./map.html') 
  ```
  - with different layers
  ```
  map_nyc = folium.Map(location=(40.7, -74), zoom_start=10, tiles='Stamen Toner')

  marker_cluster = MarkerCluster(
      locations=list(zip(df_input.Pickup_latitude, df_input.Pickup_longitude)),
      popups='pickup',
      name='pickup',
      overlay=True,
      control=True).add_to(map_nyc)

  marker_cluster = MarkerCluster(
      locations=list(zip(df_input.Dropoff_latitude, df_input.Dropoff_longitude)),
      popups='dropoff',
      name='dropoff',
      overlay=True,
      control=True).add_to(map_nyc)
  
  folium.LayerControl().add_to(map_nyc)

  map_nyc.save('./map.html')

  ```
* circle marker


## wordcloud
* sample
  ```
  from wordcloud import WordCloud, STOPWORDS
  plt.figure(figsize = (15,15))

  stopwords = set(STOPWORDS)

  wordcloud = WordCloud(
                            background_color='black',
                            stopwords=stopwords,
                            max_words=1000,
                            max_font_size=120, 
                            random_state=42
                           ).generate(str(df[str_col]))

  print(wordcloud)
  fig = plt.figure(1)
  plt.imshow(wordcloud)
  plt.title("WORD CLOUD - str_col")
  plt.axis('off')
  plt.show()
  ```
