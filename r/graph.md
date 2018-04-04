* Plots + lines with same range and axis(no x-axes)

  `plot(df$col,type='l',col='black')` `lines(df$col2,type='l',col='red')`

* Plots + lines with different range but draw in same axes

  `plot(range(data$date),range(c(data$col1,data$col2)),type='n')`
  `lines(data$date,data$col1,type='l')`
  `lines(data$date,data$col2, type = 'l', col='blue')`

  `plot()` `abline(h=mean(y$sales),v=0,col="red")`

* plot with equation/function
  - `curve(x^0,0,10, xlab = 'markting effort', ylab = 'demand', main='title string')`

* clean graph `dev.off()`

* Combine plots

  4 figures arranged 2 rows and 2 cols `par( mfrow = c(2,2))`

  1st row has 1 figure and 2nd row has 2 figures `layout( matrix( c(1,1,2,3), 2,2, byrow=TRUE))` 
  
  space: 1st col 3/(3+1), 2nd 1/(3+1) `<option> widths = c(3,1)`

  space: 1st row 1/(1+2), 2nd 2/(1+2) `Heights = c(1,2)`

* Legend

  * `legend( 'topright', legend=c('label 1','label 2'), fill=rainbow(2), col=2, cex=0.7, yjust=0, xjust=0)`
  
    fill: terrain.colors(), rainbow(), topo.colors(), heat.colors(), cm.colors() or rgb, hsv, hcl.

    col: if split into 2 cols. cex: text size. 

    xjust: move legend 0--left, 0.5--mid, 1--right. yjust: move legend 0--up, 0.5--mid, 1--bottom.

  * `legend("topleft",col=c("black","red"),lty=1,legend=c("37","14"),horiz = FALSE,bty = "n",cex=0.8,yjust = 0)`

    lty: weight of line

    horiz: horizontal or vertical

    bty: with box or not

* Multiple axis

  * 2 axis right/left

    `par(mar = c(5,4,4,5) +.1)` set space for axis    

    `plot( )`

    `par(new = TRUE)` 2nd axis can be added    

    `plot( ,type = "n", bty = "n", xaxt = "n", yaxt = "n")` no axis/label    

    `axis(4)` 2nd axis on the right =4    

    `mtext('label', side = 4, line =3)` label on the right =4 and position=3

  * More than 1 on one side

    `axis(4, line=4)`  add 3rd axis on the right =4 position=4 farther than =3 above

    `mtext( 'label', side = 4, line =6)`  position=6 further than =4

* Save or output a graph as picture

    `dev.copy(png, 'path/',width = 2000,height=800,units="px")`
    
    `dev.off()`
    
* Histogram with point below

    `hist(df$col, col='red', break=100)`
    
    `rug(df$col)` break: frequency, number of bars

* Boxplot

  `boxplot( target.col ~ class.col, data=df, col='green')` Multiple boxplots base class col

* Use category clol to colorful plot

  `with( df, plot( col1, col2, col = col2))`

#### GGPLOT2

* Barchart, show fill distribution in diff class on y

  `ggplot2::ggplot( df )` `+ ggplot2::aes( x = col1, y = col2, fill = col3)` `+ ggplot2::geom_bar( start = 'dentity')`

#### Bokeh

`library(rbokeh)`

* Plots

  `figure() %>%ly_points(col1, col2, data, color = "red", size = 15, hover = 'the speed is @col1)` 
  define color, dot size, pop label

* Straight Lines

  `figure() %>%ly_lines(lowess(data), legend = "lowess")`

* Arch lines

  `z <- lm(col1 ~ col2, data = data)`

  `p <- figure() %>%`
  	`ly_points(data, hover = data) %>%`
  	`ly_lines(lowess(data), legend = "lowess") %>%`
  	`ly_abline(z, type = 2, legend = "lm", width = 2)`

  `p` print out graph

* Complex dot size and color

  `n <- nrow(cars)`
  
  `ramp <- colorRampPalette(c("red", "blue"))(n)`  color has range
  
  `figure() %>% ly_points(cars, color = ramp, size = ceiling(cars$dist/10)*5)`  size is depending on value of dist
  
  OR `figure() %>% ly_points(cars, color = ramp, size = rank(cars$dist/10))`  size is depending on value of dist
  
#### Dygraphs



#### gridExtra

[sample codes](https://cran.r-project.org/web/packages/gridExtra/vignettes/tableGrob.html)

* simple graph `grid.table(df)`

* put color on each row
  - repeated a few colors
  ```
  t1 <- ttheme_default(core=list(
    bg_params = list(rep(c("grey95", "grey90"),length.out=4))
    ))
  grid.table(df[1:10,'col'],theme = t1)
  ```
  - color column for each row
  ```
  t1 <- ttheme_default(core=list(
    bg_params = list( fill= df$color.col)
  ))
  grid.table(df[1:10,'col'],theme = t1)
  ```
* 