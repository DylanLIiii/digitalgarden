---
{"dg-publish":true,"permalink":"/personal-page/matplotlib-and-seaborn-cookbook/"}
---

****Update Time : April 11, 2022**** 
****CheatSheet Here****
****Download dataset** **[****HERE****]****(****https://dsc.cloud/ce4dcf/stock_prices****)****, the data is from the JPX.****

<aside>

📌 ****If you want to execute the jupyter notebook, click** **[****here****]****(****https://deepnote.com/project/Untitled-project-FQNVFG8hTzu0-l1BWxxroA/%2F%E5%8F%AF%E8%A7%86%E5%8C%96%20Cookbook.ipynb****)****.****

  

</aside>

  

---

  

# Quick Start
Matplotlib provides two ways to create plots
- OOD - Users can new  a figure and an axis object and call methods of them to create plots
    - Used to create complex plots
- Pylot - Users use Pylot to create plots, and it manages all figures and axis
    - Use to create sample plots quickly.
Seaborn is a high-level package of Matplotlib, and it provides kinds of plots to accomplish the specific target  
- easier than Matplotlib and provide a more convenient
- Good for Pandas
For the sake that Seaborn is a high-level implementation of Matplotlib, it also offers two ways to create plots
- OOD : same as Matplotlob. There is also a lot of direct use of Axes in Seaborn's diagram function. Any type of this diagram has been explicitly displayed in the function name. This figure is drawn using Axes. For example, `Sns.ScatterPlot, Sns.LinePlot, Sns.Barplot,` etc. AXES drawing can set the elements of the graphs in some ways before Matplotlib.
# Matplotlib
## Anatomy of a figure
<aside>

📌 This part is partially for OOD in matplotlib

  

</aside>

- Major tick
- Minor tick
- Major tick label
- Minor tick label
- Y-axis label
- X-axis label
- Figure
- Axes
- Line
- Grid
- Legend
- Marker
## Modes

- `matplotlib.pyplot.ion()` to set the interactive mode ON

- `matplotlib.pyplot.ioff()` to switch off interactive mode

- `matplotlib.is_interactive()` to check whether interactive mode is ON(True) or OFF(False)

  

## Basic Plots

  

### Prepare

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00002-7d6bb96b-d301-457e-86e9-115a2e27ba9b?height=481](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00002-7d6bb96b-d301-457e-86e9-115a2e27ba9b?height=481)

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00004-26aa58ac-b0b3-4847-a0b6-0ec50dfd1b44?height=189.1875](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00004-26aa58ac-b0b3-4847-a0b6-0ec50dfd1b44?height=189.1875)

  

### Plot

  

[Document Here](http://t.cn/A66EVyig) — Matplotlib

  

1. Plot with a single input

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00005-cff8c697-9950-4563-beed-b34ab232a0ea?height=518.1875](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00005-cff8c697-9950-4563-beed-b34ab232a0ea?height=518.1875)

  

1. Plot with two inputs

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00008-bca6697b-78cb-41ca-ad26-c77c9d812cd5?height=500.1875](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00008-bca6697b-78cb-41ca-ad26-c77c9d812cd5?height=500.1875)

  

Using scatter is a more intelligent idea to show the correlation between the Volume and the Open.

  

### Scatter Plot

  

[Document Here](http://t.cn/A66ExAtX) — Matplotlib

  

Scatter plot is used to compare two variables and see if there is any correlation between them. If there are distinct clusters/segments within the data, it will be clear in the scatter plot.

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00011-24b32df5-3ea9-40ff-9d32-4b18b780760a?height=429.1875](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00011-24b32df5-3ea9-40ff-9d32-4b18b780760a?height=429.1875)

  

To plot a scatter plot, we can use the following code:

  

- if labels are int numbers, we can use labels as c[color] directly.

- if labels are string, we can use df.map to convert them to int numbers or other parameters c[color] can call.

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00013-e8f01280-3b08-4683-b09a-2a163d9188f4?height=636.859375](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00013-e8f01280-3b08-4683-b09a-2a163d9188f4?height=636.859375)

  

To plot a scatter by category using pylot.plot.

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00015-3e621ab1-b4c9-4452-a2dc-66163433a0ae?height=656.671875](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00015-3e621ab1-b4c9-4452-a2dc-66163433a0ae?height=656.671875)

  

### Bar Plot

  

[Document Here](http://t.cn/A66EdqSU) — Matplotlib

  

Bar plot is a graph that uses bars to compare different categories of data. Bars can be shown vertically or horizontally based on which axis is used for the categorical variable.

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00017-55e334ab-d831-467f-b825-2e56210b5090?height=416](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00017-55e334ab-d831-467f-b825-2e56210b5090?height=416)

  

Apparently, around index 1000, there is an outlier value.

  

## Heatmap

  

[Document Here](https://shrtm.nu/Eedo) — Matplotlib

  

Heatmap plot is a graph that uses a heatmap to show a matrix that displays data with color changes.

  

- Relative degree

- Correlation for lots of categories

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00020-92fd3614-d0c1-43b3-9444-6d5d69b3b02e?height=812](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00020-92fd3614-d0c1-43b3-9444-6d5d69b3b02e?height=812)

  

# Seaborn

  

In seaborn, ****drawing commands are classified according to drawing targets.****

  

- ****Basic command****

    - `sns.relplot`

        - ****relationship**** among two or more variables

    - `sns.catplot`

        - ****Classification**** diagram:

        - ****Classified distribution**** diagram: show the distribution of each category

        - ****Classification statistics**** diagram: are based on the classification, counting the number or ratio of the data under each category.

    - `sns.displot`

        - Diagram for ****single variable distribution with histogram****

    - `sns.jointplot`

        - Diagram for ****two-variable distribution****

    - `sns.pairplot`

        - Draw the diagram to show ****the relationship between several fields**** in a dataset at one time

    - `sns.regplot`

        - ****Linear regression**** diagram

    - `sns.lmplot`

    - `sns.residplot`

        - ****Residuals of linear regression**** diagram

    - `sns.heatmap`

        - [Heatmap](https://www.notion.so/Heatmap-0dde6879ab614fdbb36d758221c4818d)

- ****Advance command****

- [ ]  TODO

- ****Adjust figure****

  

Use `height` and `apsect` to adjust figure size. `height * aspect`  is length.

  

## Universal Parameters

  

Generally, These parameters are used for controlling the style of the diagram

  

- `hue` control colors of variables, Inputs for plotting long-form data

- `bin` control the number of bar, hist, or other input_point to show in the diagram

- [ ]  TODO

  

## Basic Diagram

  

1. `sns.relplot`

  

[Document Here](https://shrtm.nu/x0l2)

  

****It provides two sorts of diagrams****

  

- ********`scatterplot()`******** (with `kind="scatter"`; the default)

- ********`lineplot()`******** (with `kind="line"`)

- ****Command input parameters****

    - 

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00022-0294d4e0-46ae-461d-bfbc-5fb1108a35dc?height=1326](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00022-0294d4e0-46ae-461d-bfbc-5fb1108a35dc?height=1326)

  

1. `sns.catplot`

  

[Document Here](https://shrtm.nu/nqpX)

  

****Categorical scatterplots****:

  

- ********`stripplot()`******** (with `kind="strip"`; the default)

- ********`swarmplot()`******** (with `kind="swarm"`)

  

****Categorical distribution plots****:

  

- ********`boxplot()`******** (with `kind="box"`)

- ********`violinplot()`******** (with `kind="violin"`)

- ********`boxenplot()`******** (with `kind="boxen"`)

  

****Categorical estimate plots:****

  

- ********`pointplot()`******** (with `kind="point"`)

- ********`barplot()`******** (with `kind="bar"`)

- ********`countplot()`******** (with `kind="count"`) : bar diagram with the count of each catagory

    This block shows different kinds of CatPlot

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00023-c4e54555-3dc8-4731-add2-0f3a1cbc54ba?height=3170.2578125](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00023-c4e54555-3dc8-4731-add2-0f3a1cbc54ba?height=3170.2578125)

  

1. `sns.displot`

  

[Document Here](https://shrtm.nu/60Fe)

  

- ********`histplot()`******** (with `kind="hist"`; the default)

- ********`kdeplot()`******** (with `kind="kde"`)

- ********`ecdfplot()`******** (with `kind="ecdf"`; univariate-only)

  

When using histplot , we often hope that the hist and the KDE can be together. Using ``kde=True`` in `histplot()` to do this.

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00024-38af53f4-2387-4e98-87c2-1cc493b04df3?height=923.1875](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00024-38af53f4-2387-4e98-87c2-1cc493b04df3?height=923.1875)

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00025-bf4b6c3d-6d27-4c5d-afd9-966d8dcd7f41?height=923.1875](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00025-bf4b6c3d-6d27-4c5d-afd9-966d8dcd7f41?height=923.1875)

  

1. `sns.jointplot`

  

[Document Here](http://seaborn.pydata.org/generated/seaborn.jointplot.html#seaborn.jointplot)

  

- Kind : `“scatter” | “kde” | “hist” | “hex” | “reg” | “resid”`

  

This command can be regarded as the mix of a distribution and a kind of other plots which is in the Kind.

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00026-e672a650-b3bf-483b-9141-fdaf35dad399?height=592.1875](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00026-e672a650-b3bf-483b-9141-fdaf35dad399?height=592.1875)

  

1. `sns.pairplot`

  

[Document Here](http://seaborn.pydata.org/generated/seaborn.pairplot.html#seaborn.pairplot)

  

Pairplot can plot the relationship between several fields in a dataset at one time.

  

- ********`kind: ‘scatter’, ‘kde’, ‘hist’, ‘reg’`********

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00027-545373ca-6afa-4c70-ab31-ce00c96a2e46?height=777](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00027-545373ca-6afa-4c70-ab31-ce00c96a2e46?height=777)

  

1. `sns.regplot`

  

[Document Here](http://seaborn.pydata.org/generated/seaborn.regplot.html#seaborn.regplot)

  

- `scatter` : choose to show scatter or not

- `fit_reg`  : choose to show regression line or not

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00028-fc223f8e-e6de-4f34-b5e4-63b5afbe3be7?height=430.1875](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00028-fc223f8e-e6de-4f34-b5e4-63b5afbe3be7?height=430.1875)

  

1. `sns.heatmap`

  

[Document Here](http://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap)

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00021-436480b0-ec74-4874-83de-8e96a080bf8b?height=519.1875](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/00021-436480b0-ec74-4874-83de-8e96a080bf8b?height=519.1875)

  

## Using OOD in Seaborn

  

### FaceGrid

  

`relplot, Catplot, heatmap,` etc., these functions can be drawn in one figure, etc. by `col, row,` etc. Because their Underlying implementation uses `FacetGrid` to assemble these graphics. 

  

We can use `plt.subplot` to manage a diagram like what we do in Matplotlib.

  

- [ ]  TODO

  

Instead of this, seaborn can use `facegrid` to do just maps.

  

- Common parameter

    - `data`: data_input

    - `row, col, hue`: use for categorized , combination with col and row, hue for classifying categorize

    - `col_wrap/row_wrap`: automatically organize subplots when a single col or row exists.

    - `height/aspect:` adjusting figure size , and `wide = height * aspect`

  

[https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/0b728df8883343a989db681aeb78374d?height=610](https://embed.deepnote.com/15035514-6f21-4f3b-b4fa-5d415b1c6ba0/fbf6dbe2-5670-468d-a982-8dfd74380d9a/0b728df8883343a989db681aeb78374d?height=610)

  

[https://embed.notionlytics.com/wt/ZXlKd1lXZGxTV1FpT2lJMk16ZzJPREF6TXpZd1pqRTBZMlZqT0dJd1lUQTFaRGcyWXpZMU5tRTBNaUlzSW5kdmNtdHpjR0ZqWlZSeVlXTnJaWEpKWkNJNklrVnlaa05tWlZoSWFrYzBiMVZ5TlV0R1QwOXFJbjA9](https://embed.notionlytics.com/wt/ZXlKd1lXZGxTV1FpT2lJMk16ZzJPREF6TXpZd1pqRTBZMlZqT0dJd1lUQTFaRGcyWXpZMU5tRTBNaUlzSW5kdmNtdHpjR0ZqWlZSeVlXTnJaWEpKWkNJNklrVnlaa05tWlZoSWFrYzBiMVZ5TlV0R1QwOXFJbjA9)