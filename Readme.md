# <center> Machine Learning Tool

- This is a tool to get hands-on experience with Machine Learning concepts like Regression, Classification, Clustering.
- There are pre-loaded datasets (open-source) available within each section on the application that can be used.
- Source of the sample datasets are mentioned in the `Data Exploration` tab within the application or can be found in the [data_sources.csv](Data/data_sources.csv) file.
- The tool was built to make it as a medium to get hands-on visual experience to exploring different types of data types, building models to make predictions, evaluating the models.
- This is not the `only` way to learn/practice data science concepts. This is just a method to improve/facilitate the experience of learning data science concepts.

---

## What's inside the application?

There are 5 sections in the application, excluding the `introduction` section.

- `Data Exploration:`
    - All the sample datasets are available to use in this section.
    - The data sources (link to the source), description and the feature significances are explained here.
    - The data, once selected, is loaded on the table in order to be viewed. This is done across all the sections.
    - Once that is completed, there are 3 graphs that can be created in this page. 
        - `Scatter plot:` Create scatter plot between 2 continous variables. (**Note:** Details about how dataset is filtered automatically for continous, discrete/categorical values are explained at a later stage)
        - `Histogram plot:` Create histogram plots for continous variables. The number of bins can be changed.
        - `Count plot:` Create count plot for categorical/discrete variables. (**Note:** Only variables with `<=20` unique values can be plotted here. The length was fixed just to keep the plotting look efficient).

- `Linear Regression:`
    - Correlation matrix is plotted on this page.
    - All the features in the dataset are populated in the `select features` and can be selected as the features for the regression model
    - The data could be kept as the source data or could be normalized (The algorithms/packages used for normalization are explained later).
    - Only the continous variables are populated in the `select target` options.
    - The model can be evaluated using `Actual vs Predicted values` plot, `Residual vs Predicted Value` plot.
    - The `R^2, Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)` are displayed as well.

---

## Credits

---

Developer -  -  (@)

## License

MIT License

Copyright (c) 2019 Samira Kumar Varadharajan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.