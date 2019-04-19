# dautils
a small collection of data analysis functions for use with pandas dataframe and/or numpy arrays

# installation
python setup.py install

# functions
dautils.removeOutliers: Outlier removal from an array based on percentile thresholding.

dautils.missingStats: Identifies percentage of nans and inf values across all columns of a pandas dataframe.

dautils.sift: Quantify the relationships (if any) between a single dependent/'target' column and other columns of a dataframe using a combination of non-parametric statistical tests [1,2, 3].  Data can be numeric and/or categorical types.

dautils.relmat: Calculate a quantitative 'relationship' matrix between all columns of a dataframe.  This function aspires to be analogous a traditional correlation matrix but across mixed data types.

dautils.tladder: Construct Tukey's ladder between a dependent/'target' array and an independent/'feature' array [4].  The independent/'feature' array is assumed to have values scaled such that they are all > 0.  Quantifies the goodness of fit between each transformation via the root-mean-squared error.

examples: /tests/dautilsTests.py

References

[1] https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test

[2] https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient

[3] https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V

[4] Tukey, John Wilder (1977). Exploratory Data Analysis. Addison-Wesley.

# author
Max Fenig

# license
MIT
