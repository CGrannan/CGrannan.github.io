---
layout: post
title:      "Which Features Should I Use?"
date:       2019-07-26 20:26:30 +0000
permalink:  which_features_should_i_use
---

While I was working on a project recently, I realized how important it is to select the right features for a linear model.  Interestingly, you get a much better model from dropping some information from your data.  If you use too many features, the model can be hard to understand and be overly complex. The model can often become overfitted, showing inaccuracies when compared to a test dataset.  Furthermore, training time can increases exponentially as more features are used.  However, you need to make sure that you are dropping the right variables.  If you don't use your best features, your model will be inaccurate and will fail to show correct relationships.  It is vitally important for the strength of a model to select the right subset of features.

### So what makes a feature weak?

There are many factors that reduce the strength of a feature.  A good starting place for determining feature strength is probability value.   Since a p-value reflects the probability of rejecting the nul hypothesis, all of your features should be within a desired confidence interval.  For my project, I set a value of 0.05 as the maximum p-value that a feature could possess to be included in my model.  To determine p-values, I ran a stepwise selector function shown below.  This function would incorporate features one at a time, determine p-values, then eject any features that had a p-value greater than my limit of 0.05.  This reduced my list of features to ones that were statistically significant.
```
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

```

Another factor of feature strength is multicollinearity.  If multiple features are too highly correlated with each other, then the resulting model will show incorrect coefficients.  It may undervalue or overvalue the contributions of these features to your model.  The model should not be any less accurate due to the multicollinearity, but it will become difficult to parse the contributions of each feature.  To determine multicollinearity in my project, I ran a correlation heatmap through seaborn.  I set an upper limit of 0.75 as an acceptaple level of correlation.  Then, I dropped a column from any pair that had a correlation above my limit.  This allowed me to further trim my features down.

Yet another way to determine feature strength is through relevance to your model.  If some of your data is irrelevant or impossible to work with, those features need to be cut before modeling.  In my project, for example, one of my predictors was a date column.  These dates only represented one year, so did not have enough range to be important to my model.  Similarly, I was also working with longitude and lattitude which are very tricky to work into a linear model in a meaningful way.  I was able to quickly cut these three features and focus on features that were relevant.

One more aspect of feature strength is the correlation between a predictor and the target variable.  If a predictor correlates highly with the target, then it will give a much stronger impact on the model.  If you test the correlations of your predictors and target before modeling, you can get a good idea of which features you will want to use for your model.

### How to Select the Best Set of Features

So once you've identified your strongest features, you're ready to create your model, right?  Not quite.  Just because your features are good individually doesn't mean they will work well together.  It is important to find the best subset of your features for your model.  To find this set you will need to use a wrapper method.  A wrapper method is an algorithm that continuously creates models and evaluates their performances.  The algorithm will start with a model, then either add or drop a feature.  Then it will evaluate the change in performance and repeat.  There are two kinds of wrapper methods to consider.  The first is through forward selection.  Forward selection is a process where you start with an empty model and then add one feature at a time, noting the improvements to the model each time.  This process continues until adding a feature no longer improves the model.  Another method of selecting the optimal subset of features is backwards elimination which is effectively the opposite of forward selection.  You start with a model using all of your features. Then, you remove the least significant feature one at a time until removing features no longer improves the model.  A recursive feature elimination is a form of backward elimination, and is the method that I chose to employ in my model.  This process creates the best possible selection of features, using a number of features that you supply.  For example, If you had 30 features and you ran an RFE for 13 features,  the algorithm would create a model for every combination of 13 features, and return the best one.  The downside to this method is that the number of features used is arbitrary.  The best number of features needs to be known to use this process to its greatest extent.  To combat this weakness, I ran the RFE for multiple values of features to see what the best possible model was.  At each feature count, I displayed a graph of the model and the mean squared error.  I then compared each potential model and chose the most accurate.    One other important feature of an RFE is that it ranks the unused features on when they were dropped from the model.  This allows you to see what your most relevant unused features are.

Whichever method you use, this process allows you to pare your features down to the best selection for your model.  With the best selection of features, your model will be as accurate as possible and will show an accurate coefficient for each feature selected.   Further, it will not overfit your training data, showing similar results when compared to a test dataset.  Finally, by trimming down your features, your model will be more immediately understandable.  If you would like to learn more about feature selection, [this article](https://towardsdatascience.com/why-how-and-when-to-apply-feature-selection-e9c69adfabf2) may be helpful.

As always, thanks for reading and happy coding!
