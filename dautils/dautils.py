# -*- coding: utf-8 -*-
"""
Data analysis utilities
"""
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

try:
    from . import dautilsClasses as dauc
except Exception:
    import dautilsClasses as dauc

# remove outliers
# x numpy ndarray
# total amount of distribution at top/bottom/both to be removed
# if top is True and bottom is False, will remove only thresh of removal
# if both top and bottom are true, will remove thresh/2 of top and thresh/2 of bottom
# returns
# xRemoved is x with outliers removed
# iOutliers indicates indices where outliers were removed from x
def removeOutliers(x, thresh=0.05, top=True, bottom=True):
    iTop = 6
    iBottom = 4

    if top and bottom:
        topTail = 1.0 - (thresh/2.)
        bottomTail = thresh/2.
        xsumm = pd.Series(x).describe([bottomTail, topTail])
        iFilt = (x < xsumm[iTop]) & (x > xsumm[iBottom])
    else:
        xsumm = pd.Series(x).describe([thresh, 1.0-thresh])
        if top:
            iFilt = x < xsumm[iTop]
        elif bottom:
            iFilt = x > xsumm[iBottom]
    xRemoved = x[iFilt]
    xOutliers = x[~iFilt]
    iOutliers = np.where(~iFilt)[0]
    return xRemoved, xOutliers, iOutliers

# Show the percentage of missing/nan/inf values across all columns of a 
# pandas.DataFrame
# df: pandas.DataFrame
# verbose: True prints percentages across all columns, False skips columns that
# have no "special" values.  Default (false)
# skipCols = list indicating which columns to skip
# returns: pandas.DataFrame with summary
def missingStats(df, verbose=True, skipCols=[]):
    allCol = set(df.columns)
    cols2Proc = list(allCol.difference(set(skipCols)))
    totalPoints = float(len(df))
    numericCols = set(df.select_dtypes(include=[np.number]).columns)
    
    if verbose:
        print('Looking for missing values of input with ' + str(len(df)) + ' rows.')

    retDict = {'column': [], 'type': [],\
               'inf': [], 'null': []}
    for col in cols2Proc:
        retDict['column'].append(col)
        if col in numericCols:
            percInf = float(100*np.sum(np.isinf(df[col])))/totalPoints
            percNull = float(100*np.sum(df[col].isnull()))/totalPoints
        else:
            percInf = 0.0
            percNull = 100*len(df[df[col].isnull()])/totalPoints
        retDict['inf'].append(percInf)
        retDict['null'].append(percNull)
        retDict['type'].append(df[col].dtype)
    retDF = pd.DataFrame(retDict)
    retDF.sort_values(['null', 'inf'], ascending=[False, False], inplace=True)
    retDF.reset_index(inplace=True, drop=True)
    if verbose:
        print(retDF)
    return retDF

# Tests relationship between a dependent column of a pandas.DataFrame
# against all other ("feature") columns using data-type specific methods
# df: pandas.DataFrame
# target: column name in df corresponding to the depending 
# targetType: Either "cont" or "class" corresponding to a continuous or class dependent
# features: a list of columns in df to assess relationship to the target column, if left
# blank, then featureSift will perform analysis on all columns other than targetCol
# Testing performed:
# targetType = 'cont' and feature is continuous - Spearman correlation
# targetType = 'cont' and feature is categorical - One vs. All Other Mann-Whitney test of all unique feature values
# targetType = 'class' and feature is continuous - One vs. All Other Mann-Whitney test of all unique target values
# targetType = 'class' and feature is categorical - Pairwise binomial tests
# returns a pandas.DataFrame summarizing all features tested against the target
# including effect size, significance, and a description of the test
def sift(df, targetCol, targetType, features=[]):
    if len(features) == 0:
        features = list(set(df.columns).difference([targetCol]))

    # numerical columns
    numericCols = list(df.select_dtypes(include=[np.number]).columns.values)

    # filter the data to ensure only valid target variables
    if targetType == 'class':
        uniqueCat = list(set(df[targetCol]))
        if len(uniqueCat) <= 1: # there is nothing to target
              raise ValueError(targetCol + " column has only a single unique value and targetType is 'class'")
        dfFilt = df[~(df[targetCol].isnull())]
    elif targetType == 'cont':
        dfFilt = df[~(df[targetCol].isnull()) & ~np.isinf(df[targetCol])]
    else:
        raise ValueError("Invalid targetType argument.  Must be either 'cont' or 'class'")

    featResults = []
    y = dfFilt[targetCol].values
    featureList = []
    for feat in features:
        x = dfFilt[feat].values
        if targetType == 'cont' and feat in numericCols:
            dataPair = dauc.contCont(y, x)
        elif targetType == 'cont' and feat not in numericCols:
            dataPair = dauc.catCont(x, y) # note x and y flip
        elif targetType == 'class' and feat in numericCols:
            dataPair = dauc.catCont(y, x) 
        elif targetType == 'class' and feat not in numericCols:
            dataPair = dauc.catCat(y, x)
        else:
            raise ValueError('Invalid data types for ' + targetCol + ' and ' + feat)
        fResults = dataPair.relate()
        if type(fResults) == dict:
            featResults.append(fResults)
            featureList.append(feat)
        elif type(fResults) == list:
            featResults.extend(fResults)
            featureList.extend([feat]*len(fResults))

    outDF = pd.DataFrame(featResults)
    outDF['feature'] = featureList
    return outDF

# computes a 'relationship' matrix between columns of df
# this uses default behavior based on the data type of two features being analysed
def relMat(df, features=None):
    if features is None:
        features = df.columns.tolist()
    numericCols = list(df.select_dtypes(include=[np.number]).columns.values)
    numFeatures = len(features)
    cmat = np.zeros((numFeatures, numFeatures))
    for i in range(numFeatures): # exhaustive to allow for future non-symmetric relationship metrics
        for j in range(numFeatures):
            ifeat = df[features[i]].values
            jfeat = df[features[j]].values
            # numerical columns
            inumeric = features[i] in numericCols
            jnumeric = features[j] in numericCols
            if inumeric and jnumeric:
                dataPair = dauc.contCont(ifeat, jfeat)
                rel = dataPair.relate()
                relValue = rel['effect']
            elif ~inumeric and ~jnumeric:
                dataPair = dauc.catCat(ifeat, jfeat)
                rel = dataPair.relate()
                relValue = [x for x in rel if x['test'] == 'CramerV'][0]['effect']                
            else:
                if inumeric and not jnumeric:
                    dataPair = dauc.catCont(jfeat, ifeat)
                elif not inumeric and jnumeric:
                    dataPair = dauc.catCont(ifeat, jfeat)
                rel = dataPair.relate()
                tmp = pd.DataFrame(rel)
                try:
                    largestEffect = tmp[tmp['effect'].abs() == max(tmp['effect'].abs())]['effect'].iloc[0]
                except:
                    largestEffect = np.nan
                # take the maximum effect size
                relValue = largestEffect
            cmat[i,j] = relValue

    cmatDF = pd.DataFrame(cmat)
    cmatDF.columns = features
    cmatDF.index=features
    return cmatDF

# Tukey ladder transformation g(.) on an independent (x) variable against a 
# dependent y
# The Spearman correlation is computed also to assess
# the existence of a montonic relationship of y and x
# x and y are of type numpy.ndarray
# doPlot: detailed plots? (default=True)
# doPre: perform preprocessing to remove non-positive values of x 
# verbose: returns ladder summaries
# returns
# mr is a pandas.Dataframe that contains summary results for each transformation,
# transformed data, and the ols fit of y against the transformed x
def tladder(x, y, doPlot=True, doPre=True, verbose=False):
    # Tukey's Power Ladder
    ladder = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    # cast to take into account integer inputs
    x = x.astype(float)
    y = y.astype(float)
    
    if doPre:
        iKeep = np.where(x >= 0)
        numNP = len(x) - float(iKeep[0].shape[0])
        percNP = 100*numNP/len(x)
        # this means we have negative values
        if numNP == len(x):
            raise ValueError('All x values are non-positive.  Adjust input data.')
        elif percNP > 0:
            if verbose:
                warnings.warn('Removing ' + str(percNP) + '% of data that is non-positive before analyzing.')
            x = x[iKeep]
            y = y[iKeep]

    # spearman correlation assesses whether any monotonic relationship exists
    spMetric = ss.spearmanr(y, x, nan_policy='omit')
    mapResults = {'power': [], 'gx': [], 'rmse': [],\
                   'Spearman': [], 'gxData': []}
    for ipower in range(len(ladder)):
        if ladder[ipower] == 0:
            xTransformed = np.log(x)
            powerVal = 'log(x)'
        else:
            xTransformed = np.power(x, ladder[ipower])
            powerVal = 'x^' + str(ladder[ipower])
            
        rmse = np.sqrt(np.sum(np.power((y - xTransformed),2))/len(y))
        
        mapResults['power'].append(ladder[ipower])
        mapResults['gx'].append(powerVal)
        mapResults['rmse'].append(rmse)
        mapResults['Spearman'].append(spMetric[0])
        mapResults['gxData'].append(xTransformed)
        
    colOrder = ['power', 'gx', 'rmse', 'Spearman', 'gxData']
    mr = pd.DataFrame(mapResults)[colOrder]
    
    if doPlot:
        # if we are performing plotting, then calculate rankings
        xRanked = [ss.percentileofscore(x, z) for z in x]
        yRanked = [ss.percentileofscore(y, z) for z in y]

        plt.figure(figsize=(20, 20));
        plt.subplot(3, 3, 1);
        plt.plot(xRanked, yRanked, 'o');
        plt.title('Spearman Correlation: ' + str(np.round(spMetric[0], 2)));
        plt.xlabel('Ranked x');
        plt.ylabel('Ranked y');
        plt.grid()
        
        allRMSE = mr['rmse'].values.tolist()
        plt.subplot(3, 3, 2);     
        plt.plot(ladder, allRMSE, 'go-');
        plt.xlabel('Power');
        plt.ylabel('RMSE');
        plt.title('RMSE across ladder (0 = log)');
        plt.grid()

        for il in range(len(ladder)):
            keyValue = ladder[il]
            tmp = mr[mr['power'] == keyValue].iloc[0]
            fit = tmp['gxData']
            plt.subplot(3, 3, il + 3)
            plt.plot(x, fit, 'r-', label='g(x)');
            plt.plot(x, y, 'bo', alpha=0.5, label='y');
            plt.xlabel('x');
            plt.ylabel('y');
            plt.title('RMSE: ' + str(np.round(tmp['rmse'],2)) +\
                      ', g(x) = ' + tmp['gx'])
            plt.grid();
    
    if verbose:
        with pd.option_context('display.max_columns', 3, 'display.max_rows', 10): 
            print(mr[['gx', 'R2', 'Spearman']])
    return mr