# -*- coding: utf-8 -*-
"""
Created on Sat May 13 17:00:50 2017
Classes for use in daUtils.py
"""
import numpy as np
import scipy.stats as ss
from abc import ABCMeta, abstractmethod

# abstract class for a y and an x vector pair
class dataPair(object):
    __metaclass__ = ABCMeta
    # y is dependent variable
    # x is independent variable
    def __init__(self, y, x):
        self.x = x
        self.y = y
        if len(x) != len(y):
            raise ValueError('y and x do not have the same length.')
        self.numPoints = len(x)
    
    # compute the relationship between y and x
    @abstractmethod
    def relate(self):
        pass

# continuous-to-continuous pair
class contCont(dataPair):
    # calculate the spearman correlation between the two point
    def relate(self, nThreshold=30):
        if (len(self.y) > nThreshold) and (len(self.x) > nThreshold):
            sp = ss.spearmanr(self.y, self.x, nan_policy='omit')
            faDict = {'effectSize': sp[0], 'p-value': sp[1], 'test': 'Spearman'}
        else: # skip relationship calculation if we do not have at least nThreshold points
            faDict = {'effectSize': np.nan, 'p-value': np.nan,\
            'test': '<' + str(self.numPoints)}
        return faDict

# categorical-to-categorical
class catCat(dataPair):
    def relate(self, nThreshold=30):
        allRet = []
        # y is always assumed to be the target
        yUnique = list(set(self.y))
        # x is the feature
        xUnique = list(set(self.x))
        contingencyTable = np.zeros((len(yUnique), len(xUnique)))
        for irow in range(len(yUnique)):
            for icol in range(len(xUnique)):
                contingencyTable[irow,icol] = float(np.sum((self.x == xUnique[icol]) & (self.y == yUnique[irow])))
        
        # use this to calculate Cramer's V
        c2 = ss.chi2_contingency(contingencyTable)
        chi2Stat = c2[0]
        pvalue = c2[1]
        ctsize = contingencyTable.sum().sum()
        minDen = min(contingencyTable.shape[0] - 1, contingencyTable.shape[1] - 1)
        cv = np.sqrt((chi2Stat/ctsize)/minDen)
        allRet.append({'p-value': pvalue, 'effectSize': cv, 'test': 'CramerV'})
        
        # total count of y
        numy = float(len(self.y))
        for irow in range(len(yUnique)):
            # yvalue
            yvalue = yUnique[irow]
            totalRow = np.sum(contingencyTable[irow,:])
            priorRow = totalRow/numy
            for icol in range(len(xUnique)):
                xvalue = xUnique[icol]
                rc = contingencyTable[irow,icol]
                totalCol = np.sum(contingencyTable[:,icol])
                # p(y = yvalue | x = xvalue)
                pyx = float(rc)/totalCol
                # binomial test
                btest = ss.binom_test(rc, totalCol, priorRow)
                oddsRatio = pyx/priorRow
                allRet.append({'p-value': btest, 'effectSize': oddsRatio,\
                               'test': 'Binomial (p-value), effectSize p(y=' +\
                               str(yvalue) + '|' + 'x=' + str(xvalue) +\
                                          ') / p(y=' + str(yvalue) + ')'})
        return allRet
    
# categorical-to-continuous
class catCont(dataPair):
    # for each category, perform a one-vs-all other mann-whitney test to evaluate difference
    def relate(self, nThreshold=20):
        uniqueCat = list(set(self.y))
        allRet = []
        for uc in uniqueCat:
            utmp1 = self.x[(self.y == uc)]
            utmp0 = self.x[(self.y != uc)]
            # ensure 
            if (len(utmp1) > nThreshold) and (len(utmp0) > nThreshold):
                statistic, pvalue = ss.mannwhitneyu(utmp0, utmp1, alternative='two-sided')
                n1 = float(len(utmp1))
                n0 = float(len(utmp0))
                # rank-biserial correlation. 
                # +1 means utmp1 is greater in all possible pairs with utmp0.
                # -1 means utmp1 is less in all possible pairs with utmp0
                r1 = 1. - (2*statistic)/(n1*n0)
                faDict = {'effectSize': r1, 'p-value': pvalue,\
                'test': str(uc) + '_vs_~' + str(uc) + '_MannWhitney'}
            else:
                faDict = {'effectSize': np.nan,\
                          'test': '# of ' + str(uc) + ' < ' + str(nThreshold),
                          'p-value': np.nan}
            allRet.append(faDict)
        return allRet