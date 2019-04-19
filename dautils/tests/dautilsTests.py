# -*- coding: utf-8 -*-
"""
Tests for daUtils
"""
import unittest
import dautils
import numpy as np
import pandas as pd
from dautilsClasses import contCont, catCont, catCat

# Tests for daUtils
class dautilsTests(unittest.TestCase):
    
    # test missing value statistics
    def testMissingStats(self):
        df = pd.DataFrame({'a': [None, 1.0, np.inf, -np.inf, np.nan],\
                   'b': [2.0, 0, 0, 0, 0],\
                   'c': ['x', 'y', 'z', None, None],\
                   'd': [True, True, None, False, False],\
                    'e': [1, 1, 1, 1, np.inf],\
                    'f': ['apple', 'apple', None, 'banana', None]})
        retDF = dautils.missingStats(df, verbose=False)
        self.assertTrue(np.sum(retDF['null'] == [40, 40, 40, 20, 0, 0]) == 6)
        self.assertTrue(np.sum(retDF['inf'] == [40, 0, 0, 0, 20, 0]) == 6)

    # test correlation matrix
    def testRelMat(self):
        npoints = 1000
        means = (0, 0, 0)
        cov = [[1, 0.5, -.1], [0.5, 1, -.2], [-.1, -.2, 1]]
        dat = np.random.multivariate_normal(means, cov, (npoints))
        ucor = np.random.random(npoints)
        df = pd.DataFrame({'x': dat[:,0], 'y': dat[:,1], 'z': dat[:,2],\
        'e': ucor, 'cat': ['cat0']*npoints})
        df.loc[df['x'] < 0, 'cat'] = 'cat1'
        df['target'] = True
        df.loc[df['y'] > 1, 'target'] = False
        # generate a data frame of the correlation matrix
        cmDF = dautils.relMat(df)
        self.assertTrue((cmDF['y'].loc['x'] > 0) & (cmDF['y'].loc['z'] < 0))
        
    # test tukey's ladder fit on synthetic data
    def testTLadder(self):
        ladder = [-2, -1, -0.5, 0, 0.5, 1, 2]
        step = 0.01
        x = np.arange(step, 1, step)
        yAll = np.zeros((len(ladder), len(x)))
        for ipower in range(len(ladder)):
            if ladder[ipower] == 0:
                iy = np.log(x)
            else:
                iy = np.power(x, ladder[ipower])
            yAll[ipower,:] = iy
        
        # with direct mappings, the best rmse should be corresponding power
        counter = 0
        for ip in range(yAll.shape[0]):
            itmp = dautils.tladder(x, yAll[ip,:], doPlot=False, verbose=False)
            rmse = itmp['rmse'].iloc[ip]
            counter += rmse
        self.assertTrue(counter == 0)
                
    # test feature sifter
    def testsift(self):
        # consider a continuous dependent variable with continuous and categorical independents
        npoints = 1000
        means = (0, 0)
        cov = [[1, 0.5], [0.5, 1]]
        dat = np.random.multivariate_normal(means, cov, (npoints))
        dfCont = pd.DataFrame({'target': dat[:,0], 'xCont': dat[:,1]})
        dfCont['xCat'] = 'Cat0'
        dfCont.loc[dfCont['target'] > 1, 'xCat'] = 'Cat1'
        contAnalysis = dautils.sift(dfCont, targetCol='target', targetType='cont')
        self.assertTrue(contAnalysis['p'].iloc[0] < 1e-3)

        # consider a categorical dependent variable with continuous and categorical independents
        dbCat = pd.DataFrame({'random': np.random.random(npoints)})
        dbCat['target'] = False
        dbCat.loc[dbCat['random'] > 0.5, 'target'] = True
        numTrue = len(dbCat[dbCat['target'] == True])
        numFalse = len(dbCat) - numTrue
        dbCat['xCont'] = np.nan
        dbCat.loc[dbCat['target'] == True, 'xCont'] = np.random.randn(numTrue)
        dbCat.loc[dbCat['target'] == False, 'xCont'] = np.random.randn(numFalse)*2 + 2
        dbCat['xCat'] = 'Cat0'
        dbCat.loc[(dbCat['random'] > 0.4) & (dbCat['random'] < 0.7), 'xCat'] = 'Cat1'
        del dbCat['random']
        classAnalysis = dautils.sift(dbCat, targetCol='target', targetType='class')
        self.assertTrue(classAnalysis['p'].iloc[0] < 1)
    
    # test outlierRemoval
    def testOutlierRemoval(self):
        npoints = 998
        means = (0, 0)
        cov = [[1, 0.5], [0.5, 1]]
        dat = np.random.multivariate_normal(means, cov, (npoints))
        x = dat[:,0]
        x = np.append(x, 100)
        x = np.append(x, -100)
        xClean, iOutliers = dautils.removeOutliers(x, 2./npoints)
        self.assertTrue((npoints) in iOutliers)
        self.assertTrue((npoints+1) in iOutliers)
        
    # test contCont and contCat object
    def testContContCat(self):
        npoints = 1000
        means = (0, 0)
        cov = [[1, 0.5], [0.5, 1]]
        dat = np.random.multivariate_normal(means, cov, (npoints))
        y = dat[:,0]
        x = dat[:,1]
        cc = contCont(y, x)
        summ = cc.relate()
        # correlation should be within 10%
        self.assertTrue((summ['effect'] > 0.4) & (summ['effect'] < 0.6))

    def testCatContCat(self):
        # consider a categorical dependent variable with continuous and categorical independents
        npoints = 1000
        dbCat = pd.DataFrame({'random': np.random.random(npoints)})
        dbCat['target'] = False
        dbCat.loc[dbCat['random'] > 0.5, 'target'] = True
        numTrue = len(dbCat[dbCat['target'] == True])
        numFalse = len(dbCat) - numTrue
        dbCat['xCont'] = np.nan
        dbCat.loc[dbCat['target'] == True, 'xCont'] = np.random.randn(numTrue)
        dbCat.loc[dbCat['target'] == False, 'xCont'] = np.random.randn(numFalse)*1 + 1
        dbCat['xCat'] = 'Cat0'
        dbCat.loc[(dbCat['random'] < 0.2), 'xCat'] = 'Cat1'
        dbCat.loc[(dbCat['random'] > 0.9), 'xCat'] = 'Cat2'

        # target to continuous feature test        
        y = dbCat['target'].values
        x = dbCat['xCont'].values
        cc = catCont(y, x)
        allSumm = cc.relate()
        self.assertTrue(allSumm[0]['effect'] > 0)
        self.assertTrue(allSumm[1]['effect'] < 0)
        
        x = dbCat['xCat']
        cc = catCat(y, x)
        ccOut = cc.relate()
        self.assertTrue(ccOut[0]['p'] < 1e-3)
        self.assertTrue(ccOut[-1]['effect'] > 1)
        
if __name__ == '__main__':
    unittest.main()