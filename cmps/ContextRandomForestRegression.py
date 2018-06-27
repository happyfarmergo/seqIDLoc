import collections

import numpy as np
from sklearn.ensemble import RandomForestRegressor \
    as SklearnRandomForestRegressor
from utils import *


class ContextRandomForestRegressor:

    """ Random Forest Regressor, add context features to second layer
        back-end is scikit-learn
    """

    def __init__(self,
                 nEstimators=10,
                 nJobs=-1,
                 criterion="mse",
                 maxFeatures="auto",
                 maxDepth=None,
                 minSamplesSplit=2,
                 minSamplesLeaf=1,
                 maxLeafNodes=None):
        """ Construct function
            All parameter description can reger to `RandomForestRegressor`.
            Each parameter can be pass list with 2 elements for first layer
            and second layer
        """
        # check parameters
        nEstimators = self.checkPara(
            para=nEstimators, paraName="nEstimators")
        nJobs = self.checkPara(
            para=nJobs, paraName="nJobs")
        maxFeatures = self.checkPara(
            para=maxFeatures, paraName='maxFeatures')
        criterion = self.checkPara(
            para=criterion, paraName="criterion")
        maxDepth = self.checkPara(
            para=maxDepth, paraName="maxDepth")
        minSamplesSplit = self.checkPara(
            para=minSamplesSplit, paraName="minSamplesSplit")
        minSamplesLeaf = self.checkPara(
            para=minSamplesLeaf, paraName="minSamplesLeaf")
        maxLeafNodes = self.checkPara(
            para=maxLeafNodes, paraName="maxLeafNodes")

        # create models for two layer
        self.models = []
        for i in range(2):
            self.models.append(SklearnRandomForestRegressor(
                n_estimators=nEstimators[i],
                n_jobs=nJobs[i],
                criterion=criterion[i],
                max_features=maxFeatures[i],
                max_depth=maxDepth[i],
                min_samples_split=minSamplesSplit[i],
                min_samples_leaf=minSamplesLeaf[i],
                max_leaf_nodes=maxLeafNodes[i]))

    def checkPara(self, para, paraName):
        if isinstance(para, collections.Iterable) and \
                len(para) < 2:
            raise ValueError("if `%s` is list then len(%s) == 2" %
                             (paraName, paraName))
        para = para if isinstance(para, list) else [para] * 2
        return para

    def constructContextFeature(self, label, timestamp, trajIdx):
        assert (sum([len(t) for t in trajIdx]) == len(label))
        contextFeature = [None for _ in xrange(len(label))]
        for traj in trajIdx:
            subLabel = label[traj]
            subTimestamp = timestamp[traj]
            for i in xrange(len(traj)):
                lastLabel = subLabel[i-1] if i > 0 else subLabel[i]
                curLabel = subLabel[i]
                nextLabel = subLabel[i+1] \
                    if i < len(traj)-1 else subLabel[i]
                lastTime = subTimestamp[i-1] if i > 0 else subTimestamp[i]
                curTime = subTimestamp[i]
                nextTime = subTimestamp[i+1] \
                    if i < len(traj)-1 else subTimestamp[i]

                predLon, predLat = curLabel

                # last speed
                lastSpeed = 1. * \
                    distance(lastLabel, curLabel) / (curTime-lastTime) \
                    if curTime-lastTime > 0 else 0.

                nextSpeed = 1. * \
                    distance(nextLabel, curLabel) / (nextTime-curTime) \
                    if nextTime-curTime > 0 else 0.

                # direction
                lastDirection = azimuth(pt_a=lastLabel, pt_b=curLabel)
                nextDirection = azimuth(pt_a=curLabel, pt_b=nextLabel)
                directionChangeRate = abs(
                    (nextDirection-lastDirection) / lastDirection) if lastDirection!=0. else -999

                subContextFeature = [predLon, 
                                    predLat, 
                                    lastSpeed,
                                     nextSpeed, 
                                     lastDirection, 
                                     nextDirection,
                                     directionChangeRate]
                contextFeature[traj[i]] = subContextFeature
        contextFeature = np.array(contextFeature)
        return contextFeature

    def fit(self, feature, label, timestamp, trajIdx):
        label = np.array(label)
        timestamp = np.array(timestamp)

        # train first layer
        self.models[0].fit(feature, label)
        pred = self.models[0].predict(feature)

        # construct context feature
        contextFeature = self.constructContextFeature(
            label=pred, timestamp=timestamp, trajIdx=trajIdx)

        featureWithContext = np.hstack((feature, contextFeature))

        # train second layer
        self.models[1].fit(featureWithContext, label)

    def predict(self, feature, timestamp, trajIdx):
        timestamp = np.array(timestamp)

        # predict first layer
        pred = self.models[0].predict(feature)

        # construct context feature
        contextFeature = self.constructContextFeature(
            label=pred, timestamp=timestamp, trajIdx=trajIdx)

        featureWithContext = np.hstack((feature, contextFeature))

        # predict second layer
        pred = self.models[1].predict(featureWithContext)

        return pred
    
    def evaluate(self, te_label, te_pred):
        te_label = np.array(te_label)
        errors = []
        for idx, pred in enumerate(te_pred):
            dist = distance(te_label[idx], pred)
            errors.append(dist)
        return errors