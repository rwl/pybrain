from scipy import r_

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.rprop import RPropMinusTrainer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import one_to_n


class NFQ(ValueBasedLearner):
    """ Neuro-fitted Q-learning"""
    
    def __init__(self, maxEpochs=100):
        ValueBasedLearner.__init__(self)
        self.gamma = 0.9
        self.alpha = 1.0
        self.maxEpochs = maxEpochs
    
    def learn(self):
        # convert reinforcement dataset to NFQ supervised dataset
        supervised = SupervisedDataSet(self.module.network.indim, 1)

        for si in range(self.dataset.getNumSequences()):
            nextexperience = None
            seq = self.dataset.getSequenceIterator(si, reverse=True)

            for i, (state_, action_, reward_) in enumerate(seq):
                
                # calculate Q value for time t (not t+1)
                Q = self.module.getValue(state_, action_[0])
                
                if not nextexperience:
                    # handle last s,a,r tuple by only using r as target
                    # inp = r_[state_, one_to_n(action_[0], self.module.numActions)]
                    # tgt = reward_
                    # supervised.addSample(inp, tgt)
                    
                    # delay each experience in sequence by one
                    nextexperience = (state_, action_, reward_)
                    continue
                
                # use experience from last timestep to do Q update
                (state, action, reward) = nextexperience
                
                inp = r_[state_, one_to_n(action_[0], self.module.numActions)]
                tgt = Q + self.alpha*(reward_ + self.gamma * max(self.module.getActionValues(state)) - Q)
                supervised.addSample(inp, tgt)
                
                # update last experience with current one
                nextexperience = (state_, action_, reward_)

        # train module with backprop/rprop on dataset
        # print supervised
        trainer = RPropMinusTrainer(self.module.network, dataset=supervised, batchlearning=True, verbose=False)
        trainer.trainUntilConvergence(maxEpochs=self.maxEpochs)        
        
        # alternative: backprop, was not as stable as rprop
        # trainer = BackpropTrainer(self.module.network, dataset=supervised, learningrate=0.005, batchlearning=True, verbose=True)
        # trainer.trainUntilConvergence(maxEpochs=200)
