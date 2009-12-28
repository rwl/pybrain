from scipy import r_

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.rprop import RPropMinusTrainer
from pybrain.utilities import one_to_n


class NFQCA(ValueBasedLearner):
    """ Neuro-fitted Q-learning for continuous actions"""
    
    def __init__(self):
        ValueBasedLearner.__init__(self)
        self.gamma = 0.9
    
    def learn(self):
        # convert reinforcement dataset to NFQ supervised dataset
        supervised = SupervisedDataSet(self.module.network.indim, 1)
        
        for seq in self.dataset:
            lastexperience = None
            for state, action, reward in seq:
                if not lastexperience:
                    # delay each experience in sequence by one
                    lastexperience = (state, action, reward)
                    continue
                
                # use experience from last timestep to do Q update
                (state_, action_, reward_) = lastexperience
                inp = state_
                tgt = reward_ + self.gamma * self.module.getMaxValue(state))
                supervised.addSample(inp, tgt)
                
                # update last experience with current one
                lastexperience = (state, action, reward)

        # train module with backprop/rprop on dataset
        trainer = RPropMinusTrainer(self.module.network, dataset=supervised, batchlearning=True, verbose=False)
        
        # alternative: backprop, was not as stable as rprop
        # trainer = BackpropTrainer(self.module.network, dataset=supervised, learningrate=0.01, batchlearning=True, verbose=True)

        trainer.trainEpochs(1)
         
        

