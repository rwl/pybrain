from scipy import r_, zeros

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.rprop import RPropMinusTrainer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import one_to_n


class NFQLambda(ValueBasedLearner):
    """ Neuro-fitted Q-learning"""
    
    def __init__(self, maxEpochs=100, qlambda=0.9):
        ValueBasedLearner.__init__(self)
        self.gamma = 0.9
        self.qlambda = qlambda
        self.alpha = 0.5
        self.maxEpochs = maxEpochs
    
    def learn(self):
        # convert reinforcement dataset to NFQ supervised dataset
        supervised = SupervisedDataSet(self.module.network.indim, 1)

        for si in range(self.dataset.getNumSequences()):
            sequence_length = self.dataset.getSequenceLength(si)
            targetcollector = zeros(sequence_length)
            targetcounter = zeros(sequence_length)
            
            for cutoff in range(sequence_length):
                nextexperience = None
                seq = self.dataset.getSequenceIterator(si, reverse=True)
                for i, (state_, action_, reward_) in enumerate(seq):

                    # go back in reverse through the sequence, do a full
                    # inner loop of pairs backwards from that element, then 
                    # move on to the previous element and do another full 
                    # inner loop back.
                    # for episode of length 5, this would use the following
                    # state/action/reward indices in that order:
                    # 4+3, 3+2, 2+1, 1+0
                    #      3+2, 2+1, 1+0
                    #           2+1, 1+0
                    #                1+0
                    
                    if i < cutoff:
                        continue
                        
                    if not nextexperience:
                        # delay each experience in sequence by one
                        nextexperience = (state_, action_, reward_)
                        continue
                
                    eligibility = (self.gamma * self.qlambda) ** (i-cutoff-1)
                    
                    if i > 20 or eligibility < 0.01:
                        break
                            
                    # use experience from last timestep to do Q update
                    (state, action, reward) = nextexperience
                    
                    # Q = self.module.getValue(state_, action_[0])
                    
                    if i-cutoff == 1:
                        # calculate delta only once for full inner loop
                        delta = reward_ + self.gamma*max(self.module.getActionValues(state))
                    
                    tgt = delta * eligibility
                    
                    # collect intermediate Q updates
                    targetcollector[sequence_length-i-1] += tgt
                    # targetcounter[sequence_length-i-1] += 1

                    # update last experience with current one
                    nextexperience = (state_, action_, reward_)
            
            # now create condensed supervised dataset sample, one for each state in seq
            seq = self.dataset.getSequenceIterator(si, reverse=False)
            for i, (state_, action_, reward_) in enumerate(seq):
                if i == sequence_length - 1:
                    break
                Q = self.module.getValue(state_, action_[0])
                inp = r_[state_, one_to_n(action_[0], self.module.numActions)]
                tgt = (1-self.alpha) * Q + self.alpha * targetcollector[i] # / targetcounter[i]
                supervised.addSample(inp, tgt)
                # print tgt
                
        # train module with backprop/rprop on dataset
        trainer = RPropMinusTrainer(self.module.network, dataset=supervised, batchlearning=True, verbose=False)
        trainer.trainUntilConvergence(maxEpochs=self.maxEpochs)        
        
        # alternative: backprop, was not as stable as rprop
        # trainer = BackpropTrainer(self.module.network, dataset=supervised, learningrate=0.005, batchlearning=True, verbose=True)
        # trainer.trainUntilConvergence(maxEpochs=200)