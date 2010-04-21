from scipy import r_

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import RPropMinusTrainer, BackpropTrainer
from pybrain.rl.explorers.continuous.normal import NormalExplorer
from pybrain.auxiliary.gradientdescent import GradientDescent


class NFQCA(ValueBasedLearner):
    """ Neuro-fitted Q-learning for continuous actions"""
    
    def __init__(self):
        ValueBasedLearner.__init__(self)
        self.descent = GradientDescent()
        self.descent.alpha = 0.05 # should this be negative? we want ascend, not descend
        self.descent.rprop = False
        
        self.gamma = 0.9
        self.alpha = 0.5


    def _setModule(self, module):
        """ Set module and tell explorer about the module. """
        if self.explorer:
            self.explorer.module = module
        self._module = module
        self.descent.init(self._module.actor.params)

    def _getModule(self):
        """ Return the internal module. """
        return self._module

    module = property(_getModule, _setModule)


    
    def learn(self):
        # convert reinforcement dataset to NFQ supervised dataset
        critic_ds = SupervisedDataSet(self.module.critic.indim, 1)
        
        # create critic training dataset
        for si in range(self.dataset.getNumSequences()):
            nextexperience = None
            seq = self.dataset.getSequenceIterator(si, reverse=True)

            for i, (state_, action_, reward_) in enumerate(seq):
                # calculate Q value for time t (not t+1)
                Q = self.module.critic.activate(r_[state_, action_])

                if not nextexperience:
                    # handle last s,a,r tuple by only using r as target
                    # inp = r_[state_, action_]
                    # tgt = reward_
                    # critic_ds.addSample(inp, tgt)

                    # delay each experience in sequence by one
                    nextexperience = (state_, action_, reward_)
                    continue

                # use experience from last timestep to do Q update
                (state, action, reward) = nextexperience

                inp = r_[state_, action_]
                tgt = Q + self.alpha*(reward_ + self.gamma * self.module.getMaxValue(state) - Q)
                critic_ds.addSample(inp, tgt)

                # update last experience with current one
                nextexperience = (state_, action_, reward_)

        # train module with backprop/rprop on dataset
        trainer = RPropMinusTrainer(self.module.critic, dataset=critic_ds, batchlearning=True, verbose=True)
        trainer.trainUntilConvergence(maxEpochs=100)
        print trainer.descent.alpha       
        
        
        sumQ = 0.
        for seq in self.dataset:
            for state, action, reward in seq:
                Q = self.module.getMaxValue(state)
                sumQ += Q
        print "before training:", sumQ
        
        # calculate actor gradient
        for i in range(20):
            self.module.actor.reset()
            self.module.actor.resetDerivatives()
            for seq in self.dataset:
                for state, action, reward in seq:
                    # propagate state forward through actor and state+action through critic
                    Q = self.module.getMaxValue(state)
                    # backpropagate "1" through critic to get partial derivatives
                    dQdP = self.module.critic.backActivate(1)
                    self.module.actor.backActivate(dQdP[-self.module.actor.outdim:])
                    # print self.module.actor.derivs
            self.module.actor._setParameters(self.descent(self.module.actor.derivs))
          
            sumQ = 0.
            for seq in self.dataset:
                for state, action, reward in seq:
                    Q = self.module.getMaxValue(state)
                    sumQ += Q
            print "after training:", sumQ
            
        