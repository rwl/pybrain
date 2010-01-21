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
        self.descent.alpha = -0.1  # we want ascend, not descend
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
        for seq in self.dataset:
            lastexperience = None
            for state, action, reward in seq:
                if not lastexperience:
                    # delay each experience in sequence by one
                    lastexperience = (state, action, reward)
                    continue
                
                # use experience from last timestep to do Q update
                (state_, action_, reward_) = lastexperience
                
                # current value (forward pass)
                Q = self.module.getMaxValue(state_)
                print Q
                inp = r_[state_, action_]
                tgt = Q + self.alpha*(reward_ + self.gamma * self.module.getMaxValue(state) - Q)
                critic_ds.addSample(inp, tgt)
                
                # update last experience with current one
                lastexperience = (state, action, reward)

        # train actor network with rprop (maybe backprop until convergence?)
        # trainer = RPropMinusTrainer(self.module.critic, dataset=critic_ds, batchlearning=True, verbose=True)
        # trainer.trainEpochs(100)

        # train critic network with backprop until convergence
        trainer = BackpropTrainer(self.module.critic, dataset=critic_ds, learningrate=0.001, batchlearning=True, verbose=True)
        trainer.trainUntilConvergence(maxEpochs=300)
        
        
        sumQ = 0.
        for seq in self.dataset:
            for state, action, reward in seq:
                Q = self.module.getMaxValue(state)
                sumQ += Q
        print "before training:", sumQ
        
        # calculate actor gradient
        for i in range(100):
            self.module.actor.reset()
            self.module.actor.resetDerivatives()
            for seq in self.dataset:
                for state, action, reward in seq:
                    Q = self.module.getMaxValue(state)
                    dQdP = self.module.critic.backActivate(Q)
                    self.module.actor.backActivate(dQdP[-self.module.actor.outdim:])
            
            self.module.actor._setParameters(self.descent(self.module.actor.derivs))
          
            sumQ = 0.
            for seq in self.dataset:
                for state, action, reward in seq:
                    Q = self.module.getMaxValue(state)
                    sumQ += Q
            print "after training:", sumQ
            
        