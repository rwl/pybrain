__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"

from scipy import array

from pybrain.rl.explorers.discrete.discrete import DiscreteExplorer
from pybrain.utilities import drawGibbs

class BoltzmannExplorer(DiscreteExplorer):
    """ A discrete explorer, that executes the actions with probability 
        that depends on their action values. The boltzmann explorer has 
        a parameter epsilon (the temperature). for high epsilon, the actions are 
        nearly equiprobable. for epsilon close to 0, this action selection
        becomes greedy.
    """
    
    def __init__(self, epsilon = 2., decay = 0.9995):
        DiscreteExplorer.__init__(self)
        self.epsilon = epsilon
        self.decay = decay
        self._state = None
    
    def activate(self, state, action):
        """ The super class ignores the state and simply passes the
            action through the module. implement _forwardImplementation()
            in subclasses.
        """
        self._state = state
        return DiscreteExplorer.activate(self, state, action)
    
    
    def _forwardImplementation(self, inbuf, outbuf):
        """ Draws a random number between 0 and 1. If the number is less
            than epsilon, a random action is chosen. If it is equal or
            larger than epsilon, the greedy action is returned.
        """
        assert self.module 
        
        values = self.module.getActionValues(self._state)
        action = drawGibbs(values, self.epsilon)
        
        self.epsilon *= self.decay
        
        outbuf[:] = array([action])
