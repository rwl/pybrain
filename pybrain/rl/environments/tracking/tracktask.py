__author__ = 'Thomas Rueckstiess and Tom Schaul'

from scipy import pi, dot, array, asarray, random

from pybrain.rl.environments import EpisodicTask
from tracking import TrackingEnvironment

class TrackingTask(EpisodicTask):
    """
    """
    def __init__(self, env=None, maxsteps=126, desiredValue=0):
        """
        :key env: (optional) an instance of a TrackingEnvironment (or a subclass thereof)
        :key maxsteps: maximal number of steps (default: 1000) 
        """
        self.desiredValue = desiredValue
        if env == None:
            env = TrackingEnvironment()
        EpisodicTask.__init__(self, env) 
        self.N = maxsteps
        self.t = 0
        
        # self.sensor_limits = [(0,1), (0,1), (0,1), (0,1)]
        # self.actor_limits = [(0,1), (0,1)]
        
    def reset(self):
        EpisodicTask.reset(self)
        self.t = 0

    def performAction(self, action):
        self.t += 1
        EpisodicTask.performAction(self, action)
                       
    def isFinished(self):
        return self.t >= self.N
        
    def getReward(self):
        s = self.env.getSensors()
        
        reward = -sum((s[:2] - s[2:])**2)
        return reward
        
    def setMaxLength(self, n):
        self.N = n    
        
 
class ContinuousAbsoluteTrackingTask(TrackingTask):
    """ action is directly translated into the absolute position of the agent. """
    pass


class ContinuousRelativeTrackingTask(TrackingTask):
    """ action is scaled down (factor 20) and then added to the position of the agent. """
    
    def performAction(self, action):
        pos = self.env.getSensors()[-2:].copy()
        newpos = pos + ((asarray(action)-0.5)*2. / 20.)
        TrackingTask.performAction(self, newpos)
        

class DiscreteRelativeTrackingTask(ContinuousRelativeTrackingTask):
    """ this task assumes 4 actions (right, left, down, up) and moves the agent
        by 0.1 along the direction of that action. """
        
    @property
    def indim(self):
        return 4
    
    actions = [(0.5,1.), (0.5,0.), (1.,0.5), (0.,0.5)]
    
    def performAction(self, action):
        ContinuousRelativeTrackingTask.performAction(self, self.actions[int(action[0])])
        
        
        
        
        
        
        
        
        
        
        
