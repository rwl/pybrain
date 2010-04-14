__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from matplotlib.mlab import rk4 
from math import sin, cos
import time
from scipy import eye, matrix, random, asarray, array, r_, pi

from pybrain.rl.environments.graphical import GraphicalEnvironment


class TrackingEnvironment(GraphicalEnvironment):
    """ Description here
    """       
    
    indim = 2
    outdim = 4
        
    randomInitialization = False
    
    def __init__(self, obstacles=[]):
        GraphicalEnvironment.__init__(self)
        self.obstacles = obstacles
        self.sensors = array([0., 0., 0., 0.])
        self.t = 0
        
        # initialize the environment (randomly)
        self.reset()

    def getSensors(self):
        """ returns the 4 sensor values (x_target, y_target, x_agent, y_agent). """
        return asarray(self.sensors)
    
    def checkCollision(self, agent, obstacle):
        ll, ur = obstacle
        return all(asarray(agent) > asarray(ll)) and all(asarray(agent) < asarray(ur))     
                            
    def performAction(self, action):
        oldpos = self.sensors[-2:]
        agent = action
        
        # clipping
        for i,a in enumerate(agent):
            if a > 1.: agent[i]=1.
            if a < 0.: agent[i]=0.
        
        
        # check for any collisions
        for o in self.obstacles:
            if self.checkCollision(agent, o):
                print "collision!"
                agent = oldpos
                break
        
        # move target along    
        target = [(sin(self.t)+1)/2, (cos(self.t)+1)/2]
        self.t += 0.01
        self.t %= 2*pi

        # set new sensors values
        self.sensors = r_[target, agent]
        
        # render if requested      
        if self.hasRenderer():
            self.getRenderer().updateData(self.sensors)
            time.sleep(0.01)    
    
                        
    def reset(self):
        if self.randomInitialization:
            target = random.uniform(0., 1., 2)
            agent = random.uniform(0., 1., 2)
        else:
            target = array([0.1, 0.45])
            agent = array([0.5, 0.5])
        
        self.sensors = r_[target, agent]

