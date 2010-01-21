__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pylab import ion, figure, draw
from matplotlib.patches import Rectangle, CirclePolygon
from scipy import cos, sin, array
from pybrain.rl.environments.renderer import Renderer
import threading
import time 


class TrackingRenderer(Renderer):  
    
    def __init__(self,obstacles=[]):
        Renderer.__init__(self)
        
        self.dataLock = threading.Lock()  
        self.agent = array([0.0, 0.0])
        self.target = array([0.0, 0.0])
        self.obstacles = obstacles
        print obstacles

        self.stopRequest = False
                
    def updateData(self, sensors):
        self.dataLock.acquire()
        self.target, self.agent = sensors[:2], sensors[2:]
        self.dataLock.release()
    
    def stop(self):
        self.stopRequest = True
    
    def start(self):
        self.drawPlot()
        Renderer.start(self)
    
    def drawPlot(self):
        ion()
        fig = figure(1)
        # draw cart
        axes = fig.add_subplot(111, aspect='equal')
        
        for o in self.obstacles:
            box = Rectangle(xy=[o[0][0], o[0][1]], width=o[1][0]-o[0][0], height=o[1][1]-o[0][1], facecolor='black')
            axes.add_artist(box)
            box.set_clip_box(axes.bbox)
            
        # draw target
        self.plottarget = Rectangle(xy=self.target, width=0.04, height=0.04, facecolor='green')
        axes.add_artist(self.plottarget)
        self.plottarget.set_clip_box(axes.bbox) 
        
        # draw agent
        self.plotagent = Rectangle(xy=self.agent, width=0.04, height=0.04, facecolor='red')
        axes.add_artist(self.plotagent)
        self.plotagent.set_clip_box(axes.bbox)
        
        # set axes limits
        axes.set_xlim(-0.2, 1.2)
        axes.set_ylim(-0.2, 1.2)
        
    def _render(self): 
        while not self.stopRequest:
            self.plotagent.set_xy(self.agent)
            self.plottarget.set_xy(self.target)
            draw()
            time.sleep(0.05)
        self.stopRequest = False
