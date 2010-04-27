__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'


from pybrain.rl.environments.cartpole import CartPoleEnvironment, BalanceTask, JustBalanceTask, CartPoleRenderer
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.rl.learners.valuebased import NFQCA, PolicyValueNetwork
from pybrain.rl.explorers import NormalExplorer

from numpy import array, arange, meshgrid, pi, zeros, mean
from matplotlib import pyplot as plt

render = False

plt.ion()

env = CartPoleEnvironment()
if render:
    renderer = CartPoleRenderer()
    env.setRenderer(renderer)
    renderer.start()

module = PolicyValueNetwork(4, 1)

task = JustBalanceTask(env, 100)
learner = NFQCA()
learner.explorer = None
learner.explorer = NormalExplorer(1, -1.0)

agent = LearningAgent(module, learner)
testagent = LearningAgent(module, None)
experiment = EpisodicExperiment(task, agent)

def plotPerformance(values, fig):
    plt.figure(fig.number)
    plt.clf()
    plt.plot(values, 'o-')
    plt.gcf().canvas.draw()


performance = []

if not render:
    pf_fig = plt.figure()

# experiment.doEpisodes(50)
    
while(True):
    experiment.doEpisodes(1)
    agent.learn(1)  
    
    # test performance
    if render:
        env.delay = True
    experiment.agent = testagent
    r = mean([sum(x) for x in experiment.doEpisodes(10)])
    env.delay = False
    testagent.reset()
    experiment.agent = agent  

    performance.append(r) 
    if not render:
        plotPerformance(performance, pf_fig)
    print "reward avg", r
    # print "exploration", agent.learner.explorer.epsilon
    print "num samples", agent.history.getNumSequences()
    print "num samples", len(agent.history)
    # agent.reset()
    
