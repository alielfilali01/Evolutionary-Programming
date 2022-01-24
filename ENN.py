### simple evolutionary neural network model that implements a tournament selection theme
## Dataset = MNIST



# definition of a space
LAYER_SPACE = dict()
LAYER_SPACE['nb_units'] = (128, 1024, 'int', 0.15)
LAYER_SPACE['dropout_rate'] = (0.0, 0.7, 'float', 0.2)
LAYER_SPACE['activation'] =\
    (0,  ['linear', 'tanh', 'relu', 'sigmoid', 'elu'], 'list', 0.2)

NET_SPACE = dict()
NET_SPACE['nb_layers'] = (1, 3, 'int', 0.15)
NET_SPACE['lr'] = (0.0001, 0.1, 'float', 0.15)
NET_SPACE['weight_decay'] = (0.00001, 0.0004, 'float', 0.2)
NET_SPACE['optimizer'] =\
    (0, ['sgd', 'adam', 'adadelta', 'rmsprop'], 'list', 0.2)


#randomise a network
def random_value(space):
    """Sample  random value from the given space."""
    val = None
    if space[2] == 'int':
        val = random.randint(space[0], space[1])
    if space[2] == 'list':
        val = random.sample(space[1], 1)[0]
    if space[2] == 'float':
        val = ((space[1] - space[0]) * random.random()) + space[0]
    return {'val': val, 'id': random.randint(0, 2**10)}


def randomize_network(bounded=True):
    """Create a random network."""
    global NET_SPACE, LAYER_SPACE
    net = dict()
    for k in NET_SPACE.keys():
        net[k] = random_value(NET_SPACE[k])
    
    if bounded: 
        net['nb_layers']['val'] = min(net['nb_layers']['val'], 1)
    
    layers = []
    for i in range(net['nb_layers']['val']):
        layer = dict()
        for k in LAYER_SPACE.keys():
            layer[k] = random_value(LAYER_SPACE[k])
        layers.append(layer)
    net['layers'] = layers
    return net


#mutate a network
def mutate_net(net):
    """Mutate a network."""
    global NET_SPACE, LAYER_SPACE

    # mutate optimizer
    for k in ['lr', 'weight_decay', 'optimizer']:
        
        if random.random() < NET_SPACE[k][-1]:
            net[k] = random_value(NET_SPACE[k])
            
    # mutate layers
    for layer in net['layers']:
        for k in LAYER_SPACE.keys():
            if random.random() < LAYER_SPACE[k][-1]:
                layer[k] = random_value(LAYER_SPACE[k])
    # mutate number of layers -- RANDOMLY ADD
    if random.random() < NET_SPACE['nb_layers'][-1]:
        if net['nb_layers']['val'] < NET_SPACE['nb_layers'][1]:
            if random.random()< 0.5:
                layer = dict()
                for k in LAYER_SPACE.keys():
                    layer[k] = random_value(LAYER_SPACE[k])
                net['layers'].append(layer)
                # value & id update
                net['nb_layers']['val'] = len(net['layers'])
                net['nb_layers']['id'] +=1
            else:
                if net['nb_layers']['val'] > 1:
                    net['layers'].pop()
                    net['nb_layers']['val'] = len(net['layers'])
                    net['nb_layers']['id'] -=1
    return net


#make the network
class CustomModel():

    def __init__(self, build_info, CUDA=True):

        previous_units = 28 * 28
        self.model = nn.Sequential()
        self.model.add_module('flatten', Flatten())
        for i, layer_info in enumerate(build_info['layers']):
            i = str(i)
            self.model.add_module(
                'fc_' + i,
                nn.Linear(previous_units, layer_info['nb_units']['val'])
                )
            self.model.add_module(
                'dropout_' + i,
                nn.Dropout(p=layer_info['dropout_rate']['val'])
                )
            if layer_info['activation']['val'] == 'tanh':
                self.model.add_module(
                    'tanh_'+i,
                    nn.Tanh()
                )
            if layer_info['activation']['val'] == 'relu':
                self.model.add_module(
                    'relu_'+i,
                    nn.ReLU()
                )
            if layer_info['activation']['val'] == 'sigmoid':
                self.model.add_module(
                    'sigm_'+i,
                    nn.Sigmoid()
                )
            if layer_info['activation']['val'] == 'elu':
                self.model.add_module(
                    'elu_'+i,
                    nn.ELU()
                )
            previous_units = layer_info['nb_units']['val']

        self.model.add_module(
            'classification_layer',
            nn.Linear(previous_units, 10)
            )
        self.model.add_module('sofmax', nn.LogSoftmax())
        self.model.cpu()
        
        if build_info['optimizer']['val'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(),
                                lr=build_info['weight_decay']['val'],
                                weight_decay=build_info['weight_decay']['val'])

        elif build_info['optimizer']['val'] == 'adadelta':
            optimizer = optim.Adadelta(self.model.parameters(),
                                    lr=build_info['weight_decay']['val'],
                                    weight_decay=build_info['weight_decay']['val'])

        elif build_info['optimizer']['val'] == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(),
                                    lr=build_info['weight_decay']['val'],
                                    weight_decay=build_info['weight_decay']['val'])
        else:
            optimizer = optim.SGD(self.model.parameters(),
                                lr=build_info['weight_decay']['val'],
                                weight_decay=build_info['weight_decay']['val'],
                                momentum=0.9)
        self.optimizer = optimizer
        self.cuda = False
        if CUDA:
            self.model.cuda()
            self.cuda = True



#GP optimiser

from __future__ import absolute_import
import random
import numpy as np
from operator import itemgetter
import torch.multiprocessing as mp
from net_builder import randomize_network
import copy
from worker import CustomWorker, Scheduler

class TournamentOptimizer:
    """Define a tournament play selection process."""

    def __init__(self, population_sz, init_fn, mutate_fn, nb_workers=2, use_cuda=True):

        self.init_fn = init_fn
        self.mutate_fn = mutate_fn
        self.nb_workers = nb_workers
        self.use_cuda = use_cuda
        
        # population
        self.population_sz = population_sz
        self.population = [init_fn() for i in range(population_sz)]        
        self.evaluations = np.zeros(population_sz)
        
        # book keeping
        self.elite = []
        self.stats = []
        self.history = []

    def step(self):
        """Tournament evolution step."""
        print('\nPopulation sample:')
        for i in range(0,self.population_sz,2):
            print(self.population[i]['nb_layers'],
                  self.population[i]['layers'][0]['nb_units'])
        self.evaluate()
        children = []
        print('\nPopulation mean:{} max:{}'.format(
            np.mean(self.evaluations), np.max(self.evaluations)))
        n_elite = 2
        sorted_pop = np.argsort(self.evaluations)[::-1]
        elite = sorted_pop[:n_elite]
        
        # print top@n_elite scores
        self.elite = []
        print('\nTop performers:')
        for i,e in enumerate(elite):
            self.elite.append((self.evaluations[e], self.population[e]))    
            print("{}-score:{}".format( str(i), self.evaluations[e]))   
            children.append(self.population[e])

        p = 0.85 # winner probability 
        tournament_size = 3
        probs = [p*((1-p)**i) for i in range(tournament_size-1)]
        # a little trick to certify that probs is adding up to 1.0
        probs.append(1-np.sum(probs))
        
        while len(children) < self.population_sz:
            pop = range(len(self.population))
            sel_k = random.sample(pop, k=tournament_size)
            fitness_k = list(np.array(self.evaluations)[sel_k])
            selected = zip(sel_k, fitness_k)
            rank = sorted(selected, key=itemgetter(1), reverse=True)
            pick = np.random.choice(tournament_size, size=1, p=probs)[0]
            best = rank[pick][0]
            model = self.mutate_fn(self.population[best])
            children.append(model)

        self.population = children
        
    def evaluate(self):
        """evaluate the models."""
        
        workerids = range(self.nb_workers)
        workerpool = Scheduler(workerids, self.use_cuda )
        self.population, returns = workerpool.start(self.population)

        self.evaluations = returns
        self.stats.append(copy.deepcopy(returns))
        self.history.append(copy.deepcopy(self.population)) 




#Run the code
"""Tournament play experiment."""
from __future__ import absolute_import
import net_builder
import gp
import cPickle
# Use cuda ?
CUDA_ = True

if __name__=='__main__':
    # setup a tournament!
    nb_evolution_steps = 10
    tournament = \
        gp.TournamentOptimizer(
            population_sz=50,
            init_fn=net_builder.randomize_network,
            mutate_fn=net_builder.mutate_net,
            nb_workers=3,
            use_cuda=True)

    for i in range(nb_evolution_steps):
        print('\nEvolution step:{}'.format(i))
        print('================')
        tournament.step()
        # keep track of the experiment results & corresponding architectures
        name = "tourney_{}".format(i)
        cPickle.dump(tournament.stats, open(name + '.stats','wb'))
        cPickle.dump(tournament.history, open(name +'.pop','wb'))


