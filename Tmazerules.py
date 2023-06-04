import numpy as np

## Rules and inputs are documented here for the Tmaze task

class Tmazetask:
  def __init__(self, highrew, highcost, highconflict, rareactcoeff, pts_per_class, n_classes, n_dim):
      self.highrew = highrew
      self.highcost = highcost
      self.highconflict = highconflict
      self.rareactcoeff = rareactcoeff
      self.pts_per_class = pts_per_class
      self.n_classes = n_classes
      self.n_dim = n_dim
      self.dims = {
          'reward_left': 0,
          'reward_right': 1,
          'cost_left': 2,
          'cost_right': 3,
          'conflict_left': 4,
          'conflict_right': 5
      }
      self.actranges = {
        'turn_left': range(self.pts_per_class),
        'turn_right': range(self.pts_per_class, 2 * self.pts_per_class),
        'hesitate': range(2 * self.pts_per_class, 3 * self.pts_per_class),
        'rare_actions': range(3 * self.pts_per_class, self.n_classes*self.pts_per_class),
      }
      self.signal = self.Tmazerules()
      self.classes = np.repeat(range(self.n_classes),self.pts_per_class)
      self.simdat()


  def simdat(self):
      self.dat = self.signal + self.noise()

  def noise(self):

    # some noise everywhere
    noise = np.random.randn(self.n_classes*self.pts_per_class,self.n_dim)

    # make rare actions very different than the task at hand
    noise[self.actranges['rare_actions'], :] += \
        self.rareactcoeff * np.random.randn(len(self.actranges['rare_actions']), self.n_dim)

    return noise

  def Tmazerules(self):

    signal = np.zeros([self.n_classes * self.pts_per_class, self.n_dim])

    # approach large rewards
    signal[self.actranges['turn_left'],self.dims['reward_left']] += self.highrew  # turn left when high left reward
    signal[self.actranges['turn_right'],self.dims['reward_right']] += self.highrew  # turn right when high right reward

    # avoid large costs
    signal[self.actranges['turn_right'],self.dims['cost_left']] += self.highcost  # turn right when high left cost
    signal[self.actranges['turn_left'],self.dims['cost_right']] += self.highcost  # turn left when high right cost

    # avoid conflict
    signal[self.actranges['turn_right'],self.dims['conflict_left']] += self.highconflict  # turn right when high left conflict
    signal[self.actranges['turn_left'],self.dims['conflict_right']] += self.highconflict  # turn left when high right conflict

    return signal