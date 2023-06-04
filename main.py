import numpy as np
from LDAClassifier import LDAClassifier
from Tmazerules import Tmazetask
from Sims import Simulation


pts_per_class = 100
n_classes = 7
n_dim = 20
nsims = 100
ndim_inc = 1
classes = np.repeat(range(n_classes), pts_per_class)

highrew = 1
lowrew = .1
highcost = 1
lowcost = .1
highconflict = 1
lowconflict = .5
rareactcoeff = 5

BBinpt = np.zeros([1,n_dim])
NCBinpt = np.zeros([1,n_dim])
CBCinpt = np.zeros([1,n_dim])
BBinpt[0,:6] = [highrew,lowrew,0,0,0,0] # see dimension order in Tmazerules
NCBinpt[0,:6] = [highrew,lowrew,lowcost,highcost,highconflict,lowconflict] # see dimension order in Tmazerules
CBCinpt[0,:6] = [highrew,lowrew,highcost,lowcost,highconflict,highconflict] # see dimension order in Tmazerules

Tmaze = Tmazetask(highrew, highcost, highconflict, rareactcoeff, pts_per_class, n_classes, n_dim)
sim = Simulation(LDAClassifier, Tmaze, nsims, BBinpt)
sim.runsims()
# sim.dimimportanceplt("BB")
sim.actionbarplt("BB")

# easy_choice, difficult_choice = [np.random.randn(n_classes*pts_per_class,n_dim) for _ in range(2)]
# easy_choice[:pts_per_class,0] += .1
# difficult_choice[:pts_per_class,0] += .5
# classes = np.repeat(range(n_classes),pts_per_class)
#
# train_dataset = DataSet(difficult_choice, classes)
#
# likelihoods = np.zeros([n_classes-3,n_classes])
#
# for ndim in range(2,n_classes-1):
#     clf = LDAClassifier(projection_dim=ndim)
#     clf.fit(train_dataset.get_data_as_dict())
#     inputs, targets = train_dataset.get_data()
#     acc, predictions, proj, likelihood = clf.score(inputs,targets)
#     likelihoods[ndim-2,:] = likelihood[2,:]
#
# clf = LDAClassifier(projection_dim=2)
# clf.fit(train_dataset.get_data_as_dict())
# inputs, targets = train_dataset.get_data()
# acc, predictions, proj, likelihood = clf.score(inputs,targets)
# colors = cm.rainbow(np.linspace(0, 1, n_classes))
# plotlabels = {np.unique(classes)[c] : colors[c] for c in range(n_classes)}
#
# for point,pred in zip(proj,predictions):
#   plt.scatter(point[0],point[1],color=plotlabels[pred])
# plt.legend(np.unique(classes),title="class ID")
# plt.xlabel("Discriminant 1")
# plt.xlabel("Discriminant 2")
# plt.show()
#
# for ndim in range(likelihoods.shape[0]):
#     plt.plot(likelihoods[ndim,:])
# plt.legend(range(2,n_classes),title="# discriminants")
# plt.xlabel("Class assignment")
# plt.ylabel("Assignment probs")
# plt.show()


