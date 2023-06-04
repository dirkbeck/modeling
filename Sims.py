import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns

class Simulation:
    def __init__(self, classifier, task, nsims, x, ndiminc=1):
        self.LDAClassifier = classifier
        self.nsims = nsims
        self.X = task.dat  # historical predictors, n observations by n dim
        self.y = task.classes  # historical class membership, n observations
        self.classes = np.unique(self.y)
        self.nclasses = len(self.classes)
        self.x = x  # cortical input, 1 by n dim
        self.ndim = self.X.shape[1]
        self.task = task
        self.ndiminc = ndiminc
        self.likelihoods = pd.DataFrame(columns=['# discriminants','action','likelihood at timestep t'])
        self.discriminantvalues = pd.DataFrame(columns=['action','discriminant number','importance of discriminant'])

    def runsims(self, ndiminc=1):
        n_discrms,action_lklhd,likatt,action_dscrm,n_discrim,discrim_val = [],[],[],[],[],[]

        for sim in range(self.nsims):
            self.task.simdat()
            datset = DataSet(self.task.dat, self.y)

            # sims for likelihoods calculation
            for ndim in range(2, self.nclasses, ndiminc):
                clf = self.LDAClassifier(projection_dim=ndim)
                clf.fit(datset.get_data_as_dict())
                inputs, targets = datset.get_data()
                _, _, _, likelihood = clf.score(self.x, targets)
                for c in range(self.nclasses):
                    n_discrms.append(ndim)
                    action_lklhd.append(c)
                    likatt.append(likelihood[0,c])

            # sims for discriminant importance calculation
            clf = self.LDAClassifier(projection_dim=self.nclasses-1)
            clf.fit(datset.get_data_as_dict())
            inputs, targets = datset.get_data()
            for c in range(self.nclasses):
                # dimimportances = clf.dimimportance(datset.get_data_as_dict(),self.x)
                for d in range(self.nclasses-1):
                    action_dscrm.append(c)
                    n_discrim.append(d+1)
                    # discrim_val.append(dimimportances[c, d])

        self.likelihoods['# discriminants'] = pd.Series(n_discrms, dtype='str')
        self.likelihoods['action'] = pd.Series(action_lklhd, dtype='str')
        self.likelihoods['likelihood at timestep t'] = pd.Series(likatt, dtype='float')
        # self.discriminantvalues['discriminant number'] = pd.Series(n_discrim, dtype='int')
        # self.discriminantvalues['action'] = pd.Series(action_dscrm, dtype='str')
        # self.discriminantvalues['importance of discriminant'] = pd.Series(discrim_val, dtype='float')

    def errorrange(self, vector):
        return tuple(np.percentile(vector, [25, 75]))

    def replaceactionnames(self, dat):
        dat['action'] = dat['action'].replace(["0", "1", "2", "3", "4", "5", "6", "7"],
                                              ["turn left", "turn right", "deliberate", "rare action",
                                               "rare action", "rare action", "rare action", "rare action"])

    def actionbarplt(self, plttitle):
        # make a cleaner plot by labeling and consolidating rare actions
        lklhoods = self.likelihoods
        self.replaceactionnames(lklhoods)
        sns.barplot(data=lklhoods, x="action", y="likelihood at timestep t", hue="# discriminants",
                    hue_order=list(map(str, sorted([*range(2, self.nclasses, self.ndiminc)], reverse=True))), errorbar=self.errorrange)
        plt.title(plttitle)
        plt.show()

    def dimimportanceplt(self, plttitle):
        dsrmvals = self.discriminantvalues
        self.replaceactionnames(dsrmvals)
        sns.lineplot(dsrmvals,x='discriminant number',y='importance of discriminant',hue='action', errorbar=self.errorrange)
        plt.title(plttitle)
        plt.show()

class DataSet:
    def __init__(self, data, targets):
        self.classes = np.unique(targets)
        self.data = data
        self.targets = targets
        self.data_dict = self.to_dict(data, targets)

    def to_dict(self, data, targets):
        data_dict = {}
        for x, y in zip(data, targets):
            if y not in data_dict:
                data_dict[y] = [x.flatten()]
            else:
                data_dict[y].append(x.flatten())

        for i in self.classes:
            data_dict[i] = np.asarray(data_dict[i])

        return data_dict

    def get_data_by_class(self, class_id):
        if class_id in self.classes:
            return self.data_dict[class_id]
        else:
            raise ("Class not found.")

    def get_data_as_dict(self):
        return self.data_dict

    def get_data(self):
        data = []
        labels = []
        for label, class_i_data in self.data_dict.items():
            data.extend(class_i_data)
            labels.extend(class_i_data.shape[0] * [label])
        data = np.asarray(data)
        labels = np.asarray(labels)
        return data, labels