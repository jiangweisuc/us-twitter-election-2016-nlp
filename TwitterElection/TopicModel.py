import numpy as np
import pandas as pd
from sklearn import metrics, model_selection, ensemble, naive_bayes, svm, manifold, preprocessing
from six.moves import cPickle as pickle
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib


# Python 3.5 environment
# Last updated: Feb 19, 2017
# Written by Melissa K


class TopicModel(object):
    def __init__(self, X, target, sss_kfolds=3, test_size=0.4):
        """
        :param X: has to be a numeric numpy nd-array
        :param target: has to be a numpy 1d array containing categories/labels encoded as int
        :param sss_kfolds: k = number of iterations for cross-validation
        :param test_size: percentage of data set to retain for test/validation
        """
        self.X = X
        self.target = target
        self.sss_kfolds = sss_kfolds
        self.test_size = test_size

    @staticmethod
    def summary_metrics(target_true, target_pred):
        """
        Returns simple list of most common performance metrics for classification tasks
        :param target_true:
        :param target_pred:
        :return:
        """
        accuracy = metrics.accuracy_score(target_true, target_pred)
        precision = metrics.precision_score(target_true, target_pred, average='weighted')
        recall = metrics.recall_score(target_true, target_pred, average='weighted')
        f1_score = metrics.f1_score(target_true, target_pred, average='weighted')
        return [accuracy, precision, recall, f1_score]

    @staticmethod
    def prec_recall_min_scorer(target_true, target_pred):
        """
        Has to be of type: score_func(target_true, target_pred, **kwargs)
        Calculates the overall minimum of both precision and recall
        :param target_true:
        :param target_pred:
        :return:
        """
        # type: score_func(target_true, target_pred, **kwargs)
        precision = metrics.precision_score(target_true, target_pred, average=None)
        recall = metrics.recall_score(target_true, target_pred, average=None)
        return min(min(precision), min(recall))

    def custom_score(self):
        """
        Expects signature of score_func(target_true, target_pred, **kwargs)
        A custom scorer will be called, here overall minimum of both precision and recall.
        That minimum is maximized.
        This approach will avoid that a high accuracy hides either higher false positives or false negatives
        :return:
        """
        return metrics.make_scorer(self.prec_recall_min_scorer, greater_is_better=True)

    def cv_model_eval(self, load_best_model=None, param_search=None):
        """
        Performs model fitting and evaluation as well as parameter tuning using cross-validation
        STEPS:
        - Tune parameters for each of the three classifiers using cross-validation and custom performance metric scorer
        - Use best parameters to perform final model fitting and evaluation using cross-validation
        - Compare three models based on Confidence Interval of accuracy (printed to console)
        - Pickle model and some results
        - Plot ROC curves and confusion matrices
        :param load_best_model: if you have run this once the model will be saved to disk, so this step can be skipped the next
        time as model will be loaded from disk. Useful if you want to change plotting, but not fit the model again...
        :param param_search: if None parameters won't be optimized anddefault setting will be used
        :return:
        """
        sss = model_selection.StratifiedShuffleSplit(n_splits=self.sss_kfolds, test_size=self.test_size, random_state=0)
        sss.get_n_splits(self.X, self.target)
        clfs = [naive_bayes.GaussianNB(), ensemble.RandomForestClassifier(random_state=42),
                ensemble.AdaBoostClassifier()]
        clfs_names = ['NaiveBayesClassifier', 'RandomForestClassifier', 'AdaBoostClassifier']
        clfs_target_test = []
        clfs_target_pred = []
        clfs_probas = []

        params = [{
            # parameters for RandomForestClassifier
            'min_samples_split': stats.randint(2, 8),
            'max_depth': stats.randint(1, 6),
            'n_estimators': stats.randint(1, 100),
            'max_features': stats.randint(1, 4),
            'min_samples_leaf': stats.randint(1, 11),
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']},
            {
                # parameters for AdaBoost Classifier
                'learning_rate': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0],
                'n_estimators': [5, 10, 25, 50, 75, 100]}
        ]

        clfs_optimized = [clfs[0]]
        if param_search is not None:
            # get best model parameters using Randomized Search and cross-validation
            print("Parameter Search...")
            n_iter_search = 6
            for i, (clf, param_dist) in enumerate(zip(clfs[1:], params)):
                print('Parameter Search of...', clf)
                grid_search = model_selection.RandomizedSearchCV(clf, param_distributions=param_dist,
                                                                 cv=sss.split(self.X, self.target),
                                                                 n_iter=n_iter_search,
                                                                 # using custom scorer defined above
                                                                 # "maximize the min of precision and recall"
                                                                 scoring=self.custom_score())

                clfs_optimized.append(grid_search.fit(self.X, self.target).best_estimator_)
            clfs = clfs_optimized
            print('Best Estimators: ', clfs)

        for i, (clf, name) in enumerate(zip(clfs, clfs_names)):
            summary = []
            best_accuracy = 0
            # model selection, fit and predict and compare confidence intervals of accuracy
            if load_best_model is None:
                print('\nCross-Validation using following Classifier:\n\n%s\n\n' % clf)
                for j, (train_index, test_index) in enumerate(sss.split(self.X, self.target)):
                    X_train, X_test = self.X[train_index], self.X[test_index]
                    target_train, target_test = self.target[train_index], self.target[test_index]
                    clf = clf.fit(X_train, target_train)  # training model on train data set portion
                    target_pred = clf.predict(X_test)
                    probas = clf.predict_proba(X_test)
                    current_metrics = self.summary_metrics(target_true=target_test, target_pred=target_pred)
                    summary.append(current_metrics)
                    if current_metrics[0] > best_accuracy:
                        best_accuracy = current_metrics[0]
                        best_clf = clf
                        best_target_test = target_test
                        best_target_pred = target_pred
                        best_probas = probas
                        # get summaries and statistical confidence interval to decide which model is the best
                summary = np.array(summary)
                df = pd.DataFrame(summary, columns=['accuracy', 'precision', 'recall', 'f1_score'])
                print(df.mean(axis=0))
                print('Confidence Interval (CI) Accuracy:', stats.t.interval(0.95, len(summary[:, 0]) - 1,
                                                                             loc=np.mean(summary[:, 0]),
                                                                             scale=stats.sem(summary[:, 0],
                                                                                             ddof=1)))

                with open('model/' + name + '_best.pickle', 'wb') as f:
                    pickle.dump([best_clf, best_target_test, best_target_pred, best_probas], f, pickle.HIGHEST_PROTOCOL)

            with open('model/' + name + '_best.pickle', 'rb') as f:
                clf, target_test, target_pred, probas = pickle.load(f)
            clfs_target_test.append(target_test)
            clfs_target_pred.append(target_pred)
            clfs_probas.append(probas)

            ## plotting of best cross-validation split respectively
            # Confusion matrix
            self.plot_confusion_matrix(target_true=target_test, target_pred=target_pred, clf_name=name,
                                       fname='output/' + name + '_confusion_matrix.png')

            # ROC Curve
            self.plot_ROC_curve(target_true=target_test, probas=probas, fname='output/' + name + '_roc_curve.png')

            # 2D embedding plot (can be computationally intensive)
            # tsne = manifold.TSNE(n_components=2, init='pca', random_state=42)
            # X_tsne = tsne.fit_transform(X_test)
            # self.plot_2D_manifold(X_2D=X_tsne, target=target_pred, fname='output/' + name + '_tSNE_embedding.png')
        self.plot_confusion_matrix_all_clfs(clfs_target_test, clfs_target_pred, clfs_names)
        self.plot_ROC_curve_all_clfs(clfs_target_test, clfs_target_pred, clfs_names)

    @staticmethod
    def plot_confusion_matrix(target_true, target_pred, clf_name, fname='confusion_matrix.png'):
        """
        Single plot for each classifier
        """
        fig = plt.figure(figsize=(8, 8))
        fig = sm.graphics.plot_corr(metrics.confusion_matrix(target_true, target_pred))

        plt.title('%s Confusion Matrix' % clf_name)
        plt.ylabel('True Target', fontsize=12)
        plt.xlabel('Predicted Target', fontsize=12)
        plt.savefig(fname, dpi=600, bbox_inches='tight')

    @staticmethod
    def plot_confusion_matrix_all_clfs(l_target_true, l_target_pred, clfs_names,
                                       fname='output/all_clfs_confusion_matrix.png'):
        """
        1x3 subplots for confusion matrices of all classifiers
        """
        fig = plt.figure(figsize=(20, 10))
        ncols = len(l_target_true)
        nrows = 1
        for i in range(len(l_target_true)):
            ax = fig.add_subplot(nrows, ncols, i + 1)
            sm.graphics.plot_corr(metrics.confusion_matrix(l_target_true[i], l_target_pred[i]), ax=ax)
            ax.set_title(clfs_names[i])
            ax.set_xlabel('Predicted Target', fontsize=12)
            ax.set_ylabel('True Target', fontsize=12)
        plt.savefig(fname, dpi=600, bbox_inches='tight')

    @staticmethod
    def plot_ROC_curve(target_true=None, probas=None, fname='roc_curve.png'):
        """
        Single ROC curve for each classifier.
        Accomodates binary target of multi-class labels (for latter case target gets binarized)
        """
        # get ROC curve
        unique_classes = np.unique(target_true)
        n_classes = len(unique_classes)

        if n_classes == 2:  # binary case
            fpr, tpr, _ = metrics.roc_curve(target_true, probas[:, 1], pos_label=1)
            roc_auc = metrics.auc(fpr, tpr)
            fig = plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, label='Binary Target (AUC = %.6f)' % roc_auc)

        else:  # multi-class, express problem in multiple binary predictions, thus binarize target first
            target_binarized = preprocessing.label_binarize(target_true, classes=unique_classes)
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            fig = plt.figure(figsize=(8, 8))
            for i in range(0, n_classes):
                # OUTPUT: false positive rate (fpr) and true positive rate (tpr)
                # INPUT: Y_true (here target_binarized = binarized multiclass labels),
                # works also with binary target
                # target_score (here  probability estimates of the positive class)
                fpr[i], tpr[i], _ = metrics.roc_curve(target_binarized[:, i], probas[:, i], pos_label=1)
                roc_auc[i] = metrics.auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], label='Class {0} (AUC = {1:0.6f})'
                                               ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Pure Chance')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate = 1-Specificity', fontsize=14)
        plt.ylabel('True Positive Rate = Sensitivity', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
        plt.legend(loc="lower right")
        plt.savefig(fname, dpi=600, bbox_inches='tight')

    @staticmethod
    def plot_ROC_curve_all_clfs(l_target_true, l_probas, clfs_names, fname='output/all_clfs_roc_curve.png'):
        """
        Here compare ROC curves for different binary classifiers in one plot
        """
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fig = plt.figure(figsize=(8, 8))
        for i in range(len(l_target_true)):
            fpr[i], tpr[i], _ = metrics.roc_curve(l_target_true[i], l_probas[i], pos_label=1)
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label='%s (AUC = %.4f)'
                                           % (clfs_names[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Pure Chance')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate = 1-Specificity', fontsize=14)
        plt.ylabel('True Positive Rate = Sensitivity', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
        plt.legend(loc="lower right")
        plt.savefig(fname, dpi=600, bbox_inches='tight')

    @staticmethod
    def plot_2D_manifold(X_2D, target, fname='tsne_embedding.png'):
        """
        Manifold learning 2D embedding and plotting.
        Can be computationally expensive!
        """
        # scaling
        x_min, x_max = np.min(X_2D, 0), np.max(X_2D, 0)
        X_2D = (X_2D - x_min) / (x_max - x_min)
        # assign to DataFrame
        df = pd.DataFrame(X_2D, columns=['posx', 'posy'])
        df['target'] = target

        grouped = df.groupby('target')
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(np.unique(target))))
        for i, label in grouped:
            ax.scatter(label.posx, label.posy, s=20, alpha=0.5, color=colors[i], label='label {0}'.format(i))
            ax.set_aspect('auto')
            ax.axis('off')
            ax.legend(numpoints=1)

        plt.title('TSNE Embedding', fontsize=16)
        plt.savefig(fname, dpi=600, bbox_inches='tight')
