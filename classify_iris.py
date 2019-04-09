import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors, svm

from matplotlib import pyplot as plt

import code


class IrisData(object):
    DATA_FILENAME = 'iris.data'
    TARGET_CLASS = 'Iris-setosa'
    FEATURE_LABELS = [
        'sepal length in cm',
        'sepal width in cm',
        'petal length in cm',
        'petal width in cm',
    ]
    FEATURE_DICT = {i: label for i, label in enumerate(FEATURE_LABELS)}
    LABEL_DICT = [
        'Is NOT %s' % TARGET_CLASS,
        'Is %s' % TARGET_CLASS,
    ]

    def __init__(self):
        df = pd.io.parsers.read_csv(
            filepath_or_buffer=self.DATA_FILENAME,
            header=None,
            sep=',',
        )

        df.columns = [
            l for i, l in sorted(self.FEATURE_DICT.items())
        ] + ['class label']

        df.dropna(how="all", inplace=True) # to drop the empty line at file-end

        df.tail()

        # just look at petals, so visualization is 2D
        self.X = df.values[:,2:4]

        # just look at setosa vs. not setosa so that its binary classification
        self.y = (np.array(df['class label']) == self.TARGET_CLASS) * 1.0

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            self.X,
            self.y,
            test_size=0.2
        )

        classifier = svm.SVC(kernel='linear')
        classifier.fit(X_train, y_train)

        boundary_slope = -classifier.coef_[0][0] / classifier.coef_[0][1]
        boundary_intercept = -classifier.intercept_[0] / classifier.coef_[0][1]
        self.boundary_points = np.array([
            [0, boundary_intercept],
            [3, boundary_intercept + 3 * boundary_slope]
        ])
        self.closest_points = classifier.support_vectors_

        code.interact(local=locals())

    def plotSamples2D(self):
        ax = plt.subplot(111)
        for label, marker, color in zip(
            [0, 1],('^', 's', 'o'), ('blue', 'red')
        ):
            plt.scatter(
                x=self.X[:,0].real[self.y == label],
                y=self.X[:,1].real[self.y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=self.LABEL_DICT[label]
            )

        leg = plt.legend(loc='upper right', fancybox=True)
        leg.get_frame().set_alpha(0.5)

        # hide axis ticks
        plt.tick_params(
            axis="both",
            which="both",
            bottom="off",
            top="off",
            labelbottom="on",
            left="off",
            right="off",
            labelleft="on"
        )

        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        plt.plot(self.boundary_points[:,0], self.boundary_points[:,1])
        plt.plot(self.closest_points[:,0], self.closest_points[:,1])

        plt.grid()
        plt.tight_layout()
        plt.show()


def main():
    iris_data = IrisData()
    iris_data.plotSamples2D()


if __name__ == '__main__':
    main()
