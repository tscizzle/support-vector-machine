import numpy as np

from matplotlib import style, pyplot as plt

import code


class SupportVectorMachine:
    def __init__(self):
        self.colors = {1: 'r', -1: 'b'}

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

    # train
    def fit(self, data):
        self.data = data

        # map ||w|| to [w, b]
        vector_options = {}

        transforms = [[a, b] for a in [1, -1] for b in [1, -1]]

        self.max_feature_value = max(
            feature
            for yi_data in self.data.values()
            for datum in yi_data
            for feature in datum
        )
        self.min_feature_value = min(
            feature
            for yi_data in self.data.values()
            for datum in yi_data
            for feature in datum
        )

        step_sizes = [self.max_feature_value * 10**(-1 - c) for c in range(3)]

        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                b_top = self.max_feature_value * b_range_multiple
                b_bottom = -b_top
                for b in np.arange(b_bottom, b_top, step * b_multiple):
                    for transform in transforms:
                        w_t = w * transform
                        bad_option = False
                        for yi, yi_data in self.data.items():
                            for xi in yi_data:
                                if yi * (np.dot(w_t, xi) + b) < 1:
                                    bad_option = True
                                    break
                            if bad_option:
                                break
                        if not bad_option:
                            w_mag = np.linalg.norm(w_t)
                            vector_options[w_mag] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                else:
                    w = np.around(w - step, 8)
            min_norm = min(vector_options.keys())
            optimal_choice = vector_options[min_norm]
            self.w, self.b = optimal_choice
            latest_optimum = self.w[0] + step * 2

        code.interact(local=locals())

    # use
    def predict(self, datum):
        # sign(w.x + b)
        classification = np.sign(np.dot(np.array(datum), self.w) + self.b)

        if classification != 0:
            self.ax.scatter(
                datum[0],
                datum[1],
                s=200,
                marker='*',
                color=self.colors[classification]
            )
        else:
            print('This datum', datum, 'is right on the decision boundary.')

        return classification

    # view
    def visualize(self):
        for yi, yi_data in self.data.items():
            for xi in yi_data:
                self.ax.scatter(xi[0], xi[1], s=100, color=self.colors[yi])

        def hyperplane(x, w, b, v):
            # v = w.x + b
            return (-w[0] * x - b + v) / w[1]

        hyp_x_min = self.min_feature_value * 0.9
        hyp_x_max = self.max_feature_value * 1.1
        hyp_x_range = [hyp_x_min, hyp_x_max]

        # positive hyperplane, w.x + b = 1
        psv0 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv1 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot(hyp_x_range, [psv0, psv1], 'k')

        # negative hyperplane, w.x + b = -1
        nsv0 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv1 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot(hyp_x_range, [nsv0, nsv1], 'k')

        # negative hyperplane, w.x + b = -1
        db0 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db1 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot(hyp_x_range, [db0, db1], 'k')

        plt.show()


def main():
    data_dict = {
        -1: np.array([
            [1, 7],
            [2, 8],
            [3, 8],
        ]),
        1: np.array([
            [5, 1],
            [6, -1],
            [7, 3],
        ])
    }

    svm = SupportVectorMachine()

    svm.fit(data_dict)

    predict_us = [
        [0, 10],
        [1, 3],
        [3, 4],
        [3, 5],
        [5, 5],
        [5, 6],
        [6, -5],
        [5, 8],
    ]

    for datum in predict_us:
        svm.predict(datum)

    svm.visualize()


if __name__ == '__main__':
    main()
