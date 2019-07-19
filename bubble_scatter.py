import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder


class BubbleScatter:
    """Description"""
    def __init__(self, data, x, y, z=None, z_aggregation=None, xbins=None,
             ybins=None, x_categories=None, y_categories=None,
             max_bubble_size=400, joint_distribution=True, cmap="cool"):

        self.data = data
        self.x = x
        self.y = y
        self.z = z
        self.z_aggregation = z_aggregation
        self.max_bubble_size = max_bubble_size
        self.joint_distribution = joint_distribution
        self.cmap = cmap

        self.categories = {}
        self.categories[self.x] = x_categories
        self.categories[self.y] = y_categories

        self.bins = {}
        self.bins[self.x] = xbins
        self.bins[self.y] = ybins

        self._validate_args()

        self.x_values = None
        self.y_values = None
        self.bubble_sizes = None
        self.bubble_colors = None
        self.x_ticks = None
        self.y_ticks = None
        self.x_labels = None
        self.y_labels = None

    def plot(self, show=True):
        """Plot bubble scatter."""
        self._process()

        fig, ax = plt.subplots(1, 1)

        ax.scatter(self.x_values,
                   self.y_values,
                   s=self.bubble_sizes,
                   c=self.bubble_colors,
                   cmap=self.cmap)

        ax.set_xlabel(self.x)
        ax.set_xticks(self.x_ticks)
        ax.set_xticklabels(self.x_labels)

        ax.set_ylabel(self.y)
        ax.set_yticks(self.y_ticks)
        ax.set_yticklabels(self.y_labels)

        ax.grid()

        if not show:
            plt.close()
            return fig

    def _process(self):
        """Main calculation of plotting parameters."""

        # transform data
        transformed_data = pd.DataFrame()
        transformed_data[self.x] = self._transform(self.x)
        transformed_data[self.y] = self._transform(self.y)

        # calculate buuble sizes
        bubble_sizes = self._calc_bubble_sizes(transformed_data)
        index = bubble_sizes.index
        self.bubble_sizes = bubble_sizes.tolist()

        # calculate bubble colors
        if self.z and self.z_aggregation:
            transformed_data[self.z] = self.data[self.z]
            self.bubble_colors = self._calc_bubble_colors(transformed_data, index)

        # encode values
        xy = self._calc_xy_values(index)
        x_encoder = self._get_encoder(self.x)
        y_encoder = self._get_encoder(self.y)
        self.x_values = x_encoder.fit_transform(xy[[self.x]])
        self.y_values = y_encoder.fit_transform(xy[[self.y]])


        # calculate labels
        self.x_labels = self._calc_labels(
            self.x, x_encoder.categories_[0].tolist())
        self.y_labels = self._calc_labels(
            self.y, y_encoder.categories_[0].tolist())

        # calculate ticks
        self.x_ticks = self._calc_ticks(xy, self.x)
        self.y_ticks = self._calc_ticks(xy, self.y)

    def _calc_bubble_sizes(self, data):
        """Calculate bubble sizes."""
        normalize = not self.joint_distribution
        grouped = data.groupby(self.x)
        bubble_sizes = grouped[self.y].value_counts(normalize=normalize)

        if not normalize:
            bubble_sizes = bubble_sizes / sum(bubble_sizes)

        bubble_sizes = self.max_bubble_size * bubble_sizes

        return bubble_sizes

    def _calc_bubble_colors(self, data, index):
        """Calculate bubble colors."""
        grouped = data.groupby([self.x, self.y])
        bubble_colors = grouped.agg({self.z: self.z_aggregation})
        bubble_colors = bubble_colors.loc[index][self.z].tolist()

        return bubble_colors

    @staticmethod
    def _calc_xy_values(index):
        """Description"""
        values = pd.DataFrame(index=index)
        values.reset_index(inplace=True)

        return values

    def _transform(self, col):
        """Description"""
        if self.bins[col] is None:
            return self.data[col]

        bins = self.bins[col]
        min_value = min(self.data[col])
        max_value = max(self.data[col])

        if isinstance(bins, list):
            if bins[0] >= min_value:
                bins[0] = min_value - 0.01
            if bins[-1] < max_value:
                bins[-1] = max_value

        values = pd.cut(self.data[col], bins=bins)
        values = values.astype(object)
        first_interval = min(values)
        new_first_interval = pd.Interval(min_value, first_interval.right)
        values[values == first_interval] = new_first_interval
        values = values.apply(lambda x: x.mid)
        values = values.astype(float)

        return values

    @staticmethod
    def _calc_ticks(data, col):
        """TODO: refactoring based on categories length"""
        n = len(data[col].unique())
        ticks = list(range(n))
        return ticks

    def _get_encoder(self, axis):
        """Description"""
        if not self.bins[axis] and self.categories[axis]:
            encoder = OrdinalEncoder(categories=self.categories[axis])
        else:
            encoder = OrdinalEncoder()

        return encoder

    def _calc_labels(self, col, categories):
        """Description"""
        if self.categories[col]:
            labels = self.categories[col]
        else:
            labels = categories

        return labels

    def _validate_args(self):
        """Validate args"""
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError

