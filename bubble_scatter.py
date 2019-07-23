import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder


class BubbleScatter:
    """Description"""
    def __init__(self, data, x, y, z=None, z_aggregation=None, x_precision=2,
                 y_precision=2, xbins=None, ybins=None, x_categories=None,
                 y_categories=None, max_bubble_size=400,
                 joint_distribution=True, cmap="coolwarm"):

        self.data = data
        self.x = x
        self.y = y
        self.z = z
        self.z_aggregation = z_aggregation
        self.precision = {self.x: x_precision, self.y: y_precision}
        self.max_bubble_size = max_bubble_size
        self.joint_distribution = joint_distribution
        self.cmap = cmap

        self.categories = {}
        self.categories[self.x] = x_categories
        self.categories[self.y] = y_categories

        self.bins = {}
        self.bins[self.x] = xbins
        self.bins[self.y] = ybins

        # plotting parameters
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

        # calculate bubble sizes
        bubble_sizes = self._calc_bubble_sizes(transformed_data)
        index = bubble_sizes.index
        self.bubble_sizes = bubble_sizes.tolist()

        # calculate bubble colors
        if self.z and self.z_aggregation:
            transformed_data[self.z] = self.data[self.z]
            self.bubble_colors = self._calc_bubble_colors(
                transformed_data, index)

        # encode values
        xy = self._calc_xy_values(index)
        x_categories = self._get_encoder_categories(self.x)
        y_categories = self._get_encoder_categories(self.y)
        x_encoder = OrdinalEncoder(categories=x_categories)
        y_encoder = OrdinalEncoder(categories=y_categories)
        self.x_values = x_encoder.fit_transform(xy[[self.x]])
        self.y_values = y_encoder.fit_transform(xy[[self.y]])

        # calculate ticks and labels
        x_categories = x_encoder.categories_[0].tolist()
        y_categories = y_encoder.categories_[0].tolist()
        self.x_labels = self._calc_labels(self.x, x_categories)
        self.y_labels = self._calc_labels(self.y, y_categories)
        self.x_ticks = self._calc_ticks(x_categories)
        self.y_ticks = self._calc_ticks(y_categories)

    def _calc_bubble_sizes(self, data):
        """Calculate bubble sizes according to type of distribution."""
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
        bubble_colors = bubble_colors.loc[index][[self.z]].values

        return bubble_colors

    @staticmethod
    def _calc_xy_values(index):
        """Description"""
        values = pd.DataFrame(index=index)
        values.reset_index(inplace=True)

        return values

    def _transform(self, col):
        """Replace values with average values of intervals."""
        if self.bins[col] is None:
            return self.data[col]

        bins = self.bins[col]
        precision = self.precision[col]
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
        values = values.apply(lambda x: round(x.mid, precision))
        values = values.astype(float)

        return values

    @staticmethod
    def _calc_ticks(categories):
        n = len(categories)
        ticks = list(range(n))
        return ticks

    def _get_encoder_categories(self, axis):
        if not self.bins[axis] and self.categories[axis]:
            return self.categories[axis]

        return "auto"

    def _calc_labels(self, col, categories):
        """Description"""
        if self.categories[col]:
            return self.categories[col]

        return categories
