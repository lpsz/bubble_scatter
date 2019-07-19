import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from unittest import TestCase
from bubble_scatter import BubbleScatter


class TestBubbleScatter(TestCase):
    def test_plot_show_true(self):
        df = pd.DataFrame()
        df["x"] = [1, 1, 2, 2]
        df["y"] = [1, 2, 1, 1]

        scatter = BubbleScatter(df, "x", "y")
        result = scatter.plot(show=True)

        self.assertIsNone(result)

    def test_plot_show_false(self):
        df = pd.DataFrame()
        df["x"] = [1, 1, 2, 2]
        df["y"] = [1, 2, 1, 1]

        scatter = BubbleScatter(df, "x", "y")
        result = scatter.plot(show=False)

        self.assertIsInstance(result, plt.Figure)

    def test__process(self):
        df = pd.DataFrame()
        df["x"] = [1, 1, 2, 2]
        df["y"] = [1, 2, 1, 1]
        df["z"] = [1, 2, 3, 4]

        scatter = BubbleScatter(df, "x", "y", z="z", z_aggregation="sum",
                                max_bubble_size=1)
        scatter._process()

        true_x_ticks = [0, 1]
        true_y_ticks = [0, 1]
        true_x_labels = [1, 2]
        true_y_labels = [1, 2]
        true_bubble_sizes = [0.25, 0.25, 0.5]
        true_bubble_colors = [1, 2, 7]
        true_x_values = np.array([[0.0], [0.0], [1.0]])
        true_y_values = np.array([[0.0], [1.0], [0.0]])

        self.assertListEqual(scatter.x_ticks, true_x_ticks)
        self.assertListEqual(scatter.y_ticks, true_y_ticks)
        self.assertListEqual(scatter.x_labels, true_x_labels)
        self.assertListEqual(scatter.y_labels, true_y_labels)
        self.assertListEqual(scatter.bubble_sizes, true_bubble_sizes)
        self.assertListEqual(scatter.bubble_colors, true_bubble_colors)
        self.assertTrue(all(scatter.x_values == true_x_values))
        self.assertTrue(all(scatter.y_values == true_y_values))

    # test cacl_bubble_sizes
    # ==========================================================================

    def test__calc_bubble_sizes_joint(self):
        """Test bubble sizes for joint distribution (x,y)."""
        df = pd.DataFrame()
        df["x"] = [1, 1, 2, 2]
        df["y"] = [1, 2, 1, 1]

        scatter = BubbleScatter(df, "x", "y", joint_distribution=True,
                                max_bubble_size=1)

        index = pd.MultiIndex(
            levels=[[1, 2], [1, 2]],
            codes=[[0, 0, 1], [0, 1, 0]],
            names=["x", "y"]
        )

        true_result = pd.Series([0.25, 0.25, 0.5], index=index)
        result = scatter._calc_bubble_sizes(df)

        self.assertTrue(result.equals(true_result))

    def test__calc_bubble_sizes_conditional(self):
        """Test bubble sizes for conditional distribution (x|y)."""
        df = pd.DataFrame()
        df["x"] = [1, 1, 2, 2]
        df["y"] = [1, 2, 1, 1]

        scatter = BubbleScatter(df, "x", "y", joint_distribution=False,
                                max_bubble_size=1)

        index = pd.MultiIndex(
            levels=[[1, 2], [1, 2]],
            codes=[[0, 0, 1], [0, 1, 0]],
            names=["x", "y"]
        )

        true_result = pd.Series([0.5, 0.5, 1.0], index=index)
        result = scatter._calc_bubble_sizes(df)

        self.assertTrue(result.equals(true_result))

    # test calc_bubble_colors
    # ==========================================================================

    def test__calc_bubble_colors(self):
        df = pd.DataFrame()
        df["x"] = [1, 1, 2, 2]
        df["y"] = [1, 2, 1, 1]
        df["z"] = [1, 2, 3, 4]

        scatter = BubbleScatter(df, "x", "y", z="z", z_aggregation="sum")

        index = pd.MultiIndex(
            levels=[[1, 2], [1, 2]],
            codes=[[1, 0, 0], [0, 1, 0]],
            names=["x", "y"]
        )

        true_result = [7, 2, 1]
        result = scatter._calc_bubble_colors(df, index)

        self.assertListEqual(result, true_result)

    # test calc_xy_values
    # ==========================================================================

    def test__calc_xy_values(self):
        index = pd.MultiIndex(
            levels=[[1, 2], [1, 2]],
            codes=[[0, 0, 1], [0, 1, 0]],
            names=['x', 'y'])

        true_result = pd.DataFrame({"x": [1, 1, 2], "y": [1, 2, 1]})
        result = BubbleScatter._calc_xy_values(index)

        self.assertTrue(result.equals(true_result))

    # test transform
    # ==========================================================================

    def test__transform_no_xbins(self):
        df = pd.DataFrame()
        df["x"] = [1, 2, 3]
        df["y"] = [1, 2, 3]
        scatter = BubbleScatter(df, "x", "y")
        result = scatter._transform("x")
        true_result = df["x"]
        self.assertTrue(result.equals(true_result))

    def test__transform_no_ybins(self):
        df = pd.DataFrame()
        df["x"] = [1, 2, 3]
        df["y"] = [1, 2, 3]
        scatter = BubbleScatter(df, "x", "y")
        result = scatter._transform("y")
        true_result = df["x"]
        self.assertTrue(result.equals(true_result))

    def test__transform_with_int_xbins(self):
        df = pd.DataFrame()
        df["x"] = [0, 1, 2, 3, 4]
        df["y"] = [0, 1, 2, 3, 4]
        scatter = BubbleScatter(df, "x", "y", xbins=2)
        result = scatter._transform("x")
        true_result = pd.Series([1.0, 1.0, 1.0, 3.0, 3.0], name="x")
        self.assertTrue(result.equals(true_result))

    def test__transform_with_int_ybins(self):
        df = pd.DataFrame()
        df["x"] = [0, 1, 2, 3, 4]
        df["y"] = [0, 1, 2, 3, 4]
        scatter = BubbleScatter(df, "x", "y", ybins=2)
        result = scatter._transform("y")
        true_result = pd.Series([1.0, 1.0, 1.0, 3.0, 3.0], name="y")
        self.assertTrue(result.equals(true_result))

    def test__transform_with_interval_xbins(self):
        df = pd.DataFrame()
        df["x"] = [0, 1, 2, 3, 4]
        df["y"] = [0, 1, 2, 3, 4]
        xbins = [0, 2, 4]
        scatter = BubbleScatter(df, "x", "y", xbins=xbins)
        result = scatter._transform("x")
        true_result = pd.Series([1.0, 1.0, 1.0, 3.0, 3.0], name="x")
        self.assertTrue(result.equals(true_result))

    def test__transform_with_interval_ybins(self):
        df = pd.DataFrame()
        df["x"] = [0, 1, 2, 3, 4]
        df["y"] = [0, 1, 2, 3, 4]
        ybins = [0, 2, 4]
        scatter = BubbleScatter(df, "x", "y", ybins=ybins)
        result = scatter._transform("y")
        true_result = pd.Series([1.0, 1.0, 1.0, 3.0, 3.0], name="y")
        self.assertTrue(result.equals(true_result))

    # test calc_ticks
    # ==========================================================================

    def test__calc_ticks(self):
        df = pd.DataFrame()
        df["A"] = [1, 1, 2, 2]

        result = BubbleScatter._calc_ticks(df, "A")
        true_result = [0, 1]
        self.assertListEqual(result, true_result)

    # test get_encoder
    # ==========================================================================

    def test__get_encoder(self):
        df = pd.DataFrame(columns=["x", "y"])
        scatter = BubbleScatter(df, "x", "y")
        result = scatter._get_encoder("x")
        self.assertTrue(isinstance(result, OrdinalEncoder))

    def test__get_encoder_no_xbins_no_xcategories(self):
        df = pd.DataFrame(columns=["x", "y"])
        scatter = BubbleScatter(df, "x", "y")
        encoder = scatter._get_encoder("x")
        self.assertEqual(encoder.categories, "auto")

    def test__get_encoder_no_ybins_no_ycategories(self):
        df = pd.DataFrame(columns=["x", "y"])
        scatter = BubbleScatter(df, "x", "y")
        encoder = scatter._get_encoder("y")
        self.assertEqual(encoder.categories, "auto")

    def test__get_encoder_with_xbins_no_xcategories(self):
        df = pd.DataFrame(columns=["x", "y"])
        scatter = BubbleScatter(df, "x", "y", xbins=3)
        encoder = scatter._get_encoder("x")
        self.assertEqual(encoder.categories, "auto")

    def test__get_encoder_with_ybins_no_ycategories(self):
        df = pd.DataFrame(columns=["x", "y"])
        scatter = BubbleScatter(df, "x", "y", xbins=3)
        encoder = scatter._get_encoder("y")
        self.assertEqual(encoder.categories, "auto")

    def test__get_encoder_no_xbins_with_xcategories(self):
        df = pd.DataFrame(columns=["x", "y"])
        scatter = BubbleScatter(df, "x", "y", x_categories=["a", "b", "c"])
        encoder = scatter._get_encoder("x")
        self.assertEqual(encoder.categories, ["a", "b", "c"])

    def test__get_encoder_no_ybins_with_ycategories(self):
        df = pd.DataFrame(columns=["x", "y"])
        scatter = BubbleScatter(df, "x", "y", y_categories=["a", "b", "c"])
        encoder = scatter._get_encoder("y")
        self.assertEqual(encoder.categories, ["a", "b", "c"])

    def test__get_encoder_with_xbins_with_xcategories(self):
        df = pd.DataFrame(columns=["x", "y"])
        scatter = BubbleScatter(df, "x", "y", xbins=3,
                                x_categories=["a", "b", "c"])
        encoder = scatter._get_encoder("x")
        self.assertEqual(encoder.categories, "auto")

    def test__get_encoder_with_ybins_with_ycategories(self):
        df = pd.DataFrame(columns=["x", "y"])
        scatter = BubbleScatter(df, "x", "y", ybins=3,
                                y_categories=["a", "b", "c"])
        encoder = scatter._get_encoder("y")
        self.assertEqual(encoder.categories, "auto")

    # test calc_labels
    # ==========================================================================

    def test__calc_labels_x_without_categories(self):
        df = pd.DataFrame()

        scatter = BubbleScatter(df, "x", "y", x_categories=None)
        result = scatter._calc_labels("x", ["a", "b", "c"])
        true_result = ["a", "b", "c"]
        self.assertListEqual(result, true_result)

    def test__calc_labels_y_without_categories(self):
        df = pd.DataFrame()

        scatter = BubbleScatter(df, "x", "y", y_categories=None)
        result = scatter._calc_labels("y", ["a", "b", "c"])
        true_result = ["a", "b", "c"]
        self.assertListEqual(result, true_result)

    def test__calc_labels_x_with_categories(self):
        df = pd.DataFrame()

        scatter = BubbleScatter(df, "x", "y", x_categories=["a", "b"])
        result = scatter._calc_labels("x", ["a", "b", "c"])
        true_result = ["a", "b"]
        self.assertListEqual(result, true_result)

    def test__calc_labels_y_with_categories(self):
        df = pd.DataFrame()

        scatter = BubbleScatter(df, "x", "y", y_categories=["a", "b"])
        result = scatter._calc_labels("y", ["a", "b", "c"])
        true_result = ["a", "b"]
        self.assertListEqual(result, true_result)

    def test__validate_args(self):
        self.assertRaises(ValueError, BubbleScatter, 1, 1, 1)
