import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import pandas as pd
import math 

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                train_df, val_df, test_df,
                label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
       
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
    
    def plot(self, model=None, plot_col='remainder__visits', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

            plt.xlabel('Time [h]')

def plotModelGridError(model, figsize=(12, 6), xscale:str="linear", xscale2:str="linear", param1:str=None, param2:str=None):
    """Plot model cross-validation error along grid of hyperparameters

    Args:
        model: model to analyze
        figsize (tuple[float, float], optional): figure of plot size. Defaults to (12, 6).
        xscale (str, optional): Scale of x-axis of first plot. Defaults to "linear".
        xscale2 (str, optional): Scale of x-axis of second plot. Defaults to "linear".
        param1 (str, optional): First parameter of the grid to analyze. Defaults to None.
        param2 (str, optional): Second parameter of the grid to analyze. Defaults to None.

    Raises:
        TypeError: No hyperparameters found in grid, grid must have some hyperparameter to create plot
    """
    cv_r = model.cv_results_
    err = cv_r["mean_test_score"]
    std = cv_r["std_test_score"]
    param_names = list(model.cv_results_.keys())
    if param1 is not None and param2 is not None:
        param_names = ["param_"+param1, "param_"+param2]
    param_keys = [s for s in param_names if "param_" in s]
    params = [s.split("param_")[1] for s in param_keys]
        
    best_params = model.best_params_
    if not param_keys:
        raise TypeError("No hyperparameters encountered in grid.")
    if len(param_keys) > 1:
        grid1 = model.cv_results_[param_keys[0]].data
        cat1 = 'num'
        if not(type(grid1[0]) == int or type(grid1[0]) == float):
            grid1 = [p for p in list(grid1)]
            cat1 = 'cat'
        param_name1 = " ".join(params[0].split("__")[1].split("_"))
        grid2 = model.cv_results_[param_keys[1]].data
        cat2 = 'num'
        if not(type(grid2[0]) == int or type(grid2[0]) == float):
            grid2 = [p  for p in list(grid2)]
            cat2 = 'cat'
        param_name2 = " ".join(params[1].split("__")[1].split("_"))

        cols        = ['cv_error', 'cv_std']
        multi_index = pd.MultiIndex.from_tuples([(p1, p2) for p1, p2 in sorted(zip(grid1, grid2))], names=[param1, param2])
        dfe         = pd.DataFrame([(e, s) for e, s in zip(err, std)], columns=cols, index=multi_index)
        # First hyperparameter
        plt.figure(figsize=figsize)
        ax = plt.gca()
        dfe.unstack(level=1)['cv_error'].plot(ax=ax, style='o-', yerr=dfe.unstack(level=1)['cv_std'])
        #reset color cycle so that the marker colors match
        ax.set_prop_cycle(None)
        #plot the markers
        sc = dfe.unstack(level=1)['cv_error'].plot(figsize=(12,8), style='o-', markersize=5, ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[0:int(len(labels)/2)]
        labels = labels[0:int(len(labels)/2)]
        ax.legend(handles,labels,loc="lower right", title=param_name2)
        if not cat1 == 'cat':
            plt.plot(model.best_params_[params[0]], model.best_score_, marker="o", markersize=15, color="red")
        else:
            pos = list(dfe.unstack(level=1).index).index(model.best_params_[params[0]])
            plt.plot(pos, model.best_score_, marker="o", markersize=15, color="red")
        plt.title(f"Best model with {params[0]} = {str(best_params[params[0]])} and {params[1]} = {str(best_params[params[1]])} ")
        plt.xlabel(param_name1)
        plt.xscale(xscale)
        plt.show()
        # Second hyperparameter
        plt.figure(figsize=figsize)
        dfe.unstack(level=0)['cv_error'].plot(ax=plt.gca(), style='o-', yerr=dfe.unstack(level=1)['cv_std'])
        #reset color cycle so that the marker colors match
        plt.gca().set_prop_cycle(None)
        #plot the markers
        dfe.unstack(level=0)['cv_error'].plot(figsize=(12,8), style='o-', markersize=5, ax = plt.gca())
        handles, labels = plt.gca().get_legend_handles_labels()
        handles = handles[0:int(len(labels)/2)]
        labels = labels[0:int(len(labels)/2)]
        plt.gca().legend(handles,labels,loc="lower right", title=param_name1)
        if not cat2 == 'cat':
            plt.plot(model.best_params_[params[1]], model.best_score_, marker="o", markersize=15, color="red")
        else:
            pos = list(dfe.unstack(level=0).index).index(model.best_params_[params[1]])
            plt.plot(pos, model.best_score_, marker="o", markersize=15, color="red")
        plt.title(f"Best model with {params[0]} = {str(best_params[params[0]])} and {params[1]} = {str(best_params[params[1]])} ")
        plt.xlabel(param_name2)
        plt.xscale(xscale2)
        plt.show()
    else:
        grid=model.cv_results_[param_keys[0]].data
        if not(type(grid[0]) == int or type(grid[0]) == float):
            grid = [p for p in list(grid)]
        param_name= " ".join(params[0].split("__")[1].split("_"))
        
        plt.figure(figsize=figsize)
        plt.errorbar(grid, err, yerr=std, linestyle="None", ecolor='lightblue')
        plt.plot(grid, err, marker="o", markersize=10, c='lightblue')
        plt.plot(model.best_params_[params[0]], model.best_score_, marker="o", markersize=15, color="red")
        plt.title(f"Best model with {params[0]} = {str(best_params[params[0]])} ")
        plt.xlabel(param_name)
        plt.xscale(xscale)
        plt.show()
    return

def plotModelDiagnosis(df: pd.core.frame.DataFrame, pred: str, output_name: str, figsize=(12,6), bins:int=30, smooth_order:int=5):
    """_summary_

    Args:
        df (pd.core.frame.DataFrame): Dataframe containing input, output and prediction variables
        pred (str): Name of prediction column
        output_name (str): Name of output column
        figsize (Tuple[float, float], optional): Size of plot figure. Defaults to (12,6).
        bins (int, optional): bins used in histogram plots. Defaults to 30.
        smooth_order (int, optional): degree of spline of approximation line in scatter plot. Defaults to 5.
    """
    # Create the residuals
    df['residuals'] = df[output_name] - df[pred]
    out_num = np.where(df.columns.values == 'residuals')[0]
    nplots = df.shape[1]
    # Create subplots
    fig, axs = plt.subplots(math.floor(math.sqrt(nplots))+1, math.ceil(math.sqrt(nplots)), figsize=figsize)
    fig.tight_layout(pad=4.0)

    input_num = 0
    for ax in axs.ravel():
        if input_num < nplots:
            # Create plots
            if input_num == out_num:
                df['residuals'].plot.hist(bins=bins, ax=ax)
                ax.set_title('Histogram of residuals')
            else:
                if df.iloc[:,input_num].dtype.name == 'category':
                    sns.boxplot(x=df.columns.values.tolist()[input_num], y='residuals', data=df, ax=ax)
                    ax.set_title(df.columns.values.tolist()[input_num] + ' vs ' + 'residuals')
                else:
                    sns.regplot(x=df.columns.values.tolist()[input_num], y='residuals', data=df, ax=ax, order=smooth_order, ci=None, line_kws={'color':'navy'})
                    ax.set_title(df.columns.values.tolist()[input_num] + ' vs ' + 'residuals')

            input_num += 1
        else:
            ax.axis('off')


