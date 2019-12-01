# -*- coding: utf-8 -*-
# Updated: Nov-23-19

from bokeh.models import Panel, Tabs
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, LinearColorMapper, Legend, BasicTickFormatter, \
    LegendItem, Span, BasicTicker, LabelSet
from bokeh.models.widgets import DataTable, Select, TableColumn, Slider, MultiSelect, RadioButtonGroup, Div, Button, \
    CheckboxGroup, PreText, Paragraph, FileInput
from bokeh.layouts import column, row, widgetbox
from bokeh.palettes import Spectral6, Set1, Category20, RdBu, RdBu3, Oranges, Blues
from bokeh.transform import linear_cmap, transform
from bokeh.models.ranges import FactorRange
from bokeh.transform import factor_cmap
from bokeh.models.tickers import FixedTicker, SingleIntervalTicker
from bokeh import events
from bokeh.models.callbacks import CustomJS

from math import pi
from collections import OrderedDict

import pandas as pd
import numpy as np

from clustering import get_elbow_plot, get_tsne, clustering_data
from regression import get_corr_plot, get_regression_plot, get_colors
from logistic_regression import get_logreg_output
from classification import get_classify_output

from pybase64 import b64decode
import warnings
import os
import io

warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
CODE

"""


class plot_attributes(object):
    """[summary]

    Arguments:
        object {figure} -- Unformatted plot

    Returns:
        [figure] -- Formatted plot
    """

    def plot_format(self, plot):

        plot.background_fill_color = self.background_fill_color
        plot.border_fill_color = self.border_fill_color
        plot.xaxis.formatter = self.x_axis_format
        plot.yaxis.formatter = self.y_axis_format
        plot.title.align = self.title_align
        plot.title.text_font = self.text_font
        plot.axis.axis_label_text_font = self.axis_label_text_font
        plot.axis.axis_label_text_font_size = self.axis_label_text_font_size
        plot.title.text_font_size = self.text_font_size

        return plot


class landing_page():
    def __init__(self):
        self.note = None

    def upload_fit_data(self, attr, old, new):
            print (file_input.filename)
            decoded = b64decode(new)
            f = io.BytesIO(decoded)
            new_df = pd.read_csv(f)

    def enable_upload(self, attr, old, new):
        if self.load_data_source.active == 1:
            self.file_input.disabled = False
        else:
            self.file_input.disabled = True

    def landing_note(self):
        self.note = Div(text="""Machine Learning Tool: <br> This is a tool to get hands-on experience with Machine Learning
        concepts like Regression, Classification, Clustering. There are 2 ways to use this tool.</br></br>
        <li>If you've a dataset of your choice, you can upload the dataset below. (<b>Note:</b> At this point, please only 
        upload files that are either <i style= "color: red;">.csv/.xls</i> format and smaller in size (<30000 rows)). Larger files would  work but will
        take longer time to execute the models and making plots. </li>
        </br>
        <li>The second option is to choose the pre-loaded datasets (open-source) avaialable within each section.</li> """,
                        style={'font-size': '14pt', 'color': 'black'},
                        width=1200, sizing_mode='stretch_both', css_classes=['div_landing'])


        self.load_data_source = RadioButtonGroup(labels=["Use sample data", "Upload data"], active=0)

        self.file_input = FileInput(accept=".csv,.xls", disabled= True)
        self.file_input.disabled = True

        self.load_data_source.on_change('active', self.enable_upload)
        self.file_input.on_change('value', self.upload_fit_data)

        tab_landing = Panel(child=column(self.note, row(self.load_data_source, self.file_input)),
                            title="Home")
        return tab_landing


class eda_plots(plot_attributes):

    def __init__(self):

        self.active_df = None
        self.table_eda = None
        self.explore_data_select = None
        self.button_eda_plot = None
        self.slider_bins = None
        self.log_x_cb = None
        self.log_y_cb = None
        self.log_hist_cb = None
        self.button_hist_plot = None
        self.plots = None
        self.hover_scatter = None
        self.eda_df = None
        self.button_count_plot = None
        self.plot_count_plot = None
        self.reset_data_eda()

    def reset_data_eda(self):
        self.source_scatter.data = dict(x=[], y=[], color=[])
        self.source_histogram.data = dict(top=[], left=[], right=[])
        self.source_count_plot.data = dict(x=[], y=[])
        self.source_eda.data = {}
        self.table_eda.columns = []
        self.select_x_axis.options = ["None"]
        self.select_y_axis.options = ["None"]
        self.select_color.options = ['None']
        self.select_hist.options = ["None"]
        self.select_count_plot.options = ["None"]
        self.select_x_axis.value = "None"
        self.select_y_axis.value = "None"
        self.select_color.value = 'None'
        self.select_hist.value = "None"
        self.select_count_plot.value = "None"
        self.plot_scatter.xaxis.axis_label = ''
        self.plot_scatter.yaxis.axis_label = ''
        self.plot_hist.xaxis.axis_label = ''
        self.plot_hist.yaxis.axis_label = ''
        self.plot_count_plot.xaxis.axis_label = ''
        self.plot_count_plot.yaxis.axis_label = ''

    def create_eda_figure(self):
        active_df = self.explore_data_select.value
        select_x_axis = self.select_x_axis.value
        select_y_axis = self.select_y_axis.value

        if active_df != "Select dataset":
            ticker_x_dict, ticker_y_dict = {}, {}
            xs, ys = [], []
            if select_x_axis != "None" and select_y_axis != "None":
                if self.log_x_cb.active:
                    if self.log_x_cb.active[0] == 0:
                        xs = np.log(self.eda_df[select_x_axis].values + 1)
                else:
                    xs = self.eda_df[select_x_axis].values

                if self.log_y_cb.active:
                    if self.log_y_cb.active[0] == 0:
                        ys = np.log(self.eda_df[select_y_axis].values + 1)
                else:
                    ys = self.eda_df[select_y_axis].values

            self.plot_scatter.xaxis.axis_label = select_x_axis
            self.plot_scatter.yaxis.axis_label = select_y_axis

            color_dict = {}
            select_color = self.select_color.value

            if select_color != "None":
                color_factors = self.eda_df[select_color].unique().tolist()
                for i in range(0, len(color_factors)):
                    color_dict[str(color_factors[i])] = Category20[20][i]

                scat_color = pd.Series(
                    self.eda_df[select_color].astype(str)).map(color_dict)
                self.source_scatter.data = dict(x=xs, y=ys, color=scat_color)
            else:
                scat_color = ['dodgerblue'] * len(xs)
                self.source_scatter.data = dict(x=xs, y=ys, color=scat_color)

    def create_hist_figure(self):
        active_df = self.explore_data_select.value

        if active_df != "Select dataset":
            hist, edges = [], []
            if self.select_hist.value != 'None':
                self.plot_hist.xaxis.axis_label = self.select_hist.value
                self.plot_hist.yaxis.axis_label = 'Count'

                if self.log_hist_cb.active:
                    if self.log_hist_cb.active[0] == 0:
                        log_hist = np.log(
                            self.eda_df[self.select_hist.value].values + 1)
                else:
                    log_hist = self.eda_df[self.select_hist.value].values

                hist, edges = np.histogram(
                    log_hist, bins=self.slider_bins.value)

            self.source_histogram.data = dict(
                top=hist, left=edges[:-1], right=edges[1:])

    def create_count_figure(self):
        active_df = self.explore_data_select.value

        if active_df != "Select dataset":
            count_column, count_value = [], []
            if self.select_count_plot.value != 'None':
                self.plot_count_plot.xaxis.axis_label = self.select_count_plot.value
                self.plot_count_plot.yaxis.axis_label = 'Count'

                count_df = self.eda_df[self.select_count_plot.value].value_counts(
                ).to_frame()

                count_column, count_value = count_df.index.tolist(
                ), count_df[self.select_count_plot.value].values.tolist()
                count_column = [str(i) for i in count_column]
                self.plot_count_plot.x_range.factors = list(count_column)
                self.source_count_plot.data = dict(
                    x=list(count_column), y=list(count_value))

    def eda_table(self, attr, old, new):
        active_df = self.explore_data_select.value

        if active_df != "Select dataset":
            self.reset_data_eda()
            self.file_path = str(self.cwd + self.data_path +
                                 str(self.eda_data_source.get(active_df)))
            self.eda_df = pd.read_csv(self.file_path)
            self.eda_df = self.eda_df.fillna(self.eda_df.mean())
            self.eda_df.columns = [x.upper() for x in self.eda_df.columns]

            self.source_eda.data = dict(self.eda_df)
            self.table_eda.columns = [TableColumn(
                field=cols, title=cols, width=90) for cols in self.eda_df.columns]

            filter_objects = {}
            filter_numeric = {}
            likely_cat = {}
            for var in self.eda_df.columns:
                filter_objects[var] = self.eda_df[var].dtype == np.float64 or self.eda_df[var].dtype == np.int64
                filter_numeric[var] = str(
                    self.eda_df[var].dtype) == 'object' or self.eda_df[var].nunique() <= 20
                likely_cat[var] = self.eda_df[var].nunique() <= 20

            filter_objects = [
                k for k, v in filter_objects.items() if v is True]
            self.select_x_axis.options = ["None"] + filter_objects
            self.select_y_axis.options = ["None"] + filter_objects
            self.select_hist.options = ["None"] + filter_objects

            likely_cat = [k for k, v in likely_cat.items() if v is True]
            self.select_color.options = ['None'] + likely_cat

            filter_numeric = [
                k for k, v in filter_numeric.items() if v is True]
            self.select_count_plot.options = ["None"] + filter_numeric

        else:
            self.reset_data_eda()

    def eda_button_enable(self, attr, old, new):

        if (self.select_x_axis.value != 'None') and (self.select_y_axis.value != "None"):
            self.button_eda_plot.disabled = False
        else:
            self.button_eda_plot.disabled = True

        if self.select_hist.value != "None":
            self.button_hist_plot.disabled = False
        else:
            self.button_hist_plot.disabled = True

        if self.select_count_plot.value != "None":
            self.button_count_plot.disabled = False
        else:
            self.button_count_plot.disabled = True

    def exploration_plots(self):

        df_exploration = pd.DataFrame()
        self.source_eda = ColumnDataSource(data=dict(df_exploration))
        eda_columns = [TableColumn(field=cols, title=cols)
                       for cols in df_exploration.columns]
        self.table_eda = DataTable(source=self.source_eda, columns=eda_columns, width=1200,
                                   height=300, fit_columns=True)

        x_scat, y_scat, scat_color = [], [], []
        self.source_scatter = ColumnDataSource(
            data=dict(x=x_scat, y=y_scat, color=scat_color))
        self.hover_scatter = HoverTool(
            tooltips=[("X", "@x{1.11}"),
                      ("Y", "@y{1.11}")])

        self.plot_scatter = figure(title="Scatter Plot", plot_height=600, plot_width=800,
                                   tools=['pan,box_zoom,reset'] + [self.hover_scatter])
        self.plot_scatter.scatter(x='x', y='y', size=10, line_color="white", alpha=0.6,
                                  hover_color='white', hover_alpha=0.5, source=self.source_scatter, fill_color='color')
        self.plot_scatter = self.plot_format(self.plot_scatter)
        self.plot_scatter.min_border_left = 75
        self.plot_scatter.min_border_bottom = 75

        hist, edges = [], []

        self.source_histogram = ColumnDataSource(
            data=dict(top=hist, left=edges[:-1], right=edges[1:]))

        hover_hist = HoverTool(
            tooltips=[("X", "@left{1.11} ~ @right{1.11}"),
                      ("Count", "@top{int}")])
        self.plot_hist = figure(title='Histogram', plot_height=600, plot_width=800,
                                tools=['pan,box_zoom,reset'] + [hover_hist])
        self.plot_hist.quad(top='top', bottom=0, left='left', right='right', source=self.source_histogram,
                            fill_color='dodgerblue', line_color="white", fill_alpha=0.8)
        self.plot_hist = self.plot_format(self.plot_hist)
        self.plot_hist.min_border_left = 50
        self.plot_hist.min_border_bottom = 50

        count_column, count_value = [], []

        self.source_count_plot = ColumnDataSource(
            data=dict(x=count_column, y=count_value))

        hover_count_plot = HoverTool(tooltips=[("Category:", "@x"),
                                               ("Count:", "@y{int}")])
        self.plot_count_plot = figure(title="Count Plot", plot_height=600, plot_width=800,
                                      tools=['pan,box_zoom,reset']+[hover_count_plot], x_range=[])
        self.plot_count_plot.vbar(x='x', top='y', width=0.9, source=self.source_count_plot,
                                    fill_color='dodgerblue',
                                    line_color="white", fill_alpha=0.8)
        self.plot_count_plot.background_fill_color = self.background_fill_color
        self.plot_count_plot.border_fill_color = self.border_fill_color
        self.plot_count_plot.title.align = self.title_align
        self.plot_count_plot.title.text_font = self.text_font
        self.plot_count_plot.axis.axis_label_text_font = self.axis_label_text_font
        self.plot_count_plot.axis.axis_label_text_font_size = self.axis_label_text_font_size
        self.plot_count_plot.title.text_font_size = self.text_font_size
        self.plot_count_plot.min_border_top = 50
        self.plot_count_plot.min_border_bottom = 75
        self.plot_count_plot.xaxis.major_label_orientation = pi / 4

        self.explore_data_select = Select(title="Dataset:", value="Select dataset",
                                          options=["Select dataset"] + list(self.eda_data_source.keys()))
        self.select_x_axis = Select(
            title="X-Axis:", value="None", options=["None"])
        self.select_y_axis = Select(
            title="Y-Axis:", value="None", options=["None"])
        self.select_color = Select(
            title="Color:", value="None", options=["None"])
        self.button_eda_plot = Button(label="Draw Plot")
        self.button_eda_plot.disabled = True

        self.select_hist = Select(
            title="Histogram Value:", value="None", options=["None"])
        self.slider_bins = Slider(title="Histogram Bins", value=20, start=5.0, end=50, step=1,
                                  callback_policy='mouseup', css_classes=['custom_slider'])

        self.log_x_cb = CheckboxGroup(
            labels=["Log transform: x-axis"], active=[])
        self.log_y_cb = CheckboxGroup(
            labels=["Log transform: y-axis"], active=[])
        self.log_hist_cb = CheckboxGroup(
            labels=["Log transform axis"], active=[])

        self.button_hist_plot = Button(label="Draw Histogram")
        self.button_hist_plot.disabled = True

        self.select_count_plot = Select(
            title="Count Plot Value:", value="None", options=["None"])
        self.button_count_plot = Button(label="Draw Count Plot")
        self.button_count_plot.disabled = True

        self.select_x_axis.on_change('value', self.eda_button_enable)
        self.select_y_axis.on_change('value', self.eda_button_enable)
        self.select_hist.on_change('value', self.eda_button_enable)
        self.select_count_plot.on_change('value', self.eda_button_enable)
        self.explore_data_select.on_change("value", self.eda_table)
        self.button_eda_plot.on_click(self.create_eda_figure)
        self.button_hist_plot.on_click(self.create_hist_figure)
        self.button_count_plot.on_click(self.create_count_figure)

        tab_eda = Panel(child=column(self.explore_data_select, self.table_eda,
                                     row(column(self.select_x_axis, self.log_x_cb, self.select_y_axis, self.log_y_cb,
                                                self.select_color, self.button_eda_plot), self.plot_scatter),
                                     row(column(self.select_hist, self.log_hist_cb, self.slider_bins,
                                                self.button_hist_plot), self.plot_hist),
                                     row(column(self.select_count_plot,
                                                self.button_count_plot), self.plot_count_plot)),
                        title="Exploration")
        return tab_eda


class linear_regression(plot_attributes):

    """

    Linear Regression Tab

    """

    def __init__(self):
        self.color_bar = None
        self.plot_hist_resid = None
        self.reg_target_ms = None
        self.source_corr = None
        self.plot_corr = None
        self.table_reg = None
        self.button_reg = None
        self.hline = None
        self.hover_corr = None
        self.hover_reg = None
        self.hover_resid = None
        self.hover_resid_hist = None
        self.legend_reg = None
        self.plot_reg = None
        self.plot_resid = None
        self.reg_data_select = None
        self.reg_features_ms = None
        self.reg_scatter = None
        self.active_df = None
        self.reg_df = None
        self.normalize_linreg = None
        self.reset_data_reg()

    def reset_data_reg(self):
        self.source_reg.data = {}
        self.source_reg_scat.data = dict(actual=[], predict=[])
        self.source_reg_resid.data = dict(predict=[], residual=[])
        self.source_hist_resid.data = dict(top=[], bottom=[], right=[])
        self.legend_reg.items = []
        self.table_reg.columns = []
        self.color_bar_reg.scale_alpha = 0
        self.color_bar_reg.major_label_text_alpha = 0
        self.reg_features_ms.options = ["ALL"]
        self.reg_features_ms.value = ["ALL"]
        self.reg_target_ms.options = ['SELECT TARGET']
        self.reg_target_ms.value = 'SELECT TARGET'
        self.button_logreg.disabled = True
        top, bottom, left, right, labels, nlabels, color_list, corr = get_corr_plot(
            pd.DataFrame())
        self.corr_plot(top, bottom, left, right, labels,
                       nlabels, color_list, corr)

    def corr_plot(self, top, bottom, left, right, labels, nlabels, color_list, corr):

        self.source_corr.data = dict(
            top=top, bottom=bottom, left=left, right=right, color=color_list, corr=corr)
        self.plot_corr.x_range.start, self.plot_corr.x_range.end = 0, nlabels
        self.plot_corr.y_range.start, self.plot_corr.y_range.end = 0, nlabels
        ticks = [tick + 0.5 for tick in list(range(nlabels))]

        tick_dict = OrderedDict([[tick, labels[ii]]
                                 for ii, tick in enumerate(ticks)])

        self.color_bar_reg.scale_alpha = 1
        self.color_bar_reg.major_label_text_alpha = 1
        self.plot_corr.xaxis.ticker = ticks
        self.plot_corr.yaxis.ticker = ticks
        self.plot_corr.xaxis.major_label_overrides = tick_dict
        self.plot_corr.yaxis.major_label_overrides = tick_dict

    def reg_plot(self):

        features = self.reg_features_ms.value
        label = self.reg_target_ms.value
        active_norm = self.normalize_linreg.active

        if label != "SELECT TARGET":
            if 'ALL' in features:
                df_columns = self.reg_df.columns.values.tolist()
                df_columns.remove(label)
                features_df = self.reg_df.loc[:, df_columns]

            else:
                if label in features:
                    features.remove(label)
                    features_df = self.reg_df.loc[:, features]

                else:
                    features_df = self.reg_df.loc[:, features]

            target_df = self.reg_df.loc[:, label]

            actual_reg, predict_reg, text, MAE, RMSE, residual, \
                slope, intercept = get_regression_plot(
                    features_df, target_df, active_norm)

            self.plot_reg.x_range.start, self.plot_reg.x_range.end = actual_reg.min(), actual_reg.max()
            self.plot_reg.y_range.start, self.plot_reg.y_range.end = predict_reg.min(), predict_reg.max()

            self.plot_resid.x_range.start, self.plot_resid.x_range.end = predict_reg.min(), predict_reg.max()
            self.plot_resid.y_range.start, self.plot_resid.y_range.end = residual.min(), residual.max()

            self.source_reg_scat.data = dict(
                actual=list(actual_reg), predict=list(predict_reg))
            self.source_reg_resid.data = dict(
                predict=list(predict_reg), residual=list(residual))
            self.legend_reg.items = [LegendItem(label=text[0], renderers=[self.reg_scatter]),
                                     LegendItem(label="MAE - " + str(MAE),
                                                renderers=[self.reg_scatter]),
                                     LegendItem(label="RMSE - " + str(RMSE), renderers=[self.reg_scatter])]

            vhist, vedges = np.histogram(residual, bins=50)
            vmax = max(vhist) * 1.1

            self.plot_hist_resid.x_range.start, self.plot_hist_resid.x_range.end = 0, vmax
            self.plot_hist_resid.y_range.start, self.plot_hist_resid.y_range.end = residual.min(), residual.max()

            self.hline.line_alpha = 0.5
            self.source_hist_resid.data = dict(
                top=vedges[1:], bottom=vedges[:-1], right=vhist)

            self.error_count += 1
            self.alert_reg.text = str(self.error_count)+" Regression Completed"

    def create_figure_reg(self, attr, old, new):
        self.active_df = self.reg_data_select.value

        if self.active_df != "Select dataset":
            self.reset_data_reg()
            self.file_path = str(
                self.cwd + self.data_path + str(self.regression_data_source.get(self.active_df)))

            self.reg_df = pd.read_csv(self.file_path)
            self.reg_df = self.reg_df.fillna(self.reg_df.mean())
            self.reg_df.columns = [x.upper() for x in self.reg_df.columns]
            self.source_reg.data = dict(self.reg_df)
            self.table_reg.columns = [TableColumn(
                field=cols, title=cols, width=90) for cols in self.reg_df.columns]

            self.reg_features_ms.options = ['ALL'] + list(self.reg_df.columns)

            likely_target = {}
            for var in self.reg_df.columns:
                likely_target[var] = self.reg_df[var].nunique(
                ) > self.reg_df.shape[0]*0.1
            likely_target = [k for k, v in likely_target.items() if v is True]
            self.reg_target_ms.options = [
                'SELECT TARGET'] + list(likely_target)

            top, bottom, left, right, labels, nlabels, color_list, corr = get_corr_plot(
                self.reg_df)
            self.corr_plot(top, bottom, left, right, labels,
                           nlabels, color_list, corr)

            self.button_reg.disabled = True
        else:
            self.reset_data_reg()

    def button_enable(self, attr, old, new):
        if self.reg_target_ms.value != 'SELECT TARGET':
            self.button_reg.disabled = False
        else:
            self.button_reg.disabled = True

    def lin_reg(self):
        df_reg = pd.DataFrame()
        self.source_reg = ColumnDataSource(data=dict(df_reg))
        reg_columns = [TableColumn(field=cols, title=cols)
                       for cols in df_reg.columns]
        self.table_reg = DataTable(source=self.source_reg, columns=reg_columns, width=1200, height=300,
                                   fit_columns=True)

        top, bottom, left, right, color, corr = [], [], [], [], [], []
        self.source_corr = ColumnDataSource(
            data=dict(top=top, bottom=bottom, left=left, right=right, color=color, corr=corr))

        self.hover_corr = HoverTool(tooltips=[("Correlation", "@corr{1.11}")])

        self.plot_corr = figure(plot_width=750, plot_height=650, title="Correlation Matrix",
                                toolbar_location='left', tools=[self.hover_corr])

        self.plot_corr.quad(top='top', bottom='bottom', left='left',
                            right='right', color='color', line_color='white', source=self.source_corr)
        self.plot_corr = self.plot_format(self.plot_corr)
        self.plot_corr.xgrid.grid_line_color = None
        self.plot_corr.ygrid.grid_line_color = None
        self.plot_corr.xaxis.major_label_orientation = pi / 4
        self.plot_corr.min_border_left = 110
        self.plot_corr.min_border_bottom = 110
        self.plot_corr.y_range.flipped = True

        corr_colors = list(reversed(RdBu[9]))
        self.reg_mapper = LinearColorMapper(
            palette=corr_colors, low=-1, high=1)

        self.color_bar_reg = ColorBar(color_mapper=self.reg_mapper, location=(0, 0),
                                      ticker=BasicTicker(
                                          desired_num_ticks=len(corr_colors)),
                                      scale_alpha=0, major_label_text_alpha=0)
        self.plot_corr.add_layout(self.color_bar_reg, 'right')
        self.color_bar_reg.background_fill_color = 'whitesmoke'

        actual_reg, predict_reg = [], []
        self.source_reg_scat = ColumnDataSource(
            data=dict(actual=actual_reg, predict=predict_reg))

        self.hover_reg = HoverTool(tooltips=[("Actual", "@actual{int}"),
                                             ("Predicted", "@predict{int}")])

        self.plot_reg = figure(plot_height=500, plot_width=900,
                               tools=['pan,box_zoom,reset,wheel_zoom'] + [self.hover_reg])

        self.reg_scatter = self.plot_reg.scatter(x='actual', y='predict', size=7, line_color="white", alpha=0.6,
                                                 hover_color='white',
                                                 hover_alpha=0.5, source=self.source_reg_scat,
                                                 fill_color='dodgerblue', )

        self.legend_reg = Legend(items=[LegendItem(label="", renderers=[
                                 self.reg_scatter])], location='bottom_right')
        self.plot_reg.add_layout(self.legend_reg)
        self.plot_reg = self.plot_format(self.plot_reg)
        self.plot_reg.xaxis.axis_label = "Actual Value"
        self.plot_reg.yaxis.axis_label = "Predicted Value"

        residual, predict_reg = [], []
        self.source_reg_resid = ColumnDataSource(
            data=dict(predict=predict_reg, residual=residual))

        self.hover_resid = HoverTool(tooltips=[("Predicted", "@predict{int}"),
                                               ("Residual", "@residual{int}")],
                                     names=['resid'])

        self.plot_resid = figure(plot_height=500, plot_width=700,
                                 tools=['pan,box_zoom,reset,wheel_zoom'] + [self.hover_resid])

        self.hline = Span(location=0, dimension='width', line_color='black', line_width=3,
                          line_alpha=0, line_dash="dashed")
        self.plot_resid.renderers.extend([self.hline])

        self.plot_resid.scatter(x='predict', y='residual', size=7, line_color="white", alpha=0.6, hover_color='white',
                                hover_alpha=0.5, source=self.source_reg_resid, fill_color='dodgerblue', name='resid')
        self.plot_resid = self.plot_format(self.plot_resid)
        self.plot_resid.xaxis.axis_label = "Predicted Value"
        self.plot_resid.yaxis.axis_label = "Residual Value"

        vhist, vedges = [], []

        self.source_hist_resid = ColumnDataSource(
            data=dict(top=vedges[1:], bottom=vedges[:-1], right=vhist))
        self.hover_resid_hist = HoverTool(tooltips=[("Count", "@right{int}")])
        self.plot_hist_resid = figure(toolbar_location=None, plot_width=200, plot_height=self.plot_resid.plot_height,
                                      y_range=self.plot_resid.y_range, min_border=10, y_axis_location="right",
                                      tools=[self.hover_resid_hist] + ['pan'])
        self.plot_hist_resid.quad(left=0, bottom='bottom', top='top', right='right', color="dodgerblue",
                                  line_color="white", source=self.source_hist_resid)

        self.plot_hist_resid.ygrid.grid_line_color = None
        self.plot_hist_resid.xaxis.major_label_orientation = np.pi / 4
        self.plot_hist_resid = self.plot_format(self.plot_hist_resid)

        self.reg_data_select = Select(title="Dataset:", value="Select dataset",
                                      options=["Select dataset"] + list(self.regression_data_source.keys()))
        self.reg_features_ms = MultiSelect(
            title="Select features:", value=["ALL"], options=["ALL"])
        self.normalize_linreg = RadioButtonGroup(
            labels=["Actual Data", "Normalize Data"], active=0)

        self.reg_target_ms = Select(title="Select target for regression:", value="SELECT TARGET",
                                    options=["SELECT TARGET"])
        self.button_reg = Button(label="Calculate regression")
        self.button_reg.disabled = True

        self.reg_data_select.on_change("value", self.create_figure_reg)
        self.reg_target_ms.on_change('value', self.button_enable)
        self.button_reg.on_click(self.reg_plot)

        self.div_whitespace = Div(text="""""", height=100)

        self.alert_reg = Div(text='', css_classes=['hidden'], visible=False)

        self.alert_reg.js_on_change('text', self.callback_notification)

        tab_reg = Panel(child=column(self.reg_data_select, self.table_reg, self.plot_corr,
                                     row(column(self.reg_features_ms, self.normalize_linreg,
                                                self.reg_target_ms, self.button_reg),
                                         column(self.plot_reg, row(self.plot_resid, self.plot_hist_resid),
                                                self.alert_reg, self.div_whitespace))),
                        title="Linear Regression")

        return tab_reg


class logistic_regression(plot_attributes):
    """
    Tab for Logistic Regression

    """

    def __init__(self):

        self.active_df = None
        self.logreg_df = None
        self.legend_roc = None
        self.roc_line = None
        self.hover_logreg_cm = None
        self.color_bar_logreg_cm = None
        self.table_class_rep = None
        self.button_logreg = None
        self.hover_logreg_roc = None
        self.labels_logreg_cm = None
        self.logreg_roc_plot = None
        self.normalize_logreg = None
        self.div_report_title = None
        self.reset_data_logreg()

    def reset_data_logreg(self):
        self.source_logreg.data = {}
        self.source_class_rep_logreg.data = {}
        self.source_logreg_cm.data = dict(Actual=[], Prediction=[], value=[])
        self.source_logreg_roc.data = dict(fpr_roc=[], tpr_roc=[])
        self.source_logreg_const_roc.data = dict(
            const_roc_x=[], const_roc_y=[])
        self.table_logreg.columns = []
        self.table_class_rep_logreg.columns = []
        self.legend_roc.items = []
        self.color_bar_logreg_cm.scale_alpha = 0
        self.color_bar_logreg_cm.major_label_text_alpha = 0
        self.logreg_features_ms.options = ["ALL"]
        self.logreg_features_ms.value = ["ALL"]
        self.logreg_target_ms.options = ['SELECT TARGET']
        self.logreg_target_ms.value = 'SELECT TARGET'
        self.button_logreg.disabled = True

    def logreg_button_enable(self, attr, old, new):

        if self.logreg_target_ms.value != 'SELECT TARGET':
            self.button_logreg.disabled = False
        else:
            self.button_logreg.disabled = True

    def create_figure_logreg(self, attr, old, new):
        self.active_df = self.logreg_data_select.value

        if self.active_df != "Select dataset":
            self.reset_data_logreg()

            self.file_path = str(self.cwd + self.data_path +
                                 str(self.logreg_data_source.get(self.active_df)))

            logreg_df = pd.read_csv(self.file_path)
            logreg_df = logreg_df.fillna(logreg_df.mean())
            logreg_df.columns = [x.upper() for x in logreg_df.columns]
            self.logreg_df = logreg_df

            self.source_logreg.data = dict(logreg_df)
            self.table_logreg.columns = [TableColumn(field=cols, title=cols, width=90) for cols in
                                         self.logreg_df.columns]

            self.logreg_features_ms.options = [
                "ALL"] + logreg_df.columns.values.tolist()

            likely_cat = {}
            for var in logreg_df.columns:
                likely_cat[var] = logreg_df[var].nunique() == 2 and set(
                    logreg_df[var].unique()) == set([0, 1])
            likely_cat = [k for k, v in likely_cat.items() if v is True]

            self.logreg_target_ms.options = ['SELECT TARGET'] + likely_cat
            self.button_logreg.disabled = True
        else:
            self.reset_data_logreg()

    def logreg_plot(self):
        features = self.logreg_features_ms.value
        label = self.logreg_target_ms.value
        logreg_df = self.logreg_df
        active_norm = self.normalize_logreg.active

        if label != "SELECT TARGET":
            if 'ALL' in features:
                df_columns = logreg_df.columns.values.tolist()
                df_columns.remove(label)
                features_df = logreg_df.loc[:, df_columns]
            else:
                if label in features:
                    features.remove(label)
                    features_df = logreg_df.loc[:, features]
                else:
                    features_df = logreg_df.loc[:, features]

            target_df = logreg_df.loc[:, label]

        accuracy_score, class_report_df, confusion_df, \
            logit_roc_auc, fpr, tpr, thresholds = get_logreg_output(
                features_df, target_df, active_norm)

        self.source_class_rep_logreg.data = dict(class_report_df)
        self.table_class_rep_logreg.columns = [TableColumn(field=cols, title=cols, width=90) for cols in
                                               class_report_df.columns]
        self.table_class_rep_logreg.index_position = None

        self.logreg_cm_mapper.low, self.logreg_cm_mapper.high = confusion_df.value.values.min(
        ), confusion_df.value.values.max()
        self.color_bar_logreg_cm.scale_alpha = 1
        self.color_bar_logreg_cm.major_label_text_alpha = 1

        self.logreg_cm_plot.x_range.start, self.logreg_cm_plot.x_range.end = confusion_df.Actual.min(), \
            confusion_df.Actual.max()
        self.logreg_cm_plot.y_range.start, self.logreg_cm_plot.y_range.end = confusion_df.Prediction.min(), \
            confusion_df.Prediction.max()

        self.logreg_cm_plot.xaxis.ticker = sorted(target_df.unique())
        self.logreg_cm_plot.yaxis.ticker = sorted(target_df.unique())
        self.logreg_cm_plot.xaxis.axis_label = "Actual"
        self.logreg_cm_plot.yaxis.axis_label = "Predicted"

        self.source_logreg_cm.data = confusion_df
        self.source_logreg_roc.data = dict(fpr_roc=fpr, tpr_roc=tpr)
        self.logreg_roc_plot.xaxis.axis_label = "False Positive Rate"
        self.logreg_roc_plot.yaxis.axis_label = "True Positive Rate"
        self.legend_roc.items = [LegendItem(label="Logistic Regression (area = " + str(logit_roc_auc) + ")",
                                            renderers=[self.roc_line])]
        self.source_logreg_const_roc.data = dict(
            const_roc_x=[0, 1], const_roc_y=[0, 1])

        self.error_count += 1
        self.alert_logreg.text = str(
            self.error_count)+" Logistic Regression Completed"

    def logreg(self):

        df_logreg = pd.DataFrame()
        self.source_logreg = ColumnDataSource(data=dict(df_logreg))
        logreg_columns = [TableColumn(field=cols, title=cols)
                          for cols in df_logreg.columns]
        self.table_logreg = DataTable(source=self.source_logreg, columns=logreg_columns, width=1200, height=300,
                                      fit_columns=True)

        df_class_report = pd.DataFrame()
        self.source_class_rep_logreg = ColumnDataSource(
            data=dict(df_class_report))
        class_rep_columns_logreg = [TableColumn(
            field=cols, title=cols) for cols in df_class_report.columns]
        self.table_class_rep_logreg = DataTable(source=self.source_class_rep_logreg, columns=class_rep_columns_logreg,
                                                width=600, height=200, fit_columns=True)

        logreg_cm_colors = list(reversed(Blues[9]))
        actual_cm, predicted_cm, value_cm = [], [], []
        self.source_logreg_cm = ColumnDataSource(
            data=dict(Actual=actual_cm, Prediction=predicted_cm, value=value_cm))

        self.logreg_cm_mapper = LinearColorMapper(
            palette=logreg_cm_colors, low=0, high=100)

        self.labels_logreg_cm = LabelSet(x='Actual', y='Prediction', text='value', level='overlay', x_offset=0,
                                         y_offset=-10,
                                         source=self.source_logreg_cm, render_mode='canvas', text_align='center',
                                         text_font='times',
                                         text_color='#FF0000', text_font_style='bold', text_font_size='16px')

        self.hover_logreg_cm = HoverTool(tooltips=[("Actual", "@Actual"),
                                                   ("Predicted", "@Prediction"),
                                                   ("Value", "@value")])
        self.logreg_cm_plot = figure(plot_width=400, plot_height=300, title="Confusion Matrix", toolbar_location=None,
                                     tools=[self.hover_logreg_cm], x_axis_location="above")

        self.logreg_cm_plot.rect(x="Actual", y="Prediction", width=.9, height=.9, source=self.source_logreg_cm,
                                 line_color='black', fill_color=transform('value', self.logreg_cm_mapper))
        self.logreg_cm_plot.y_range.flipped = True

        self.color_bar_logreg_cm = ColorBar(color_mapper=self.logreg_cm_mapper, location=(0, 0),
                                            ticker=BasicTicker(
                                                desired_num_ticks=len(logreg_cm_colors)),
                                            scale_alpha=0, major_label_text_alpha=0)

        self.logreg_cm_plot.add_layout(self.color_bar_logreg_cm, 'right')
        self.color_bar_logreg_cm.background_fill_color = "whitesmoke"

        self.logreg_cm_plot = self.plot_format(self.logreg_cm_plot)
        self.logreg_cm_plot.add_layout(self.labels_logreg_cm)
        self.logreg_cm_plot.min_border_left = 50
        self.logreg_cm_plot.min_border_top = 50

        self.hover_logreg_roc = HoverTool(tooltips=[("False Positive Rate", "@fpr_roc"),
                                                    ("True Positive Rate", "@tpr_roc")],
                                          names=['roc'])

        fpr_roc, tpr_roc = [], []

        self.source_logreg_roc = ColumnDataSource(
            data=dict(fpr_roc=fpr_roc, tpr_roc=tpr_roc))

        const_roc_x, const_roc_y = [], []
        self.source_logreg_const_roc = ColumnDataSource(
            data=dict(const_roc_x=const_roc_x, const_roc_y=const_roc_y))

        self.logreg_roc_plot = figure(plot_width=500, plot_height=500, title="ROC AUC", toolbar_location=None,
                                      tools=[self.hover_logreg_roc], x_range=(-0.04, 1.04), y_range=(-0.04, 1.04))

        self.roc_line = self.logreg_roc_plot.line(x="fpr_roc", y="tpr_roc", line_width=4, source=self.source_logreg_roc,
                                                  line_color='dodgerblue', name='roc')
        self.logreg_roc_plot.line(x="const_roc_x", y="const_roc_y", line_width=2, line_dash='dashed',
                                  source=self.source_logreg_const_roc, line_color='orangered')
        self.legend_roc = Legend(items=[LegendItem(label="", renderers=[
                                 self.roc_line])], location='bottom_right')
        self.logreg_roc_plot.add_layout(self.legend_roc)
        self.logreg_roc_plot = self.plot_format(self.logreg_roc_plot)
        self.logreg_roc_plot.min_border_left = 50
        self.logreg_roc_plot.min_border_bottom = 50

        self.logreg_data_select = Select(title="Dataset:", value="Select dataset",
                                         options=["Select dataset"] + list(self.logreg_data_source.keys()))
        self.logreg_features_ms = MultiSelect(
            title="Select features:", value=["ALL"], options=["ALL"])
        self.normalize_logreg = RadioButtonGroup(
            labels=["Actual Data", "Normalize Data"], active=0)

        self.logreg_target_ms = Select(title="Select target for Logistic regression:", value="SELECT TARGET",
                                       options=["SELECT TARGET"])
        self.button_logreg = Button(label="Calculate regression")
        self.button_logreg.disabled = True

        self.logreg_data_select.on_change("value", self.create_figure_logreg)
        self.logreg_target_ms.on_change('value', self.logreg_button_enable)
        self.button_logreg.on_click(self.logreg_plot)

        self.div_report_title = Div(
            text="""<center>Classification Report</center>""", width=600)

        self.alert_logreg = Div(text='', css_classes=['hidden'], visible=False)

        self.alert_logreg.js_on_change('text', self.callback_notification)

        tab_logreg = Panel(child=column(self.logreg_data_select, self.table_logreg,
                                        row(column(self.logreg_features_ms, self.normalize_logreg,
                                                   self.logreg_target_ms, self.button_logreg),
                                            column(self.div_report_title, self.table_class_rep_logreg, self.logreg_cm_plot,
                                                   self.logreg_roc_plot, self.alert_logreg))),
                           title="Logistic Regression")

        return tab_logreg


class classification(plot_attributes):

    def __init__(self):
        self.source_classify = None

    def create_figure_classify(self, attr, old, new):
        self.active_df = self.classify_data_select.value

        if self.active_df != "Select dataset":
            self.file_path = str(
                self.cwd + self.data_path + str(self.classify_data_source.get(self.active_df)))
            classify_df = pd.read_csv(self.file_path)
            classify_df = classify_df.fillna(classify_df.mean())
            classify_df.columns = [x.upper() for x in classify_df.columns]
            self.classify_df = classify_df

            self.source_classify.data = dict(classify_df)
            self.table_classify.columns = [TableColumn(field=cols, title=cols, width=90) for cols in
                                           self.classify_df.columns]

            self.classify_features_ms.options = [
                "ALL"] + classify_df.columns.values.tolist()

            likely_cat = {}
            for var in classify_df.columns:
                likely_cat[var] = classify_df[var].nunique() <= 20
            likely_cat = [k for k, v in likely_cat.items() if v is True]

            self.classify_target_ms.options = ['SELECT TARGET'] + likely_cat

            self.button_classify.disabled = True

        else:
            self.source_classify.data = {}
            self.table_classify.columns = []
            self.classify_features_ms.options = ["ALL"]
            self.classify_features_ms.value = ["ALL"]
            self.classify_target_ms.options = ['SELECT TARGET']
            self.classify_target_ms.value = 'SELECT TARGET'
            self.button_classify.disabled = True
            self.source_classify_cm.data = {}
            self.source_classify_fi.data = {}
            self.source_class_rep_classify.data = {}

    def classify_button_enable(self, attr, old, new):
        if self.classify_target_ms.value != "SELECT TARGET":
            self.button_classify.disabled = False
        else:
            self.button_classify.disabled = True

    def classify_plot(self):
        features = self.classify_features_ms.value
        label = self.classify_target_ms.value
        classify_df = self.classify_df
        active_norm = self.normalize_classify.active

        if label != "SELECT TARGET":
            if 'ALL' in features:
                df_columns = classify_df.columns.values.tolist()
                df_columns.remove(label)
                features_df = classify_df.loc[:, df_columns]
            else:
                if label in features:
                    features.remove(label)
                    features_df = classify_df.loc[:, features]
                else:
                    features_df = classify_df.loc[:, features]

            target_df = classify_df.loc[:, label]

        accuracy_score, class_report_df, confusion_df, \
            rf_feature_labels, rf_feature_importance = get_classify_output(
                features_df, target_df, active_norm)

        self.source_class_rep_classify.data = dict(class_report_df)
        self.table_class_rep_classify.columns = [TableColumn(field=cols, title=cols, width=90) for cols in
                                                 class_report_df.columns]
        self.table_class_rep_classify.index_position = None

        self.classify_cm_mapper.low, self.classify_cm_mapper.high = confusion_df.value.values.min(), \
            confusion_df.value.values.max()
        self.color_bar_classify_cm.scale_alpha = 1
        self.color_bar_classify_cm.major_label_text_alpha = 1

        if str(confusion_df['Actual'].dtype) == 'object' or str(confusion_df['Prediction'].dtype) == 'object':

            self.classify_cm_plot.xaxis.ticker = list(
                set(pd.Categorical(confusion_df['Actual']).codes))
            ticker_x_dict = dict(
                enumerate(pd.Categorical(confusion_df['Actual']).categories))
            confusion_df['Actual'] = pd.Categorical(
                confusion_df['Actual']).codes
            self.classify_cm_plot.xaxis.major_label_overrides = ticker_x_dict
            self.classify_cm_plot.xaxis.major_label_orientation = pi / 4

            self.classify_cm_plot.yaxis.ticker = list(
                set(pd.Categorical(confusion_df['Prediction']).codes))
            ticker_y_dict = dict(enumerate(pd.Categorical(
                confusion_df['Prediction']).categories))
            confusion_df['Prediction'] = pd.Categorical(
                confusion_df['Prediction']).codes
            self.classify_cm_plot.yaxis.major_label_overrides = ticker_y_dict

        else:
            self.classify_cm_plot.x_range.start, self.classify_cm_plot.x_range.end = confusion_df.Actual.min(), \
                confusion_df.Actual.max()
            self.classify_cm_plot.y_range.start, self.classify_cm_plot.y_range.end = confusion_df.Prediction.min(), \
                confusion_df.Prediction.max()
            self.classify_cm_plot.xaxis.ticker = sorted(target_df.unique())
            self.classify_cm_plot.yaxis.ticker = sorted(target_df.unique())

        self.classify_cm_plot.xaxis.axis_label = "Actual"
        self.classify_cm_plot.yaxis.axis_label = "Predicted"

        self.source_classify_cm.data = confusion_df
        rf_df = pd.DataFrame(dict({'rf_features': rf_feature_labels,
                                   'rf_importance': rf_feature_importance})).nlargest(15, "rf_importance")
        self.source_classify_fi.data = dict(rf_df)
        self.classify_fi_plot.x_range.factors = rf_df['rf_features'].values.tolist(
        )

        self.error_count += 1
        self.alert_classify.text = str(
            self.error_count)+" Classification completed"

    def classify(self):
        df_classify = pd.DataFrame()
        self.source_classify = ColumnDataSource(data=dict(df_classify))
        classify_columns = [TableColumn(field=cols, title=cols)
                            for cols in df_classify.columns]
        self.table_classify = DataTable(source=self.source_classify, columns=classify_columns, width=1200, height=300,
                                        fit_columns=True)

        df_class_report = pd.DataFrame()
        self.source_class_rep_classify = ColumnDataSource(
            data=dict(df_class_report))
        class_rep_columns_classify = [TableColumn(
            field=cols, title=cols) for cols in df_class_report.columns]
        self.table_class_rep_classify = DataTable(source=self.source_class_rep_classify, columns=class_rep_columns_classify, width=600, height=200,
                                                  fit_columns=True)

        classify_cm_colors = list(reversed(Blues[9]))
        actual_cm, predicted_cm, value_cm = [], [], []
        self.source_classify_cm = ColumnDataSource(data=dict(Actual=actual_cm, Prediction=predicted_cm,
                                                             value=value_cm))

        self.classify_cm_mapper = LinearColorMapper(
            palette=classify_cm_colors, low=0, high=100)
        self.labels_classify_cm = LabelSet(x='Actual', y='Prediction', text='value', level='overlay', x_offset=0,
                                           y_offset=-10,
                                           source=self.source_classify_cm, render_mode='canvas', text_align='center',
                                           text_font='times',
                                           text_color='#FF0000', text_font_style='bold', text_font_size='16px')

        self.hover_classify_cm = HoverTool(tooltips=[("Actual", "@Actual"),
                                                     ("Predicted", "@Prediction"),
                                                     ("Value", "@value")])
        self.classify_cm_plot = figure(plot_width=600, plot_height=550, title="Confusion Matrix", toolbar_location=None,
                                       tools=[self.hover_logreg_cm], x_axis_location="above")
        self.classify_cm_plot.rect(x="Actual", y="Prediction", width=.9, height=.9, source=self.source_classify_cm,
                                   line_color='black', fill_color=transform('value', self.classify_cm_mapper))
        self.classify_cm_plot.y_range.flipped = True

        self.color_bar_classify_cm = ColorBar(color_mapper=self.classify_cm_mapper, location=(0, 0),
                                              ticker=BasicTicker(
                                                  desired_num_ticks=len(classify_cm_colors)),
                                              scale_alpha=0, major_label_text_alpha=0)

        self.classify_cm_plot.add_layout(self.color_bar_classify_cm, 'right')
        self.color_bar_classify_cm.background_fill_color = "whitesmoke"
        self.classify_cm_plot = self.plot_format(self.classify_cm_plot)
        self.classify_cm_plot.add_layout(self.labels_classify_cm)
        self.classify_cm_plot.min_border_left = 100
        self.classify_cm_plot.min_border_top = 100
        self.classify_cm_plot.min_border_bottom = 50

        rf_features = []
        rf_importance = []

        self.hover_classify_fi = HoverTool(tooltips=[("Feature", "@rf_features"),
                                                     ("Importance Score", "@rf_importance{0.02f}")])
        self.source_classify_fi = ColumnDataSource(
            data=dict(rf_features=rf_features, rf_importance=rf_importance))
        self.classify_fi_plot = figure(x_range=[], plot_width=600, plot_height=600, toolbar_location=None,
                                       title="Feature Importance", tools=[self.hover_classify_fi])
        self.classify_fi_plot.vbar(x='rf_features', top='rf_importance', bottom=0, width=0.9,
                                   source=self.source_classify_fi, line_color='white', fill_color='dodgerblue')
        self.classify_fi_plot.background_fill_color = self.background_fill_color
        self.classify_fi_plot.border_fill_color = self.border_fill_color
        self.classify_fi_plot.yaxis.formatter = self.x_axis_format
        self.classify_fi_plot.title.align = self.title_align
        self.classify_fi_plot.title.text_font = self.text_font
        self.classify_fi_plot.axis.axis_label_text_font = self.axis_label_text_font
        self.classify_fi_plot.axis.axis_label_text_font_size = '8pt'
        self.classify_fi_plot.title.text_font_size = self.text_font_size
        self.classify_fi_plot.xaxis.major_label_orientation = pi / 4
        self.classify_fi_plot.min_border_left = 50
        self.classify_fi_plot.min_border_bottom = 100

        self.classify_data_select = Select(title="Dataset:", value="Select dataset",
                                           options=["Select dataset"] + list(self.classify_data_source.keys()))
        self.classify_features_ms = MultiSelect(
            title="Select features:", value=["ALL"], options=["ALL"])
        self.normalize_classify = RadioButtonGroup(
            labels=["Actual Data", "Normalize Data"], active=0)

        self.classify_target_ms = Select(title="Select target for Classification:", value="SELECT TARGET",
                                         options=["SELECT TARGET"])
        self.button_classify = Button(label="Perform classification")
        self.button_classify.disabled = True

        self.classify_data_select.on_change(
            'value', self.create_figure_classify)
        self.classify_target_ms.on_change("value", self.classify_button_enable)
        self.button_classify.on_click(self.classify_plot)

        self.div_report_title = Div(
            text="""<center>Classification Report</center>""", width=600)

        self.alert_classify = Div(text='', css_classes=[
                                  'hidden'], visible=False)

        self.alert_classify.js_on_change('text', self.callback_notification)

        tab_classify = Panel(child=column(self.classify_data_select, self.table_classify,
                                          row(column(self.classify_features_ms, self.normalize_classify, self.classify_target_ms,
                                                     self.button_classify),
                                              column(self.div_report_title, self.table_class_rep_classify, column(self.classify_cm_plot, self.classify_fi_plot, self.alert_classify)))),
                             title="Classification")

        return tab_classify


class clustering(plot_attributes):
    """

    Tab for Clustering

    """

    def __init__(self):
        self.source_clustering = None
        self.clust_df = None
        self.source_clust = None
        self.mapper = None
        self.clust_scat = None
        self.clust_slider = None
        self.button_cluster = None
        self.clus_data_select = None
        self.clust_features_ms = None
        self.clust_norm_rbg = None
        self.hover_clust = None
        self.table_clustering = None

    def cluster_plot(self):

        active_features = self.clust_features_ms.value
        active_norm = self.clust_norm_rbg.active
        active_clust_no = self.clust_slider.value

        source_clust_data = clustering_data(self.clust_df, active_features, active_norm, active_clust_no,
                                            self.clustering_data_source, self.mapper, self.clust_scat)
        self.source_clust.data = source_clust_data
        self.error_count += 1
        self.alert_cluster.text = str(self.error_count)+" Clustering Completed"

    def clustering_plot(self, attr, old, new):
        self.active_df = str(self.clus_data_select.value)

        if self.active_df != "Select dataset":
            self.button_cluster.disabled = False
            self.file_path = str(
                self.cwd + self.data_path + str(self.clustering_data_source.get(self.active_df)))

            clust_df = pd.read_csv(self.file_path)
            clust_df = clust_df.fillna(clust_df.mean())
            clust_df.columns = [x.upper() for x in clust_df.columns]

            self.clust_df = clust_df

            self.source_clustering.data = dict(clust_df)
            self.table_clustering.columns = [TableColumn(field=cols, title=cols, width=90) for cols in
                                             self.clust_df.columns]
            self.clust_features_ms.options = ['ALL'] + list(clust_df.columns)

        else:
            self.button_cluster.disabled = True

    def cluster(self):
        df_clustering = pd.DataFrame()
        self.source_clustering = ColumnDataSource(data=dict(df_clustering))
        clust_columns = [TableColumn(field=cols, title=cols)
                         for cols in df_clustering.columns]
        self.table_clustering = DataTable(source=self.source_clustering, columns=clust_columns, width=1200, height=300,
                                          fit_columns=True)

        tsne_x, tsne_y, cluster_col = [0], [0], [0]
        self.source_clust = ColumnDataSource(
            data=dict(x=tsne_x, y=tsne_y, cluster=cluster_col))

        self.hover_clust = HoverTool(tooltips=[("User", "$index"),
                                               ("Cluster", "@cluster")])
        self.mapper = linear_cmap(field_name='cluster', palette=Set1[9], low=min(
            cluster_col), high=max(cluster_col))
        self.clust_scat = figure(plot_height=600, plot_width=850, tools=[
                                 'pan,box_zoom,reset,tap'] + [self.hover_clust])
        self.clust_scat.scatter(
            "x", 'y', source=self.source_clust, color=self.mapper, size=10, legend='cluster')
        self.clust_scat.axis.major_tick_line_color = None
        self.clust_scat.axis.minor_tick_line_color = None
        self.clust_scat.xaxis.axis_label = "Dimension 1"
        self.clust_scat.yaxis.axis_label = "Dimension 2"
        self.clust_scat.title.text_font_size = '12pt'
        self.clust_scat.min_border_top = 40
        self.clust_scat = self.plot_format(self.clust_scat)

        self.clus_data_select = Select(title="Dataset:", value="Select dataset",
                                       options=["Select dataset"] + list(self.clustering_data_source.keys()))
        self.clust_features_ms = MultiSelect(
            title="Select features for clustering:", value=["ALL"], options=["ALL"])
        self.clust_norm_rbg = RadioButtonGroup(
            labels=["Actual Data", "Normalize Data"], active=0)
        self.clust_slider = Slider(title="Total Clusters", value=5, start=1, end=20, step=1,
                                   callback_policy='mouseup', css_classes=['custom_slider'])
        self.button_cluster = Button(
            label="Calculate and plot clusters", css_classes=['button'])
        self.button_cluster.disabled = True

        self.clus_data_select.on_change("value", self.clustering_plot)
        self.button_cluster.on_click(self.cluster_plot)

        self.alert_cluster = Div(text='', css_classes=[
                                 'hidden'], visible=False)

        self.alert_cluster.js_on_change('text', self.callback_notification)

        tab_cluster = Panel(child=column(self.clus_data_select, self.table_clustering,
                                         row(column(self.clust_features_ms, self.clust_norm_rbg, self.clust_slider,
                                                    self.button_cluster),
                                             column(self.clust_scat), self.alert_cluster)), title="Clustering")

        return tab_cluster


class main_tool(landing_page, eda_plots, linear_regression, logistic_regression, clustering, classification):

    """
        Add datasets in each dictionary based on the algorithm
        eda_data_source: Can be any dataset for exploratory analysis
        clustering_data_source: Dataset for clustering algorithm
        regression_data_source: Dataset for linear regression algorithm
        logreg_data_source: Dataset for logistic regression algorithm
        classify_data_source: Dataset for multilabel classification algorithm

    """

    def __init__(self):
        self.cwd = str(os.getcwd())
        self.data_path = "/ML/Data/"
        self.eda_data_source = {"Credit Card (Clustering)": "CC GENERAL.csv",
                                "House Sales (Lin. Reg.)": "HOUSING PRICE.csv",
                                "Diabetes (Log. Reg.)": "DIABETES.csv",
                                "Glass Type (Classification)": "GLASS.csv",
                                "Census Income (Classification)": "CENSUS_INCOME.csv"}
        self.clustering_data_source = {"Credit Card": "CC GENERAL.csv"}
        self.regression_data_source = {"House Sales": "HOUSING PRICE.csv"}
        self.logreg_data_source = {"Diabetes": "DIABETES.csv"}
        self.classify_data_source = {"Glass Type": "GLASS.csv", "Mobile Prices": "MOBILE.csv",
                                     "Census Income": "CENSUS_INCOME.csv"}

        self.background_fill_color = 'whitesmoke'
        self.border_fill_color = 'whitesmoke'
        self.x_axis_format = BasicTickFormatter(use_scientific=False)
        self.y_axis_format = BasicTickFormatter(use_scientific=False)
        self.title_align = 'center'
        self.text_font_size = '12pt'
        self.text_font = 'times'
        self.axis_label_text_font = 'times'
        self.axis_label_text_font_size = "12pt"
        self.error_count = 0
        self.callback_notification = CustomJS(args={}, code="""var x = document.getElementById("toast")
                x.className = "show";
                s = cb_obj.text
                document.getElementById("desc").innerHTML = s.substr(s.indexOf(' ')+1);
                setTimeout(function(){ x.className = x.className.replace("show", ""); }, 5000);""")

    def run_tool(self):
        landing_tab = self.landing_note()
        eda_tab = self.exploration_plots()
        linreg_tab = self.lin_reg()
        logreg_tab = self.logreg()
        cluster_tab = self.cluster()
        classify_tab = self.classify()

        tabs = Tabs(tabs=[landing_tab, eda_tab, linreg_tab, logreg_tab, classify_tab, cluster_tab],
                    tabs_location='above', sizing_mode='scale_both', active=0)

        return tabs


tabs = main_tool().run_tool()

curdoc().add_root(tabs)
curdoc().title = "ML APP"
