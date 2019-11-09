from bokeh.models import Panel, Tabs
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, LinearColorMapper
from bokeh.models.widgets import DataTable, Select, TableColumn, Slider, MultiSelect, RadioButtonGroup, Div, Button
from bokeh.layouts import column, row, gridplot
from bokeh.palettes import Spectral6, Set1, Category20, RdBu
from bokeh.transform import linear_cmap
from bokeh.models import NumeralTickFormatter

from math import pi
from collections import OrderedDict

import pandas as pd
import numpy as np

from clustering import get_elbow_plot, get_tsne, clustering_data
from regression import get_corr_plot, get_regression_plot

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data_path = "ml_tool/data/"

eda_data_source = {"Credit Card":"CC GENERAL.csv", "House Sales":"kc_house_data.csv"}
clustering_data_source = {"Credit Card":"CC GENERAL.csv"}
regression_data_source = {"House Sales":"kc_house_data.csv"}

x_scat, y_scat = [], []
source_scatter = ColumnDataSource(data=dict(x=x_scat, y=y_scat))
hover_scatter = HoverTool(
        tooltips=[("X", "@x{1.11}"),
                  ("Y", "@y{1.11}")])

plot_scatter = figure(plot_height=500, plot_width=700, tools=['pan,box_zoom,reset']+[hover_scatter])
plot_scatter.scatter(x='x', y='y', size=14, line_color="white", alpha=0.6, hover_color='white',
                     hover_alpha=0.5, source=source_scatter, fill_color='dodgerblue',)

plot_scatter.background_fill_color = "whitesmoke"
plot_scatter.border_fill_color = "whitesmoke"

hist, edges = [], []

source_histogram = ColumnDataSource(data=dict(top=hist, left=edges[:-1], right=edges[1:]))

hover_hist = HoverTool(
        tooltips=[("X", "@left{1.11} ~ @right{1.11}"),
                  ("Y", "@top{int}")])
plot_hist = figure(plot_height=500, plot_width=700, tools=['pan,box_zoom,reset']+[hover_hist])
plot_hist.quad(top='top', bottom=0, left='left', right='right', source=source_histogram,
               fill_color='dodgerblue', line_color="white", fill_alpha=0.8)

plot_hist.background_fill_color = "whitesmoke"
plot_hist.border_fill_color = "whitesmoke"

df_exploration = pd.DataFrame()
source_eda = ColumnDataSource(data=dict(df_exploration))
eda_columns = [TableColumn(field=cols, title=cols) for cols in df_exploration.columns]
table_eda = DataTable(source=source_eda, columns=eda_columns, width=1200, height=300, fit_columns=False)


def update_figure(attr, old, new):
    active_df = explore_data_select.value

    eda_df = pd.read_csv(data_path + str(eda_data_source.get(active_df)))
    eda_df = eda_df.fillna(eda_df.mean())




def create_figure(attr, old, new):
    active_df = explore_data_select.value

    if active_df != "Select dataset":
        eda_df = pd.read_csv(data_path + str(eda_data_source.get(active_df)))
        eda_df = eda_df.fillna(eda_df.mean())
        eda_df.columns = [x.upper() for x in eda_df.columns]

        source_eda.data = dict(eda_df)
        table_eda.columns = [TableColumn(field=cols, title=cols, width=90) for cols in eda_df.columns]

        select_x_axis.options = ["None"]+eda_df.columns.values.tolist()
        select_y_axis.options = ["None"]+eda_df.columns.values.tolist()
        select_hist.options = ["None"]+eda_df.columns.values.tolist()

    xs, ys = [], []

    if select_x_axis.value != "None" and select_y_axis.value != "None":
        xs = eda_df[select_x_axis.value].values
        ys = eda_df[select_y_axis.value].values

    plot_scatter.xaxis.axis_label = select_x_axis.value
    plot_scatter.yaxis.axis_label = select_y_axis.value
    plot_hist.xaxis.axis_label = select_hist.value
    plot_hist.yaxis.axis_label = 'Count'
    source_scatter.data = dict(x=xs, y=ys)

    hist, edges = [], []
    if select_hist.value != 'None':
        hist, edges = np.histogram(eda_df[select_hist.value].values, bins=slider_bins.value)

    source_histogram.data = dict(top=hist, left=edges[:-1], right=edges[1:])


explore_data_select = Select(title="Dataset:", value="Select dataset",
                             options=["Select dataset"]+list(eda_data_source.keys()))
explore_data_select.on_change("value", create_figure)

select_x_axis = Select(title="X-Axis:", value="None", options=["None"])
select_x_axis.on_change('value', create_figure)

select_y_axis = Select(title="Y-Axis:", value="None", options=["None"])
select_y_axis.on_change('value', create_figure)

select_hist = Select(title="Histogram Value:", value="None", options=["None"])
select_hist.on_change('value', create_figure)

slider_bins = Slider(title="Histogram Bins", value=20, start=5.0, end=50, step=1, callback_policy='mouseup')
slider_bins.on_change('value_throttled', create_figure)

tab_eda = Panel(child=column(explore_data_select, table_eda, row(column(select_x_axis, select_y_axis), plot_scatter),
                             row(column(select_hist, slider_bins), plot_hist)), title="Exploration")


"""
Regression
"""

df_reg = pd.DataFrame()
source_reg = ColumnDataSource(data=dict(df_reg))
reg_columns = [TableColumn(field=cols, title=cols) for cols in df_reg.columns]
table_reg = DataTable(source=source_reg, columns=reg_columns, width=1200, height=300, fit_columns=False)

top, bottom, left, right, color, corr = [], [], [], [], [], []
source_corr = ColumnDataSource(data=dict(top=top, bottom=bottom, left=left, right=right, color=color, corr=corr))

hover_corr = HoverTool(
        tooltips=[("Correlation", "@corr{1.11}")])

plot_corr = figure(plot_width=750, plot_height=650, title="Correlation Coefficient",
                   toolbar_location='left', tools=[hover_corr])

plot_corr.quad(top='top', bottom='bottom', left='left',
               right='right', color='color', line_color='white', source=source_corr)
plot_corr.xgrid.grid_line_color = None
plot_corr.ygrid.grid_line_color = None
plot_corr.xaxis.major_label_orientation = pi/4
plot_corr.background_fill_color = "whitesmoke"
plot_corr.border_fill_color = "whitesmoke"
plot_corr.min_border_left = 110
plot_corr.min_border_bottom = 110

colors = list(reversed(RdBu[9]))
mapper = LinearColorMapper(palette=colors, low=-1, high=1)
color_bar = ColorBar(color_mapper=mapper, location=(0, 0))
plot_corr.add_layout(color_bar, 'right')
color_bar.background_fill_color = 'whitesmoke'


actual_reg, predict_reg = [], []
source_reg_scat = ColumnDataSource(data=dict(actual=actual_reg, predict=predict_reg))

hover_reg = HoverTool(
        tooltips=[("Actual", "@actual{int}"),
                  ("Predicted", "@predict{int}")])

plot_reg = figure(plot_height=500, plot_width=700, tools=['pan,box_zoom,reset']+[hover_reg])

plot_reg.scatter(x='actual', y='predict', size=10, line_color="white", alpha=0.6, hover_color='white',
                 hover_alpha=0.5, source=source_reg_scat, fill_color='dodgerblue',)

plot_reg.xaxis.axis_label = "Actual Value"
plot_reg.yaxis.axis_label = "Predicted Value"
plot_reg.background_fill_color = "whitesmoke"
plot_reg.border_fill_color = "whitesmoke"
plot_reg.xaxis.formatter = NumeralTickFormatter(format="00")
plot_reg.yaxis.formatter = NumeralTickFormatter(format="00")

def corr_plot(top, bottom, left, right, labels, nlabels, color_list, corr):
    source_corr.data = dict(top=top, bottom=bottom, left=left, right=right, color=color_list, corr=corr)
    plot_corr.x_range.start, plot_corr.x_range.end = 0, nlabels
    plot_corr.y_range.start, plot_corr.y_range.end = 0, nlabels
    ticks = [tick + 0.5 for tick in list(range(nlabels))]

    tick_dict = OrderedDict([[tick, labels[ii]] for ii, tick in enumerate(ticks)])

    plot_corr.xaxis.ticker = ticks
    plot_corr.yaxis.ticker = ticks
    plot_corr.xaxis.major_label_overrides = tick_dict
    plot_corr.yaxis.major_label_overrides = tick_dict


def reg_plot():

    features = reg_features_ms.value
    label = reg_target_ms.value
    active_df = reg_data_select.value

    if active_df != "Select dataset":
        reg_df = pd.read_csv(data_path + str(regression_data_source.get(active_df)))
        reg_df = reg_df.fillna(reg_df.mean())
        reg_df.columns = [x.upper() for x in reg_df.columns]

    if label != "SELECT TARGET":

        if 'ALL' in features:
            df_columns = reg_df.columns.values.tolist()
            df_columns.remove(label)
            features_df = reg_df.loc[:, df_columns]

        else:
            if label in features:
                features.remove(label)
                features_df = reg_df.loc[:, features]

            else:
                features_df = reg_df.loc[:, features]


        target_df = reg_df.loc[:, label]

        actual_reg, predict_reg = get_regression_plot(features_df, target_df)
        plot_reg.x_range.start, plot_reg.x_range.end = actual_reg.min(), actual_reg.max()
        plot_reg.y_range.start, plot_reg.y_range.end = predict_reg.min(), predict_reg.max()

        source_reg_scat.data = dict(actual=list(actual_reg), predict=list(predict_reg))


def create_figure_reg(attr, old, new):
    active_df = reg_data_select.value

    if active_df != "Select dataset":
        reg_df = pd.read_csv(data_path + str(regression_data_source.get(active_df)))
        reg_df = reg_df.fillna(reg_df.mean())
        reg_df.columns = [x.upper() for x in reg_df.columns]

        source_reg.data = dict(reg_df)
        table_reg.columns = [TableColumn(field=cols, title=cols, width=90) for cols in reg_df.columns]
        # if active_df == "House Sales":
        #     cat_col_reg = ['bedrooms','bathrooms','floors','waterfront','view','condition','grade']
        #
        #     reg_df = pd.concat([pd.get_dummies(reg_df[cat_col_reg].astype('category'), drop_first=True),
        #                reg_df.drop(columns=cat_col_reg)], axis=1, sort=False)

        reg_features_ms.options = ['ALL'] + list(reg_df.columns)
        reg_target_ms.options = ['SELECT TARGET'] + list(reg_df.columns)

        features = reg_features_ms.options
        label = reg_target_ms.value

        # reg_plot(reg_df, features, label)

        top, bottom, left, right, labels, nlabels, color_list, corr = get_corr_plot(reg_df)
        corr_plot(top, bottom, left, right, labels, nlabels, color_list, corr)


reg_data_select = Select(title="Dataset:", value="Select dataset",
                             options=["Select dataset"]+list(regression_data_source.keys()))
reg_data_select.on_change("value", create_figure_reg)


reg_features_ms = MultiSelect(title="Select features for regression:"
                                    "(Features will exclude target even if selected through 'ALL' or explicitly)",
                              value=["ALL"], options=["ALL"])
# reg_features_ms.on_change("value", create_figure_reg)

reg_target_ms = Select(title="Select target for regression:", value="SELECT TARGET", options=["SELECT TARGET"])
# reg_target_ms.on_change("value", create_figure_reg)

button_reg = Button(label="Calculate regression")
button_reg.on_click(reg_plot)

tab_reg = Panel(child=column(reg_data_select, table_reg, plot_corr,
                             row(column(reg_features_ms, reg_target_ms, button_reg), plot_reg)),
                title="Linear Regression")


"""
Clustering
"""

tsne_x, tsne_y, cluster_col = [0], [0], [0]
source_clust = ColumnDataSource(data=dict(x=tsne_x, y=tsne_y, cluster=cluster_col))

hover_clus = HoverTool(
        tooltips=[
            ("User", "$index"),
            ("Cluster", "@cluster")])
mapper = linear_cmap(field_name='cluster', palette=Set1[9], low=min(cluster_col), high=max(cluster_col))
clust_scat = figure(plot_height=600, plot_width=850, tools=['pan,box_zoom,reset,tap']+[hover_clus])
clust_scat.scatter("x", 'y', source=source_clust, color=mapper, size=10, legend='cluster')
clust_scat.axis.major_tick_line_color = None
clust_scat.axis.minor_tick_line_color = None
clust_scat.xaxis.axis_label = "Dimension 1"
clust_scat.yaxis.axis_label = "Dimension 2"
clust_scat.background_fill_color = "whitesmoke"
clust_scat.border_fill_color = "whitesmoke"
clust_scat.title.text_font_size = '12pt'


def cluster_plot():
    active_df = str(clus_data_select.value)
    active_features = clust_features_ms.value
    active_norm = clust_norm_rbg.active
    active_clust_no = clust_slider.value

    source_clust_data = clustering_data(data_path, active_df, active_features, active_norm, active_clust_no,
                                        clustering_data_source, source_clustering, table_clustering,
                                        clust_features_ms, mapper, clust_scat, div_loading)
    source_clust.data = source_clust_data

def clustering_plot(attr, old, new):
    active_df = str(clus_data_select.value)
    clust_df = pd.read_csv(data_path + str(clustering_data_source.get(active_df)))
    clust_df = clust_df.fillna(clust_df.mean())
    clust_df.columns = [x.upper() for x in clust_df.columns]

    source_clustering.data = dict(clust_df)
    table_clustering.columns = [TableColumn(field=cols, title=cols, width=90) for cols in clust_df.columns]
    clust_features_ms.options = ['ALL'] + list(clust_df.columns)
    # active_df = str(clus_data_select.value)
    # active_features = clust_features_ms.value
    # active_norm = clust_norm_rbg.active
    # active_clust_no = clust_slider.value
    #
    # source_clust_data = clustering_data(data_path, active_df, active_features, active_norm, active_clust_no,
    #                                     clustering_data_source, source_clustering, table_clustering,
    #                                     clust_features_ms, mapper, clust_scat, div_loading)
    # source_clust.data = source_clust_data


clus_data_select = Select(title="Dataset:", value="Select dataset",
                          options=["Select dataset"]+list(clustering_data_source.keys()))
clus_data_select.on_change("value", clustering_plot)

df_clustering = pd.DataFrame()
source_clustering = ColumnDataSource(data=dict(df_clustering))
clust_columns = [TableColumn(field=cols, title=cols) for cols in df_clustering.columns]
table_clustering = DataTable(source=source_clustering, columns=clust_columns, width=1200, height=300, fit_columns=False)

clust_features_ms = MultiSelect(title="Select features for clustering:", value=["ALL"], options=["ALL"])
# clust_features_ms.on_change("value", clustering_plot)

clust_norm_rbg = RadioButtonGroup(labels=["Actual Data", "Normalize Data"], active=0)
# clust_norm_rbg.on_change("active", clustering_plot)

clust_slider = Slider(title="Total Clusters", value=5, start=1, end=20, step=1, callback_policy='mouseup')
# clust_slider.on_change('value_throttled', clustering_plot)

button_cluster = Button(label="Calculate clusters")
button_cluster.on_click(cluster_plot)

div_loading = Div(text="""""")
tab_cluster = Panel(child=column(clus_data_select, table_clustering, row(column(clust_features_ms,
                          clust_norm_rbg,clust_slider, button_cluster, div_loading), column(clust_scat))), title="Clustering")

tabs = Tabs(tabs=[tab_eda, tab_reg, tab_cluster], tabs_location='above')

curdoc().add_root(tabs)
curdoc().title = "ML APP"