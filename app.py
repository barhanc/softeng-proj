"""
NOTE: THIS CODE IS UNSAFE GARBAGE BUT CODING GUIs IS GARBAGE SO IT FITS
"""

# # macOS packaging support
# from multiprocessing import freeze_support  # noqa

# freeze_support()  # noqa

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_svg

plt.style.use("ggplot")

from io import StringIO
from typing import List
from dataclasses import dataclass

from nicegui import ui, events, native
from engine.preprocess import PreprocessModule
from engine.cluster import ClusterModule
from engine.pca import PCAModule

from pandas.api.types import is_numeric_dtype

@dataclass
class State:
    df: pd.DataFrame
    prev: List["State"]


class App:
    def __init__(self):
        self.state = State(None, None)

    def main(self):
        def page_preprocess():
            def csv_file_handler(e: events.UploadEventArguments):
                ui.notify(f"Uploaded {e.name}")
                try:
                    content = e.content.read().decode("utf-8")
                    self.state.df = pd.read_csv(StringIO(content), delimiter=";", decimal=",")
                    self.state.prev = None
                    refresh()
                except Exception as ex:
                    ui.notify(f"Error occurred while opening file: {str(ex)}")

            def show_table():
                df = self.state.df
                df = df.rename(columns={col: f"{col} ({df.dtypes[col]})" for col in df.columns})

                table_container.clear()
                with table_container:
                    ui.table(
                        columns=[{"name": col, "label": col, "field": col} for col in df.columns],
                        rows=df.to_dict(orient="records"),
                        pagination={"rowsPerPage": 10},
                    )

            def change_types():
                def on_click():
                    try:
                        for col in cols.value:
                            self.state = State(df=self.state.df.astype({col: typ.value}), prev=self.state)
                            ui.notify(f"Successfully converted '{col}' into type '{typ.value}'")
                        show_table()
                    except Exception as e:
                        ui.notify(f"Error: {e}")

                change_types_container.clear()
                with change_types_container, ui.card().classes("w-80"), ui.scroll_area():
                    with ui.row():
                        ui.button("", icon="check", on_click=on_click)
                        ui.label("Change type")
                    cols = ui.select([col for col in self.state.df.columns], multiple=True)
                    typ = ui.select(["int", "float", "bool", "object"], value="object")


            def rename_columns():
                def on_enter(e, col: str, new: str):
                    if not new:
                        ui.notify(f"No name was provided")
                        return
                    try:
                        self.state = State(self.state.df.rename(columns={col: new}), prev=self.state)
                        ui.notify(f"Successfully renamed '{col}' into '{new}'")
                        refresh()
                    except Exception as ex:
                        ui.notify(f"Could not rename '{col}' into '{new}'")

                def on_click(e):
                    on_enter(e, col.value, name_input.value)

                rename_columns_container.clear()
                with rename_columns_container, ui.card().classes("w-80"), ui.scroll_area():
                    with ui.row():
                        ui.button("", icon="check", on_click=on_click)
                        ui.label("Rename columns")
                    col = ui.select([col for col in self.state.df.columns])
                    name_input = ui.input(label="New name")
                    name_input.on("keydown.enter", handler=lambda e: on_enter(e, col.value, name_input.value))

            def apply_scaler():
                def on_change(e):
                    desc.set_content(PreprocessModule.scalers[e.value]["docstr"])
                    desc.props("size=80")

                def on_click(e):
                    if len(col.value) == 0:
                        ui.notify("No columns have been chosen")
                        return

                    try:
                        self.state = State(
                            df=PreprocessModule.scale(X=self.state.df, columns=col.value, method=scr.value),
                            prev=self.state,
                        )
                        ui.notify(f"Successfully applied scaler")
                        show_table()
                    except Exception as ex:
                        ui.notify("Error occurred while applying scaler")

                apply_scaler_container.clear()
                with apply_scaler_container, ui.card().classes("w-80"), ui.scroll_area():
                    keys = list(PreprocessModule.scalers.keys())
                    with ui.row():
                        ui.button("", icon="check", on_click=on_click)
                        ui.label("Apply scaler")
                    col = ui.select([col for col in self.state.df.columns], multiple=True)
                    scr = ui.select(keys, value=keys[0], on_change=on_change)

                    desc = ui.markdown(PreprocessModule.scalers[keys[0]]["docstr"]).props("size=80")

            def apply_imputer():
                def on_change(e):
                    desc.set_content(PreprocessModule.imputers[e.value]["docstr"])
                    desc.props("size=80")

                def on_click(e):
                    if len(col.value) == 0:
                        ui.notify("No columns have been chosen")
                        return

                    try:
                        self.state = State(
                            df=PreprocessModule.impute(X=self.state.df, columns=col.value, method=imp.value),
                            prev=self.state,
                        )
                        ui.notify(f"Successfully applied imputer")
                        show_table()
                    except Exception as ex:
                        ui.notify("Error occurred while applying imputer")

                apply_imputer_container.clear()
                with apply_imputer_container, ui.card().classes("w-80"), ui.scroll_area():
                    keys = list(PreprocessModule.imputers.keys())
                    with ui.row():
                        ui.button("", icon="check", on_click=on_click)
                        ui.label("Apply imputer")
                    col = ui.select([col for col in self.state.df.columns], multiple=True)
                    imp = ui.select(keys, value=keys[0], on_change=on_change)

                    desc = ui.markdown(PreprocessModule.imputers[keys[0]]["docstr"]).props("size=80")

            def apply_pca():
                apply_pca_container.clear()
                with apply_pca_container, ui.card().classes("w-80"), ui.scroll_area():
                    with ui.row():
                        ui.button("", icon="chevron_right", on_click=lambda e: ui.navigate.to("/pca"))
                        ui.label("Perform PCA")
                    ui.markdown(PCAModule.desc).props("size=80")

            def apply_cluster():
                apply_cluster_container.clear()
                with apply_cluster_container, ui.card().classes("w-80"), ui.scroll_area():
                    with ui.row():
                        ui.button("", icon="chevron_right", on_click=lambda e: ui.navigate.to("/cluster"))
                        ui.label("Perform clustering")
                    ui.markdown(ClusterModule.desc).props("size=80")

            def refresh():
                show_table()
                change_types()
                rename_columns()
                apply_cluster()
                apply_imputer()
                apply_scaler()
                apply_pca()

            def revert():
                if self.state.prev is not None:
                    self.state = self.state.prev
                    refresh()
                    ui.notify(f"Reverted changes")
                else:
                    ui.notify(f"Nothing to revert")

            # ==========================================================================================
            # ==========================================================================================

            # Upload button
            ui.upload(on_upload=csv_file_handler, max_files=1, auto_upload=True).props("accept=.csv").classes(
                "max-w-full"
            )

            # Preprocessing cards
            with ui.row():
                rename_columns_container = ui.element()
                change_types_container = ui.element()
                apply_pca_container = ui.element()

            with ui.row():
                apply_imputer_container = ui.element()
                apply_scaler_container = ui.element()
                apply_cluster_container = ui.element()

            # Revert button
            ui.button("", on_click=revert, icon="undo")

            # Container for the table
            table_container = ui.element()

        @ui.page("/pca")
        def page_pca():
            columns: list[str] = []
            exclude: bool = False
            components: tuple[int, int] = (1, 2)

            def select_columns():
                def on_click():
                    nonlocal columns, exclude, components
                    columns = col.value
                    exclude = {"Include": False, "Exclude": True}[met.value]
                    components = (int(comp1.value), int(comp2.value))
                    if exclude or len(columns) >= 2 and components[0] < components[1] < len(columns):
                        refresh()
                    else:
                        ui.notify(f"At least 2 columns must be selected and components 1 have to be greater then components 2 and also smaller than number of selected columns")

                select_columns_container.clear()
                with select_columns_container, ui.card().classes("w-80"), ui.scroll_area():
                    with ui.row():
                        ui.button("", icon="check", on_click=on_click)
                        ui.label("Select columns")
                    met = ui.select(["Include", "Exclude"], value="Include")
                    col = ui.select([col for col in self.state.df.columns], multiple=True)
                    comp1 = ui.select([str(i) for i in range(1, len(self.state.df.columns)-1)], value="1", label="Component 1").props("style='width: 100%'")
                    comp2 = ui.select([str(i) for i in range(1, len(self.state.df.columns)-1)], value="2", label="Component 2").props("style='width: 100%'")

            def chart_xvar():
                def on_click():
                    fig.savefig(fname="graph.png")
                    ui.download(src="./graph.png")

                chart_xvar_containter.clear()
                with chart_xvar_containter, ui.row():
                    with ui.matplotlib(figsize=(10, 6)).figure as fig:
                        df = self.state.df
                        df = df.loc[:, ~df.columns.isin(columns)] if exclude else df.loc[:, df.columns.isin(columns)]
                        PCAModule.visualize_xvar(df, fig)

                        plt.title('Explained Variance by Principal Components')
                        plt.xlabel('Principal Components')
                        plt.ylabel('Variance Explained')

                    ui.button("", icon="save", on_click=on_click)

            def table_loadings():
                table_loadings_container.clear()
                with table_loadings_container:
                    df = self.state.df
                    df = df.loc[:, ~df.columns.isin(columns)] if exclude else df.loc[:, df.columns.isin(columns)]
                    ui.table.from_pandas(
                        PCAModule.loadings(df), title="Loadings table", pagination={"rowsPerPage": 10}
                    )

            def chart_loadings_hm():
                def on_click():
                    fig.savefig(fname="graph.png")
                    ui.download(src="./graph.png")

                chart_loadings_hm_container.clear()
                with chart_loadings_container, ui.row():
                    with ui.matplotlib(figsize=(10, 8)).figure as fig:
                        df = self.state.df
                        df = df.loc[:, ~df.columns.isin(columns)] if exclude else df.loc[:, df.columns.isin(columns)]
                        PCAModule.visualize_loadings_hm(df, fig)

                        plt.title('Heatmap of PCA Loadings')
                        plt.xlabel('Principal Components')
                        plt.ylabel('Features')

                    ui.button("", icon="save", on_click=on_click)

            def chart_loadings():
                def on_click():
                    fig.savefig(fname="graph.png")
                    ui.download(src="./graph.png")

                chart_loadings_container.clear()
                with chart_loadings_container, ui.row():
                    with ui.matplotlib(figsize=(10, 6)).figure as fig:
                        df = self.state.df
                        df = df.loc[:, ~df.columns.isin(columns)] if exclude else df.loc[:, df.columns.isin(columns)]
                        PCAModule.visualize_loadings(df, fig, components=components)

                        plt.title('PCA Loadings')
                        plt.xlabel(f'Principal Component {components[0]}')
                        plt.ylabel(f'Principal Component {components[1]}')
                        plt.legend(title='Features')

                    ui.button("", icon="save", on_click=on_click)

            def refresh():
                try:
                    chart_xvar()
                except Exception as e:
                    ui.notify(
                        "Error occurred while computing PCA. Make sure there are no NaNs in the selected columns and that all selected columns have numeric type."
                    )
                    print(e)
                try:
                    chart_loadings()
                except Exception as e:
                    print(e)
                try:
                    chart_loadings_hm()
                except Exception as e:
                    print(e)
                try:
                    table_loadings()
                except Exception as e:
                    print(e)

            ui.button("", icon="chevron_left", on_click=lambda e: ui.navigate.back())

            if self.state.df is None:
                ui.notify("Data has not been uploaded")
                return

            select_columns_container = ui.element()
            chart_xvar_containter = ui.element()
            chart_loadings_container = ui.element()
            chart_loadings_hm_container = ui.element()
            table_loadings_container = ui.element()

            select_columns()

        @ui.page("/cluster")
        def page_cluster():
            param_desc = None
            param_input_field = None
            kwarg = None
            model_kwargs = {name: {k: None for k in model["kwargs"].keys()} for name, model in ClusterModule.models.items()}
            columns: list[str] = []
            exclude: bool = False
            met, col = None, None

            def compute_scores():
                nonlocal columns, exclude, met, col

                columns = col.value
                exclude = {"Include": False, "Exclude": True}[met.value]

                if exclude:
                    col_list = [x for x in self.state.df.columns if x not in columns]
                else:
                    col_list = [x for x in self.state.df.columns if x in columns]

                if len(col_list) == 0:
                    ui.notify("No columns selected.")
                    return
                
                df_pass = self.state.df[col_list]

                if not all([is_numeric_dtype(df_pass[x]) for x in df_pass.columns]):

                    for col_name in df_pass.columns:
                        if not is_numeric_dtype(df_pass[col_name]):
                            ui.notify(f"Column {col_name} has type {df_pass[col_name].dtype} should be float or int.")
                    return

                hopkins_container.clear()
                with hopkins_container, ui.card().classes("w-80"), ui.scroll_area():
                    ui.label("Hopkins Statistics")

                    try:
                        ui.label(f"Score {ClusterModule.hopkins(df_pass):.2f}")
                        ui.markdown(
"""Computes Hopkins statistics H value to estimate cluster tendency of data set `X`.

It acts as a statistical hypothesis test where the null hypothesis is that the data is
generated by a Poisson point process and are thus uniformly randomly distributed. Under the
null hypothesis of spatial randomness, this statistic has a Beta(m,m) distribution and will
always lie between 0 and 1. The interpretation of H follows these guidelines:

- Low values of H indicate repulsion of the events in X away from each other.
- Values of H near 0.5 indicate spatial randomness of the events in X.
- High values of H indicate possible clustering of the events in X. Values of H>0.75
    indicate a clustering tendency at the 90% confidence level

We calculate Hopkins statistic `samples` times and then calculate the mean value of Hopkins
statistics.

For details see: https://journal.r-project.org/articles/RJ-2022-055/"""
                        )
                    
                    except:
                        ui.notify("Try imputing the data first.")

            def select_columns():
                nonlocal met, col

                select_columns_container.clear()
                with select_columns_container, ui.card().classes("w-80"), ui.scroll_area():
                    with ui.row():
                        ui.button("Scores", on_click=compute_scores)
                        ui.label("Select columns")

                    met = ui.select(["Include", "Exclude"], value="Include")
                    col = ui.select([col for col in self.state.df.columns], multiple=True)

            def algorithm_setup():
                nonlocal model_kwargs
                
                model_keys = list(ClusterModule.models.keys())
                model = ClusterModule.models[model_keys[0]]

                def create_input(model, kwarg, variable):
                    nonlocal model_kwargs

                    if isinstance(variable, str):
                        var_type, default_value = variable.split(',')
                        if var_type == 'int':
                            default_value = int(default_value)
                        elif var_type == 'float':
                            default_value = float(default_value)

                        default_value = model_kwargs[model][kwarg] if model_kwargs[model][kwarg] is not None else default_value
                        input_field = ui.input(label=f'Enter {var_type}', value=str(default_value))

                    elif isinstance(variable, list):
                        default_value =  model_kwargs[model][kwarg] if model_kwargs[model][kwarg] is not None else variable[0]
                        input_field = ui.select(variable, label='Choose an option', value=str(default_value))
                        input_field.props('style="width: 90%;"')

                    return input_field
                
                def accept_param_button_callback():
                    nonlocal model
                    nonlocal model_kwargs
                    nonlocal param_input_field

                    arg_type = model["kwargs"][kwarg]["type"]

                    if isinstance(arg_type, str):
                        var_type, _ = arg_type.split(',')
                        if var_type == 'int':
                            model_kwargs[model["name"]][kwarg] = int(param_input_field.value)
                        elif var_type == 'float':
                            model_kwargs[model["name"]][kwarg] = float(param_input_field.value)

                    else:
                        model_kwargs[model["name"]][kwarg] = param_input_field.value

                # alg params
                def setup_alg_params():
                    nonlocal model
                    nonlocal param_desc
                    nonlocal param_input_field
                    nonlocal kwarg

                    set_parameters_container.clear()
                    with set_parameters_container, ui.card().classes("w-80"), ui.scroll_area():
                        kwarg_keys = list(model["kwargs"].keys())
                        kwarg = kwarg_keys[0]

                        ui.label("Change Parameters")

                        param_selector = ui.select(kwarg_keys, value=kwarg, on_change=pick_kwarg)
                        param_desc = ui.markdown(model["kwargs"][kwarg]["docstr"]).props("size=80")

                        accept_param_button = ui.button(text="Accept Parameter", on_click=accept_param_button_callback)

                        param_type = model["kwargs"][kwarg]["type"]
                        param_input_field = create_input(model["name"], kwarg, param_type)

                def change_algorithm(e):
                    nonlocal model

                    model = ClusterModule.models[e.value]

                    model_desc.set_content(model["docstr"])
                    model_desc.props("size=80")

                    setup_alg_params()

                def pick_kwarg(e):
                    nonlocal model
                    nonlocal param_desc
                    nonlocal param_input_field
                    nonlocal kwarg
                    
                    param_desc.set_content(model["kwargs"][e.value]["docstr"])
                    param_desc.props("size=80")
                    param_type = model["kwargs"][e.value]["type"]

                    param_input_field.delete()
                    param_input_field = create_input(model["name"], e.value, param_type)

                    kwarg = e.value

                def run_clustering():
                    nonlocal model
                    nonlocal model_kwargs
                    nonlocal col, met

                    columns = col.value
                    exclude = {"Include": False, "Exclude": True}[met.value]

                    if exclude:
                        col_list = [x for x in self.state.df.columns if x not in columns]
                    else:
                        col_list = [x for x in self.state.df.columns if x in columns]

                    if len(col_list) <= 1:
                        ui.notify("Select at least 2 columns.")
                        return 
                    
                    df_pass = self.state.df[col_list]

                    kwargs = model_kwargs[model["name"]]
                    passed_kw = {k: v for k, v in kwargs.items() if v is not None}

                    try:
                        visualise_container.clear()
                        with visualise_container:

                            if not all([is_numeric_dtype(df_pass[x]) for x in df_pass.columns]):

                                for col_name in df_pass.columns:
                                    if not is_numeric_dtype(df_pass[col_name]):
                                        ui.notify(f"Column {col_name} has type {df_pass[col_name].dtype} should be float or int.")
                                return

                            labels = ClusterModule.cluster(df_pass, method=model["name"], **passed_kw)

                            with ui.row():

                                def save_clustering():
                                    fig.savefig(fname="graph.png")
                                    ui.download(src="./graph.png")

                                with ui.matplotlib(figsize=(10, 8)).figure as fig:
                                    ClusterModule.visualize(df_pass, labels, fig)

                                    plt.title('Cluster Visualization')
                                    plt.xlabel('Feature 1')
                                    plt.ylabel('Feature 2')
                                    plt.legend(title='Clusters')

                                ui.button("", icon="save", on_click=save_clustering)

                            with ui.row():
                                pick_label_containder = ui.element()
                                pick_label_containder.clear()
                                
                                with pick_label_containder, ui.card().classes("w-80"), ui.scroll_area():
                                    def update_desc_table(e):
                                        label = e.value

                                        table_containder.clear()
                                        with table_containder:
                                            description = ClusterModule.describe(df_pass, labels)[label]["stat"]
                                            description.insert(0, "Metric", description.index.tolist())

                                            ui.table.from_pandas(description, title="Descriptive Statistics", pagination={"rowsPerPage": 10})

                                    ui.label("Descriptive Statistics")
                                    picked_label = labels[0]
                                    label_selector = ui.select(list(set(labels.tolist())), value=labels[0], on_change=update_desc_table)
                                    ui.markdown("Select label of the cluster that you want to see descriptive statistics (count, mean, std, min, max, q1, q2, q3) for.")

                                table_containder = ui.element()
                                table_containder.clear()
                                with table_containder:

                                    description = ClusterModule.describe(df_pass, labels)[picked_label]["stat"]
                                    description.insert(0, "Metric", description.index.tolist())

                                    ui.table.from_pandas(description, title="Descriptive Statistics", pagination={"rowsPerPage": 10})

                            davies_container.clear()
                            with davies_container, ui.card().classes("w-80"), ui.scroll_area():
                                ui.label("Davies-Bouldin Score")

                                if len(np.unique(labels).tolist()) > 1:
                                    db_score = ClusterModule.evaluate(df_pass, labels, method='Davies-Bouldin')
                                    ui.label(f"Score: {db_score:.2f}")

                                else:
                                    ui.label(f"Cannot calculate score for one cluster.")

                                ui.markdown(ClusterModule.scores["Davies-Bouldin"]["docstr"])

                            silhouette_container.clear()
                            with silhouette_container, ui.card().classes("w-80"), ui.scroll_area():
                                ui.label("Silhouette Score")

                                if len(np.unique(labels).tolist()) > 1:
                                    sil_score = ClusterModule.evaluate(df_pass, labels, method='Silhouette')
                                    ui.label(f"Score: {sil_score:.2f}")

                                else:
                                    ui.label(f"Cannot calculate score for one cluster.")

                                ui.markdown(ClusterModule.scores["Silhouette"]["docstr"])

                            calinski_container.clear()
                            with calinski_container, ui.card().classes("w-80"), ui.scroll_area():
                                ui.label("Calinski-Harabasz Score")

                                if len(np.unique(labels).tolist()) > 1:
                                    ch_score = ClusterModule.evaluate(df_pass, labels, method='Calinski-Harabasz')
                                    ui.label(f"Score: {ch_score:.2f}")

                                else:
                                    ui.label(f"Cannot calculate score for one cluster.")

                                ui.markdown(ClusterModule.scores["Calinski-Harabasz"]["docstr"])

                            save_container.clear()
                            with save_container, ui.card().classes("w-80"), ui.scroll_area():

                                def save_file():
                                    state = self.state
                                    while state.prev is not None:
                                        state = state.prev

                                    original_df = state.df.copy()
                                    original_df["Labels"] = labels

                                    original_df.to_csv(save_name.value, index=False, sep=";", decimal=",")
                                    ui.notify("csv file has been saved")

                                ui.label("Save Clustering")
                                save_button = ui.button("SAVE", on_click=save_file)
                                save_name = ui.input(label=f'Enter filename', value="output.csv")

                                ui.markdown("")

                    except:
                        ui.notify("Couldn't cluster. Try imputing the data.")
                        return

                # alg
                choose_algorithm_container.clear()
                with choose_algorithm_container, ui.card().classes("w-80"), ui.scroll_area():

                    with ui.row():
                        ui.button("Run", on_click=run_clustering)
                        ui.label("Select algorithm")

                    alg_selector = ui.select(model_keys, value=model_keys[0], on_change=change_algorithm)
                    model_desc = ui.markdown(model["docstr"]).props("size=80")

                setup_alg_params()

            ui.button("", icon="chevron_left", on_click=lambda e: ui.navigate.back())

            if self.state.df is None:
                ui.notify("Data has not been uploaded")
                return
            
            with ui.row():
                select_columns_container = ui.element()
                choose_algorithm_container = ui.element()
                set_parameters_container = ui.element()

            with ui.row():
                hopkins_container = ui.element()

            visualise_container = ui.element()

            with ui.row():
                davies_container = ui.element()
                silhouette_container = ui.element()
                calinski_container = ui.element()

            save_container = ui.element()
            
            select_columns()
            algorithm_setup()

        page_preprocess()

    def mainloop(self):
        self.main()
        ui.run(reload=True, native=False, port=native.find_open_port(), title="App")


App().mainloop()
