"""
NOTE: THIS CODE IS UNSAFE GARBAGE BUT CODING GUIs IS GARBAGE SO IT FITS
"""

# # macOS packaging support
# from multiprocessing import freeze_support  # noqa

# freeze_support()  # noqa

import numpy
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
                    self.state.df = pd.read_csv(StringIO(content))
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
                        self.state = State(df=self.state.df.astype({col.value: typ.value}), prev=self.state)
                        ui.notify(f"Successfully converted '{col.value}' into type '{typ.value}'")
                        show_table()
                    except Exception as e:
                        ui.notify(f"Error: {e}")

                change_types_container.clear()
                with change_types_container, ui.card().classes("w-80"), ui.scroll_area():
                    with ui.row():
                        ui.button("", icon="check", on_click=on_click)
                        ui.label("Change dtype")

                    col = ui.select([col for col in self.state.df.columns], value=self.state.df.columns[0])
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

                rename_columns_container.clear()
                with rename_columns_container, ui.card().classes("w-80"), ui.scroll_area():
                    ui.label("Rename columns")
                    col = ui.select([col for col in self.state.df.columns], value=self.state.df.columns[0])
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

            def select_columns():
                def on_click():
                    nonlocal columns, exclude
                    columns = col.value
                    exclude = {"Include": False, "Exclude": True}[met.value]
                    if exclude or len(columns) >= 2:
                        refresh()
                    else:
                        ui.notify(f"At least 2 columns must be selected")

                select_columns_container.clear()
                with select_columns_container, ui.card().classes("w-80"), ui.scroll_area():
                    with ui.row():
                        ui.button("", icon="check", on_click=on_click)
                        ui.label("Select columns")
                    met = ui.select(["Include", "Exclude"], value="Include")
                    col = ui.select([col for col in self.state.df.columns], multiple=True)

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
                        PCAModule.visualize_loadings(df, fig, components=(0, 1))

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
            columns: list[str] = []
            exclude: bool = False

            def select_columns():
                select_columns_container.clear()
                with select_columns_container, ui.card().classes("w-80"), ui.scroll_area():
                    ui.label("Select columns")
                    met = ui.select(["Include", "Exclude"], value="Include")
                    col = ui.select([col for col in self.state.df.columns], multiple=True)

            def choose_algorithm():
                def on_change(e):
                    desc.set_content(ClusterModule.models[e.value]["docstr"])
                    desc.props("size=80")

                choose_algorithm_container.clear()
                with choose_algorithm_container, ui.card().classes("w-80"), ui.scroll_area():
                    keys = list(ClusterModule.models.keys())
                    with ui.row():
                        ui.button("", icon="check")
                        ui.label("Select algorithm")

                    alg = ui.select(keys, value=keys[0], on_change=on_change)
                    desc = ui.markdown(ClusterModule.models[keys[0]]["docstr"]).props("size=80")

            def refresh():
                pass

            ui.button("", icon="chevron_left", on_click=lambda e: ui.navigate.back())

            if self.state.df is None:
                ui.notify("Data has not been uploaded")
                return

            with ui.row():
                select_columns_container = ui.element()
                choose_algorithm_container = ui.element()

            select_columns()
            choose_algorithm()

        page_preprocess()

    def mainloop(self):
        self.main()
        ui.run(reload=True, native=False, port=native.find_open_port(), title="App")


App().mainloop()
