import pandas as pd


from io import StringIO
from typing import List
from dataclasses import dataclass

from nicegui import ui, events, native
from engine.preprocess import PreprocessModule


@dataclass
class State:
    df: pd.DataFrame
    prev: List["State"]


class App:
    def __init__(self):
        self.state = State(None, None)

    def page_preprocess(self):
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
            def on_change(col: str, t: str):
                try:
                    self.state = State(df=self.state.df.astype({col: t}), prev=self.state)
                    ui.notify(f"Successfully converted '{col}' into type '{t}'")
                    show_table()
                except Exception as e:
                    ui.notify(f"Could not convert '{col}' into type '{t}'")

            change_types_container.clear()
            with change_types_container, ui.card().classes("w-80"), ui.scroll_area():
                ui.label("Change dtype")
                ui.markdown("Simply choose a column and appropriate dtype to change it.")
                col = ui.select([col for col in self.state.df.columns], value=self.state.df.columns[0])
                ui.select(["int", "float", "bool", "object"], on_change=lambda t: on_change(col.value, t.value))

        def rename_columns():
            def on_enter(e, col: str, new: str):
                try:
                    self.state = State(self.state.df.rename(columns={col: new}), prev=self.state)
                    ui.notify(f"Successfully renamed '{col}' into '{new}'")
                    refresh()
                except Exception as ex:
                    ui.notify(f"Could not rename '{col}' into '{new}'")

            rename_columns_container.clear()
            with rename_columns_container, ui.card().classes("w-80"), ui.scroll_area():
                ui.label("Rename columns")
                ui.markdown("Simpy choose the column, input the new name and press ENTER.")
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
                    ui.button("", icon="check", on_click=lambda e: ui.navigate.to("/pca"))

                    ui.label("Perform PCA")
                ui.markdown(
                    """
Principal component analysis (PCA).

Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a
lower dimensional space. The input data is centered but not scaled for each feature before applying
the SVD."""
                ).props("size=80")

        def refresh():
            # Clear previous table content and display new table
            show_table()

            # Clear datatypes dropdown menu and display new one
            change_types()

            # Clear rename columns field and display new one
            rename_columns()

            # Clear imputer dropdown menu and display new on
            apply_imputer()

            # Clear scaler dropdown menu and display new one
            apply_scaler()

            # Clear
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
        ui.upload(on_upload=csv_file_handler, max_files=1, auto_upload=True).props("accept=.csv").classes("max-w-full")

        # Preprocessing cards
        with ui.row():
            change_types_container = ui.element()

            rename_columns_container = ui.element()

        with ui.row():
            apply_imputer_container = ui.element()

            apply_scaler_container = ui.element()

            apply_pca_container = ui.element()

        # Revert button
        ui.button("", on_click=revert, icon="undo")

        # Container for the table
        table_container = ui.element()

    @ui.page("/pca")
    def page_pca():
        ui.button("", icon="arrow_back_ios", on_click=lambda e: ui.navigate.back())

    def mainloop(self):
        self.page_preprocess()
        ui.run(reload=True, port=native.find_open_port(), title="App")


App().mainloop()
