# routes.py
from flask import render_template, request
from datetime import datetime
from .prediction.pipeline import run_prediction_pipeline

def register_routes(app):

    @app.route("/", methods=["GET", "POST"])
    def index():
        selected_date = app.config["DEFAULT_DATE"]
        pred_dfs, results_df = None, None

        if request.method == "POST":
            selected_date_str = request.form.get("game_date")
            selected_date = datetime.strptime(selected_date_str, "%Y-%m-%d").date()

        cache = app.config["CACHE"]
        if selected_date in cache:
            pred_dfs, results_df = cache[selected_date]
        else:
            pred_dfs, results_df = run_prediction_pipeline(
                app.config["MODELS"],
                selected_date,
                app.config["CONFIG"]
            )
            cache[selected_date] = (pred_dfs, results_df)

        for df in pred_dfs or []:
            df["predictions_html"] = df["predictions"].to_html(
                index=False, classes="table table-sm table-striped align-middle text-center"
            )

        results_df_html = (
            results_df.to_html(
                index=False,
                classes="table table-sm table-bordered align-middle text-center"
            )
            if results_df is not None else None
        )

        return render_template(
            "index.html",
            min_date=app.config["MIN_DATE"],
            max_date=app.config["MAX_DATE"],
            pred_dfs=pred_dfs,
            results_df_html=results_df_html,
            selected_date=selected_date
        )
