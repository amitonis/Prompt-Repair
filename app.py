import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    # PromptRepair

    Targeted prompt correction for structured text extraction.

    Status: scaffold. This app is a thin front end over `src/`; the console
    pipeline (evaluator, defect injection, patch guard, runner) is built
    first, per the methodology guide. Wire the workspace, development
    comparison, evaluation dashboard, and live demo panels to `src/` on
    Days 7-12.
    """
    )
    return


if __name__ == "__main__":
    app.run()
