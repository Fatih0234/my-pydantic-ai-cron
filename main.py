import asyncio

import typer

from mycli.jobs.ev_charging_growth import run_ev_charging_growth

app = typer.Typer(help="GenAI CLI for investment investigations.")

@app.command("ev-charging-growth")
def ev_charging_growth_cli(
    months: int = 5,
    metals_etf: str = "PPLT"
):
    """
    Investigate correlation between EV charging growth data and a given metals ETF.
    """
    asyncio.run(run_ev_charging_growth(months, metals_etf))

if __name__ == "__main__":
    app()
