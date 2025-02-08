import csv
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from collections import defaultdict

import httpx
from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from rich.console import Console
from rich.panel import Panel

from settings import settings

console = Console()


class EvChargingGrowthResult(BaseModel):
    """Result model for EV charging growth analysis."""
    model_config = ConfigDict(frozen=True)

    metals_etf_symbol: str = Field(..., description="Symbol of the analyzed metals ETF")
    correlation: float = Field(..., description="Correlation coefficient between EV growth and metals price")
    recommendation: str = Field(..., description="Analysis-based recommendation")


# Obviously a file is a very basic persistence/caching mechanism. Feel free to use
# a database instead. I just wanted something simple that didn't require additional setup
# and could be easily synced via the CI job.
def already_fetched_today(csv_path: Path, metals_etf: str) -> bool:
    """Check if data for the specified ETF has already been fetched today."""
    today_str = date.today().isoformat()

    if not csv_path.exists():
        return False

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return any(
            row[0][:10] == today_str and row[1] == metals_etf
            for row in csv.reader(f)
        )


async def fetch_ev_charger_growth(ctx: RunContext[None], months: int) -> List[float]:
    """Fetch monthly EV charger creation counts from Open Charge Map for the UK."""
    base_url = "https://api.openchargemap.io/v3/poi"
    days_ago = months * 30
    since_date = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()

    async with httpx.AsyncClient() as client:
        response = await client.get(
            base_url,
            params={
                "key": settings.OPENCHARGEMAP_API_KEY,
                "countrycode": "GB",
                "modifiedsince": since_date,
                "maxresults": 5000,
                "compact": "true",
                "verbose": "false"
            },
            timeout=20.0
        )
        response.raise_for_status()
        data = response.json()

    monthly_counts = defaultdict(int)
    now_utc = datetime.now(timezone.utc)
    cutoff_dt = now_utc - timedelta(days=days_ago)

    for point_of_interest in data:
        if created_str := point_of_interest.get("DateCreated"):
            try:
                created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                if created_dt >= cutoff_dt:
                    month_key = created_dt.strftime("%Y-%m")
                    monthly_counts[month_key] += 1
            except ValueError:
                continue

    return [
        float(monthly_counts.get(
            (now_utc - timedelta(days=30 * (months - 1 - i))).strftime("%Y-%m"),
            0
        ))
        for i in range(months)
    ]


async def fetch_metals_prices(ctx: RunContext[None], symbol: str, months: int) -> List[float]:
    """Fetch monthly EOD prices for a metals ETF from Marketstack."""
    base_url = "https://api.marketstack.com/v2/eod"
    now_utc = datetime.now(timezone.utc)

    params = {
        "access_key": settings.MARKETSTACK_API_KEY,
        "symbols": symbol,
        "date_from": (now_utc - timedelta(days=months * 30)).strftime("%Y-%m-%d"),
        "date_to": now_utc.strftime("%Y-%m-%d"),
        "limit": 200,
        "sort": "ASC"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(base_url, params=params, timeout=20.0)
        response.raise_for_status()
        payload = response.json()

    if not (daily := payload.get("data")):
        raise ValueError(f"No data found for symbol {symbol}")

    step = max(1, len(daily) // months)
    results = [daily[i]["close"] for i in range(0, len(daily), step)][:months]

    # Pad with last value if needed
    return results + [results[-1]] * (months - len(results)) if results else []


def finalize_result(data: EvChargingGrowthResult) -> EvChargingGrowthResult:
    """Validate and finalize the analysis result."""
    return EvChargingGrowthResult(
        metals_etf_symbol=data.metals_etf_symbol,
        correlation=max(-1.0, min(1.0, data.correlation)),
        recommendation=data.recommendation
    )


def create_agent(model: OpenAIModel) -> Agent:
    """Create and configure the analysis agent."""
    agent = Agent(
        model,
        deps_type=None,
        result_type=EvChargingGrowthResult,
        system_prompt="""
        You are analyzing EV charging growth rates vs. a metals ETF's monthly prices.
        Call the provided tools to fetch real data, compute correlation, 
        then return a final JSON with metals_etf_symbol, correlation, recommendation.
        """
    )

    agent.tool(fetch_ev_charger_growth)
    agent.tool(fetch_metals_prices)
    agent.result_validator(finalize_result)

    return agent


async def run_ev_charging_growth(
        months: int = 5,
        metals_etf: str = "PPLT",
) -> Optional[EvChargingGrowthResult]:
    """
    Run EV charging growth analysis correlated with metals ETF prices.

    Args:
        months: Number of months of historical data to analyze
        metals_etf: Symbol of the metals ETF to analyze

    Returns:
        Analysis result if successful, None if already fetched today
    """
    storage_path = Path(settings.STORAGE_FILE_PATH)

    if already_fetched_today(storage_path, metals_etf):
        console.print(
            Panel(
                f"[yellow]Skipping analysis: already have data for {metals_etf} today",
                title="Analysis Status"
            )
        )
        return None

    console.print(
        Panel(
            f"[blue]Analyzing EV charging growth correlation with {metals_etf} over {months} months",
            title="Analysis Status"
        )
    )

    model = OpenAIModel("gpt-4o", api_key=settings.OPENAI_API_KEY)
    agent = create_agent(model)

    prompt = f"""
    Analyze correlation between monthly EV growth and {metals_etf} prices over {months} months:
    1) Call fetch_ev_charger_growth({months})
    2) Call fetch_metals_prices(symbol={metals_etf}, {months})
    3) Compute correlation
    4) Return JSON with (metals_etf_symbol, correlation, recommendation)
    """

    result = await agent.run(prompt)
    final_data = result.data

    console.print(
        Panel(
            f"[green]Correlation: {final_data.correlation:.3f}\n"
            f"Recommendation: {final_data.recommendation}",
            title=f"Analysis Results for {metals_etf}"
        )
    )

    storage_path.parent.mkdir(parents=True, exist_ok=True)
    with storage_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            final_data.metals_etf_symbol,
            final_data.correlation,
            final_data.recommendation
        ])

    return final_data