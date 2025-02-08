import pytest
from mycli.jobs.ev_charging_growth import (
    fetch_ev_charger_growth,
    fetch_metals_prices,
)


@pytest.mark.asyncio
async def test_fetch_ev_charger_growth():
    """Test the tool call as a normal sync function."""
    # You can pass a dummy 'ctx' if needed; or None if your code doesn't rely on ctx
    fake_ctx = None  
    months = 5
    growth_rates = await fetch_ev_charger_growth(fake_ctx, months=months)

    assert isinstance(growth_rates, list)
    assert len(growth_rates) == 5
    for rate in growth_rates:
        assert isinstance(rate, float)
        assert 0 <= rate <= 200.0


@pytest.mark.asyncio
async def test_fetch_metals_prices():
    """Test the tool call as a normal sync function."""
    fake_ctx = None
    prices = await fetch_metals_prices(fake_ctx, symbol="PPLT", months=3)

    assert isinstance(prices, list)
    assert len(prices) == 3
    for price in prices:
        assert isinstance(price, float)
        assert 50.0 <= price <= 1000.0
