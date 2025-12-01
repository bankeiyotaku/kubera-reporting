"""Tests for reporter module."""

from datetime import datetime, timezone

import pytest

from kubera_reporting.reporter import PortfolioReporter
from kubera_reporting.types import PortfolioSnapshot


@pytest.fixture
def current_snapshot() -> PortfolioSnapshot:
    """Create current snapshot for testing."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "portfolio_id": "test-123",
        "portfolio_name": "Test Portfolio",
        "currency": "USD",
        "net_worth": {"amount": 105000.0, "currency": "USD"},
        "total_assets": {"amount": 155000.0, "currency": "USD"},
        "total_debts": {"amount": 50000.0, "currency": "USD"},
        "accounts": [
            {
                "id": "account-1",
                "name": "Stocks",
                "institution": "Broker",
                "value": {"amount": 15000.0, "currency": "USD"},
                "category": "asset",
                "sheet_name": "Investments",
            },
            {
                "id": "account-2",
                "name": "Savings",
                "institution": "Bank",
                "value": {"amount": 5000.0, "currency": "USD"},
                "category": "asset",
                "sheet_name": "Cash",
            },
        ],
    }


@pytest.fixture
def previous_snapshot() -> PortfolioSnapshot:
    """Create previous snapshot for testing."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "portfolio_id": "test-123",
        "portfolio_name": "Test Portfolio",
        "currency": "USD",
        "net_worth": {"amount": 100000.0, "currency": "USD"},
        "total_assets": {"amount": 150000.0, "currency": "USD"},
        "total_debts": {"amount": 50000.0, "currency": "USD"},
        "accounts": [
            {
                "id": "account-1",
                "name": "Stocks",
                "institution": "Broker",
                "value": {"amount": 10000.0, "currency": "USD"},
                "category": "asset",
                "sheet_name": "Investments",
            },
            {
                "id": "account-2",
                "name": "Savings",
                "institution": "Bank",
                "value": {"amount": 5000.0, "currency": "USD"},
                "category": "asset",
                "sheet_name": "Cash",
            },
        ],
    }


def test_calculate_deltas(current_snapshot, previous_snapshot):
    """Test calculating deltas between snapshots."""
    reporter = PortfolioReporter()
    report_data = reporter.calculate_deltas(current_snapshot, previous_snapshot)

    # Check net worth change
    assert report_data["net_worth_change"] is not None
    assert report_data["net_worth_change"]["amount"] == 5000.0

    # Check asset changes
    assert len(report_data["asset_changes"]) == 2

    # Find the stocks account change
    stocks_change = next((c for c in report_data["asset_changes"] if c["name"] == "Stocks"), None)
    assert stocks_change is not None
    assert stocks_change["change"]["amount"] == 5000.0
    assert stocks_change["change_percent"] == 50.0

    # Find the savings account (no change)
    savings_change = next((c for c in report_data["asset_changes"] if c["name"] == "Savings"), None)
    assert savings_change is not None
    assert savings_change["change"]["amount"] == 0.0


def test_calculate_deltas_no_previous(current_snapshot):
    """Test calculating deltas with no previous snapshot."""
    reporter = PortfolioReporter()
    report_data = reporter.calculate_deltas(current_snapshot, None)

    assert report_data["net_worth_change"] is None
    assert len(report_data["asset_changes"]) == 2


def test_generate_html_report(current_snapshot, previous_snapshot):
    """Test HTML report generation."""
    reporter = PortfolioReporter()
    report_data = reporter.calculate_deltas(current_snapshot, previous_snapshot)

    html = reporter.generate_html_report(report_data)

    assert "<!DOCTYPE html>" in html
    assert "Net worth" in html
    # Net worth is now recalculated from parent accounts, not API total
    assert "$20,000" in html  # Sum of Stocks ($15k) + Savings ($5k)
    assert "Stocks" in html
    assert "Assets" in html


def test_account_rollover_shows_in_deltas():
    """Test that accounts going to zero (e.g., rollovers) show in deltas.

    This tests the fix for issue where 401k was rolled over to IRA:
    - Previous: 401k had $100k, IRA had $0
    - Current: 401k has $0, IRA has $105k (with $5k market gain)
    - Report should show both: 401k -$100k and IRA +$105k
    """
    reporter = PortfolioReporter()

    # Previous snapshot: 401k has balance, IRA is zero
    previous: PortfolioSnapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "portfolio_id": "test-123",
        "portfolio_name": "Test Portfolio",
        "currency": "USD",
        "net_worth": {"amount": 100000.0, "currency": "USD"},
        "total_assets": {"amount": 100000.0, "currency": "USD"},
        "total_debts": {"amount": 0.0, "currency": "USD"},
        "accounts": [
            {
                "id": "401k-account",
                "name": "401k",
                "institution": "Fidelity",
                "value": {"amount": 100000.0, "currency": "USD"},
                "category": "asset",
                "sheet_name": "Retirement",
            },
            {
                "id": "ira-account",
                "name": "Rollover IRA",
                "institution": "Fidelity",
                "value": {"amount": 0.0, "currency": "USD"},
                "category": "asset",
                "sheet_name": "Retirement",
            },
        ],
    }

    # Current snapshot: 401k is zero (rolled over), IRA has balance
    current: PortfolioSnapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "portfolio_id": "test-123",
        "portfolio_name": "Test Portfolio",
        "currency": "USD",
        "net_worth": {"amount": 105000.0, "currency": "USD"},
        "total_assets": {"amount": 105000.0, "currency": "USD"},
        "total_debts": {"amount": 0.0, "currency": "USD"},
        "accounts": [
            {
                "id": "401k-account",
                "name": "401k",
                "institution": "Fidelity",
                "value": {"amount": 0.0, "currency": "USD"},
                "category": "asset",
                "sheet_name": "Retirement",
            },
            {
                "id": "ira-account",
                "name": "Rollover IRA",
                "institution": "Fidelity",
                "value": {"amount": 105000.0, "currency": "USD"},
                "category": "asset",
                "sheet_name": "Retirement",
            },
        ],
    }

    report_data = reporter.calculate_deltas(current, previous)

    # Both accounts should appear in deltas
    assert len(report_data["asset_changes"]) == 2

    # Find the 401k change (should show -$100k decrease)
    change_401k = next((c for c in report_data["asset_changes"] if c["id"] == "401k-account"), None)
    assert change_401k is not None, "401k account should appear in deltas even though it's now zero"
    assert change_401k["previous_value"]["amount"] == 100000.0
    assert change_401k["current_value"]["amount"] == 0.0
    assert change_401k["change"]["amount"] == -100000.0
    assert change_401k["change_percent"] == -100.0

    # Find the IRA change (should show +$105k increase)
    change_ira = next((c for c in report_data["asset_changes"] if c["id"] == "ira-account"), None)
    assert change_ira is not None
    assert change_ira["previous_value"]["amount"] == 0.0
    assert change_ira["current_value"]["amount"] == 105000.0
    assert change_ira["change"]["amount"] == 105000.0

    # Net worth change should be +$5k (the actual market gain)
    assert report_data["net_worth_change"]["amount"] == 5000.0
