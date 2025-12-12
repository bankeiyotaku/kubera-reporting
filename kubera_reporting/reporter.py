"""Generate portfolio reports."""

import base64
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from kubera_reporting import currency_format, html_formatter
from kubera_reporting.allocation import calculate_asset_allocation_by_organization
from kubera_reporting.chart_generator import generate_allocation_chart
from kubera_reporting.exceptions import ReportGenerationError
from kubera_reporting.prompts import AI_SUMMARY_PROMPT_NO_AMOUNTS, AI_SUMMARY_PROMPT_WITH_AMOUNTS
from kubera_reporting.types import (
    AccountDelta,
    MoneyValue,
    PortfolioSnapshot,
    ReportData,
    ReportType,
)


class PortfolioReporter:
    """Generates portfolio reports."""

    def __init__(self) -> None:
        """Initialize the reporter with Jinja2 environment."""
        # Get the templates directory relative to this file
        templates_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(loader=FileSystemLoader(str(templates_dir)))
        self._chart_bytes: bytes | None = None
    
    def get_chart_bytes(self) -> bytes | None:
        """Get chart bytes from last generate_html_report call.
        
        Returns:
            Chart image bytes if a chart was generated, None otherwise
        """
        return self._chart_bytes

    def _aggregate_holdings_to_accounts(
        self, snapshot: PortfolioSnapshot, previous: PortfolioSnapshot | None = None,
        filter_holdings: bool = False
    ) -> PortfolioSnapshot:
        """Process accounts, optionally filtering out individual holdings.

        Raw API data contains both parent accounts (e.g., "Account Name - Bonds - 1234")
        and their individual holdings (e.g., 500 stocks with IDs like {uuid}_isin-xxx).

        Args:
            snapshot: Snapshot containing raw API data with both parents and holdings
            previous: Optional previous snapshot to check for accounts that had value
            filter_holdings: If True, filter out individual holdings and keep only parent accounts.
                            If False, keep all accounts for detailed reporting.

        Returns:
            Snapshot with processed accounts
        """
        # Build lookup of previous account values if available
        prev_account_values = {}
        if previous:
            for account in previous["accounts"]:
                prev_account_values[account["id"]] = account["value"]["amount"]

        # Filter accounts based on filter_holdings parameter
        filtered_accounts = []

        for account in snapshot["accounts"]:
            acc_id = account["id"]

            # Check if this is a holding (has underscore in ID)
            # All holdings have underscore in ID (parent_child pattern)
            # Parent accounts are pure UUIDs without underscores
            is_holding = "_" in acc_id

            # Check if account has zero value (should be skipped)
            has_zero_value = account["value"]["amount"] == 0

            # Determine if we should keep this account
            if filter_holdings:
                # Legacy mode: Keep only parent accounts with non-zero value
                # (or accounts that had previous value)
                had_previous_value = prev_account_values.get(acc_id, 0) != 0
                should_keep = not is_holding and (not has_zero_value or had_previous_value)
            else:
                # Detailed mode: Keep all accounts (parents and holdings) with non-zero value
                should_keep = not has_zero_value

            if should_keep:
                # Add is_holding flag to account for downstream processing
                account_with_flag = {**account, "is_holding": is_holding}
                filtered_accounts.append(account_with_flag)

        # Recalculate totals from ALL accounts (not just filtered parents)
        # This ensures our totals match the sum of what's actually in the portfolio
        all_assets = sum(
            acc["value"]["amount"]
            for acc in snapshot["accounts"]
            if acc["category"] == "asset" and "_" not in acc["id"]
        )
        all_debts = sum(
            acc["value"]["amount"]
            for acc in snapshot["accounts"]
            if acc["category"] == "debt" and "_" not in acc["id"]
        )

        # Return new snapshot with filtered accounts and recalculated totals
        return {
            **snapshot,
            "accounts": filtered_accounts,
            "total_assets": {"amount": all_assets, "currency": snapshot["currency"]},
            "total_debts": {"amount": all_debts, "currency": snapshot["currency"]},
            "net_worth": {"amount": all_assets - all_debts, "currency": snapshot["currency"]},
        }

    def calculate_deltas(
        self, current: PortfolioSnapshot, previous: PortfolioSnapshot | None
    ) -> ReportData:
        """Calculate changes between current and previous snapshots.

        Args:
            current: Current snapshot
            previous: Previous snapshot (may be None)

        Returns:
            Report data with calculated deltas
        """
        # Save original snapshot for allocation (needs child holdings with detailed subTypes)
        current_original = current

        # Process accounts - keep detailed holdings for comprehensive reporting
        # Pass filter_holdings=False to keep all holdings visible in the report
        if previous:
            previous = self._aggregate_holdings_to_accounts(previous, filter_holdings=False)
        current = self._aggregate_holdings_to_accounts(current, previous, filter_holdings=False)

        # Calculate net worth change and percentage
        net_worth_change: MoneyValue | None = None
        net_worth_change_percent: float | None = None
        if previous:
            net_worth_change = {
                "amount": current["net_worth"]["amount"] - previous["net_worth"]["amount"],
                "currency": current["currency"],
            }
            if previous["net_worth"]["amount"] != 0:
                net_worth_change_percent = (
                    net_worth_change["amount"] / abs(previous["net_worth"]["amount"])
                ) * 100

        # Build account lookup for previous snapshot
        prev_accounts = {}
        if previous:
            for account in previous["accounts"]:
                prev_accounts[account["id"]] = account

        # Calculate account deltas
        asset_changes: list[AccountDelta] = []
        debt_changes: list[AccountDelta] = []

        for account in current["accounts"]:
            prev_account = prev_accounts.get(account["id"])

            if prev_account:
                change_amount = account["value"]["amount"] - prev_account["value"]["amount"]
                change_percent = None
                if prev_account["value"]["amount"] != 0:
                    change_percent = (change_amount / abs(prev_account["value"]["amount"])) * 100

                delta: AccountDelta = {
                    "id": account["id"],
                    "name": account["name"],
                    "institution": account["institution"],
                    "category": account["category"],
                    "sheet_name": account["sheet_name"],
                    "section_name": account.get("section_name"),
                    "sub_type": account.get("sub_type"),
                    "asset_class": account.get("asset_class"),
                    "account_type": account.get("account_type"),
                    "geography": account.get("geography"),
                    "current_value": account["value"],
                    "previous_value": prev_account["value"],
                    "change": {"amount": change_amount, "currency": current["currency"]},
                    "change_percent": change_percent,
                    "is_holding": account.get("is_holding", False),
                }
            else:
                # New account
                delta = {
                    "id": account["id"],
                    "name": account["name"],
                    "institution": account["institution"],
                    "category": account["category"],
                    "sheet_name": account["sheet_name"],
                    "section_name": account.get("section_name"),
                    "sub_type": account.get("sub_type"),
                    "asset_class": account.get("asset_class"),
                    "account_type": account.get("account_type"),
                    "geography": account.get("geography"),
                    "current_value": account["value"],
                    "previous_value": {"amount": 0.0, "currency": current["currency"]},
                    "change": account["value"],
                    "change_percent": None,
                    "is_holding": account.get("is_holding", False),
                }

            if account["category"] == "asset":
                asset_changes.append(delta)
            else:
                debt_changes.append(delta)

        # Sort by absolute change amount (largest first)
        asset_changes.sort(key=lambda x: abs(x["change"]["amount"]), reverse=True)
        debt_changes.sort(key=lambda x: abs(x["change"]["amount"]), reverse=True)

        return {
            "current": current,
            "current_unaggregated": current_original,
            "previous": previous,
            "net_worth_change": net_worth_change,
            "net_worth_change_percent": net_worth_change_percent,
            "asset_changes": asset_changes,
            "debt_changes": debt_changes,
        }

    def generate_ai_summary(
        self,
        report_data: ReportData,
        report_type: ReportType = ReportType.DAILY,
        hide_amounts: bool = False,
    ) -> str | None:
        """Generate AI summary of portfolio changes.

        Args:
            report_data: Report data with deltas
            report_type: Type of report being generated
            hide_amounts: If True, ask LLM to avoid specific dollar amounts

        Returns:
            AI-generated summary or None if generation fails
        """
        try:
            from kubera_reporting.llm_client import LLMClient

            # Only generate summary if we have previous data
            if not report_data["previous"] or not report_data["net_worth_change"]:
                return None

            llm = LLMClient()

            # Get configurable threshold for "also notable" changes (default: $250)
            notable_threshold = float(os.getenv("KUBERA_AI_NOTABLE_THRESHOLD", "250"))

            # Format period description based on report type
            period_descriptions = {
                ReportType.DAILY: "daily",
                ReportType.WEEKLY: "weekly",
                ReportType.MONTHLY: "monthly",
                ReportType.QUARTERLY: "quarterly",
                ReportType.YEARLY: "yearly",
            }
            period = period_descriptions.get(report_type, "daily")

            # Type narrowing - we know net_worth_change is not None here
            net_worth_change = report_data["net_worth_change"]
            previous = report_data["previous"]

            # For AI summary, calculate deltas on UNAGGREGATED data to capture individual holdings
            # This allows AI to see big moves in individual stocks/crypto, not just parent accounts
            current_unagg = report_data.get("current_unaggregated", report_data["current"])

            # Build lookup for previous accounts (unaggregated)
            prev_accounts_lookup = {}
            if previous:
                for account in previous["accounts"]:
                    prev_accounts_lookup[account["id"]] = account

            # Calculate deltas for ALL accounts (including individual holdings)
            # Exclude physical assets (real estate, cars, domains) - no daily volatility
            all_asset_deltas = []
            for account in current_unagg["accounts"]:
                if account["category"] != "asset":
                    continue

                # Skip physical assets - no meaningful daily volatility
                account_type = (account.get("account_type") or "").lower()
                sub_type = (account.get("sub_type") or "").lower()
                sheet_name = (account.get("sheet_name") or "").lower()

                # Filter out real estate
                if account_type == "property":
                    continue
                if "real estate" in sheet_name:
                    continue
                if sub_type in ["primary residence", "investment property"]:
                    continue

                # Filter out vehicles
                if "vehicle" in sub_type or "car" in sub_type:
                    continue

                # Filter out domain names and other digital property
                if "domain" in sub_type or "domain" in account["name"].lower():
                    continue

                prev_account = prev_accounts_lookup.get(account["id"])
                if not prev_account:
                    continue  # Skip new accounts for AI summary (focus on changes)

                change_amount = account["value"]["amount"] - prev_account["value"]["amount"]
                if change_amount == 0:
                    continue  # Skip accounts with no change

                change_percent = None
                if prev_account["value"]["amount"] != 0:
                    change_percent = (change_amount / abs(prev_account["value"]["amount"])) * 100

                all_asset_deltas.append(
                    {
                        "name": account["name"],
                        "sheet": account["sheet_name"],
                        "current_value": account["value"]["amount"],
                        "previous_value": prev_account["value"]["amount"],
                        "change": change_amount,
                        "percent": change_percent,
                        "is_holding": "_"
                        in account["id"],  # Individual holding if ID has underscore
                    }
                )

            # Build TWO perspectives for AI:
            # 1. Top movers by dollar amount (what drove net worth change)
            # 2. Top movers by percentage (what had notable swings, even if small positions)

            # Sort by absolute change amount for net worth impact
            by_dollar = sorted(
                all_asset_deltas,
                key=lambda x: abs(float(x["change"])),  # type: ignore[arg-type]
                reverse=True,
            )

            # Sort by absolute percentage change for notable moves
            by_percent = sorted(
                [d for d in all_asset_deltas if d["percent"] is not None],
                key=lambda x: abs(float(x["percent"])),  # type: ignore[arg-type]
                reverse=True,
            )

            # Build context for AI with both perspectives
            top_dollar_movers = [
                {
                    "name": d["name"],
                    "sheet": d["sheet"],
                    "current_value": round(float(d["current_value"]), 2),  # type: ignore[arg-type]
                    "previous_value": round(float(d["previous_value"]), 2),  # type: ignore[arg-type]
                    "change": round(float(d["change"]), 2),  # type: ignore[arg-type]
                    "percent": (
                        round(float(d["percent"]), 2)  # type: ignore[arg-type]
                        if d["percent"] is not None
                        else None
                    ),
                    "is_holding": d["is_holding"],
                }
                for d in by_dollar[:3]
            ]

            # Filter percent movers by notable threshold (absolute change amount)
            notable_percent_movers = [
                d
                for d in by_percent[:10]  # Check top 10 by percent
                if abs(float(d["change"])) >= notable_threshold  # type: ignore[arg-type]
            ]

            top_percent_movers = [
                {
                    "name": d["name"],
                    "sheet": d["sheet"],
                    "current_value": round(float(d["current_value"]), 2),  # type: ignore[arg-type]
                    "change": round(float(d["change"]), 2),  # type: ignore[arg-type]
                    "percent": round(float(d["percent"]), 2),  # type: ignore[arg-type]
                    "is_holding": d["is_holding"],
                }
                for d in notable_percent_movers[:3]  # Take top 3 above threshold
            ]

            # Note: For now, debt movers use aggregated data (less common to have individual debts)
            top_debt_movers = [
                {
                    "name": d["name"],
                    "current_value": round(d["current_value"]["amount"], 2),
                    "previous_value": round(d["previous_value"]["amount"], 2),
                    "change": round(d["change"]["amount"], 2),
                    "percent": (
                        round(d["change_percent"], 2) if d["change_percent"] is not None else None
                    ),
                }
                for d in report_data["debt_changes"][:3]
            ]

            # Calculate asset allocation (use unaggregated for accurate categorization)
            snapshot_for_allocation = report_data.get(
                "current_unaggregated", report_data["current"]
            )
            allocation = calculate_asset_allocation_by_organization(snapshot_for_allocation)

            # Build comprehensive portfolio data for prompt
            # Round allocation percentages to 2 decimal places
            allocation_rounded = {k: round(v, 2) for k, v in allocation.items()}

            # Get currency from report data
            currency = report_data["current"]["currency"]
            currency_symbol = currency_format.get_currency_symbol(currency)

            if hide_amounts:
                # Omit dollar amounts, include only percentages
                portfolio_data = {
                    "currency": currency,
                    "net_worth": {
                        "change_percent": round(
                            (net_worth_change["amount"] / previous["net_worth"]["amount"] * 100)
                            if previous["net_worth"]["amount"] != 0
                            else 0,
                            2,
                        ),
                    },
                    "asset_allocation": allocation_rounded,
                    "top_dollar_movers": [
                        {
                            "name": d["name"],
                            "sheet": d["sheet"],
                            "percent": d["percent"],
                            "is_holding": d["is_holding"],
                        }
                        for d in top_dollar_movers
                    ],
                    "top_percent_movers": [
                        {
                            "name": d["name"],
                            "sheet": d["sheet"],
                            "percent": d["percent"],
                            "is_holding": d["is_holding"],
                        }
                        for d in top_percent_movers
                    ],
                    "top_debt_movers": [
                        {
                            "name": d["name"],
                            "percent": (
                                round(d["change_percent"], 2)
                                if d["change_percent"] is not None
                                else None
                            ),
                        }
                        for d in report_data["debt_changes"][:3]
                    ],
                }

                prompt = AI_SUMMARY_PROMPT_NO_AMOUNTS.format(
                    period=period,
                    currency=currency,
                    portfolio_data=json.dumps(portfolio_data, indent=2),
                )
            else:
                # Include full amounts with currency
                portfolio_data = {
                    "currency": currency,
                    "net_worth": {
                        "current": round(report_data["current"]["net_worth"]["amount"], 2),
                        "change": round(net_worth_change["amount"], 2),
                        "change_percent": round(
                            (net_worth_change["amount"] / previous["net_worth"]["amount"] * 100)
                            if previous["net_worth"]["amount"] != 0
                            else 0,
                            2,
                        ),
                    },
                    "asset_allocation": allocation_rounded,
                    "top_dollar_movers": top_dollar_movers,
                    "top_percent_movers": top_percent_movers,
                    "top_debt_movers": top_debt_movers,
                }

                prompt = AI_SUMMARY_PROMPT_WITH_AMOUNTS.format(
                    period=period,
                    currency=currency,
                    currency_symbol=currency_symbol,
                    portfolio_data=json.dumps(portfolio_data, indent=2),
                )

            # Generate summary
            from litellm import completion

            response = completion(
                model=llm.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )

            return response.choices[0].message.content

        except Exception as e:
            # If AI generation fails, just skip it
            print(f"Warning: AI summary generation failed: {e}")
            return None

    def generate_html_report(
        self,
        report_data: ReportData,
        report_type: ReportType = ReportType.DAILY,
        top_n: int = 20,
        ai_summary: str | None = None,
        recipient_name: str | None = None,
        hide_amounts: bool = False,
        is_export: bool = False,
    ) -> str:
        """Generate HTML email report with embedded charts.

        Args:
            report_data: Report data
            report_type: Type of report (daily, weekly, monthly, etc.)
            top_n: Number of top movers to show (default: 20)
            ai_summary: Optional AI-generated summary
            recipient_name: Optional name for greeting (default: "Portfolio Report")
            hide_amounts: If True, mask dollar amounts (show "$XX" instead)
            is_export: If True, add collapse/expand functionality and use base64 for charts

        Returns:
            HTML report string

        Raises:
            ReportGenerationError: If generation fails
        """
        try:
            # Group assets by sheet > section (two-level hierarchy)
            # First level: sheet_name
            assets_by_sheet: dict[str, dict[str, list[AccountDelta]]] = defaultdict(
                lambda: defaultdict(list)
            )

            for asset in report_data["asset_changes"]:
                sheet = asset["sheet_name"] or "Uncategorized"
                # Use section_name for sub-grouping, fall back to "Default" if not available
                section = asset.get("section_name") or "Default"

                assets_by_sheet[sheet][section].append(asset)

            # Sort accounts within each section
            # Group parent accounts first, then holdings alphabetically
            for sheet_sections in assets_by_sheet.values():
                for section_accounts in sheet_sections.values():
                    section_accounts.sort(
                        key=lambda x: (
                            x.get("is_holding", False),  # Parents first (False < True)
                            -x["current_value"]["amount"]  # Then by value descending
                        )
                    )

            # Calculate totals for both sheet and section levels
            sheet_totals: dict[str, dict[str, float | int | None]] = {}
            section_totals: dict[str, dict[str, dict[str, float | int | None]]] = defaultdict(dict)

            for sheet_name, sheet_sections in assets_by_sheet.items():
                sheet_current_total = 0.0
                sheet_change_total = 0.0 if report_data["previous"] else None
                sheet_previous_total = 0.0
                sheet_account_count = 0

                for section_name, accounts in sheet_sections.items():
                    # Calculate section-level totals
                    # Only sum parent accounts to avoid double-counting (holdings are children of parents)
                    parent_accounts = [a for a in accounts if not a.get("is_holding", False)]
                    section_current = sum(a["current_value"]["amount"] for a in parent_accounts)
                    section_change = (
                        sum(a["change"]["amount"] for a in parent_accounts)
                        if report_data["previous"]
                        else None
                    )
                    section_previous = sum(a["previous_value"]["amount"] for a in parent_accounts)

                    # Calculate section percentage change
                    section_change_percent = None
                    if (
                        report_data["previous"]
                        and section_change is not None
                        and section_previous != 0
                    ):
                        section_change_percent = (section_change / section_previous) * 100

                    section_totals[sheet_name][section_name] = {
                        "count": len(accounts),
                        "total_value": section_current,
                        "total_change": section_change,
                        "change_percent": section_change_percent,
                    }

                    # Accumulate for sheet totals
                    sheet_current_total += section_current
                    if sheet_change_total is not None and section_change is not None:
                        sheet_change_total += section_change
                    sheet_previous_total += section_previous
                    sheet_account_count += len(accounts)

                # Calculate sheet-level percentage change
                sheet_change_percent = None
                if (
                    report_data["previous"]
                    and sheet_change_total is not None
                    and sheet_previous_total != 0
                ):
                    sheet_change_percent = (sheet_change_total / sheet_previous_total) * 100

                sheet_totals[sheet_name] = {
                    "count": sheet_account_count,
                    "total_value": sheet_current_total,
                    "total_change": sheet_change_total,
                    "change_percent": sheet_change_percent,
                }

            # Keep top movers for backwards compatibility
            asset_movers = report_data["asset_changes"][:top_n]
            debt_movers = report_data["debt_changes"][:top_n]

            # Validate that sum of sheet totals matches sum of displayed PARENT accounts
            # (excluding holdings to avoid double-counting)
            sum_of_sheet_totals = sum(
                sheet["total_value"]
                for sheet in sheet_totals.values()
                if isinstance(sheet["total_value"], (int, float))
            )
            # Only sum parent accounts for validation
            sum_of_displayed_assets = sum(
                asset["current_value"]["amount"] 
                for asset in report_data["asset_changes"]
                if not asset.get("is_holding", False)
            )
            sum_of_displayed_debts = sum(
                debt["current_value"]["amount"] 
                for debt in report_data["debt_changes"]
                if not debt.get("is_holding", False)
            )

            # Allow small floating point differences (< $0.01)
            asset_diff = abs(sum_of_sheet_totals - sum_of_displayed_assets)
            if asset_diff > 0.01:
                raise ReportGenerationError(
                    f"Report validation failed: Sum of sheet totals (${sum_of_sheet_totals:,.2f}) "
                    f"does not match sum of displayed parent assets (${sum_of_displayed_assets:,.2f}). "
                    f"Difference: ${asset_diff:,.2f}. This indicates accounts are being "
                    f"incorrectly filtered from the report."
                )

            # Use API snapshot totals for headline numbers (not filtered account sums)
            # The API totals are the source of truth and include all holdings
            current_total_assets = report_data["current"]["total_assets"]
            current_total_debts = report_data["current"]["total_debts"]
            current_net_worth = report_data["current"]["net_worth"]

            # Validate that displayed accounts match the API snapshot totals
            # (They should match since we filter in _aggregate_holdings_to_accounts)
            api_asset_diff = abs(current_total_assets["amount"] - sum_of_displayed_assets)
            api_debt_diff = abs(current_total_debts["amount"] - sum_of_displayed_debts)

            if api_asset_diff > 0.01:
                raise ReportGenerationError(
                    f"Report validation failed: API snapshot total assets "
                    f"(${current_total_assets['amount']:,.2f}) does not match sum of "
                    f"displayed assets (${sum_of_displayed_assets:,.2f}). "
                    f"Difference: ${api_asset_diff:,.2f}"
                )

            if api_debt_diff > 0.01:
                raise ReportGenerationError(
                    f"Report validation failed: API snapshot total debts "
                    f"(${current_total_debts['amount']:,.2f}) does not match sum of "
                    f"displayed debts (${sum_of_displayed_debts:,.2f}). "
                    f"Difference: ${api_debt_diff:,.2f}"
                )

            # Calculate changes from API snapshot totals
            total_asset_change_percent = None
            total_debt_change_percent = None
            net_worth_change_percent = None

            if report_data["previous"]:
                total_asset_change = (
                    report_data["current"]["total_assets"]["amount"]
                    - report_data["previous"]["total_assets"]["amount"]
                )
                total_debt_change = (
                    report_data["current"]["total_debts"]["amount"]
                    - report_data["previous"]["total_debts"]["amount"]
                )
                net_worth_change_amount = (
                    report_data["current"]["net_worth"]["amount"]
                    - report_data["previous"]["net_worth"]["amount"]
                )

                # Calculate percentages
                if report_data["previous"]["total_assets"]["amount"] != 0:
                    total_asset_change_percent = (
                        total_asset_change / report_data["previous"]["total_assets"]["amount"]
                    ) * 100
                if report_data["previous"]["total_debts"]["amount"] != 0:
                    total_debt_change_percent = (
                        total_debt_change / report_data["previous"]["total_debts"]["amount"]
                    ) * 100
                if report_data["previous"]["net_worth"]["amount"] != 0:
                    net_worth_change_percent = (
                        net_worth_change_amount
                        / abs(report_data["previous"]["net_worth"]["amount"])
                    ) * 100
            else:
                total_asset_change = 0
                total_debt_change = 0
                net_worth_change_amount = 0

            # Calculate asset allocation (use unaggregated for accurate categorization)
            snapshot_for_allocation = report_data.get(
                "current_unaggregated", report_data["current"]
            )
            allocation = calculate_asset_allocation_by_organization(snapshot_for_allocation)

            # Generate chart and embed it
            chart_src = ""
            self._chart_bytes = None  # Store for later retrieval
            if allocation:
                chart_bytes = generate_allocation_chart(allocation)
                self._chart_bytes = chart_bytes
                
                # For export: use base64 data URI (works in browsers)
                # For email: use CID reference (more reliable in email clients)
                if is_export:
                    chart_b64 = base64.b64encode(chart_bytes).decode("utf-8")
                    chart_src = f"data:image/png;base64,{chart_b64}"
                else:
                    # Use CID for email
                    chart_src = "cid:chart_image"

            # Format greeting based on recipient name and report type
            if recipient_name:
                name_part = f"Hi {recipient_name},"
            else:
                name_part = "Hi,"

            if report_data["previous"]:
                # With comparison data - mention the period
                if report_type == ReportType.DAILY:
                    period_desc = "here's a recap of account balances that changed yesterday"
                elif report_type == ReportType.WEEKLY:
                    period_desc = "here's a recap of changes over the past week"
                elif report_type == ReportType.MONTHLY:
                    period_desc = "here's a recap of changes over the past month"
                elif report_type == ReportType.QUARTERLY:
                    period_desc = "here's a recap of changes over the past quarter"
                elif report_type == ReportType.YEARLY:
                    period_desc = "here's a recap of changes over the past year"
                else:
                    period_desc = "here's your portfolio report"
                greeting = f"{name_part} {period_desc}."
            else:
                # No comparison - just show snapshot
                greeting = f"{name_part} here's a snapshot of your current account balances."

            # Sort sheets by total value (descending)
            sorted_sheets = dict(
                sorted(
                    assets_by_sheet.items(),
                    key=lambda x: sheet_totals[x[0]]["total_value"] or 0,
                    reverse=True,
                )
            )

            # Format report type for display
            report_type_display = report_type.value.capitalize()

            # Create wrapper functions that respect hide_amounts flag
            def format_money_wrapper(value: MoneyValue) -> str:
                return html_formatter.format_money(value, hide_amounts=hide_amounts)

            def format_net_worth_wrapper(value: MoneyValue) -> str:
                return html_formatter.format_net_worth(value, hide_amounts=hide_amounts)

            def format_change_wrapper(
                change: MoneyValue, change_percent: float | None = None
            ) -> tuple[str, str]:
                return html_formatter.format_change(
                    change, change_percent, hide_amounts=hide_amounts
                )

            # Convert AI summary newlines to HTML <br> tags for email compatibility
            # Email clients don't reliably support white-space: pre-line
            ai_summary_html = ai_summary.replace("\n", "<br>") if ai_summary else None

            template = self.jinja_env.get_template("report_template.html")
            return template.render(
                current=report_data["current"],
                previous=report_data["previous"],
                current_net_worth=current_net_worth,
                net_worth_change={
                    "amount": net_worth_change_amount,
                    "currency": report_data["current"]["currency"],
                }
                if report_data["previous"]
                else None,
                net_worth_change_percent=net_worth_change_percent,
                asset_movers=asset_movers,
                debt_movers=debt_movers,
                assets_by_sheet=sorted_sheets,
                sheet_totals=sheet_totals,
                section_totals=section_totals,
                current_total_assets=current_total_assets,
                current_total_debts=current_total_debts,
                total_asset_change=total_asset_change,
                total_asset_change_percent=total_asset_change_percent,
                total_debt_change=total_debt_change,
                total_debt_change_percent=total_debt_change_percent,
                allocation=allocation,
                chart_src=chart_src,
                ai_summary=ai_summary_html,
                greeting=greeting,
                report_date=datetime.now().strftime("%b %d, %Y"),
                report_type=report_type_display,
                format_money=format_money_wrapper,
                format_net_worth=format_net_worth_wrapper,
                format_change=format_change_wrapper,
                is_export=is_export,
            )
            
            return html
            
        except Exception as e:
            raise ReportGenerationError(f"Failed to generate HTML report: {e}") from e
