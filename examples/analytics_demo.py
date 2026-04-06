#!/usr/bin/env python3
"""Analytics Demo - Showcase Wauldo SDK analytics capabilities.

This demo uses MockHttpClient to demonstrate the analytics features
without requiring a running server or API key.

Usage:
    python examples/analytics_demo.py

Output:
    Formatted analytics report with ROI insights, cache performance,
    and tenant traffic summary.
"""

from wauldo import MockHttpClient


def format_number(n: int) -> str:
    """Format number with thousand separators."""
    return f"{n:,}"


def format_percent(p: float) -> str:
    """Format percentage with 1 decimal place."""
    return f"{p:.1f}%"


def format_usd(amount: float) -> str:
    """Format USD amount with 2 decimal places."""
    return f"${amount:.2f}"


def print_roi_insights(client: MockHttpClient) -> None:
    """Print ROI insights section."""
    insights = client.get_insights()

    print("=" * 50)
    print("📊 ROI INSIGHTS")
    print("=" * 50)
    print(f"Total requests:       {format_number(insights.total_requests)}")
    print(f"Intelligence calls:   {format_number(insights.intelligence_requests)}")
    print(f"Fallback calls:       {format_number(insights.fallback_requests)}")
    print()
    print("💰 Cost Savings")
    print(f"  Tokens saved:       {format_number(insights.tokens.saved_total)}")
    print(f"  Baseline tokens:    {format_number(insights.tokens.baseline_total)}")
    print(f"  Real tokens:        {format_number(insights.tokens.real_total)}")
    print(f"  Avg savings:        {format_percent(insights.tokens.saved_percent_avg)}")
    if insights.tokens.saved_percent_min and insights.tokens.saved_percent_max:
        print(f"  Savings range:      {format_percent(insights.tokens.saved_percent_min)} - {format_percent(insights.tokens.saved_percent_max)}")
    print(f"  Est. savings:       {format_usd(insights.cost.estimated_usd_saved)}")
    print()


def print_cache_performance(client: MockHttpClient) -> None:
    """Print cache performance section."""
    analytics = client.get_analytics(minutes=60)

    print("=" * 50)
    print("⚡ CACHE PERFORMANCE (Last 60 minutes)")
    print("=" * 50)
    print(f"Total requests:       {format_number(analytics.cache.total_requests)}")
    print(f"Cache hit rate:       {format_percent(analytics.cache.cache_hit_rate * 100)}")
    print(f"Avg latency:          {analytics.cache.avg_latency_ms:.0f}ms")
    print(f"P95 latency:          {analytics.cache.p95_latency_ms:.0f}ms")
    print()
    print("🎯 Token Efficiency")
    print(f"  Baseline:           {format_number(analytics.tokens.total_baseline)}")
    print(f"  Real usage:         {format_number(analytics.tokens.total_real)}")
    print(f"  Saved:              {format_number(analytics.tokens.total_saved)}")
    print(f"  Efficiency:         {format_percent(analytics.tokens.avg_savings_percent)}")
    print(f"  Uptime:             {analytics.uptime_secs // 60} minutes")
    print()


def print_traffic_summary(client: MockHttpClient) -> None:
    """Print traffic summary section."""
    traffic = client.get_analytics_traffic()

    print("=" * 50)
    print("🌐 TRAFFIC SUMMARY (Today)")
    print("=" * 50)
    print(f"Total requests:       {format_number(traffic.total_requests_today)}")
    print(f"Total tokens:         {format_number(traffic.total_tokens_today)}")
    print(f"Error rate:           {format_percent(traffic.error_rate * 100)}")
    print(f"Avg latency:          {traffic.avg_latency_ms}ms")
    print(f"P95 latency:          {traffic.p95_latency_ms}ms")
    print(f"Uptime:               {traffic.uptime_secs // 3600} hours")
    print()
    print("🏆 Top Tenants")
    print("-" * 50)
    for i, tenant in enumerate(traffic.top_tenants, 1):
        print(f"  {i}. {tenant.tenant_id}")
        print(f"     Requests: {format_number(tenant.requests_today):>8} | "
              f"Tokens: {format_number(tenant.tokens_used):>10} | "
              f"Success: {format_percent(tenant.success_rate * 100)}")
    print()


def main() -> None:
    """Run the analytics demo."""
    print("\n" + "=" * 50)
    print("🚀 Wauldo SDK Analytics Demo")
    print("=" * 50)
    print("Using MockHttpClient - no server needed!\n")

    # Create mock client with default settings
    client = MockHttpClient()

    # Print all analytics sections
    print_roi_insights(client)
    print_cache_performance(client)
    print_traffic_summary(client)

    print("=" * 50)
    print("✅ Demo complete!")
    print("=" * 50)
    print("\nTo use real analytics, switch to HttpClient:")
    print('  client = HttpClient(base_url="http://localhost:3000")')
    print()


if __name__ == "__main__":
    main()
