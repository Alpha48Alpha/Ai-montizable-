#!/usr/bin/env python3
"""
Monetization Engine — AI-Monetizable Revenue & Strategy Calculator
===================================================================
Provides revenue estimates and monetization strategies for:
  - Movie / animation productions  (ad revenue, subscriptions, licensing,
                                    merchandise, crowdfunding)
  - Product marketing packages      (margin analysis, affiliate revenue,
                                    upsell / bundles, platform recommendations)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Revenue constants
# ---------------------------------------------------------------------------

# YouTube CPM rates (USD per 1,000 views) — conservative mid-range estimates
YOUTUBE_CPM: dict[str, float] = {
    "anime":   3.50,
    "sci-fi":  4.00,
    "comedy":  3.00,
    "drama":   3.50,
    "action":  3.75,
    "default": 3.00,
}

# Patreon net revenue share after platform fees (~12 %)
PATREON_REVENUE_SHARE = 0.88

# Typical content licensing one-time fees (USD)
LICENSING_RATES: dict[str, int] = {
    "music_sync_indie":      250,
    "music_sync_commercial": 1_500,
    "character_ip_small":    500,
    "character_ip_medium":   2_500,
    "story_option":          1_000,
}

# Affiliate commission rates by shoe category
AFFILIATE_RATES: dict[str, float] = {
    "running":   0.08,   # 8 %
    "lifestyle": 0.07,   # 7 %
    "hiking":    0.09,   # 9 %
    "default":   0.07,
}

# Estimated cost-of-goods ratio by shoe type (fraction of retail price)
COGS_RATIO: dict[str, float] = {
    "running shoe":              0.35,
    "lifestyle / casual shoe":   0.40,
    "hiking boot":               0.38,
    "default":                   0.40,
}

# Type-specific ad copy hooks for shoe marketing
SHOE_COPY_HOOKS: dict[str, dict[str, str]] = {
    "running shoe": {
        "hook":    "engineered for runners who refuse to slow down",
        "benefit": "every stride feels effortless",
    },
    "lifestyle / casual shoe": {
        "hook":    "designed for everyday comfort without compromise",
        "benefit": "you'll never want to take them off",
    },
    "hiking boot": {
        "hook":    "built for adventurers who push beyond the trail",
        "benefit": "every summit is within reach",
    },
    "default": {
        "hook":    "crafted for those who demand the best",
        "benefit": "every step is a statement",
    },
}


# ===========================================================================
# Movie / animation monetization
# ===========================================================================

def estimate_movie_ad_revenue(views: int, genre: str = "default") -> float:
    """Return estimated ad revenue (USD) for a YouTube video given view count."""
    genre_key = genre.lower().split("/")[0].strip()
    cpm = YOUTUBE_CPM.get(genre_key, YOUTUBE_CPM["default"])
    return round((views / 1_000) * cpm, 2)


def estimate_movie_subscription_revenue(
    subscribers: int, price_per_month: float = 5.00
) -> float:
    """Return estimated monthly net revenue (USD) from a Patreon subscription."""
    return round(subscribers * price_per_month * PATREON_REVENUE_SHARE, 2)


def generate_movie_monetization_plan(movie: dict) -> dict:
    """Build a full monetization plan dictionary for a movie / animation project."""
    genre = movie.get("genre", "default")
    title = movie["title"]
    genre_key = genre.lower().split("/")[0].strip()
    cpm = YOUTUBE_CPM.get(genre_key, YOUTUBE_CPM["default"])

    return {
        "title": title,
        "ad_revenue": {
            "platform":             "YouTube (ad-supported)",
            "model":                "CPM advertising",
            "estimated_cpm_usd":    cpm,
            "revenue_at_10k_views": estimate_movie_ad_revenue(10_000, genre),
            "revenue_at_100k_views": estimate_movie_ad_revenue(100_000, genre),
            "revenue_at_1m_views":  estimate_movie_ad_revenue(1_000_000, genre),
        },
        "subscription": {
            "platform":               "Patreon",
            "model":                  "Monthly subscriber fee ($5/month)",
            "monthly_at_100_subs":    estimate_movie_subscription_revenue(100),
            "monthly_at_500_subs":    estimate_movie_subscription_revenue(500),
            "monthly_at_1000_subs":   estimate_movie_subscription_revenue(1_000),
        },
        "licensing": {
            "model":                    "IP / sync licensing (one-time fees)",
            "music_sync_indie_usd":     LICENSING_RATES["music_sync_indie"],
            "music_sync_commercial_usd": LICENSING_RATES["music_sync_commercial"],
            "character_ip_small_usd":   LICENSING_RATES["character_ip_small"],
            "character_ip_medium_usd":  LICENSING_RATES["character_ip_medium"],
            "story_option_usd":         LICENSING_RATES["story_option"],
        },
        "merchandise": {
            "recommended_items": [
                f"{title} — character art prints",
                f"{title} — enamel pin set",
                f"{title} — digital art book (PDF)",
                f"{title} — OST download",
            ],
            "platform_suggestions": ["Redbubble", "Printful + Shopify", "Gumroad"],
        },
        "crowdfunding": {
            "recommended_platforms": ["Kickstarter", "Indiegogo"],
            "target_range_usd": "$5,000 – $25,000",
            "suggested_tiers": [
                {
                    "name":       "Supporter",
                    "price_usd":  5,
                    "reward":     "Digital wallpaper pack",
                },
                {
                    "name":       "Fan",
                    "price_usd":  25,
                    "reward":     "HD download + name in credits",
                },
                {
                    "name":       "Producer",
                    "price_usd":  100,
                    "reward":     "Executive producer credit + digital art book",
                },
            ],
        },
    }


def format_movie_monetization_report(plan: dict) -> str:
    """Return a formatted multi-line string of the movie monetization plan."""
    ad = plan["ad_revenue"]
    sub = plan["subscription"]
    lic = plan["licensing"]
    merch = plan["merchandise"]
    crowd = plan["crowdfunding"]

    lines = [
        "\n  ── Monetization Plan ──",
        f"  Project: {plan['title']}",
        "",
        "  Ad Revenue Estimates (YouTube):",
        f"    CPM rate        : ${ad['estimated_cpm_usd']:.2f}",
        f"    At   10,000 views: ${ad['revenue_at_10k_views']:>8.2f}",
        f"    At  100,000 views: ${ad['revenue_at_100k_views']:>8.2f}",
        f"    At 1,000,000 views: ${ad['revenue_at_1m_views']:>8.2f}",
        "",
        "  Subscription Revenue (Patreon, $5/month, net after fees):",
        f"    100 subscribers : ${sub['monthly_at_100_subs']:>8.2f}/month",
        f"    500 subscribers : ${sub['monthly_at_500_subs']:>8.2f}/month",
        f"  1,000 subscribers : ${sub['monthly_at_1000_subs']:>8.2f}/month",
        "",
        "  Licensing Revenue (per deal):",
        f"    Music sync — indie     : ${lic['music_sync_indie_usd']:,}",
        f"    Music sync — commercial: ${lic['music_sync_commercial_usd']:,}",
        f"    Character IP — small   : ${lic['character_ip_small_usd']:,}",
        f"    Character IP — medium  : ${lic['character_ip_medium_usd']:,}",
        f"    Story option           : ${lic['story_option_usd']:,}",
        "",
        "  Merchandise Opportunities:",
    ]
    for item in merch["recommended_items"]:
        lines.append(f"    • {item}")
    lines.append(f"    Platforms: {', '.join(merch['platform_suggestions'])}")

    lines += [
        "",
        "  Crowdfunding:",
        f"    Platforms: {', '.join(crowd['recommended_platforms'])}",
        f"    Target   : {crowd['target_range_usd']}",
        "    Tiers:",
    ]
    for tier in crowd["suggested_tiers"]:
        lines.append(
            f"      ${tier['price_usd']:>4}  {tier['name']:<14} — {tier['reward']}"
        )

    return "\n".join(lines)


# ===========================================================================
# Product / shoe monetization
# ===========================================================================

def _normalize_shoe_type(shoe: dict) -> str:
    """Return the lower-cased shoe type string for lookup purposes."""
    return shoe["type"].lower()


def estimate_product_margin(shoe: dict) -> dict:
    """Return a margin breakdown dictionary for a shoe product."""
    price = shoe["price_usd"]
    cogs_ratio = COGS_RATIO.get(_normalize_shoe_type(shoe), COGS_RATIO["default"])
    estimated_cogs = round(price * cogs_ratio, 2)
    gross_margin = round(price - estimated_cogs, 2)
    margin_pct = round((gross_margin / price) * 100, 1)
    return {
        "retail_price_usd":   price,
        "estimated_cogs_usd": estimated_cogs,
        "gross_margin_usd":   gross_margin,
        "gross_margin_pct":   margin_pct,
    }


def estimate_affiliate_revenue(
    shoe: dict,
    monthly_clicks: int = 1_000,
    conversion_rate: float = 0.02,
) -> dict:
    """Return estimated affiliate commission figures for a shoe product."""
    rate_key = _normalize_shoe_type(shoe).split("/")[0].strip()
    commission_rate = AFFILIATE_RATES.get(rate_key, AFFILIATE_RATES["default"])
    monthly_sales = int(monthly_clicks * conversion_rate)
    monthly_commission = round(monthly_sales * shoe["price_usd"] * commission_rate, 2)
    return {
        "monthly_clicks":               monthly_clicks,
        "conversion_rate_pct":          round(conversion_rate * 100, 1),
        "estimated_monthly_sales":      monthly_sales,
        "commission_rate_pct":          round(commission_rate * 100, 1),
        "monthly_commission_usd":       monthly_commission,
        "annual_commission_usd":        round(monthly_commission * 12, 2),
    }


def generate_product_monetization_plan(shoe: dict) -> dict:
    """Build a full monetization plan dictionary for a shoe product."""
    price = shoe["price_usd"]
    return {
        "name":                   shoe["name"],
        "margin_analysis":        estimate_product_margin(shoe),
        "affiliate":              estimate_affiliate_revenue(shoe),
        "upsell_recommendations": [
            f"Premium insoles                      +${price * 0.12:.2f}",
            f"Matching socks pack (3-pair)          +${price * 0.08:.2f}",
            f"Shoe care & cleaning kit              +${price * 0.10:.2f}",
        ],
        "bundle_deal": {
            "description":    "2-pair bundle — same model, different colour",
            "bundle_price_usd": round(price * 1.80, 2),
            "saving_usd":     round(price * 0.20, 2),
        },
        "subscription": {
            "name":        "Annual Shoe Club",
            "price_usd":   round(price * 0.85 * 4, 2),
            "includes":    "4 pairs/year, priority new-release access, free shipping",
        },
        "platform_recommendations": [
            "Shopify storefront — full margin retained, owned customer data",
            "Amazon (FBM) — broad reach, established trust",
            "Instagram / TikTok Shop — social commerce, impulse buys",
            "Running / outdoor specialty retailers — offline distribution",
        ],
    }


def format_product_monetization_report(plan: dict) -> str:
    """Return a formatted multi-line string of the product monetization plan."""
    margin = plan["margin_analysis"]
    aff    = plan["affiliate"]
    bundle = plan["bundle_deal"]
    sub    = plan["subscription"]

    lines = [
        "\n💹  MONETIZATION ANALYSIS",
        "",
        "  Margin Analysis:",
        f"    Retail price   : ${margin['retail_price_usd']:.2f}",
        f"    Est. COGS      : ${margin['estimated_cogs_usd']:.2f}",
        f"    Gross margin   : ${margin['gross_margin_usd']:.2f}  "
        f"({margin['gross_margin_pct']}%)",
        "",
        "  Affiliate Revenue Estimate (1,000 clicks/month, 2 % conversion):",
        f"    Commission rate      : {aff['commission_rate_pct']:.1f}%",
        f"    Est. monthly sales   : {aff['estimated_monthly_sales']} units",
        f"    Monthly commission   : ${aff['monthly_commission_usd']:.2f}",
        f"    Annual commission    : ${aff['annual_commission_usd']:.2f}",
        "",
        "  Upsell Recommendations:",
    ]
    for item in plan["upsell_recommendations"]:
        lines.append(f"    • {item}")

    lines += [
        "",
        "  Bundle Deal:",
        f"    {bundle['description']}",
        f"    Bundle price: ${bundle['bundle_price_usd']:.2f}  "
        f"(save ${bundle['saving_usd']:.2f})",
        "",
        f"  Subscription — {sub['name']}:",
        f"    ${sub['price_usd']:.2f}/year",
        f"    Includes: {sub['includes']}",
        "",
        "  Platform Recommendations:",
    ]
    for platform in plan["platform_recommendations"]:
        lines.append(f"    • {platform}")

    return "\n".join(lines)


# ===========================================================================
# Shoe marketing copy helpers (used by shoe_demo.py)
# ===========================================================================

def get_shoe_copy_hooks(shoe_type: str) -> dict[str, str]:
    """Return type-specific marketing copy hooks for a shoe."""
    return SHOE_COPY_HOOKS.get(shoe_type.lower(), SHOE_COPY_HOOKS["default"])
