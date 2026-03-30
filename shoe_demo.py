#!/usr/bin/env python3
"""
Shoe Demo — AI-Monetizable Content Generator
=============================================
Generates a complete AI-powered shoe product package:
  - Product description
  - Marketing copy (taglines, ad copy)
  - Social-media captions
  - SEO keywords
  - Pricing strategy recommendation

Run:
    python shoe_demo.py
"""

import json
import textwrap

from monetization import (
    generate_product_monetization_plan,
    format_product_monetization_report,
    get_shoe_copy_hooks,
)

# ---------------------------------------------------------------------------
# Shoe data
# ---------------------------------------------------------------------------
SHOES = [
    {
        "name": "AeroStride X1",
        "type": "Running Shoe",
        "material": "recycled mesh upper, carbon-fibre plate",
        "colours": ["Midnight Black", "Solar Orange", "Ice White"],
        "price_usd": 149.99,
        "features": [
            "Energy-return foam midsole",
            "Carbon-fibre propulsion plate",
            "Recycled-mesh upper (30 % post-consumer material)",
            "Reflective heel tab for night runs",
            "True-to-size fit with wide-toe box",
        ],
    },
    {
        "name": "UrbanWalker Pro",
        "type": "Lifestyle / Casual Shoe",
        "material": "full-grain leather upper, rubber outsole",
        "colours": ["Caramel Brown", "Slate Grey", "Cream"],
        "price_usd": 119.99,
        "features": [
            "Cushioned memory-foam insole",
            "Slip-resistant rubber outsole",
            "Full-grain leather upper — ages beautifully",
            "Padded collar for all-day comfort",
            "Available in half sizes",
        ],
    },
    {
        "name": "PeakClimb 3000",
        "type": "Hiking Boot",
        "material": "waterproof nubuck leather, Vibram outsole",
        "colours": ["Forest Green", "Desert Tan", "Obsidian"],
        "price_usd": 199.99,
        "features": [
            "Waterproof nubuck leather with sealed seams",
            "Vibram Megagrip outsole for technical terrain",
            "Ankle-support shank",
            "Removable ortholite footbed",
            "GORE-TEX lining",
        ],
    },
]

# ---------------------------------------------------------------------------
# Content-generation helpers
# ---------------------------------------------------------------------------

def generate_product_description(shoe: dict) -> str:
    features_block = "\n".join(f"  • {f}" for f in shoe["features"])
    colours = ", ".join(shoe["colours"])
    lines = [
        f"{shoe['name']} — {shoe['type']}",
        "-----------------------------------------------",
        f"Built from {shoe['material']}, the {shoe['name']} delivers",
        "performance and style in equal measure.",
        "",
        "Key features:",
        features_block,
        "",
        f"Available colours: {colours}",
        f"Price: ${shoe['price_usd']:.2f}",
    ]
    return "\n".join(lines) + "\n"


def generate_marketing_copy(shoe: dict) -> dict:
    name = shoe["name"]
    shoe_type = shoe["type"]
    shoe_type_lower = shoe_type.lower()
    top_feature = shoe["features"][0]
    colour = shoe["colours"][0]
    hooks = get_shoe_copy_hooks(shoe_type_lower)

    return {
        "tagline": f"{name} — {hooks['hook'].capitalize()}.",
        "ad_copy": (
            f"Introducing the {name} — the ultimate {shoe_type_lower} "
            f"{hooks['hook']}. "
            f"Featuring {top_feature.lower()}, {hooks['benefit']}. "
            f"Now available in {colour} and more."
        ),
        "call_to_action": f"Shop {name} today at ${shoe['price_usd']:.2f} →",
    }


def generate_social_captions(shoe: dict) -> list:
    name = shoe["name"]
    colour = shoe["colours"][0]
    feature = shoe["features"][0].lower()
    price = shoe["price_usd"]

    return [
        f"👟 Meet the {name}. {feature.capitalize()}. All day, every day. #NewRelease #Footwear",
        f"🔥 Fresh drop alert! {name} in {colour} — limited stock. Grab yours for ${price:.2f}. #ShoeOfTheDay",
        f"💡 Why settle for ordinary? The {name} redefines what a {shoe['type'].lower()} can be. #Innovation #Style",
    ]


def generate_seo_keywords(shoe: dict) -> list:
    base = [
        shoe["name"].lower(),
        shoe["type"].lower(),
        shoe["material"].split(",")[0].strip().lower(),
    ]
    extra = [c.lower() for c in shoe["colours"]]
    generic = ["buy shoes online", "best shoes 2025", "premium footwear"]
    return list(dict.fromkeys(base + extra + generic))  # deduplicated, order-preserved


def generate_pricing_strategy(shoe: dict) -> str:
    price = shoe["price_usd"]
    if price < 100:
        tier, rationale = "Value", "competitive entry-level positioning"
    elif price < 160:
        tier, rationale = "Mid-range", "strong quality-to-price ratio"
    else:
        tier, rationale = "Premium", "aspirational branding with margin headroom"

    return (
        f"Pricing tier : {tier}\n"
        f"Rationale    : ${price:.2f} supports {rationale}.\n"
        f"Suggested promo : 10 % launch discount → ${price * 0.90:.2f}"
    )


# ---------------------------------------------------------------------------
# Package builder
# ---------------------------------------------------------------------------

def build_shoe_package(shoe: dict) -> dict:
    marketing = generate_marketing_copy(shoe)
    return {
        "product_description": generate_product_description(shoe),
        "tagline": marketing["tagline"],
        "ad_copy": marketing["ad_copy"],
        "call_to_action": marketing["call_to_action"],
        "social_captions": generate_social_captions(shoe),
        "seo_keywords": generate_seo_keywords(shoe),
        "pricing_strategy": generate_pricing_strategy(shoe),
        "monetization_plan": generate_product_monetization_plan(shoe),
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 60


def print_package(shoe: dict) -> None:
    pkg = build_shoe_package(shoe)

    print(SEPARATOR)
    print(f"  SHOE PRODUCT PACKAGE — {shoe['name'].upper()}")
    print(SEPARATOR)

    print("\n📝  PRODUCT DESCRIPTION")
    print(pkg["product_description"])

    print("📣  MARKETING COPY")
    print(f"  Tagline : {pkg['tagline']}")
    print(f"  Ad copy : {pkg['ad_copy']}")
    print(f"  CTA     : {pkg['call_to_action']}")

    print("\n📱  SOCIAL-MEDIA CAPTIONS")
    for i, caption in enumerate(pkg["social_captions"], 1):
        print(f"  [{i}] {caption}")

    print("\n🔍  SEO KEYWORDS")
    print("  " + " | ".join(pkg["seo_keywords"]))

    print("\n💰  PRICING STRATEGY")
    for line in pkg["pricing_strategy"].splitlines():
        print(f"  {line}")

    print(format_product_monetization_report(pkg["monetization_plan"]))

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n🤖  AI-MONETIZABLE SHOE DEMO  🤖")
    print("Generating full product packages for 3 shoe models …\n")

    for shoe in SHOES:
        print_package(shoe)

    # Optionally export one package as JSON
    sample_pkg = build_shoe_package(SHOES[0])
    output_path = "shoe_demo_output.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump({"shoe": SHOES[0]["name"], "package": sample_pkg}, fh, indent=2)

    print(f"✅  Demo complete. Sample package exported to '{output_path}'.")


if __name__ == "__main__":
    main()
