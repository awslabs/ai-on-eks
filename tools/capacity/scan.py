# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-region GPU capacity scanner using PUBLIC AWS APIs only.

Public APIs: get_spot_placement_scores, describe_spot_price_history,
describe_instance_type_offerings, pricing get_products.
"""
from __future__ import annotations

import json

import boto3

PRICING_REGION = "us-east-1"


def _default_factory(service: str, region: str):
    return boto3.client(service, region_name=region)


def _enabled_regions(client_factory) -> list[str]:
    ec2 = client_factory("ec2", PRICING_REGION)
    resp = ec2.describe_regions(AllRegions=False)
    return sorted(r["RegionName"] for r in resp["Regions"])


def _placement_scores(client_factory, instance_types, regions) -> dict[str, int]:
    ec2 = client_factory("ec2", PRICING_REGION)
    resp = ec2.get_spot_placement_scores(
        InstanceTypes=instance_types,
        TargetCapacity=1,
        RegionNames=regions,
        SingleAvailabilityZone=False,
    )
    return {s["Region"]: s["Score"] for s in resp.get("SpotPlacementScores", [])}


def _is_offered(client_factory, region, instance_type) -> bool:
    ec2 = client_factory("ec2", region)
    resp = ec2.describe_instance_type_offerings(
        LocationType="region",
        Filters=[{"Name": "instance-type", "Values": [instance_type]}],
    )
    return bool(resp.get("InstanceTypeOfferings"))


def _spot_price(client_factory, region, instance_type) -> float | None:
    ec2 = client_factory("ec2", region)
    resp = ec2.describe_spot_price_history(
        InstanceTypes=[instance_type],
        ProductDescriptions=["Linux/UNIX"],
        MaxResults=1,
    )
    history = resp.get("SpotPriceHistory", [])
    if not history:
        return None
    return float(history[0]["SpotPrice"])


def _on_demand_price(client_factory, region, instance_type) -> float | None:
    pricing = client_factory("pricing", PRICING_REGION)
    resp = pricing.get_products(
        ServiceCode="AmazonEC2",
        Filters=[
            {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
            {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
            {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
        ],
        MaxResults=1,
    )
    price_list = resp.get("PriceList", [])
    if not price_list:
        return None
    doc = json.loads(price_list[0])
    for sku in doc.get("terms", {}).get("OnDemand", {}).values():
        for term in sku.values():
            for dim in term.get("priceDimensions", {}).values():
                usd = dim.get("pricePerUnit", {}).get("USD")
                if usd is not None:
                    return round(float(usd), 4)
    return None


def _effective_price(option: dict) -> float:
    if option["spot_price_usd_hr"] is not None:
        return option["spot_price_usd_hr"]
    if option["on_demand_price_usd_hr"] is not None:
        return option["on_demand_price_usd_hr"]
    return float("inf")


def scan(
    instance_types: list[str],
    regions: list[str] | None = None,
    client_factory=None,
) -> list[dict]:
    """regions=None -> all enabled regions. Returns list sorted by price asc."""
    client_factory = client_factory or _default_factory
    if regions is None:
        regions = _enabled_regions(client_factory)

    scores = _placement_scores(client_factory, instance_types, regions)

    options: list[dict] = []
    for region in regions:
        for instance_type in instance_types:
            if not _is_offered(client_factory, region, instance_type):
                continue
            spot = _spot_price(client_factory, region, instance_type)
            on_demand = _on_demand_price(client_factory, region, instance_type)
            if spot is not None:
                availability = "spot"
            elif on_demand is not None:
                availability = "on-demand"
            else:
                availability = "none"
            options.append({
                "region": region,
                "instance_type": instance_type,
                "spot_price_usd_hr": spot,
                "on_demand_price_usd_hr": on_demand,
                "spot_placement_score": scores.get(region),
                "availability": availability,
            })

    options.sort(key=_effective_price)
    return options
