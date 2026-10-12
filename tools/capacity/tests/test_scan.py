# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import boto3
from botocore.stub import Stubber

from tools.capacity.scan import scan


def _stubbed_factory():
    """Return (factory, stubbers) with canned responses for one region scan."""
    clients: dict[tuple[str, str], object] = {}
    stubbers: dict[tuple[str, str], Stubber] = {}

    def make(service, region):
        key = (service, region)
        if key in clients:
            return clients[key]
        client = boto3.client(service, region_name=region)
        stub = Stubber(client)
        clients[key] = client
        stubbers[key] = stub
        return client

    return make, clients, stubbers


def test_scan_single_region_ranks_and_maps_fields():
    factory, clients, stubbers = _stubbed_factory()

    # 1) placement scores queried from us-east-1 across the given regions
    ec2_use1 = factory("ec2", "us-east-1")
    stubbers[("ec2", "us-east-1")].add_response(
        "get_spot_placement_scores",
        {"SpotPlacementScores": [{"Region": "us-east-2", "Score": 7}]},
        {
            "InstanceTypes": ["g6e.12xlarge"],
            "TargetCapacity": 1,
            "RegionNames": ["us-east-2"],
            "SingleAvailabilityZone": False,
        },
    )

    # 2) offering check in us-east-2
    ec2_use2 = factory("ec2", "us-east-2")
    stubbers[("ec2", "us-east-2")].add_response(
        "describe_instance_type_offerings",
        {"InstanceTypeOfferings": [
            {"InstanceType": "g6e.12xlarge", "LocationType": "region", "Location": "us-east-2"}
        ]},
        {"LocationType": "region",
         "Filters": [{"Name": "instance-type", "Values": ["g6e.12xlarge"]}]},
    )
    # 3) spot price history in us-east-2
    stubbers[("ec2", "us-east-2")].add_response(
        "describe_spot_price_history",
        {"SpotPriceHistory": [
            {"InstanceType": "g6e.12xlarge", "SpotPrice": "4.21",
             "ProductDescription": "Linux/UNIX", "AvailabilityZone": "us-east-2a"}
        ]},
        {"InstanceTypes": ["g6e.12xlarge"],
         "ProductDescriptions": ["Linux/UNIX"],
         "MaxResults": 1},
    )

    # 4) on-demand price from pricing (us-east-1)
    price_doc = {
        "terms": {"OnDemand": {"sku.jr": {"sku.jr.term": {
            "priceDimensions": {"sku.jr.term.dim": {
                "pricePerUnit": {"USD": "10.4900000000"}}}}}}}
    }
    factory("pricing", "us-east-1")
    stubbers[("pricing", "us-east-1")].add_response(
        "get_products",
        {"PriceList": [json.dumps(price_doc)]},
        {"ServiceCode": "AmazonEC2",
         "Filters": [
             {"Type": "TERM_MATCH", "Field": "instanceType", "Value": "g6e.12xlarge"},
             {"Type": "TERM_MATCH", "Field": "regionCode", "Value": "us-east-2"},
             {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
             {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
             {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
             {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
         ],
         "MaxResults": 1},
    )

    for stub in stubbers.values():
        stub.activate()

    result = scan(["g6e.12xlarge"], regions=["us-east-2"], client_factory=factory)

    assert result == [{
        "region": "us-east-2",
        "instance_type": "g6e.12xlarge",
        "spot_price_usd_hr": 4.21,
        "on_demand_price_usd_hr": 10.49,
        "spot_placement_score": 7,
        "availability": "spot",
    }]


def test_scan_marks_none_when_not_offered():
    factory, clients, stubbers = _stubbed_factory()
    stubbers[("ec2", "us-east-1")] = None  # placeholder; created below

    ec2_use1 = factory("ec2", "us-east-1")
    stubbers[("ec2", "us-east-1")].add_response(
        "get_spot_placement_scores",
        {"SpotPlacementScores": []},
        {"InstanceTypes": ["p5.48xlarge"], "TargetCapacity": 1,
         "RegionNames": ["eu-west-1"], "SingleAvailabilityZone": False},
    )
    ec2_euw1 = factory("ec2", "eu-west-1")
    stubbers[("ec2", "eu-west-1")].add_response(
        "describe_instance_type_offerings",
        {"InstanceTypeOfferings": []},
        {"LocationType": "region",
         "Filters": [{"Name": "instance-type", "Values": ["p5.48xlarge"]}]},
    )
    for stub in stubbers.values():
        if stub is not None:
            stub.activate()

    result = scan(["p5.48xlarge"], regions=["eu-west-1"], client_factory=factory)
    assert result == []
