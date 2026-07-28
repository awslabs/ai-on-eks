---
title: GPU Capacity Scanner
sidebar_label: Capacity Scanner
---

# GPU Capacity Scanner

GPU capacity is scarce and priced differently in every region. The `aoe-capacity`
tool scans all enabled regions and returns a ranked list of where a given instance
type is available and what it costs, using only public AWS APIs.

## Usage

    aoe-capacity find g6e.12xlarge --regions all
    aoe-capacity find p5.48xlarge --regions us-east-2,us-west-2 --json

Each row reports the region, spot price, on-demand price, the spot placement score
(1-10, higher is better), and availability (`spot`, `on-demand`, or `none`). Results
are sorted by effective price ascending so the cheapest viable option is first.

## APIs used

`get_spot_placement_scores`, `describe_spot_price_history`,
`describe_instance_type_offerings`, and pricing `get_products`. No internal or
account-privileged APIs are used, so the tool runs anywhere with standard EC2 and
Pricing read permissions.
