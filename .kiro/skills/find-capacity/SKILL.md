---
name: find-capacity
description: Find where GPU capacity is available or cheapest across AWS regions. Use when the user asks "where can I get 8xH100 cheapest?" or "is g6e.12xlarge available in us-west-2?". Wraps the aoe-capacity scanner (public AWS APIs only).
---

# Find GPU Capacity

When the user asks where GPUs are available or cheapest:

1. Map the request to an EC2 instance type (8xH100 -> p5.48xlarge, L40S -> g6e.\*).
2. `aoe-capacity find <instance_type> --regions all` (or a comma list of regions).
3. Report the ranking cheapest-first (region, spot $/hr, on-demand $/hr, placement
   score, availability) and name the single best option. Public AWS APIs only.
