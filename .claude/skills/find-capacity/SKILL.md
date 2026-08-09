---
name: find-capacity
description: Use when the user asks where GPU capacity is available or cheapest (e.g. "where can I get 8xH100 cheapest?", "is g6e.12xlarge available in us-west-2?"). Wraps the aoe-capacity scanner.
---

# Find GPU Capacity

Answer "where in the world can I get these GPUs, and for how much?" using public AWS
APIs only.

## Steps

1.  **Map the request to an instance type.** If the user names a GPU count/type
    ("8xH100"), translate to the EC2 instance type (8xH100 -> `p5.48xlarge`;
    1xH100 -> `p5.4xlarge`; L40S -> `g6e.*`). If they already gave an instance type, use
    it directly.

2.  **Scan.** For all regions:

        aoe-capacity find <instance_type> --regions all

    For specific regions:

        aoe-capacity find <instance_type> --regions us-east-2,us-west-2

3.  **Report the ranking.** Present the top rows (region, spot $/hr, on-demand $/hr, spot
    placement score, availability), cheapest first. Call out the single best option and
    whether it is spot or on-demand. Add `--json` if the user wants machine-readable
    output.
