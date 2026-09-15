# GPU Capacity Scanner (`aoe-capacity`)

Answers "where in the world can I get these GPUs right now, and for how much?"
using only public AWS APIs: `get_spot_placement_scores`,
`describe_spot_price_history`, `describe_instance_type_offerings`, and pricing
`get_products`.

## Install

    pip install -e .

## Use

    aoe-capacity find g6e.12xlarge --regions all
    aoe-capacity find p5.48xlarge --regions us-east-2,us-west-2 --json

Output is ranked by effective price ascending (spot when available, else on-demand).
`--regions all` (the default) scans every enabled region in the account.

## Library

    from tools.capacity.scan import scan
    scan(["g6e.12xlarge"], regions=None)   # -> list[CapacityOption] sorted by price
