import boto3  
from datetime import datetime
from collections import defaultdict

# IAM Role needs below permissions:
# {
#     "Version": "2012-10-17",
#     "Statement": [
#         {
#             "Effect": "Allow",
#             "Action": [
#                 "ec2:DescribeRegions",
#                 "ec2:DescribeInstanceTypeOfferings",
#                 "ec2:DescribeInstanceTypes",
#                 "ec2:DescribeAvailabilityZones",
#                 "ssm:GetParameter"
#             ],
#             "Resource": "*"
#         }
#     ]
# }

instance_types = ["g6e.xlarge", "g5.xlarge"] # Edit this line to change the instance types displayed  
print(f"checking for instance_types: {instance_types}")
regions = [region["RegionName"] for region in boto3.client("ec2").describe_regions()["Regions"]]
supported_regions = defaultdict(list)
  
for region in regions:  
   ec2_region = boto3.client("ec2", region_name=region)  
   response = ec2_region.describe_instance_type_offerings(  
      # LocationType="availability-zone",  
      Filters=[  
        {
          "Name": "instance-type", 
          "Values": instance_types
        } 
      ]  
   )
   if response["InstanceTypeOfferings"]:  
      supported_regions[region] = [offer["InstanceType"] for offer in response["InstanceTypeOfferings"]]  

print("# Supported Regions as of",datetime.now().strftime("%B %d, %Y"))
print("================")  


client = boto3.client("ssm")

for region, instance_types in supported_regions.items():
    try:
        response = client.get_parameter(
          Name=f"/aws/service/global-infrastructure/regions/{region}/longName"
        )
        region_long_name = response["Parameter"]["Value"]
    except (client.exceptions.ParameterNotFound, KeyError):
        region_long_name = region
    print(f"* {region}: {region_long_name}")
    for instance_type in instance_types:
      print(f"  - {instance_type}")
    print("\n")