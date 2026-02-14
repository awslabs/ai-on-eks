#!/bin/bash


REGION=ap-southeast-2
CLUSTER_NAME=llm-eks-cluster
ALB_NAME=llama-cpp-cpu-lb
export AWS_DEFAULT_REGION=$REGION

CLUSTER_SG=$(aws eks describe-cluster \
    --name $CLUSTER_NAME \
    --query 'cluster.resourcesVpcConfig.clusterSecurityGroupId' \
    --output text)

ALB_SG=$(aws elbv2 describe-load-balancers \
    --names $ALB_NAME \
    --query 'LoadBalancers[0].SecurityGroups[0]' \
    --output text)    

aws ec2 authorize-security-group-ingress \
    --group-id $CLUSTER_SG \
    --source-group $ALB_SG \
    --protocol tcp \
    --port-range 0-65535