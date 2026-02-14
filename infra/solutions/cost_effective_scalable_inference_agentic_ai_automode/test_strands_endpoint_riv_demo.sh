echo "START SCALE INFERENCE AND AGENTIC APPS TEST at: $(date)"
echo "---------------------"
#Health check: curl -X GET "http://k8s-default-strandsd-853813bef3-1508533455.us-east-1.elb.amazonaws.com/health"
#Embed knowledge: curl -X POST "http://k8s-default-strandsd-853813bef3-1508533455.us-east-1.elb.amazonaws.com/embed" -H "Content-Type: application/json" -d '{"force_refresh": false}'
#Complex query: curl -i -X POST "http://k8s-default-strandsd-853813bef3-1508533455.us-east-1.elb.amazonaws.com/query" -H "Content-Type: application/json" -d '{"question": "Find information about \"What was the purpose of the study on encainide and flecainide in patients with supraventricular arrhythmias\". Summarize this information and create a comprehensive story.Save the story and important information to a file named \"test1.md\" in the output directory as a beautiful markdown file.", "top_k": 3}' --max-time 600

# Get the Application Load Balancer endpoint
ALB_ENDPOINT=$(kubectl get ingress strandsdk-rag-ingress-alb -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

echo "DISCOVERED STRANDS AGENT AI ENDPOINT: ${ALB_ENDPOINT}"

# Test the health endpoint
echo "Testing  health of the Agentic endpoint: ${ALB_ENDPOINT}/health"
echo ""
curl -X GET "http://${ALB_ENDPOINT}/health"
echo "\----------"

# Test knowledge embedding
echo "Testing knowledge embedding of Agentic ENDPOINT: ${ALB_ENDPOINT}/embed.."
echo ""
curl -X POST "http://${ALB_ENDPOINT}/embed" \
  -H "Content-Type: application/json" -d '{"force_refresh": false}'
echo
echo "--------"
#-H "Content-Type: application/json" -d '{"force_refresh": false}'


# Test a simple query
echo "Testing simple query endpoint: http://${ALB_ENDPOINT}/query"
echo "question: What is the weather is Arnold, CA today?"
echo
curl -X POST "http://${ALB_ENDPOINT}/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the weather is Arnold, CA today?",
    "include_web_search": true
  }'
#-H "Content-Type: application/json" -d '{"question": "Find information about
echo
echo "---------"



# Test with a more complex medical query
echo "Test with a more complex medical query: What is the most important aspect of initial treatment for Bell's palsy?"
echo ""
#"question": "Find information about \"What was the purpose of the study on encainide and flecainide in patients with supraventricular arrhythmias\". Summarize this information and create a comprehensi$

curl -X POST "http://${ALB_ENDPOINT}/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the most important aspect of initial treatment for Bell'\''s palsy?. Summarize this information and create a comprehensive story",
    "top_k": 3
  }' \
  --max-time 600
echo
echo "---------"
echo "FINISHED SCALABLE INFERENCE AND AGENTIC AI APPS TEST at: $(date)"
