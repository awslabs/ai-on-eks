# Model Download Job
resource "kubectl_manifest" "mistral_model_download" {
  depends_on = [
    module.eks,
    kubectl_manifest.s3_sync_service_account,
    aws_s3_bucket.model_storage
  ]

  yaml_body = <<-YAML
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: model-download
      namespace: default
    spec:
      backoffLimit: 3
      activeDeadlineSeconds: 7200
      template:
        spec:
          restartPolicy: Never
          serviceAccountName: model-storage-sa
          containers:
          - name: downloader
            image: python:3.11-slim
            command: ["/bin/bash", "-c"]
            args:
            - |
              set -e
              pip install -q huggingface_hub[cli] boto3

              echo "Downloading Mistral-7B-Instruct-v0.3 from HuggingFace..."
              hf download mistralai/Mistral-7B-Instruct-v0.3 --local-dir /tmp/mistral-7b

              echo "Uploading to S3 bucket: ${aws_s3_bucket.model_storage.bucket}"
              python3 << 'EOF'
              import boto3
              import os
              from pathlib import Path

              s3 = boto3.client('s3')
              bucket = "${aws_s3_bucket.model_storage.bucket}"
              local_dir = Path("/tmp/mistral-7b")

              for file_path in local_dir.rglob("*"):
                  if file_path.is_file():
                      s3_key = f"mistral-7b-v0-3/{file_path.relative_to(local_dir)}"
                      print(f"Uploading {file_path.name}...")
                      s3.upload_file(str(file_path), bucket, s3_key)

              print("Upload complete!")
              EOF
            resources:
              requests:
                memory: "4Gi"
                cpu: "2"
              limits:
                memory: "8Gi"
                cpu: "4"
  YAML
}
