# Model Download Job
resource "kubectl_manifest" "mistral_model_download" {
  depends_on = [
    module.eks,
    kubectl_manifest.s3_sync_service_account,
    aws_s3_bucket.model_storage
  ]

  force_new = true
  yaml_body = <<-YAML
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: model-download
      namespace: default
    spec:
      backoffLimit: 3
      activeDeadlineSeconds: 7200
      ttlSecondsAfterFinished: 86400
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
              pip install -q huggingface_hub boto3

              echo "Downloading Ministral-3-8B-Instruct-2512 from HuggingFace..."
              python3 -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Ministral-3-8B-Instruct-2512', local_dir='/tmp/mistral', allow_patterns=['*.json', '*.txt', '*.md', '*.model', 'consolidated.safetensors'])"

              echo "Uploading to S3 bucket: ${aws_s3_bucket.model_storage.bucket}"
              python3 << 'EOF'
              import boto3
              import os
              from pathlib import Path

              s3 = boto3.client('s3')
              bucket = "${aws_s3_bucket.model_storage.bucket}"
              local_dir = Path("/tmp/mistral")

              for file_path in local_dir.rglob("*"):
                  if file_path.is_file():
                      # Skip .cache directories and their contents
                      if '.cache' in file_path.parts:
                          continue
                      s3_key = f"Ministral-3-8B-Instruct-2512/{file_path.relative_to(local_dir)}"
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
