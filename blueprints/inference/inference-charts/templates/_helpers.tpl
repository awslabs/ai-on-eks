{{/*
Expand the name of the chart.
*/}}
{{- define "inference-charts.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "inference-charts.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "inference-charts.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "inference-charts.labels" -}}
helm.sh/chart: {{ include "inference-charts.chart" . }}
{{ include "inference-charts.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "inference-charts.selectorLabels" -}}
app.kubernetes.io/name: {{ include "inference-charts.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Component labels for VLLM
*/}}
{{- define "inference-charts.vllmComponentLabels" -}}
app.kubernetes.io/component: {{.Values.inference.serviceName}}
{{- end }}

{{/*
Component labels for Ray-VLLM
*/}}
{{- define "inference-charts.rayVllmComponentLabels" -}}
app.kubernetes.io/component: {{.Values.inference.serviceName}}
{{- end }}

{{- define "inference-charts.modelParameters" -}}
{{- $modelParameters := .Values.modelParameters -}}
{{- $args := list -}}
{{- range $key, $value := $modelParameters -}}
  {{- $args = append $args (printf "--%s %v" ($key | kebabcase) $value) -}}
{{- end -}}
{{- if eq .Values.inference.framework "aibrix" }}
    {{- $args = append $args (printf "--served-model-name %s" .Values.inference.serviceName) }}
{{- end }}
{{- if .Values.vllm.loadFormat }}
    {{- $args = append $args (printf "--load-format %s" .Values.vllm.loadFormat ) }}
{{- end}}
{{- printf "%s" (join " " $args) | trimSuffix " " -}}
{{- end -}}
