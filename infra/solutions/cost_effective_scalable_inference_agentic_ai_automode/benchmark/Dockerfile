# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o benchmark perf_benchmark.go

# Run stage
FROM alpine:latest

WORKDIR /app
COPY --from=builder /app/benchmark .
COPY prompts.txt /app

ENV URL="http://localhost:8000/v1/chat/completions"
ENV REQUESTS_PER_PROMPT=10
ENV NUM_WARMUP_REQUESTS=3

CMD ["./benchmark"]
