#!/usr/bin/env bash
# Per-rank RDMA transfer windows + node-aggregate (full-model) transfer time and throughput.
# Usage: bash rdma_ranks.sh /path/to/worker.log
# Kept as a script (not inline in SKILL.md) because skill argument substitution rewrites
# awk field references like $0 / $1 in the skill body.
set -euo pipefail
LOG="$1"

grep -E "Worker_TP[0-9]+[ _].*(RDMA transfer complete:|\[TIMING\] register_tensors:)" "$LOG" \
  | sed -E 's/^\[pod[^]]*\] //' \
  | awk '
function fmt(s){ return sprintf("%02d:%02d:%06.3f", int(s/3600), int((s%3600)/60), s-int(s/60)*60) }
{
  line=$0
  # first ISO timestamp on the line -> seconds of day
  match(line, /T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9]+/)
  split(substr(line, RSTART+1, RLENGTH-1), t, ":"); ts=t[1]*3600+t[2]*60+t[3]
  match(line, /Worker_TP[0-9]+/); r=substr(line, RSTART+9, RLENGTH-9)
  if (line ~ /register_tensors:/) {
    match(line, /register_tensors: [0-9.]+s/); d=substr(line, RSTART+18, RLENGTH-19)+0
    rs=ts-d; if (!nr++ || rs<frs) frs=rs
    next
  }
  match(line, /[0-9.]+ GB, [0-9.]+s, [0-9.]+ Gbps/); split(substr(line, RSTART, RLENGTH), f, /[ ,s]+/)
  gb=f[1]; d=f[3]; g=f[4]; st=ts-d
  printf "%012.3f TP%-2s start=%s end=%s  %.3f s  %6.1f Gbps  %5.1f GB/s\n", st, r, fmt(st), fmt(ts), d, g, gb/d
  n++; tot+=gb; sum+=g; dsum+=d; if(d>max)max=d
  if(n==1||st<fs)fs=st; if(n==1||st>ls)ls=st; if(n==1||ts<fe)fe=ts; if(n==1||ts>le)le=ts
}
END{
  if (!n) { print "zzz no RDMA transfer complete lines (not an RDMA receiver?)"; exit }
  w=le-fs
  printf "zzz1 ranks=%d  per-rank mean=%.1f Gbps (%.1f GB/s)  time-weighted=%.1f Gbps  start-skew=%.3f s  all-ranks-overlap=%.3f s\n", n, sum/n, sum/n/8, tot*8/dsum, ls-fs, (fe>ls?fe-ls:0)
  printf "zzz2 FULL-MODEL TRANSFER: %.2f GB in %.3f s wall (%s -> %s) = %.1f GB/s (%.2f Tbps)\n", tot, w, fmt(fs), fmt(le), tot/w, tot*8/w/1000
  if (nr) printf "zzz3 incl. RDMA setup (first register_tensors start -> last end): %.3f s = %.1f GB/s (%.2f Tbps)\n", le-frs, tot/(le-frs), tot*8/(le-frs)/1000
  printf "zzz4 no-skew rate = %.1f Gbps (slowest rank %.3f s)\n", tot*8/max, max
  printf "zzz5 sum of per-rank = %.1f Gbps (upper bound)\n", sum
}' | sort | sed -E 's/^[0-9.]+ //; s/^zzz[0-9]* //'
