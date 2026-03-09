llama-server \
  -m ./LFM2.5-VL-1.6B-Q4_0.gguf \
  --mmproj ./mmproj-LFM2.5-VL-1.6b-F16.gguf \
  -b 4 \
  -c 1024 \
  --threads 6 \
  --host 0.0.0.0 \
  --port 9876  \
  --no-warmup
