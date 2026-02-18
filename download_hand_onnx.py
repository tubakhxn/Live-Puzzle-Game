import urllib.request

url = "https://github.com/onnx/models/raw/main/vision/body_analysis/handtracking/models/handtracking-10.onnx"
output_path = "handtracking-10.onnx"

print("Downloading ONNX hand detection model...")
urllib.request.urlretrieve(url, output_path)
print("Download complete! Saved as handtracking-10.onnx")
