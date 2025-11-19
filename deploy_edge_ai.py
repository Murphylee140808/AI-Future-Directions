# deploy_edge_ai.py
import cv2
import time
import numpy as np
import tensorflow as tf
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='recyclable_model_quant.tflite', help='Path to .tflite model')
parser.add_argument('--labels', type=str, default='labels.json', help='JSON file with class_names list')
parser.add_argument('--cam', type=int, default=0, help='Camera index')
parser.add_argument('--width', type=int, default=224, help='Input width')
parser.add_argument('--height', type=int, default=224, help='Input height')
parser.add_argument('--normalize', action='store_true', help='If used, scale pixels to [0,1]')
args = parser.parse_args()

# Load labels
if os.path.exists(args.labels):
    with open(args.labels, 'r') as f:
        class_names = json.load(f)
else:
    # fallback: try simple class_0..N if labels file missing
    print("Warning: labels.json not found. Using placeholders.")
    class_names = []

print("Loaded model:", args.model)
print("Number of class names:", len(class_names))

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=args.model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# Determine input type & scale
input_dtype = input_details[0]['dtype']
print("Interpreter input dtype:", input_dtype)

# Open webcam
cap = cv2.VideoCapture(args.cam)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera index {}".format(args.cam))

prev_time = time.time()
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Empty frame, skipping")
            continue

        # Preprocess: BGR->RGB, resize
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.width, args.height))

        # Convert dtype
        if input_dtype == np.uint8:
            # quantized input: scale to [0,255] uint8
            in_tensor = img.astype(np.uint8)
        else:
            # float input: normalize if requested, else scale to [0,1]
            arr = img.astype(np.float32)
            if args.normalize:
                arr = arr / 255.0
            else:
                arr = arr / 255.0
            in_tensor = arr

        # Expand batch dim
        input_data = np.expand_dims(in_tensor, axis=0)

        # If interpreter expects quantized input with scale/zero_point, apply them
        if input_dtype == np.uint8 and 'quantization' in input_details[0] and input_details[0]['quantization'][0] != 0:
            scale, zero_point = input_details[0]['quantization']
            # Convert float -> quantized
            # If our in_tensor already uint8, we trust it; otherwise do the quantization step
            if in_tensor.dtype != np.uint8:
                input_data = (input_data / scale + zero_point).astype(np.uint8)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        # Flatten / safe extraction
        if output_data.ndim == 2:
            probs = output_data[0]
        else:
            probs = output_data.flatten()

        # Get predicted index safely
        top_idx = int(np.argmax(probs))
        if len(class_names) == 0:
            label = f"class_{top_idx}"
        else:
            if 0 <= top_idx < len(class_names):
                label = class_names[top_idx]
            else:
                # safety fallback
                top3 = np.argsort(probs)[-3:][::-1]
                label = f"unknown (top:{top_idx})"
                print("WARNING: predicted index out of range:", top_idx, "top3:", top3)

        # FPS
        frame_count += 1
        if frame_count >= 5:
            now = time.time()
            fps = frame_count / (now - prev_time)
            prev_time = now
            frame_count = 0
        else:
            fps = None

        # Draw label
        text = f"{label}"
        if fps:
            text += f" | {fps:.1f} FPS"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Show frame
        cv2.imshow("Edge AI - Press q or ESC to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Exiting.")
