# Visual Wake Words – INT8 MobileNetV2

## 1. Project Overview

This project implements a person-presence image classification system using MobileNetV2. It classifies an input image into one of two classes: **Person** or **Not-Person**.

The project explores model optimization for edge devices by converting a trained FP32 model into a fully integer-quantized INT8 TensorFlow Lite model.

## 2. Objectives

- Build a binary image classification model.
- Convert the trained model to TensorFlow Lite format.
- Apply post-training INT8 quantization.
- Compare model size, validation accuracy, and inference latency.
- Study the trade-offs involved in deploying machine learning models on resource-constrained devices.

## 3. Technologies Used

- Python
- TensorFlow and Keras
- MobileNetV2
- TensorFlow Lite
- NumPy and Pandas
- Google Colab

## 4. Project Workflow

1. Prepare and preprocess the image dataset.
2. Train the MobileNetV2 classification model.
3. Evaluate the FP32 baseline model.
4. Convert the model to TensorFlow Lite format.
5. Apply post-training INT8 quantization.
6. Evaluate both models on the same validation dataset.
7. Benchmark inference latency and compare model size.

## 5. Baseline vs INT8 Results

| Metric | FP32 Baseline | INT8 Optimized |
|---|---:|---:|
| Model size | 1584.92 KB | 627.90 KB |
| Validation accuracy | 75.83% | 78.33% |
| Mean inference latency | 1.004 ms | 4.130 ms |
| Median inference latency | 1.004 ms | 3.872 ms |

### Observations

- Model size decreased by approximately **60.38%** after quantization.
- INT8 validation accuracy was 2.50 percentage points higher on the 120-example validation set.
- INT8 inference was slower than FP32 inference in the measured Google Colab CPU benchmark.
- Latency and memory performance may differ on other hardware and runtimes.

## 6. Repository Files

- `vww_fp32.tflite` — FP32 TensorFlow Lite baseline model.
- `vww_int8.tflite` — INT8 TensorFlow Lite model.
- `baseline_vs_int8_results.csv` — measured comparison results.
- `Visual_Wake_Words_INT8_MobileNetV2.ipynb` — source-code notebook, once uploaded.

## 7. Execution Environment

- Platform: Google Colab
- Operating system: Linux
- Python: 3.13.16
- TensorFlow: 2.21.0
- Device used: CPU

## 8. Conclusion

The experiment demonstrates that post-training INT8 quantization can reduce model size substantially. In this experiment, the smaller model had slightly higher measured validation accuracy but slower inference on the Colab CPU. Therefore, model optimization should be evaluated using measured size, accuracy, and latency rather than model size alone.

## 9. Author

Apsara G
