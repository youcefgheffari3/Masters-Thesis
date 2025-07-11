# 🎙️ Arabic Speech Emotion Recognition System (Master's Thesis)

This repository contains the official source code and documentation for my Master's thesis. The research explores, implements, and evaluates a range of deep learning architectures for audio-based tasks, comparing traditional models with state-of-the-art approaches like Transformers and pre-trained systems.

---

## 📖 Thesis Abstract

For a complete understanding of the project's motivation, methodology, experimental setup, and detailed results, please refer to the full thesis document included in this repository:

📄 **[Master's Thesis.pdf](./Master's%20Thesis.pdf)**

---

## 🚀 Models Explored

This research investigates several distinct deep learning architectures to determine their effectiveness on the given audio task. The implementations are self-contained in the Python scripts.

1.  **Baseline CNN (`eyase_just_cnn.py`)**
    * A foundational Convolutional Neural Network designed to extract features from audio spectrograms. This model serves as a performance baseline.

2.  **Hybrid CNN + Attention + LSTM (`eyase_using_parallel_cnn_attention_lstm.py`)**
    * A more complex, hybrid architecture that combines a CNN for spatial feature extraction with an LSTM network for modeling temporal dependencies. An attention mechanism is incorporated to help the model focus on the most relevant parts of the audio sequence.

3.  **Hybrid CNN + Transformer (`eyase_using_parallel_cnn_transformer.py`)**
    * This model replaces the LSTM with a Transformer Encoder. It leverages the self-attention mechanism of the Transformer to capture long-range dependencies in the audio data, representing a more modern approach to sequence modeling.

4.  **Fine-tuned Wav2Vec2 (`eyase_wav2vec2_0.py`)**
    * This approach utilizes `Wav2Vec2`, a large-scale, pre-trained model for speech representation learning. The model is fine-tuned on the specific task of this thesis, demonstrating the power of transfer learning in the audio domain.

---

## 🛠️ Core Concepts & Technologies

This project is built using Python and the PyTorch deep learning framework. Key technologies and concepts include:

* **Framework:** PyTorch, Hugging Face Transformers
* **Audio Processing:** Libraries like `torchaudio` or `librosa` for loading and transforming audio signals into spectrograms.
* **Architectures:** CNNs, LSTMs, Attention Mechanisms, Transformers.
* **Transfer Learning:** Fine-tuning a large-scale, pre-trained model (Wav2Vec2) to adapt it to a specialized task.

---

## ⚙️ Requirements

To replicate the experiments, you need to install the following primary libraries. Please see the individual scripts for any other specific dependencies.

```bash
pip install torch torchaudio
pip install transformers
pip install numpy
# Add any other libraries like scikit-learn, pandas, etc. if used
```

---

## ▶️ Usage

Each Python script (`eyase_*.py`) is an executable that runs a specific experiment. To run a model, execute its corresponding script:

```bash
# Example for running the baseline CNN model
python eyase_just_cnn.py

# Example for running the CNN + Transformer model
python eyase_using_parallel_cnn_transformer.py
```

*Note: You may need to adjust script parameters, such as data paths or hyperparameters, directly within the files before running.*

---

## 📊 Results & Conclusion

The performance (e.g., accuracy, loss, F1-score) of each model is detailed extensively in the thesis document. The conclusion of the thesis provides a comparative analysis of the different architectures and discusses their respective strengths and weaknesses for the target audio task.

Please refer to the **[Master's Thesis.pdf](./Master's%20Thesis.pdf)** for the full analysis.
