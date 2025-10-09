# Quantum Long Short-Term Memory (QLSTM) 🚀

A hybrid quantum-classical implementation of LSTM networks that leverages quantum circuits for enhanced representation learning.

## 🎯 Overview

This project modernizes traditional LSTM networks by replacing classical linear transformations with **Variational Quantum Circuits (VQCs)**. The QLSTM maintains the core LSTM structure while using quantum computing principles for potentially improved learning capabilities.

## 🔬 Key Features

- **Hybrid Architecture**: Combines classical LSTM gates with quantum circuits
- **Variational Quantum Circuits**: Uses parameterized quantum circuits for gate computations
- **Flexible Backend Support**: Compatible with various quantum simulators
- **PyTorch Integration**: Seamless integration with PyTorch ecosystem
- **Batch Processing**: Efficient batch processing for practical applications

## 🏗️ Architecture

### Classical LSTM vs QLSTM

**Traditional LSTM:**
```
Input → Linear Layer → Activation → Output
```

**QLSTM:**
```
Input → Linear Layer → Quantum Circuit → Linear Layer → Activation → Output
```

### Gate Structure

Each LSTM gate (forget, input, update, output) is enhanced with:
1. **Classical preprocessing**: Linear transformation to quantum dimension
2. **Quantum processing**: Variational quantum circuit with:
   - Angle embedding for input encoding
   - Parameterized quantum layers for feature transformation
   - Pauli-Z measurements for output extraction
3. **Classical postprocessing**: Linear transformation back to hidden dimension

## 📦 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd qstm
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python qlstm_improved.py
```

## 🚀 Usage

### Basic Usage

```python
import torch
from qlstm_improved import QLSTM

# Create model
model = QLSTM(
    input_size=10,      # Input feature dimension
    hidden_size=20,     # Hidden state dimension
    n_qubits=15,        # Number of qubits (should be ≥ input_size + hidden_size)
    n_qlayers=2,        # Number of quantum layers
    return_sequences=True,
    dropout=0.1
)

# Sample data
batch_size, seq_length, input_size = 32, 50, 10
x = torch.randn(batch_size, seq_length, input_size)

# Forward pass
output = model(x)
print(f"Output shape: {output.shape}")
```

### Advanced Configuration

```python
# Custom backend and parameters
model = QLSTM(
    input_size=5,
    hidden_size=10,
    n_qubits=20,
    n_qlayers=3,
    backend="default.qubit",  # or "qiskit.basicaer"
    batch_first=True,
    return_sequences=False,
    return_state=True,
    dropout=0.2
)

# With initial states
h0 = torch.zeros(batch_size, 10)
c0 = torch.zeros(batch_size, 10)
output, (h_n, c_n) = model(x, init_states=(h0, c0))
```

## 🔧 Key Improvements in Enhanced Version

### 1. **Robust Error Handling**
- Input validation for parameters
- Graceful quantum circuit error handling
- Fallback to classical processing if quantum fails

### 2. **Better Quantum Circuit Design**
- Proper input padding/truncation for qubit constraints
- Unique wire naming to prevent conflicts
- Improved angle embedding

### 3. **Enhanced Features**
- Dropout regularization
- Proper weight initialization
- Gradient-friendly design
- Comprehensive documentation

### 4. **Batch Processing Optimization**
- Efficient batch handling for quantum circuits
- Memory-conscious processing
- Device compatibility

### 5. **Flexibility**
- Multiple return options (sequences, states)
- Configurable backends
- Adjustable quantum parameters

## 📊 Technical Details

### Quantum Circuit Structure

Each gate uses the following quantum circuit:
```
|0⟩ ─── RY(θ₁) ─── CNOT ─── RY(φ₁) ─── ⟨Z⟩
|0⟩ ─── RY(θ₂) ─── CNOT ─── RY(φ₂) ─── ⟨Z⟩
...
```

Where:
- `θᵢ` are input angles from angle embedding
- `φᵢ` are trainable parameters
- `⟨Z⟩` represents Pauli-Z expectation values

### Mathematical Foundation

For each LSTM gate `g ∈ {forget, input, update, output}`:

1. **Classical preprocessing**: `y = W₁(h_{t-1} ⊕ x_t) + b₁`
2. **Quantum processing**: `q = VQC(y; θ)`
3. **Classical postprocessing**: `g_t = σ(W₂q + b₂)`

Where `VQC(y; θ)` represents the variational quantum circuit.

## 🧪 Testing

Run the built-in tests:
```bash
python qlstm_improved.py
```

Expected output:
```
Testing QLSTM implementation...
Quantum weight shapes: (n_qlayers, n_qubits) = (1, 6)
Model: QLSTM(input_size=3, hidden_size=4, n_qubits=6, n_qlayers=1, backend=default.qubit)
Total parameters: 42
Input shape: torch.Size([2, 5, 3])
Output sequences shape: torch.Size([2, 5, 4])
Final hidden state shape: torch.Size([2, 4])
Final cell state shape: torch.Size([2, 4])
✅ QLSTM test passed!
✅ Gradient computation successful!
```

## 🔍 Common Issues & Solutions

### Issue 1: "concat_size > n_qubits"
**Solution**: Increase `n_qubits` to at least `input_size + hidden_size`

### Issue 2: Quantum backend errors
**Solution**: Ensure PennyLane is properly installed:
```bash
pip install pennylane --upgrade
```

### Issue 3: Slow performance
**Solutions**:
- Reduce `n_qubits` and `n_qlayers`
- Use `batch_first=True` for better memory access
- Consider using quantum simulators with GPU support

## 🔮 Future Enhancements

- [ ] Support for multiple QLSTM layers
- [ ] Integration with quantum hardware backends
- [ ] Attention mechanisms with quantum circuits
- [ ] Automated hyperparameter optimization
- [ ] Comparison benchmarks with classical LSTM

## 📚 References

- [Quantum Machine Learning](https://pennylane.ai/qml/)
- [LSTM Neural Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Variational Quantum Circuits](https://arxiv.org/abs/1801.00862)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is open source. Please check the license file for details.
