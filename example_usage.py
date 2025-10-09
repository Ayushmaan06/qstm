"""
QLSTM Example Usage and Comparison
==================================

This script demonstrates how to use the QLSTM model and compares it with classical LSTM.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from qlstm_improved import QLSTM


class ClassicalLSTM(nn.Module):
    """Classical LSTM for comparison"""
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=batch_first)
        
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output


def generate_sine_data(batch_size=10, seq_length=20, input_size=1):
    """Generate synthetic sine wave data for testing"""
    t = torch.linspace(0, 4*torch.pi, seq_length).unsqueeze(0).repeat(batch_size, 1)
    # Add some noise and multiple frequencies
    data = torch.sin(t).unsqueeze(-1)
    if input_size > 1:
        for i in range(1, input_size):
            freq = (i + 1) * 0.5
            data = torch.cat([data, torch.sin(freq * t).unsqueeze(-1)], dim=-1)
    
    return data


def compare_models():
    """Compare QLSTM with classical LSTM"""
    print("🔬 Comparing QLSTM vs Classical LSTM")
    print("=" * 50)
    
    # Parameters
    batch_size = 5
    seq_length = 15
    input_size = 2
    hidden_size = 4
    n_qubits = input_size + hidden_size + 2  # Extra qubits for safety
    
    # Generate data
    data = generate_sine_data(batch_size, seq_length, input_size)
    print(f"📊 Data shape: {data.shape}")
    
    # Create models
    print(f"\n🏗️ Creating models...")
    classical_lstm = ClassicalLSTM(input_size, hidden_size)
    qlstm = QLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        n_qubits=n_qubits,
        n_qlayers=1,
        return_sequences=True,
        dropout=0.0  # No dropout for fair comparison
    )
    
    print(f"Classical LSTM parameters: {sum(p.numel() for p in classical_lstm.parameters())}")
    print(f"QLSTM parameters: {sum(p.numel() for p in qlstm.parameters())}")
    
    # Forward pass timing
    print(f"\n⏱️ Timing comparison...")
    
    # Classical LSTM
    start_time = time.time()
    with torch.no_grad():
        classical_output = classical_lstm(data)
    classical_time = time.time() - start_time
    
    # QLSTM
    start_time = time.time()
    with torch.no_grad():
        quantum_output = qlstm(data)
    quantum_time = time.time() - start_time
    
    print(f"Classical LSTM time: {classical_time:.4f}s")
    print(f"QLSTM time: {quantum_time:.4f}s")
    print(f"Speed ratio (Classical/Quantum): {classical_time/quantum_time:.2f}")
    
    print(f"\n📈 Output shapes:")
    print(f"Classical LSTM output: {classical_output.shape}")
    print(f"QLSTM output: {quantum_output.shape}")
    
    # Basic statistics
    print(f"\n📊 Output statistics:")
    print(f"Classical LSTM - Mean: {classical_output.mean():.4f}, Std: {classical_output.std():.4f}")
    print(f"QLSTM - Mean: {quantum_output.mean():.4f}, Std: {quantum_output.std():.4f}")


def training_example():
    """Demonstrate training a QLSTM model"""
    print("\n" + "="*50)
    print("🎯 Training Example")
    print("="*50)
    
    # Parameters
    batch_size = 8
    seq_length = 10
    input_size = 3
    hidden_size = 5
    n_qubits = input_size + hidden_size + 1
    epochs = 5
    
    # Create model and data
    model = QLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        n_qubits=n_qubits,
        n_qlayers=1,
        return_sequences=False,  # Only final output
        dropout=0.1
    )
    
    # Generate training data (simple regression task)
    X = torch.randn(batch_size, seq_length, input_size)
    y = torch.randn(batch_size, hidden_size)  # Random target
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"🏋️ Training for {epochs} epochs...")
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(X)
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    print(f"✅ Training completed!")
    print(f"Final loss: {losses[-1]:.6f}")
    
    return model, losses


def visualize_quantum_weights(model):
    """Visualize the quantum circuit weights"""
    print(f"\n🔍 Quantum Circuit Weights Analysis")
    print("="*50)
    
    weights = model.get_quantum_weights()
    
    for gate_name, gate_weights in weights.items():
        print(f"\n{gate_name.upper()} Gate:")
        print(f"Shape: {gate_weights.shape}")
        print(f"Mean: {gate_weights.mean():.4f}")
        print(f"Std: {gate_weights.std():.4f}")
        print(f"Min: {gate_weights.min():.4f}")
        print(f"Max: {gate_weights.max():.4f}")


def main():
    """Main function to run all examples"""
    print("🌟 QLSTM Demonstration")
    print("="*50)
    
    try:
        # Comparison
        compare_models()
        
        # Training example
        trained_model, losses = training_example()
        
        # Analyze quantum weights
        visualize_quantum_weights(trained_model)
        
        print(f"\n🎉 All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n💡 Possible solutions:")
        print("1. Install required dependencies: pip install -r requirements.txt")
        print("2. Check that PennyLane is properly installed")
        print("3. Reduce n_qubits if memory issues occur")


if __name__ == "__main__":
    main()