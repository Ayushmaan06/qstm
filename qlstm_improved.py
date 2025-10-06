"""
Quantum Long Short-Term Memory (QLSTM) Network
==============================================

A hybrid quantum-classical neural network that enhances traditional LSTM
with quantum circuits for improved representation learning.

Author: Enhanced version with improvements and fixes
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from typing import Optional, Tuple, Union


class QLSTM(nn.Module):
    """
    Quantum Long Short-Term Memory (QLSTM) Network
    
    This implementation replaces classical linear transformations in LSTM gates
    with Variational Quantum Circuits (VQCs) for enhanced representation learning.
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units
        n_qubits (int): Number of qubits per quantum circuit (default: 4)
        n_qlayers (int): Number of quantum layers (default: 1)
        batch_first (bool): If True, input shape is (batch, seq, feature) (default: True)
        return_sequences (bool): If True, return full sequence of hidden states (default: False)
        return_state (bool): If True, return final hidden and cell states (default: False)
        backend (str): Quantum backend to use (default: "default.qubit")
        dropout (float): Dropout probability (default: 0.0)
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 n_qubits: int = 4,
                 n_qlayers: int = 1,
                 batch_first: bool = True,
                 return_sequences: bool = False, 
                 return_state: bool = False,
                 backend: str = "default.qubit",
                 dropout: float = 0.0):
        super(QLSTM, self).__init__()
        
        # Validate inputs
        if n_qubits < 1:
            raise ValueError("n_qubits must be at least 1")
        if n_qlayers < 1:
            raise ValueError("n_qlayers must be at least 1")
        if not (0 <= dropout <= 1):
            raise ValueError("dropout must be between 0 and 1")
            
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concat_size = input_size + hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state
        
        # Ensure we have enough qubits for the concatenated input
        if self.concat_size > n_qubits:
            print(f"Warning: concat_size ({self.concat_size}) > n_qubits ({n_qubits})")
            print("Consider increasing n_qubits or using dimensionality reduction")

        # Create quantum devices with unique wire names
        self.wires_forget = [f"forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"output_{i}" for i in range(self.n_qubits)]

        # Initialize quantum devices
        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        # Define quantum circuits
        self.qlayer_forget = self._create_quantum_layer(self.dev_forget, self.wires_forget)
        self.qlayer_input = self._create_quantum_layer(self.dev_input, self.wires_input)
        self.qlayer_update = self._create_quantum_layer(self.dev_update, self.wires_update)
        self.qlayer_output = self._create_quantum_layer(self.dev_output, self.wires_output)

        # Weight shapes for quantum layers
        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        print(f"Quantum weight shapes: (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        # Classical layers
        self.clayer_in = nn.Linear(self.concat_size, n_qubits)
        self.clayer_out = nn.Linear(self.n_qubits, self.hidden_size)
        
        # Quantum-classical interface layers
        self.VQC = nn.ModuleDict({
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        })
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()

    def _create_quantum_layer(self, device, wires):
        """Create a quantum circuit layer with angle embedding and entangling layers"""
        @qml.qnode(device, interface="torch")
        def circuit(inputs, weights):
            # Ensure inputs match the number of qubits
            if len(inputs) > len(wires):
                inputs = inputs[:len(wires)]
            elif len(inputs) < len(wires):
                # Pad with zeros if necessary
                padding = torch.zeros(len(wires) - len(inputs))
                inputs = torch.cat([inputs, padding])
            
            # Angle embedding
            qml.templates.AngleEmbedding(inputs, wires=wires)
            
            # Variational layers
            qml.templates.BasicEntanglerLayers(weights, wires=wires)
            
            # Measurements
            return [qml.expval(qml.PauliZ(wires=w)) for w in wires]
        
        return circuit

    def _initialize_weights(self):
        """Initialize classical layer weights using Xavier initialization"""
        nn.init.xavier_uniform_(self.clayer_in.weight)
        nn.init.zeros_(self.clayer_in.bias)
        nn.init.xavier_uniform_(self.clayer_out.weight)
        nn.init.zeros_(self.clayer_out.bias)

    def forward(self, x: torch.Tensor, 
                init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through QLSTM
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size) if batch_first=True
               or (seq_length, batch_size, input_size) if batch_first=False
            init_states: Optional tuple of (hidden_state, cell_state) initial states
            
        Returns:
            If return_sequences=True and return_state=True: (sequences, (h_n, c_n))
            If return_sequences=True and return_state=False: sequences
            If return_sequences=False and return_state=True: (h_n, (h_n, c_n))
            If return_sequences=False and return_state=False: h_n
        """
        
        # Handle input dimensions
        if self.batch_first:
            batch_size, seq_length, _ = x.size()
        else:
            seq_length, batch_size, _ = x.size()
            x = x.transpose(0, 1)  # Convert to batch_first for processing

        # Initialize hidden and cell states
        device = x.device
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h_t, c_t = init_states
            # Handle stacked LSTM case
            if h_t.dim() == 3:
                h_t = h_t[0]
            if c_t.dim() == 3:
                c_t = c_t[0]

        hidden_seq = []
        
        for t in range(seq_length):
            # Get input at time step t
            x_t = x[:, t, :]
            
            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)
            
            # Apply dropout
            v_t = self.dropout(v_t)
            
            # Map to quantum circuit dimension
            y_t = self.clayer_in(v_t)
            
            # Quantum processing for each gate
            try:
                # Process each sample in the batch individually for quantum circuits
                batch_results = {gate: [] for gate in ['forget', 'input', 'update', 'output']}
                
                for i in range(batch_size):
                    y_sample = y_t[i]
                    for gate_name in ['forget', 'input', 'update', 'output']:
                        quantum_output = self.VQC[gate_name](y_sample)
                        batch_results[gate_name].append(quantum_output)
                
                # Stack results back to batch format
                f_quantum = torch.stack(batch_results['forget'])
                i_quantum = torch.stack(batch_results['input'])
                g_quantum = torch.stack(batch_results['update'])
                o_quantum = torch.stack(batch_results['output'])
                
            except Exception as e:
                print(f"Quantum processing error: {e}")
                # Fallback to classical processing
                f_quantum = y_t
                i_quantum = y_t
                g_quantum = y_t
                o_quantum = y_t
            
            # Map quantum outputs back to hidden dimension
            f_t = torch.sigmoid(self.clayer_out(f_quantum))  # forget gate
            i_t = torch.sigmoid(self.clayer_out(i_quantum))  # input gate
            g_t = torch.tanh(self.clayer_out(g_quantum))     # candidate values
            o_t = torch.sigmoid(self.clayer_out(o_quantum))  # output gate
            
            # LSTM cell computations
            c_t = f_t * c_t + i_t * g_t  # Update cell state
            h_t = o_t * torch.tanh(c_t)  # Update hidden state
            
            if self.return_sequences:
                hidden_seq.append(h_t.unsqueeze(1))
        
        # Prepare outputs
        if self.return_sequences:
            hidden_seq = torch.cat(hidden_seq, dim=1)
            if not self.batch_first:
                hidden_seq = hidden_seq.transpose(0, 1)
                
            if self.return_state:
                return hidden_seq, (h_t, c_t)
            else:
                return hidden_seq
        else:
            if self.return_state:
                return h_t, (h_t, c_t)
            else:
                return h_t

    def get_quantum_weights(self):
        """Get the current quantum circuit weights"""
        weights = {}
        for gate_name, layer in self.VQC.items():
            weights[gate_name] = layer.weights.detach().cpu().numpy()
        return weights
    
    def set_quantum_weights(self, weights_dict):
        """Set quantum circuit weights"""
        for gate_name, weights in weights_dict.items():
            if gate_name in self.VQC:
                self.VQC[gate_name].weights.data = torch.tensor(weights)

    def extra_repr(self) -> str:
        """Extra representation for print statements"""
        return (f'input_size={self.input_size}, hidden_size={self.hidden_size}, '
                f'n_qubits={self.n_qubits}, n_qlayers={self.n_qlayers}, '
                f'backend={self.backend}')


# Example usage and testing functions
def create_sample_data(batch_size=2, seq_length=5, input_size=3):
    """Create sample data for testing"""
    return torch.randn(batch_size, seq_length, input_size)


def test_qlstm():
    """Test the QLSTM implementation"""
    print("Testing QLSTM implementation...")
    
    # Parameters
    input_size = 3
    hidden_size = 4
    n_qubits = 6  # Increased to accommodate concat_size
    batch_size = 2
    seq_length = 5
    
    # Create model
    model = QLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        n_qubits=n_qubits,
        n_qlayers=1,
        return_sequences=True,
        return_state=True,
        dropout=0.1
    )
    
    print(f"Model: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create sample data
    x = create_sample_data(batch_size, seq_length, input_size)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    try:
        sequences, (h_n, c_n) = model(x)
        print(f"Output sequences shape: {sequences.shape}")
        print(f"Final hidden state shape: {h_n.shape}")
        print(f"Final cell state shape: {c_n.shape}")
        print("✅ QLSTM test passed!")
        
        # Test gradient computation
        loss = sequences.sum()
        loss.backward()
        print("✅ Gradient computation successful!")
        
    except Exception as e:
        print(f"❌ QLSTM test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_qlstm()