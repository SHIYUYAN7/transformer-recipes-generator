import torch
import torch.nn as nn

class CustomLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weights_input_hidden = nn.Parameter(torch.randn(4 * hidden_dim, input_dim))
        self.weights_hidden_hidden = nn.Parameter(torch.randn(4 * hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.randn(4 * hidden_dim))

    def forward(self, input_tensor, hidden_state):
        h_t, c_t = hidden_state
        combined_gates = (torch.mm(input_tensor, self.weights_input_hidden.t()) +
                          torch.mm(h_t, self.weights_hidden_hidden.t()) + self.bias)
        i_gate, f_gate, g_gate, o_gate = combined_gates.chunk(4, 1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)
        c_next = (f_gate * c_t) + (i_gate * g_gate)
        h_next = o_gate * torch.tanh(c_next)
        return h_next, c_next

class EnhancedLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout_rate=0.1):
        super(EnhancedLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm_cells = nn.ModuleList([CustomLSTMCell(embedding_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.final_fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, sequence_length, _ = x.size()
        initial_states = [(torch.zeros(batch_size, self.lstm_cells[0].hidden_dim),
                           torch.zeros(batch_size, self.lstm_cells[0].hidden_dim)) for _ in range(len(self.lstm_cells))]
        x = self.embedding(x)
        output_sequence = []
        
        for t in range(sequence_length):
            layer_input = x[:, t, :]
            for layer_idx, lstm_cell in enumerate(self.lstm_cells):
                h_t, c_t = initial_states[layer_idx]
                h_t, c_t = lstm_cell(layer_input, (h_t, c_t))
                layer_input = h_t
                initial_states[layer_idx] = (h_t, c_t)
            output_sequence.append(h_t)
        
        lstm_outputs = torch.stack(output_sequence, dim=1)
        lstm_outputs = self.dropout(lstm_outputs)
        final_output = self.final_fc(lstm_outputs.contiguous().view(-1, lstm_outputs.shape[2]))
        return final_output.view(-1, sequence_length, self.final_fc.out_features)