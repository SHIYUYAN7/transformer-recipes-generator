import torch
import torch.nn as nn

class MyLSTM(nn.Module):
    def __init__(self, units, input_size):
        super(MyLSTM, self).__init__()
        self.units = units
        self.input_size = input_size

        # Create trainable weight variables for this layer.
        self.kernel = nn.Parameter(torch.Tensor(input_size, 4 * units))
        self.recurrent_kernel = nn.Parameter(torch.Tensor(units, 4 * units))
        self.bias = nn.Parameter(torch.Tensor(4 * units))

        # Initialize weights and biases
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent_kernel)
        nn.init.zeros_(self.bias)

    def forward(self, inputs, initial_state=None):
        if initial_state is None:
            ht = torch.zeros(inputs.shape[1], self.units, dtype=torch.float32, device=inputs.device)
            ct = torch.zeros(inputs.shape[1], self.units, dtype=torch.float32, device=inputs.device)
        else:
            ht, ct = initial_state

        outputs = []
        # Process each time step iteratively
        for i in range(inputs.shape[0]):  # assuming inputs shape is [seq_len, batch, input_size]
            xt = inputs[i]
            z_total = torch.matmul(xt, self.kernel) + torch.matmul(ht, self.recurrent_kernel) + self.bias

            z_i, z_f, z_c, z_o = z_total.chunk(4, 1)
            input_gate = torch.sigmoid(z_i)
            forget_gate = torch.sigmoid(z_f)
            cell_state = torch.tanh(z_c)
            output_gate = torch.sigmoid(z_o)

            ct = forget_gate * ct + input_gate * cell_state
            ht = output_gate * torch.tanh(ct)

            outputs.append(ht.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (ht, ct)