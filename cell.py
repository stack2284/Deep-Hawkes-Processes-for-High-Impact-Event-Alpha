import torch 
import torch.nn as nn
import torch.nn.functional as F 


class CTLSTMCell(nn.Module) : 

    """
        Contineous-Time Lstm cell 
        handles async events by decaying hidden memory between timestamps
    """

    def __init__(self , input_size , hidden_state): 
        super(CTLSTMCell , self).__init__() 
        self.input_size = input_size 
        self.hidden_state = hidden_state 

        """
            input , forget , output , cellcandidate
        """
        self.lstm_weights = nn.Linear(input_size + hidden_state , 4*hidden_state)

        # decay gate computes the rate of exponential decay delta 
        self.decay_weight = nn.Linear(input_size + hidden_state , hidden_state)

    def forward(self, x , dt , h_prev, c_prev): 
        """
            x : event future / Embedding at time t_i [BAtch , input_size] 
            dt : time elapsed since the previous event t_i-1 [BATCH] 
            h_prev: Hidden state exactly after the previous event [Batch, Hidden_Size]
            c_prev: Cell state exactly after the previous event [Batch, Hidden_Size]
        """
        combined_decay = torch.cat([x , h_prev] , dim=1) 
        decay_rate = F.softplus(self.decay_weight(combined_decay)) 

        # decay the previous state by time elapsed 
        c_decayed = c_prev * torch.exp(-decay_rate * dt)


        combined_lstm = torch.cat([x , h_prev] , dim =1) 
        gates = self.lstm_weights(combined_lstm) 

        i_gate , f_gate , o_gate , z_gate = torch.chunk(gates, 4 , dim=1) 

        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)
        z = torch.tanh(z_gate)
        # update the new cell state 
        c_new = f * c_decayed + i*z 
        # output the new hidden state 
        h_new = o*torch.tanh(c_new) 


        return h_new, c_new, decay_rate


def test () : 
    batch_size= 4 
    input_size = 16 
    hidden_size = 32 

    cell = CTLSTMCell(input_size , hidden_size) 

    # random inputs 

    x_mock = torch.randn(batch_size , input_size) 
    # time gaps 
    dt_mock = torch.tensor([[0.001], [0.5], [2.1], [0.0001]]) 
    h_mock = torch.randn(batch_size, hidden_size)
    c_mock = torch.randn(batch_size, hidden_size)

    # ForAard pass
    h_out, c_out, decay_out = cell(x_mock, dt_mock, h_mock, c_mock)

    print(f"Hidden State Output Shape: {h_out.shape}")      # Expected: [4, 32]
    print(f"Cell State Output Shape:   {c_out.shape}")      # Expected: [4, 32]
    print(f"Decay Rate Output Shape:   {decay_out.shape}")  # Expected: [4, 32]
    print(f"Sample Decay Rate (Min/Max): {decay_out.min().item():.4f} / {decay_out.max().item():.4f}")

if __name__ == "__main__" : 
    test() 

