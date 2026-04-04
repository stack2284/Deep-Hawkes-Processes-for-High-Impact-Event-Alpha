import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from cell import CTLSTMCell 

class NeuralHawkesModel (nn.Module) : 
    def __init__(self, num_event_types , embedding_dim , hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim 
        self.num_event_types = num_event_types 

        """
            event Embedding maps dicrete events 0 1 2 to dense vectors 
        """
        self.embedding = nn.Embedding(num_event_types , embedding_dim) 
        # contineous time LSTM cell 
        self.cell = CTLSTMCell(embedding_dim , hidden_dim) 
        # 
        # intensity projection hidden state to event probablities 
        self.intensity_layer = nn.Linear(hidden_dim , num_event_types)

    def forward (self , event_types , time_deltas) : 
        """
            event types [batch , seqlen] 
            time deltas [batch , time since prev event] 
        """ 


        #initializing hidden state and cell state to 0s 
        batch_size , seq_len = event_types.shape 
        h = torch.zeros(batch_size , self.hidden_dim , device=event_types.device) 
        c = torch.zeros(batch_size , self.hidden_dim , device=event_types.device)

        x_seq = self.embedding(event_types) 
        hidden_state = []
        decay_rate = [] 

        for t in range (seq_len) : 
            x_t = x_seq[: , t , :] 
            dt_t = time_deltas[:, t].unsqueeze(1)
            h , c , decay = self.cell(x_t , dt_t , h ,c) 

            """
                we dont calculate the loss inside this loop the loss requires integrating the contineous 
                space between events we just collect the states for now 
            """
            hidden_state.append(h) 
            decay_rate.append(decay)

        hidden_state = torch.stack(hidden_state , dim=1) 
        decay_rate = torch.stack(decay_rate , dim=1) 

        base_intensities = F.softplus(self.intensity_layer(hidden_state)) 
        return hidden_state , decay_rate , base_intensities
    


def test () : 
    batch_size =32 
    seq_len = 50 
    num_of_event = 3 
    embedding_din = 64 
    hidden_dim = 64
    model = NeuralHawkesModel(num_event_types=num_of_event , embedding_dim=embedding_din , hidden_dim=hidden_dim)

    mock_events = torch.randint(0, num_of_event, (batch_size, seq_len))
    mock_times = torch.rand(batch_size, seq_len) * 0.1 

    h_out , decay_out , intensity_out = model(mock_events , mock_times) 

    print(f"Hidden States Shape: {h_out.shape}")       # Expected: [32, 50, 64]
    print(f"Decay Rates Shape:   {decay_out.shape}")   # Expected: [32, 50, 64]
    print(f"Intensities Shape:   {intensity_out.shape}") # Expected: [32, 50, 3]
    
    print("\nSample Intensities (Batch 0, Step 0):")
    print(intensity_out[0, 0, :].detach().numpy())
    


if __name__ == "__main__" : 
    test() 


