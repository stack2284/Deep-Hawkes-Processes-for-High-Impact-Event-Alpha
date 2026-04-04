import torch 
import torch.nn as nn 
 

class HawkesLogLikelihoodLoss(nn.Module) : 
    
    """
        computing negative log likilihood for a Neural hawkes process 
    """

    def __init__(self , eps=1e-9):
        super().__init__() 
        self.eps = eps
    
    def forward (self , intensities , even_types , time_deltas , mask) : 
        """
            intensities [batch , seq_len , number_event_types ] (predicted lambda) 
            event_types : [batch , seq_len] 
            time_deltas [Batch , seq_len ] time to event 
            mask if valid sample or not 
        """
        # ---------------------------------------------------------
        # The Reward (Log-Likelihood of the actual events)
        # ---------------------------------------------------------
        # We only care about the intensity of the event that ACTUALLY happened.
        # .gather() extracts the specific intensity for the true event type.

        event_types_expanded = even_types.unsqueeze(2) 
        true_event_intensisties = intensities.gather(dim=2 , index=event_types_expanded).squeeze(2)

        #calc log (lambda) apply mask to zero out padding 
        event_ll = torch.log(true_event_intensisties + self.eps) * mask 

        # ---------------------------------------------------------
        # PART 2: The Penalty (Integral of all intensities over the time gap)
        # ---------------------------------------------------------
        # The model must pay a penalty for predicting high intensity when nothing happened.
        # We sum the intensity of ALL possible events.

        total_intensities = intensities.sum(dim=2) 
        integral_penalty = (total_intensities * time_deltas) * mask 

        # ---------------------------------------------------------
        # PART 3: Combine and Average
        # ---------------------------------------------------------
        # Total Log-Likelihood = sum(Rewards) - sum(Penalties)
        # We want to MAXIMIZE Log-Likelihood, which means MINIMIZING Negative Log-Likelihood
        log_likelihood = event_ll - integral_penalty
        num_valid_event = mask.sum() 
        loss = - log_likelihood.sum() / (num_valid_event + self.eps) 

        return loss 
    

def test() : 
   batch_size = 32
   seq_len = 50
   num_event_types = 3

   criterion = HawkesLogLikelihoodLoss()
   mock_intensities = torch.rand(batch_size, seq_len, num_event_types) + 0.1
   mock_events = torch.randint(0, num_event_types, (batch_size, seq_len))
   mock_time_deltas = torch.rand(batch_size, seq_len) * 0.05
   mock_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
   mock_mask[:, -5:] = False

   #calc loss here 
   loss = criterion(mock_intensities, mock_events, mock_time_deltas, mock_mask)
   print(f"NLL Loss: {loss.item():.4f}")
   exploding_intensities = mock_intensities * 1000
   exploding_loss = criterion(exploding_intensities, mock_events, mock_time_deltas, mock_mask)
   print(f"Loss with Higher Intensities should be Higher : {exploding_loss.item():.4f}")


if __name__ == "__main__" : 
    test() 




