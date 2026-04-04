import torch 
import torch.optim as optim 
from  torch.utils.data import DataLoader 

from data import LobsterHawkesDataset , Hawkes_collate_fn 
from model import NeuralHawkesModel 
from loss import HawkesLogLikelihoodLoss 


def train_hawkes_model(
        NUM_EVENT_TYPES = 5 , 
        DATA_PATH = "Dataset/AMZN_2012-06-21_34200000_57600000_message_5.csv" , 
        BATCH_SIZE = 64 , 
        SEQ_LEN= 50,  
        EMBEDDING_DIM=16 ,
        HIDDEN_DIM =64 ,
        EPOCHS=30 ,
        LEARNING_RATE = 1e-3 ,
        MAX_GRAD_NORM=1.0 ) :
    

    if torch.mps.is_available() : 
        device = torch.device("mps") 
    elif torch.cuda.is_available() : 
        device = torch.device("cuda") 
    else : 
        device = torch.device("cpu")


    print(f"Using {device}") 

    try : 
        dataset = LobsterHawkesDataset(DATA_PATH , seq_len=SEQ_LEN) 
        dataLoader = DataLoader(dataset , batch_size=BATCH_SIZE , shuffle=True, collate_fn =Hawkes_collate_fn)
    except FileNotFoundError: 
        print(f"ERROR FINDING FILE {DATA_PATH}")
        return 
    
    # 3 intiialization of architecture 

    model = NeuralHawkesModel(
            num_event_types=NUM_EVENT_TYPES, 
            embedding_dim=EMBEDDING_DIM , 
            hidden_dim=HIDDEN_DIM 
        ).to(device)

    criterion= HawkesLogLikelihoodLoss().to(device) 

    optimizer = optim.Adam(model.parameters() , lr=LEARNING_RATE) 

    print("---TRAINING STARTS---") 

    for epoch in range(EPOCHS) : 
        model.train() 
        total_epoch_loss = 0.0 
        for batch_idx , batch in enumerate(dataLoader) : 
            time_deltas = batch["time_deltas"].to(device) 
            event_types = batch["event_types"].to(device) 

            mask = batch["mask"].to(device) 

            # ensure 0 gadient 
            optimizer.zero_grad()

            _ , _ , intensisties = model(event_types , time_deltas) 
            
            loss = criterion(intensisties, event_types , time_deltas , mask)  

            loss.backward() 
            # Prevents  generating infinite gradients 
            torch.nn.utils.clip_grad_norm_(model.parameters() , MAX_GRAD_NORM) 
            optimizer.step() 
            total_epoch_loss += loss.item() 

            if batch_idx%5 == 0 : 
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(dataLoader)}] | Loss (NLL): {loss.item():.4f}")
            
        avg_loss = total_epoch_loss / len(dataLoader) 
        print(f"==> End of Epoch {epoch+1} | Average Loss: {avg_loss:.4f}\n")

    print("--- Training Complete. Model is ready for Inference. ---")
    torch.save(model.state_dict(), 'hawkes_weights.pth')
    print("Saved model weights to 'hawkes_weights.pth'")


if __name__ == "__main__":
    train_hawkes_model()
            



    

