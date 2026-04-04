import time 
import pandas as pd 
import numpy as np 
import torch  
import torch.nn.functional as F 

from model import NeuralHawkesModel 

def simulate_live_market(message_path,
                        orderbook_path ,
                        model , 
                        device, 
                        msg_cols = ['Time', 'Type', 'OrderID', 'Size', 'Price', 'Direction'] ,
                        ob_cols = ['AskP1', 'AskS1', 'BidP1', 'BidS1'],
                        cash = 100000.00,
                        FEE_PER_TRADE = 0.5,
                        HAWKES_VOLATILITY_THRESHOLD = 15, 
                        OBI_CONFIDENCE_THRESHOLD = 0.3
                        ) : 
    
    start_time = time.time() 
    df_msg = pd.read_csv(message_path , names=msg_cols)
    df_msg['TimeDelta'] = df_msg['Time'].diff().fillna(0.0) 
    df_msg['TimeDelta'] =np.clip(df_msg['Time'] , a_min=1e-9 ,a_max=None ) 

    def map_event(row):
        e_type, direction = row['Type'], row['Direction']
        if e_type == 1 and direction == 1: return 0
        if e_type == 1 and direction == -1: return 1
        if e_type in [2, 3]: return 2
        if e_type in [4, 5] and direction == 1: return 3
        if e_type in [4, 5] and direction == -1: return 4
        return 2
    df_msg['MappedType'] = df_msg.apply(map_event, axis=1)


    df_ob = pd.read_csv(orderbook_path , names=ob_cols) 

    df_ob['OBI'] = (df_ob['BidS1'] - df_ob['AskS1']) / (df_ob['BidS1'] + df_ob['AskS1'] + 1e-9)

    ask_prices = (df_ob['AskP1'] / 10000.0).values
    bid_prices = (df_ob['BidP1'] / 10000.0).values
    types = df_msg['Type'].values
    directions = df_msg['Direction'].values
    mapped_types = df_msg['MappedType'].values
    time_deltas = df_msg['TimeDelta'].values
    obi_array = df_ob['OBI'].values

    inventory = 0
    fees_paid = 0.0
    trades_executed = 0

    model.eval()
    hidden_dim = model.hidden_dim
    h = torch.zeros(1, hidden_dim, device=device)
    c = torch.zeros(1, hidden_dim, device=device)

    print("--- Streaming Market Data---")

    with torch.no_grad():

        for idx in range(len(types)):

            if idx > 0 and idx% 50 == 0:
                h = torch.zeros(1, hidden_dim, device=device)
                c = torch.zeros(1, hidden_dim, device=device)
                
            current_ask = ask_prices[idx]
            current_bid = bid_prices[idx]


            event_idx = torch.tensor([[mapped_types[idx]]], dtype=torch.long, device=device)
            dt = torch.tensor([[time_deltas[idx]]], dtype=torch.float32, device=device)
            
            x = model.embedding(event_idx).squeeze(1) 
            h, c, _ = model.cell(x, dt, h, c)

            intensities = F.softplus(model.intensity_layer(h)) 
            buy_intensity = intensities[0, 3].item()
            sell_intensity = intensities[0, 4].item()

            current_obi = obi_array[idx]

            if buy_intensity > HAWKES_VOLATILITY_THRESHOLD and current_obi > OBI_CONFIDENCE_THRESHOLD and inventory == 0:
                inventory = 100 
                cash -= (inventory * current_ask)
                cash -= FEE_PER_TRADE
                trades_executed += 1
                fees_paid += FEE_PER_TRADE
            

            elif sell_intensity > HAWKES_VOLATILITY_THRESHOLD and current_obi < -OBI_CONFIDENCE_THRESHOLD and inventory > 0:
                cash += (inventory * current_bid)
                cash -= FEE_PER_TRADE
                inventory = 0
                trades_executed += 1
                fees_paid += FEE_PER_TRADE

            if idx % 2000 == 0:
                mtm_value = cash + (inventory * current_bid)
                print(f"Tick {idx:06d} | PnL: ${mtm_value:.2f} | Pos: {inventory} | OBI: {current_obi:+.2f} | Buy_Int: {buy_intensity:.4f} | Sell_Int: {sell_intensity:.4f} | Trades : {trades_executed}")
            
    
    final_value = cash + (inventory * current_bid)
    elapsed = time.time() - start_time
    
    print("\n--- Backtest Complete ---")
    print(f"Time Elapsed:     {elapsed:.2f} seconds")
    print(f"Total Trades:     {trades_executed}")
    print(f"Total Fees Paid:  ${fees_paid:.2f}")
    print(f"Ending Capital:   ${final_value:.2f}")
    print(f"Net Profit:       ${final_value - 100000.00:.2f}")



if __name__ == "__main__" : 
    NUM_EVENT_TYPES = 5 
    EMBEDDING_DIM = 16 
    HIDDEN_DIM = 64 

    device = torch.device( "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralHawkesModel(
        num_event_types=NUM_EVENT_TYPES, 
        embedding_dim=EMBEDDING_DIM, 
        hidden_dim=HIDDEN_DIM
    ).to(device)

    try:
        model.load_state_dict(torch.load('hawkes_weights.pth', map_location=device))
        print("Successfully loaded trained weights from 'hawkes_weights.pth'")
    except FileNotFoundError:
        print("WARNING: 'hawkes_weights.pth' not found. Running with random weights.")

    simulate_live_market("Dataset/AMZN_2012-06-21_34200000_57600000_message_5.csv", "Dataset/AMZN_2012-06-21_34200000_57600000_orderbook_5.csv", model, device)
            



            















