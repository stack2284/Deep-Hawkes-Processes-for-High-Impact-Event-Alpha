import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import your architecture
from model import NeuralHawkesModel

def plot_model_brain(message_path, model_weights_path, device):
    print("--- Loading Data for Visualization ---")
    
    cols = ['Time', 'Type', 'OrderID', 'Size', 'Price', 'Direction']
    df = pd.read_csv(message_path, names=cols, nrows=1000)
    
    df['TimeDelta'] = df['Time'].diff().fillna(0.0)
    df['TimeDelta'] = np.clip(df['TimeDelta'], a_min=1e-9, a_max=None)
    
    def map_event(row):
        e_type, direction = row['Type'], row['Direction']
        if e_type == 1 and direction == 1: return 0
        if e_type == 1 and direction == -1: return 1
        if e_type in [2, 3]: return 2
        if e_type in [4, 5] and direction == 1: return 3 # Market Buy
        if e_type in [4, 5] and direction == -1: return 4 # Market Sell
        return 2
    df['MappedType'] = df.apply(map_event, axis=1)

    # 2. Setup Model
    NUM_EVENT_TYPES = 5
    EMBEDDING_DIM = 16
    HIDDEN_DIM = 64
    
    model = NeuralHawkesModel(NUM_EVENT_TYPES, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    prices = (df['Price'] / 10000.0).values
    mapped_types = df['MappedType'].values
    time_deltas = df['TimeDelta'].values
    times = df['Time'].values

    # 4. Run Inference to collect Intensities
    hidden_dim = model.hidden_dim
    h = torch.zeros(1, hidden_dim, device=device)
    c = torch.zeros(1, hidden_dim, device=device)

    buy_intensities = []
    sell_intensities = []

    print("--- Running Inference ---")
    with torch.no_grad():
        for idx in range(len(mapped_types)):
            if idx > 0 and idx % 50 == 0:
                h = torch.zeros(1, hidden_dim, device=device)
                c = torch.zeros(1, hidden_dim, device=device)

            event_idx = torch.tensor([[mapped_types[idx]]], dtype=torch.long, device=device)
            dt = torch.tensor([[time_deltas[idx]]], dtype=torch.float32, device=device)
            
            x = model.embedding(event_idx).squeeze(1)
            h, c, _ = model.cell(x, dt, h, c)
            
            intensities = F.softplus(model.intensity_layer(h))
            
            buy_intensities.append(intensities[0, 3].item())
            sell_intensities.append(intensities[0, 4].item())

    print("--- Rendering Plot ---")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    

    ax1.plot(times, prices, color='black', linewidth=1)
    ax1.set_ylabel("Price ($)")
    ax1.set_title("LOBSTER Microstructure & Neural Hawkes Predictions")
    ax1.grid(True, alpha=0.3)

    buy_exec_times = times[mapped_types == 3]
    buy_exec_prices = prices[mapped_types == 3]
    sell_exec_times = times[mapped_types == 4]
    sell_exec_prices = prices[mapped_types == 4]

    ax2.scatter(buy_exec_times, [1]*len(buy_exec_times), color='green', marker='^', label='True Buy Exec')
    ax2.scatter(sell_exec_times, [-1]*len(sell_exec_times), color='red', marker='v', label='True Sell Exec')
    ax2.set_yticks([-1, 1])
    ax2.set_yticklabels(["Sell Exec", "Buy Exec"])
    ax2.set_ylabel("Market Events")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    ax3.plot(times, buy_intensities, color='green', alpha=0.7, label='Predicted Buy Intensity (\u03BB)')
    ax3.plot(times, sell_intensities, color='red', alpha=0.7, label='Predicted Sell Intensity (\u03BB)')
    ax3.set_ylabel("Intensity")
    ax3.set_xlabel("Time (Seconds past midnight)")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Update these paths to point to your files
    MSG_PATH = "Dataset/AMZN_2012-06-21_34200000_57600000_message_5.csv"
    WEIGHTS_PATH = "hawkes_weights.pth"
    
    plot_model_brain(MSG_PATH, WEIGHTS_PATH, device)