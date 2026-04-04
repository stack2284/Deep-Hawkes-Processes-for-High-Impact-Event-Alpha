import torch 
from torch.utils.data import Dataset , DataLoader 
import numpy as np 
import pandas as pd 

class LobsterHawkesDataset(Dataset) : 
    def __init__(self , message_csv_path , seq_len = 100):
        """
            Parses a lobster message file into contineous time tensors. 
            seq_len how many events to look at per batch sequence 
        """
        self.seq_len = seq_len 
        self.data = self._parse_lobster(message_csv_path)
    
    def _parse_lobster (self , filepath , cols = ['Time', 'Type', 'OrderID', 'Size', 'Price', 'Direction']) : 
        """
            here cols is default as i checked dataset 
            if yo uare using anyother data source please initilize this in init 
        """

        df = pd.read_csv(filepath , names=cols)

        """
            timedelta is just change instead of timestamps 
        """
        df['TimeDelta'] = df['Time'].diff() 
        df['TimeDelta'] = df['TimeDelta'].fillna(0.0)

        """
            adding this to make sure multiple orders are handled as many orders at the same time stamp causes 
            improper shuffling 
        """
        epsilon = 1e-9 
        ### time change is between eps and infinity now 
        df['TimeDelta'] = np.clip(df['TimeDelta'] , a_min=epsilon , a_max=None) 


        # LOBSTER: 1(add), 2/3(cancel), 4/5(execute)
        # Our Vocab: 0(Add), 1(Cancel), 2(Execute)
        def map_event(row) : 
            e_type = row['Type'] 
            direction = row['Direction'] 

            if e_type == 1 and direction == 1 : return 0 
            if e_type == 1 and direction == -1 : return -1 

            if e_type in [2 , 3] : 
                return 2
            if e_type in [4 , 5] : 
                if direction == 1 : return 3 
                else : return 4 
            return 2 
        
        df['MappedType'] = df.apply(map_event , axis=1) 

        time_deltas = df['TimeDelta'].values.astype(np.float32)
        event_type = df['MappedType'].values.astype(np.int64)

        num_chunks = len(df) // self.seq_len 

        dataset = [] 

        for i in range(num_chunks) : 
                startind = i * self.seq_len 
                endind = startind + self.seq_len 
                chunk_td = torch.tensor(time_deltas[startind:endind])
                chunk_ev = torch.tensor(event_type[startind:endind])
                dataset.append({
                        'time_deltas' : chunk_td,
                        'event_types' : chunk_ev,
                    })
        
        print(f"created dataset {len(dataset)}")
        return dataset
    
    def __len__(self):
         return len(self.data) 

    def __getitem__(self, index):
         return self.data[index] 


def Hawkes_collate_fn (batch) : 
    # diffrent databases might need padding but mine 
    # as it is chucnked perfectely 
    # not needed for this dataset but future use 
    time_deltas = torch.stack([item['time_deltas'] for item in batch]) 
    event_types = torch.stack([item['event_types'] for item in batch])
    mask = torch.ones_like(event_types , dtype=torch.bool)

    return {
            'time_deltas' : time_deltas, 
            'event_types' : event_types, 
            'mask' : mask 
        }


import os
import glob

def Test (): 
    files = glob.glob("Dataset/*message*.csv")
    if files:
        SAMPLE_FILE = files[0]


    try: 
        dataset = LobsterHawkesDataset(SAMPLE_FILE , seq_len=50)
        dataloader = DataLoader(dataset , batch_size=32 , shuffle=True, collate_fn=Hawkes_collate_fn)
        batch = next(iter(dataloader)) 
        print(f"Batch Time Deltas: {batch['time_deltas'].shape}") # Expected: [32, 50]
        print(f"Batch Event Types: {batch['event_types'].shape}") # Expected: [32, 50]
        print(f"Mean Time Delta:   {batch['time_deltas'].mean().item():.6f} seconds")
    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__" : 
    Test()
