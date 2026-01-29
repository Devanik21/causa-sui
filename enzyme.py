
import torch
import os

class Enzyme:
    @staticmethod
    def digest(file_path, chunk_size=4096):
        '''
        Reads a file as raw bytes and converts it to a standard tensor.
        No interpretation. No file format bias. Just raw information.
        '''
        try:
            # 1. READ RAW BYTES
            # We open in 'rb' (Read Binary) mode to ignore file formats
            with open(file_path, 'rb') as f:
                raw_data = f.read(chunk_size)
                
            if not raw_data:
                return None
                
            # 2. CONVERT TO INTEGERS (0-255)
            # Every byte is just a number. This is the universal language.
            byte_values = list(raw_data)
            
            # 3. CREATE TENSOR SPIKE
            # Shape: [1, Length] (Batch size of 1)
            tensor_spike = torch.tensor(byte_values, dtype=torch.long).unsqueeze(0)
            
            return tensor_spike
            
        except Exception as e:
            # If the file is locked by the OS, we can't eat it.
            return None
