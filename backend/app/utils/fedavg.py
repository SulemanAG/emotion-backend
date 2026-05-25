import numpy as np
from typing import List, Union, Dict, Any

def perform_fedavg(weight_paths: List[str], sample_counts: List[int]) -> Any:
    """
    Perform Federated Averaging (FedAvg) over a list of client model weights files.
    Supports weighted averaging based on client sample counts.
    
    Formula:
    global_weight = sum(n_i * w_i) / sum(n_i)
    
    Args:
        weight_paths: List of file paths to client weights saved as .npy files.
        sample_counts: List of sample counts corresponding to each client.
        
    Returns:
        Aggregated weights in the same structure (list, dict, or array) as client weights.
    """
    if not weight_paths or not sample_counts:
        raise ValueError("Weight paths and sample counts must not be empty.")
    
    if len(weight_paths) != len(sample_counts):
        raise ValueError("The number of weight paths must match the number of sample counts.")
    
    total_samples = sum(sample_counts)
    if total_samples <= 0:
        raise ValueError("Total sample count across all clients must be greater than zero.")
        
    # Load all client weights
    client_weights = []
    for idx, path in enumerate(weight_paths):
        try:
            # Allow pickle is required since model weights might contain lists or custom dicts
            weights = np.load(path, allow_pickle=True)
            
            # If the weights object was saved as an object array wrapping a list (numpy behavior), unwrap it
            if isinstance(weights, np.ndarray) and weights.dtype == object and not isinstance(weights.tolist(), np.ndarray):
                weights = weights.tolist()
                
            client_weights.append(weights)
        except Exception as e:
            raise ValueError(f"Failed to load or parse NumPy weights file for client at index {idx} ({path}): {str(e)}")
            
    first_weight = client_weights[0]
    
    # Perform aggregation based on structure type
    if isinstance(first_weight, dict):
        return aggregate_dict(client_weights, sample_counts, total_samples)
    elif isinstance(first_weight, list):
        return aggregate_list(first_weight, client_weights, sample_counts, total_samples)
    elif isinstance(first_weight, np.ndarray):
        # In case first_weight is an object array, it might be wrapping lists or dicts.
        if first_weight.dtype == object:
            # Try to convert to list and aggregate
            try:
                unwrapped_list = [w.tolist() if isinstance(w, np.ndarray) and w.dtype == object else w for w in client_weights]
                return aggregate_list(first_weight.tolist(), unwrapped_list, sample_counts, total_samples)
            except Exception as e:
                raise TypeError(f"Object numpy array aggregation failed: {str(e)}")
        return aggregate_ndarray(client_weights, sample_counts, total_samples)
    else:
        raise TypeError(f"Unsupported weight structure type: {type(first_weight)}")

def aggregate_dict(client_weights: List[Dict[str, Any]], sample_counts: List[int], total_samples: int) -> Dict[str, Any]:
    aggregated = {}
    keys = client_weights[0].keys()
    
    for key in keys:
        # Check all clients have this key and shapes match
        key_weights = []
        for idx, client_w in enumerate(client_weights):
            if not isinstance(client_w, dict):
                raise TypeError(f"Client at index {idx} has weights type {type(client_w)}, expected dict.")
            if key not in client_w:
                raise KeyError(f"Key '{key}' is missing from client {idx} weights.")
            
            val = client_w[key]
            # Ensure value is numpy array
            if not isinstance(val, np.ndarray):
                val = np.array(val)
            key_weights.append(val)
            
        # Ensure all arrays for this key have the same shape
        shape = key_weights[0].shape
        for idx, w in enumerate(key_weights):
            if w.shape != shape:
                raise ValueError(f"Shape mismatch for key '{key}' in client {idx}: expected {shape}, got {w.shape}")
                
        # Perform weighted average
        weighted_sum = sum(sample_counts[i] * key_weights[i] for i in range(len(client_weights)))
        aggregated[key] = weighted_sum / total_samples
        
    return aggregated

def aggregate_list(first_weight_list: List[Any], client_weights: List[Any], sample_counts: List[int], total_samples: int) -> List[np.ndarray]:
    aggregated = []
    num_layers = len(first_weight_list)
    
    for i in range(num_layers):
        layer_weights = []
        for idx, client_w in enumerate(client_weights):
            if not isinstance(client_w, list):
                raise TypeError(f"Client at index {idx} has weights type {type(client_w)}, expected list.")
            if len(client_w) != num_layers:
                raise ValueError(f"Layer count mismatch in client {idx}: expected {num_layers}, got {len(client_w)}")
            
            val = client_w[i]
            if not isinstance(val, np.ndarray):
                val = np.array(val)
            layer_weights.append(val)
            
        # Check shapes
        shape = layer_weights[0].shape
        for idx, w in enumerate(layer_weights):
            if w.shape != shape:
                raise ValueError(f"Shape mismatch at layer {i} for client {idx}: expected {shape}, got {w.shape}")
                
        # Weighted average
        weighted_sum = sum(sample_counts[idx] * layer_weights[idx] for idx in range(len(client_weights)))
        aggregated.append(weighted_sum / total_samples)
        
    return aggregated

def aggregate_ndarray(client_weights: List[np.ndarray], sample_counts: List[int], total_samples: int) -> np.ndarray:
    shape = client_weights[0].shape
    for idx, w in enumerate(client_weights):
        if not isinstance(w, np.ndarray):
            w = np.array(w)
        if w.shape != shape:
            raise ValueError(f"Shape mismatch in client {idx} weights array: expected {shape}, got {w.shape}")
            
    weighted_sum = sum(sample_counts[idx] * client_weights[idx] for idx in range(len(client_weights)))
    return weighted_sum / total_samples
