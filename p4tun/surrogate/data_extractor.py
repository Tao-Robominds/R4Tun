"""
Data Extraction Module for Surrogate Model Training

Extracts parameter vectors and scores from existing BO JSON logs
in p4tun/bo/results/ directory.
"""

import os
import json
import glob
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ExtractedData:
    """Container for extracted training data."""
    X: np.ndarray  # Parameter vectors (n_samples, n_features)
    y: np.ndarray  # Scores (n_samples,)
    param_names: List[str]  # Parameter names
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_samples(self) -> int:
        return len(self.y)
    
    @property
    def n_features(self) -> int:
        return self.X.shape[1] if len(self.X.shape) > 1 else 0
    
    def __repr__(self) -> str:
        return f"ExtractedData(n_samples={self.n_samples}, n_features={self.n_features})"


class DataExtractor:
    """
    Extract training data from BO JSON logs.
    
    The BO logs contain two types of files:
    - Main results: {tunnel_id}_{stage}_{timestamp}.json 
      Contains best_params, all_scores, convergence
    - History: {tunnel_id}_{stage}_{timestamp}_history.json
      Contains full parameter vectors for each evaluation
    """
    
    def __init__(
        self,
        results_dir: str = 'p4tun/bo/results',
        stage: str = 'detection',
        metric: str = 'mIoU',
    ):
        """
        Initialize DataExtractor.
        
        Args:
            results_dir: Directory containing BO results JSON files
            stage: Stage to extract data for ('detection' or 'sam')
            metric: Metric to use as target ('mIoU', 'OA', 'F1')
        """
        self.results_dir = results_dir
        self.stage = stage
        self.metric = metric
        
        # Define key parameters for each stage based on sensitivity analysis
        self.key_params = {
            'detection': [
                'binary_threshold',
                'hough_oblique_threshold',
                'angle_positive_min',
                'angle_positive_max',
                'hough_vertical_threshold',
                'dilation_kernel_size',
                'dilation_iterations',
                'hough_oblique_min_length',
                'hough_oblique_max_gap',
                'hough_horizontal_threshold',
                'hough_horizontal_min_length',
                'hough_horizontal_max_gap',
                'merge_distance_threshold',
            ],
            'sam': [
                'segment_width',
                'k_height',
                'ab_height',
                'angle_deg',
                'k_outer_ring',
                'k_middle_ring',
                'k_inner_ring',
                'k_center_ring',
                'ab_outer_ring',
                'ab_middle_ring',
                'ab_inner_ring',
                'ab_center_ring',
                'ab_fine_spacing',
                'ab_ultra_fine',
                'ab_edge_ring',
                'ab_edge_spacing',
                'ab_level_1',
                'ab_level_2',
                'ab_level_3',
                'ab_level_4',
                'ab_level_5',
                'ab_level_6',
                'ab_level_7',
                'k_mask_width',
                'k_mask_height_pos',
                'k_mask_height_neg',
                'ab_mask_width',
                'ab_mask_height',
                'min_quality_threshold',
                'padding',
                'crop_margin',
            ],
        }
    
    def find_log_files(self, tunnel_id: Optional[str] = None) -> List[str]:
        """
        Find all relevant log files.
        
        Args:
            tunnel_id: Optional tunnel ID to filter by
            
        Returns:
            List of file paths for history JSON files
        """
        pattern = f'*_{self.stage}_*_history.json'
        if tunnel_id:
            pattern = f'{tunnel_id}_{self.stage}_*_history.json'
        
        files = glob.glob(os.path.join(self.results_dir, pattern))
        return sorted(files)
    
    def extract_from_history(self, history_file: str) -> Optional[ExtractedData]:
        """
        Extract training data from a history JSON file.
        
        Args:
            history_file: Path to history JSON file
            
        Returns:
            ExtractedData object or None if extraction fails
        """
        try:
            with open(history_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {history_file}: {e}")
            return None
        
        # Extract history entries
        history = data.get('history', [])
        if not history:
            print(f"No history entries in {history_file}")
            return None
        
        # Get parameter names from first entry
        first_entry = history[0]
        if 'params' not in first_entry:
            print(f"No params in history entries: {history_file}")
            return None
        
        param_names = list(first_entry['params'].keys())
        
        # Extract X (parameters) and y (scores)
        X_list = []
        y_list = []
        
        for entry in history:
            params = entry.get('params', {})
            score = entry.get('score')
            
            if params and score is not None:
                # Build parameter vector
                param_vec = [params.get(name, 0.0) for name in param_names]
                X_list.append(param_vec)
                y_list.append(score)
        
        if not X_list:
            print(f"No valid entries extracted from {history_file}")
            return None
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Metadata
        metadata = {
            'tunnel_id': data.get('tunnel_id', 'unknown'),
            'stage': data.get('stage', self.stage),
            'metric': data.get('metric', self.metric),
            'best_score': data.get('best_score'),
            'source_file': history_file,
        }
        
        return ExtractedData(X=X, y=y, param_names=param_names, metadata=metadata)
    
    def extract_from_main_file(self, main_file: str) -> Optional[ExtractedData]:
        """
        Extract best parameters and score from main results file.
        Useful for single-point extraction when history is not available.
        
        Args:
            main_file: Path to main results JSON file
            
        Returns:
            ExtractedData with single sample or None
        """
        try:
            with open(main_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {main_file}: {e}")
            return None
        
        best_params = data.get('best_params', {})
        best_score = data.get('best_score')
        
        if not best_params or best_score is None:
            return None
        
        param_names = list(best_params.keys())
        X = np.array([[best_params[name] for name in param_names]])
        y = np.array([best_score])
        
        metadata = {
            'tunnel_id': data.get('tunnel_id', 'unknown'),
            'stage': data.get('stage', self.stage),
            'metric': data.get('metric', self.metric),
            'best_score': best_score,
            'source_file': main_file,
        }
        
        return ExtractedData(X=X, y=y, param_names=param_names, metadata=metadata)
    
    def extract_all(
        self,
        tunnel_ids: Optional[List[str]] = None,
        filter_negative: bool = True,
        min_score: float = 0.0,
    ) -> ExtractedData:
        """
        Extract data from all available log files.
        
        Args:
            tunnel_ids: Optional list of tunnel IDs to include
            filter_negative: Remove samples with negative/zero scores
            min_score: Minimum score threshold
            
        Returns:
            Combined ExtractedData from all sources
        """
        all_X = []
        all_y = []
        all_metadata = {
            'source_files': [],
            'tunnel_ids': set(),
            'stage': self.stage,
            'metric': self.metric,
        }
        param_names = None
        
        # Find history files
        if tunnel_ids:
            files = []
            for tid in tunnel_ids:
                files.extend(self.find_log_files(tid))
        else:
            files = self.find_log_files()
        
        print(f"Found {len(files)} history files for stage '{self.stage}'")
        
        for file_path in files:
            extracted = self.extract_from_history(file_path)
            if extracted is None:
                continue
            
            # Initialize param_names from first successful extraction
            if param_names is None:
                param_names = extracted.param_names
            elif extracted.param_names != param_names:
                print(f"Skipping {file_path}: param names mismatch")
                continue
            
            all_X.append(extracted.X)
            all_y.append(extracted.y)
            all_metadata['source_files'].append(file_path)
            all_metadata['tunnel_ids'].add(extracted.metadata.get('tunnel_id'))
        
        if not all_X:
            raise ValueError(f"No data extracted for stage '{self.stage}'")
        
        # Combine all data
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        
        # Filter samples
        if filter_negative:
            mask = y > min_score
            X = X[mask]
            y = y[mask]
            print(f"Filtered to {len(y)} samples with score > {min_score}")
        
        all_metadata['tunnel_ids'] = list(all_metadata['tunnel_ids'])
        all_metadata['n_total_samples'] = len(y)
        
        print(f"Extracted {len(y)} total samples with {len(param_names)} parameters")
        
        return ExtractedData(X=X, y=y, param_names=param_names, metadata=all_metadata)
    
    def extract_key_parameters(
        self,
        data: ExtractedData,
        param_subset: Optional[List[str]] = None,
    ) -> ExtractedData:
        """
        Extract only the most sensitive parameters.
        
        Args:
            data: Full extracted data
            param_subset: Optional list of parameter names to include.
                         If None, uses predefined key_params for the stage.
                         
        Returns:
            ExtractedData with reduced feature set
        """
        if param_subset is None:
            param_subset = self.key_params.get(self.stage, data.param_names)
        
        # Find indices of requested parameters
        indices = []
        filtered_names = []
        for name in param_subset:
            if name in data.param_names:
                idx = data.param_names.index(name)
                indices.append(idx)
                filtered_names.append(name)
        
        if not indices:
            print("Warning: No matching parameters found, returning original data")
            return data
        
        X_filtered = data.X[:, indices]
        
        metadata = data.metadata.copy()
        metadata['original_n_features'] = data.n_features
        metadata['filtered_params'] = filtered_names
        
        print(f"Reduced from {data.n_features} to {len(filtered_names)} parameters")
        
        return ExtractedData(
            X=X_filtered,
            y=data.y,
            param_names=filtered_names,
            metadata=metadata,
        )
    
    def save_extracted_data(self, data: ExtractedData, output_path: str):
        """
        Save extracted data to JSON for later use.
        
        Args:
            data: ExtractedData to save
            output_path: Output file path
        """
        save_dict = {
            'X': data.X.tolist(),
            'y': data.y.tolist(),
            'param_names': data.param_names,
            'metadata': data.metadata,
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        print(f"Saved extracted data to {output_path}")
    
    def load_extracted_data(self, input_path: str) -> ExtractedData:
        """
        Load previously saved extracted data.
        
        Args:
            input_path: Path to saved JSON file
            
        Returns:
            ExtractedData object
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        return ExtractedData(
            X=np.array(data['X']),
            y=np.array(data['y']),
            param_names=data['param_names'],
            metadata=data.get('metadata', {}),
        )


def main():
    """Example usage of DataExtractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract training data from BO logs')
    parser.add_argument('--stage', '-s', default='detection',
                       choices=['detection', 'sam'],
                       help='Stage to extract data for')
    parser.add_argument('--results-dir', default='p4tun/bo/results',
                       help='Directory containing BO results')
    parser.add_argument('--output', '-o', default=None,
                       help='Output file path')
    parser.add_argument('--tunnel', '-t', default=None,
                       help='Specific tunnel ID to extract')
    parser.add_argument('--key-only', action='store_true',
                       help='Extract only key sensitive parameters')
    
    args = parser.parse_args()
    
    extractor = DataExtractor(
        results_dir=args.results_dir,
        stage=args.stage,
    )
    
    tunnel_ids = [args.tunnel] if args.tunnel else None
    data = extractor.extract_all(tunnel_ids=tunnel_ids)
    
    if args.key_only:
        data = extractor.extract_key_parameters(data)
    
    print(f"\nExtracted data summary:")
    print(f"  Samples: {data.n_samples}")
    print(f"  Features: {data.n_features}")
    print(f"  Score range: [{data.y.min():.3f}, {data.y.max():.3f}]")
    print(f"  Parameters: {data.param_names[:5]}...")
    
    if args.output:
        extractor.save_extracted_data(data, args.output)


if __name__ == '__main__':
    main()
