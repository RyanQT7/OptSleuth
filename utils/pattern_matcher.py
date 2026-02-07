import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class PatternMatcher:
    """
    Enhanced Pattern Matching Module.
    
    Key Features:
    1. Supports direct pattern matching using pre-computed embeddings.
    2. Returns detailed template descriptions for better interpretability.
    3. Optimized storage structure for the template library.
    """

    def __init__(self, transformer_model: torch.nn.Module, input_feature_dim: int = 6, device: torch.device = None):
        """
        Initializes the Pattern Matcher.
        
        Args:
            transformer_model: Pre-trained Transformer encoder for feature extraction.
            input_feature_dim: Dimensionality of the input time-series features.
            device: Computation device (CPU or GPU).
        """
        self.model = transformer_model
        self.input_feature_dim = input_feature_dim
        self.model.eval()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model.to(self.device)

        self.template_library = {} 
        self.next_template_id = 0
        self.category_templates = defaultdict(list) 

        self._template_features = None
        self._template_categories = None
        self._template_descriptions = None
        self._template_ids = None

    def build_template_library(self, template_data: List[Dict]) -> None:
        """
        Constructs the template library from a list of raw data templates.
        This method extracts features for each template and stores them efficiently.
        """
        for i, template_info in enumerate(template_data):
            template_window = template_info['data'] 
            category = template_info['category']
            description = template_info.get('description', f'template_{i}')

            if template_window.ndim != 2:
                raise ValueError(f"Template data must be 2-dimensional, current dimension: {template_window.ndim}")

            T, feature_dim = template_window.shape
            if feature_dim != self.input_feature_dim:
                raise ValueError(f"Template feature dimension {feature_dim} does not match expected dimension {self.input_feature_dim}")

            input_data = torch.FloatTensor(template_window).unsqueeze(0)
            input_data = input_data.to(self.device)

            with torch.no_grad():
                template_feature = self.model.feature_extractor(input_data) 
                template_feature = template_feature.squeeze(0)

            template_id = self.next_template_id
            self.template_library[template_id] = {
                'feature': template_feature.cpu(),
                'category': category,
                'description': description
            }
            self.category_templates[category].append(template_id)
            self.next_template_id += 1

        self._build_template_matrix()

    def _build_template_matrix(self):
        """
        Compiles individual templates into a tensor matrix for batch processing.
        """
        if not self.template_library:
            return

        template_features = []
        template_categories = []
        template_descriptions = []
        template_ids = []

        for template_id, info in self.template_library.items():
            template_features.append(info['feature'])
            template_categories.append(info['category'])
            template_descriptions.append(info['description'])
            template_ids.append(template_id)

        self._template_features = torch.stack(template_features).to(self.device)
        self._template_categories = torch.tensor(template_categories, dtype=torch.long, device=self.device)
        self._template_descriptions = template_descriptions
        self._template_ids = torch.tensor(template_ids, dtype=torch.long, device=self.device)

    def _extract_device_features(self, window_data: torch.Tensor) -> torch.Tensor:
        """
        Extracts feature embeddings for the input time windows using the Transformer model.
        """
        with torch.no_grad():
            device_features = self.model.feature_extractor(window_data)
        return device_features

    def _compute_dtw_similarity(self, query_features: torch.Tensor, template_features: torch.Tensor) -> torch.Tensor:
        """
        Computes the Dynamic Time Warping (DTW) similarity between query features and template features.
        """
        if query_features.dim() == 2:
            query_np = query_features.cpu().numpy()
            template_np = template_features.cpu().numpy()

            num_queries = query_np.shape[0]
            num_templates = template_np.shape[0]
            similarities = np.zeros((num_queries, num_templates))

            for i in range(num_queries):
                for j in range(num_templates):
                    vec1 = query_np[i].flatten().reshape(-1, 1)
                    vec2 = template_np[j].flatten().reshape(-1, 1)
                    
                    distance, _ = fastdtw(vec1, vec2, dist=euclidean)
                    dtw_sim = np.exp(-0.1 * distance)
                    similarities[i, j] = dtw_sim

            return torch.tensor(similarities, device=self.device)
        else:
            batch_size, num_queries, feature_dim = query_features.shape
            num_templates = template_features.size(0)
            similarities = np.zeros((batch_size, num_queries, num_templates))

            query_np = query_features.cpu().numpy()
            template_np = template_features.cpu().numpy()

            for b in range(batch_size):
                for i in range(num_queries):
                    for j in range(num_templates):
                        vec1 = query_np[b, i].flatten().reshape(-1, 1)
                        vec2 = template_np[j].flatten().reshape(-1, 1)
                        
                        distance, _ = fastdtw(vec1, vec2, dist=euclidean)
                        dtw_sim = np.exp(-0.1 * distance)
                        similarities[b, i, j] = dtw_sim

            return torch.tensor(similarities, device=self.device)

    def match_patterns(self, window_data: torch.Tensor, return_description: bool = False) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, List[str]]]:
        """
        Main entry point for pattern matching using raw time-series data.
        """
        if not self.template_library:
            raise ValueError("Template library is empty. Please call build_template_library first.")

        window_data = window_data.to(self.device)
        num_devices, T, feature_dim = window_data.shape

        if feature_dim != self.input_feature_dim:
            raise ValueError(f"Input feature dimension {feature_dim} does not match model expectation {self.input_feature_dim}")

        device_features = self._extract_device_features(window_data)
        similarity_matrix = self._compute_dtw_similarity(device_features, self._template_features)

        device_scores, best_indices = torch.max(similarity_matrix, dim=1)
        device_categories = self._template_categories[best_indices]

        if not return_description:
            return device_scores.cpu(), device_categories.cpu()

        best_indices_cpu = best_indices.cpu().numpy()
        device_descriptions = [self._template_descriptions[idx] for idx in best_indices_cpu]
        return device_scores.cpu(), device_categories.cpu(), device_descriptions

    def match_patterns_from_features(self, device_features: torch.Tensor, return_description: bool = False) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, List[str]]]:
        """
        Performs pattern matching using pre-computed feature embeddings.
        """
        if not self.template_library:
            raise ValueError("Template library is empty. Please call build_template_library first.")

        device_features = device_features.to(self.device)

        if device_features.dim() != 2:
            raise ValueError(f"Input feature tensor must be 2-dimensional, but got {device_features.dim()}")

        num_devices, embedding_dim = device_features.shape
        template_embedding_dim = self._template_features.size(1)
        
        if embedding_dim != template_embedding_dim:
            raise ValueError(f"Input embedding dimension {embedding_dim} does not match template dimension {template_embedding_dim}")

        similarity_matrix = self._compute_dtw_similarity(device_features, self._template_features)

        device_scores, best_indices = torch.max(similarity_matrix, dim=1)
        device_categories = self._template_categories[best_indices]

        if not return_description:
            return device_scores.cpu(), device_categories.cpu()

        best_indices_cpu = best_indices.cpu().numpy()
        device_descriptions = [self._template_descriptions[idx] for idx in best_indices_cpu]
        return device_scores.cpu(), device_categories.cpu(), device_descriptions

    def get_template_by_id(self, template_id: int) -> Optional[Dict]:
        if template_id in self.template_library:
            info = self.template_library[template_id].copy()
            info['id'] = template_id
            return info
        return None

    def get_top_k_matches(self, window_data: torch.Tensor, k: int = 3) -> List[List[Dict]]:
        if not self.template_library:
            raise ValueError("Template library is empty.")

        window_data = window_data.to(self.device)
        device_features = self._extract_device_features(window_data)
        similarity_matrix = self._compute_dtw_similarity(device_features, self._template_features)

        top_k_scores, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1, largest=True, sorted=True)
        num_devices = device_features.size(0)
        top_k_matches = []

        for i in range(num_devices):
            device_matches = []
            for j in range(k):
                template_idx = top_k_indices[i, j].item()
                match_info = {
                    'score': top_k_scores[i, j].item(),
                    'category': self._template_categories[template_idx].item(),
                    'description': self._template_descriptions[template_idx],
                    'template_id': template_idx
                }
                device_matches.append(match_info)
            top_k_matches.append(device_matches)

        return top_k_matches