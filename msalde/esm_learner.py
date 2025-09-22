import gc
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

from .model import ModelPrediction, Variant
from .learner import Learner, LearnerFactory
from typing import Optional


def print_elapsed(operation: str, start: float, end: float):
    elapsed = end - start
    print(f"{operation} took {elapsed:.2f} seconds")

def get_esm_model_and_tokenizer(base_model_name: str):
    """
    We garbage collect previous model/tokenizer if any to free memory
    on small memory systems.
    We assume that caller has dropped references to previous model/tokenizer.
    """
    gc.collect()
    # torch.cuda.empty_cache()
    esm_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    esm_model = AutoModel.from_pretrained(base_model_name)
    return esm_model, esm_tokenizer


class ESM2RandomForestLearnerInternal(nn.Module):

    def __init__(self,
                 base_model,
                 num_layers_to_unfreeze: int,
                 use_pooling: bool):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_pooling = use_pooling
        self._base_model = base_model
        self._regression_head = nn.Sequential(
            nn.Linear(self._base_model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output a single value for regression
        ).to(self._device)

    def forward(self, input_ids, attention_mask):
        outputs = self._base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token representation
        # cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)

        # Get last hidden state
        embeddings = outputs.last_hidden_state
        # Apply pooling if requested
        if self._use_pooling:
            # Mean pooling (excluding special tokens)
            embeddings = torch.sum(
                embeddings * attention_mask.unsqueeze(-1), dim=1
            ) / torch.sum(attention_mask, dim=1, keepdim=True)
        else:
            # Use CLS token
            embeddings = embeddings[:, 0]

        return self._regression_head(embeddings)


class ESM2MLPLearnerHelper(nn.Module):

    def __init__(self,
                 base_model,
                 use_pooling: bool):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_pooling = use_pooling
        self._base_model = base_model
        self._regression_head = nn.Sequential(
            nn.Linear(self._base_model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output a single value for regression
        ).to(self._device)

    def forward(self, input_ids, attention_mask):
        outputs = self._base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token representation
        # cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)

        # Get last hidden state
        embeddings = outputs.last_hidden_state
        # Apply pooling if requested
        if self._use_pooling:
            # Mean pooling (excluding special tokens)
            embeddings = torch.sum(
                embeddings * attention_mask.unsqueeze(-1), dim=1
            ) / torch.sum(attention_mask, dim=1, keepdim=True)
        else:
            # Use CLS token
            embeddings = embeddings[:, 0]

        return self._regression_head(embeddings)


#class ESM2RandomForestLearner(Learner):
class ESM2Learner(Learner):

    def __init__(self, input_dim: Optional[int] = None,
                 random_state1: Optional[int] = None,
                 random_state2: Optional[int] = None,
                 model=None,
                 base_model=None,
                 tokenizer=None,
                 num_layers_to_unfreeze: int = 2,
                 batch_size: int = None,
                 num_epochs: int = None):  # hidden_size depends on the ESM2 variant
        super().__init__(input_dim, random_state1, random_state2)
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)
        # Freeze all layers first
        for param in base_model.parameters():
            param.requires_grad = False

        # Unfreeze top N layers (e.g., last 2 transformer blocks)
        encoder_layers = base_model.encoder.layer
        for layer in encoder_layers[-num_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True

    def fit_model(
        self,
        variants: list[Variant],
        scores: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
    ) -> None:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                             self._model.parameters()), lr=1e-4)
        loss_fn = nn.MSELoss()

        inputs = self._tokenizer([v.sequence for v in variants],
                                 padding=True, return_tensors="pt").to(
            self._device
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)
        scores_tensor = torch.tensor(scores, dtype=torch.float32).to(self._device)
        dataset = TensorDataset(input_ids, attention_mask, scores_tensor)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        self._model.train()
        for epoch in range(self._num_epochs):
            print(f"Starting epoch {epoch+1}/{self._num_epochs}")
            total_loss = 0
            for batch in loader:
                ids, mask, target = [x.to(self._device) for x in batch]
                optimizer.zero_grad()
                start = time.perf_counter()
                output = self._model(ids, mask)
                end = time.perf_counter()
                print_elapsed("forward pass", start, end)
                loss = loss_fn(output, target)
                start = time.perf_counter()
                loss.backward()
                start = time.perf_counter()
                print_elapsed("backward pass", start, end)
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    def predict(
        self,
        variants: list[Variant],
    ) -> list[ModelPrediction]:

        # Tokenize
        encoded = self._tokenizer([v.sequence for v in variants],
                                  padding=True,
                                  return_tensors="pt").to(self._device)
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        self._model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            predictions = self._model(input_ids, attention_mask)
        end = time.perf_counter()
        print_elapsed("predictions: " + str(len(variants)), start, end)
        predictions = predictions.detach().squeeze().cpu().numpy()
        print("Predicted values:", predictions)
        variant_scores = zip(variants, predictions)
        return [ModelPrediction(variant_id=v.id, score=score)
                for v, score in variant_scores]


class ESM2RandomForestLearnerFactory(LearnerFactory):
    """
    Factory class for creating RidgeLearner instances.
    This is a placeholder for the actual implementation.
    """
    def create_instance(self, **kwargs) -> Learner:
        """
        Create a RidgeLearner instance with the given parameters.
        """
        base_model_name = kwargs.pop("base_model_name")
        use_pooling = kwargs.pop("use_pooling")
        # hidden_size = kwargs.pop("hidden_size")
        base_model, tokenizer = get_esm_model_and_tokenizer(
            base_model_name)
        model = ESM2RandomForestLearnerInternal(
            base_model, use_pooling)

        return ESM2Learner(model=model,
                           base_model=base_model,
                           tokenizer=tokenizer,
                           **kwargs)


class ESM2MLPLearnerFactory(LearnerFactory):
    """
    Factory class for creating RidgeLearner instances.
    This is a placeholder for the actual implementation.
    """
    def create_instance(self, **kwargs) -> Learner:
        """
        Create a RidgeLearner instance with the given parameters.
        """

        base_model_name = kwargs.pop("base_model_name")
        use_pooling = kwargs.pop("use_pooling")
        # hidden_size = kwargs.pop("hidden_size")
        base_model, tokenizer = get_esm_model_and_tokenizer(
            base_model_name)
        model = ESM2MLPLearnerHelper(
            base_model, use_pooling)

        return ESM2Learner(model=model,
                           base_model=base_model,
                           tokenizer=tokenizer,
                           **kwargs)

