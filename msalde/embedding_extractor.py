import csv
from Bio import SeqIO
from pathlib import Path
import esm
from .embedder import ProteinEmbedderFactory


class EmbeddingExtractor:
    def __init__(
            self,
            protein_embedder_factories: dict[str, type[ProteinEmbedderFactory]],
            embedder_config,
            datasets_config: dict[str, dict],
    ):
        self._datasets_config = datasets_config
        if "type" not in embedder_config:
            raise ValueError(f"Embedder type not specified in config")
        embedder_factory = protein_embedder_factories.get(embedder_config.type)
        if embedder_factory is None:
            raise ValueError(
                f"Unknown embedder type: {embedder_config.type}. "
                f"Available types: {list(protein_embedder_factories.keys())}"
            )
        self._embedder = embedder_factory.create_instance(
            embedder_config, None)
        self._output_dir = embedder_config.get("output_dir")
        self._model_name = embedder_config.get("model_name")
        
    def extract_by_fasta_file(self, dataset_name: str, fasta_file: str,
                              output_dir: str):
        sequences = [(record.id, str(record.seq))
                      for record in SeqIO.parse(fasta_file, "fasta")]
        seq_embeddings = self._embedder.embed_sequences([seq for _, seq in sequences])
        model_name = self._model_name.split("/")[-1]  # Get model name without path
        output_path = Path(output_dir) / f"{dataset_name}_{model_name}.csv"
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id"] + [f"e_{i}" for i in range(len(seq_embeddings[0]))])
            for id, embedding in zip([id for id, _ in sequences], seq_embeddings):
                writer.writerow([id] + embedding.tolist())

    def extract_by_dataset_name(self, dataset_name: str | list[str]):
        if isinstance(dataset_name, list):
            for name in dataset_name:
                self.extract_by_dataset_name(name)
            return
        dataset_config = self._datasets_config.get(dataset_name)
        if dataset_config is None:
            raise ValueError(f"Dataset '{dataset_name}' not found in config")
        fasta_file = dataset_config.get("fasta_file")
        if fasta_file is None:
            raise ValueError(f"FASTA file not specified for dataset '{dataset_name}'")
        self.extract_by_fasta_file(dataset_name, fasta_file, self._output_dir)


