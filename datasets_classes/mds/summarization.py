from datasets_classes.base import Dataset
from rouge_score import rouge_scorer


class SummarizationDataset(Dataset):
    def __init__(self, name, dir_path, split_name):
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeSU'], use_stemmer=True)
        super().__init__(name, dir_path, split_name)
