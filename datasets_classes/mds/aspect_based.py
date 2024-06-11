from datasets import load_dataset
import pandas as pd
from datasets_classes.mds.summarization import SummarizationDataset


class AspectBasedSummarizationDataset(SummarizationDataset):

    def load(self):
        data_files_text, data_files_paths = self._read_files('jsonl')
        openasp = load_dataset('json', data_files=data_files_paths)
        df_openasp = {}
        for split_name, split_data in openasp.items():
            df_openasp[split_name] = pd.DataFrame(split_data)
        return df_openasp

    def pre_process(self):
        all_samples = {}
        for split_name, split_data in self.all_data.items():
            results_data = []
            for sample_id, row in split_data.iterrows():
                guid = row['guid']
                aspect_label = row['aspect_label']
                src_docs = [' '.join(doc["text"]) for doc in row['documents']]
                msgs = self.get_sample2msg(src_docs, aspect_label=aspect_label)
                results_data.append({
                    'guid': guid,
                    'final_msgs': msgs,
                    'documents': src_docs,
                    'aspect_label': aspect_label,
                    'target': '\n'.join(row['summary_text'])
                })
            all_samples[split_name] = results_data
        return all_samples

    def get_sample(self, instructions, documents, **kwargs):
        documents = self.get_shuffled_documents(documents)
        aspect_label = kwargs.get("aspect_label", "")
        doc_strings = "\n".join([f"Document {j}: ```{doc}```" for j, doc in enumerate(documents, start=1)])
        target = kwargs.get("target", "")
        sample = f"Aspect label: {aspect_label}\n\nThe provided documents:\n{doc_strings}\nAspect-based summary: {target}"
        return sample
