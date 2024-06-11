
from datasets_classes.base import Dataset


class SciCo(Dataset):

    def get_max_num_docs(self):
        return max([self.all_data[k]["doc_ids"].apply(max).max() for k in self.all_data])

    def load(self):
        return self._load_from_hf("allenai/scico")

    def pre_process(self):
        samples = {}
        for category in self.all_data:
            samples[category] = []
            df = self.all_data[category]
            for i, row in df.iterrows():
                tokens = row['tokens']
                mentions = row['mentions']
                clusters = {}
                names = []
                mentions_to_docs = {}
                random_generator = self.common_data['random']
                all_mention_ids = random_generator.sample(range(1, 100), k=len(mentions))
                for k, (paragraph_id, start, end, cluster_id) in enumerate(mentions):
                    id_mention = all_mention_ids[k]
                    names.append(tokens[paragraph_id][start:end+1])
                    tokens[paragraph_id][start] = f'[{tokens[paragraph_id][start]}'
                    tokens[paragraph_id][end] = f'{tokens[paragraph_id][end]}]({id_mention})'
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(id_mention)
                    mentions_to_docs[id_mention] = row['doc_ids'][paragraph_id]
                paragraphs = [' '.join(par) for par in tokens]
                documents = []
                for index_paragraph, doc_id in enumerate(row['doc_ids']):
                    if len(documents) == doc_id:
                        documents.append("")
                    documents[doc_id] += f"{paragraphs[index_paragraph]}\n"
                msgs = self.get_sample2msg(documents)
                samples[category].append({
                    'sample_id': row['id'],
                    'targets': sorted([clusters[k] for k in clusters]),
                    'all_docs': documents,
                    'final_msgs': msgs
                })
        return samples



