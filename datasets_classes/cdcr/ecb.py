
from datasets_classes.base import Dataset
import json
import pandas as pd


class ECB(Dataset):

    def get_max_num_docs(self):
        return 1  # will update this later

    def load(self):
        data_text, data_paths = self._read_files('json')
        data_as_json = {}
        for file_name, content in data_text.items():
            data_as_json[file_name] = json.loads(content)
        return data_as_json

    def update_shuffled_doc_ids(self, processed_data):
        self.max_num_docs = max([max([processed_data[k][category]["docs"].apply(len).max()
                                      for category in processed_data[k]])
                                 for k in processed_data])
        self.shuffled_doc_ids = self.common_data["random"].sample(range(self.max_num_docs), k=self.max_num_docs)

    def pre_process(self):
        processed_data = self._mark_entities_and_events()
        self.update_shuffled_doc_ids(processed_data)
        all_samples = {}
        for split_name in processed_data:
            for category in processed_data[split_name]:
                df = processed_data[split_name][category]
                data = []
                for _, row in df.iterrows():
                    sorted_docs = sorted(row['docs'])
                    msgs = self.get_sample2msg(sorted_docs)
                    data.append({
                        'topic_id': row['topic_id'],
                        'final_msgs': msgs,
                        'targets': sorted([list(val) for k, val in row['targets'].items()]),
                        'documents': sorted_docs
                    })
                all_samples[f"{split_name}_{category}"] = data
        return all_samples

    def _mark_entities_and_events(self):
        processed_data = {}
        for split_name in ["train", "dev", "test"]:
            processed_data[split_name] = {'events': {}, 'entities': {}}
            for category in processed_data[split_name]:
                all_out = self._process_single_dataset(category, split_name)
                processed_data[split_name][category] = pd.DataFrame(all_out)
        return processed_data

    def _process_single_dataset(self, category, split_name):
        df = pd.DataFrame(self.all_data[f'{split_name}_{category}'])
        df["mention_id"] = -1

        group_by_topic = df.groupby("topic").groups
        topics_to_docs = {}
        topics_to_clusters = {}
        for topic_id, indices in group_by_topic.items():
            df_topic = df.loc[indices]
            mention_ids = list(range(len(df_topic)))
            random_generator = self.common_data['random']
            shuffle_indices = random_generator.sample(mention_ids, k=len(mention_ids))
            df.loc[indices, "mention_id"] = shuffle_indices
            topics_to_docs[topic_id] = set(df_topic['doc_id'].unique())
            topics_to_clusters[topic_id] = set(df_topic['cluster_id'].unique())
        cluster_to_mentions = {}
        for cluster_id, indices in df.groupby("cluster_id").groups.items():
            cluster_to_mentions[cluster_id] = list(df.loc[indices]["mention_id"])

        docs_df = {doc_id: pd.DataFrame(self.all_data[split_name][doc_id],
                                        columns=["sent_id", "token_id", "token_text", "is_annotated", "pos"])
                   for doc_id in df['doc_id'].unique()}

        for _, row in df.iterrows():
            doc_id = row['doc_id']
            start = row['tokens_ids'][0]
            end = row['tokens_ids'][-1]
            doc_df = docs_df[doc_id]
            df_index_start = doc_df[doc_df["token_id"] == start].index[0]
            text = doc_df.iloc[df_index_start]["token_text"]
            doc_df.loc[df_index_start, "token_text"] = f"[{text}"

            df_index_end = doc_df[doc_df["token_id"] == end].index[0]
            text = doc_df.iloc[df_index_end]["token_text"]
            doc_df.loc[df_index_end, "token_text"] = f"{text}]({row['mention_id']})"

        all_docs_str = {}
        for doc_id in docs_df:
            annotated_span_ids = docs_df[doc_id].loc[docs_df[doc_id]["is_annotated"], "token_id"]
            start, end = (annotated_span_ids.min(), annotated_span_ids.max())
            doc_to_str = docs_df[doc_id].loc[docs_df[doc_id]["token_id"].isin(range(start, end + 1))]
            sorted_by_token_id = doc_to_str.sort_values("token_id")
            doc_str = " ".join(sorted_by_token_id["token_text"])
            all_docs_str[doc_id] = doc_str

        all_out = self._prepare_output(all_docs_str, cluster_to_mentions, topics_to_docs, topics_to_clusters)
        return all_out

    @staticmethod
    def _prepare_output(all_docs_str, cluster_mentions, topics_to_docs, topics_to_clusters):
        all_out = []
        for topic_id in topics_to_clusters:
            docs = [all_docs_str[doc_id] for doc_id in topics_to_docs[topic_id]]
            clusters = topics_to_clusters[topic_id]
            targets = {cluster_id: set(cluster_mentions[cluster_id]) for cluster_id in clusters}
            all_out.append({"topic_id": topic_id, "targets": targets, "docs": docs})
        return all_out
