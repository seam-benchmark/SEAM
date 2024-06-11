
from datasets_classes.mds.aspect_based import SummarizationDataset


class MultiNews(SummarizationDataset):
    def __init__(self, name, split_name):
        self.sep_ = '|||||'
        super().__init__(name, dir_path="", split_name=split_name)

    def load(self):
        mult = self._load_from_hf("multi_news")
        for split_name in mult:
            mult[split_name]["documents"] = mult[split_name]["document"].apply(lambda x: x.split(self.sep_))
        return mult

    def pre_process(self):
        all_samples = {}
        for split_name, split_data in self.all_data.items():
            results_data = []
            for sample_id, row in split_data.iterrows():
                guid = sample_id
                target = row["summary"]
                documents = row["documents"]
                msgs = self.get_sample2msg(documents)
                results_data.append({
                    'id': guid,
                    'final_msgs': msgs,
                    'target': target
                })
            all_samples[split_name] = results_data
        return all_samples




