from datasets_classes.base import Dataset
import json
import numpy as np
import re


class FuseReviews(Dataset):
    TOTAL_REVIEWS = 8

    def get_max_num_docs(self):
        return self.TOTAL_REVIEWS

    def load(self):
        return self._load_from_hf("lovodkin93/FuseReviews")

    @staticmethod
    def _extract_highlights(highlighted_rev, highlight_start_tkn, highlight_end_tkn):
        pattern = fr'{highlight_start_tkn}(.*?){highlight_end_tkn}'
        return re.findall(pattern, highlighted_rev, re.DOTALL)

    def _make_demo(self, instance, prompt_dict, with_CoT, is_test):
        instance_prompt = prompt_dict["prompt_structure"]
        instance_prompt = instance_prompt.replace("{INST}", prompt_dict["instructions"])  # add instructions
        instance_prompt = instance_prompt.replace("{H_REV}", f"\n".join(
            [f"{rev_name}: {rev}" for rev_name, rev in instance["reviews"].items()]))  # add highlighted reviews
        instance_prompt = instance_prompt.replace("{HS}", prompt_dict['highlight_start']).replace("{HE}", prompt_dict[
            'highlight_end'])

        # add list of highlights
        highlights_to_string_dict = {rev_name: "\n ".join([f"{i + 1}. {highlight}" for i, highlight in enumerate(
            self._extract_highlights(highlighted_rev, "{HS}", "{HE}"))]) for rev_name, highlighted_rev in
                                     instance["reviews"].items()}
        highlights_to_string = "\n".join(
            f"{rev_name}:\n {highlight_lst}" for rev_name, highlight_lst in highlights_to_string_dict.items())
        instance_prompt = instance_prompt.replace("{HIGHLIGHTS}", highlights_to_string)

        # add CoT
        if with_CoT:
            if not is_test:
                alignments_to_string_lst = ["Spans " + " ; ".join(
                    [f'{",".join([str(i + 1) for i in highlights])} ({rev_name})' for rev_name, highlights in
                     alignment['alignments'].items() if
                     highlights]) + f" are combined to form sentence {j + 1}: {alignment['output_sentence']}" for
                                            j, alignment in enumerate(instance['highlights_output_alignment'])]
            alignments_to_string = "\n".join(alignments_to_string_lst) if not is_test else ""
            instance_prompt = instance_prompt.replace("{COT}", prompt_dict["cot_structure"])
            instance_prompt = instance_prompt.replace("{COT_STEPS}", alignments_to_string)
            if is_test:
                instance_prompt = instance_prompt.replace("So, the answer is: {A}", "")
        else:
            instance_prompt = instance_prompt.replace("{COT}", "")

        # add answer for demonstrations
        answer = instance['output'] if not is_test else ""
        instance_prompt = instance_prompt.replace("{A}", answer)
        return instance_prompt.strip(), highlights_to_string

    @staticmethod
    def _add_highlights(text, highlights, highlight_start_tkn, highlight_end_tkn):
        if not highlights:
            return text
        highlights = sorted(highlights, key=lambda x: x[0])
        highlighted_text = text[:highlights[0][0]]  # start with the text until first highlight

        for i, span in enumerate(highlights):
            end_idx_non_highlighted = highlights[i + 1][0] if i < len(highlights) - 1 else len(
                text)  # if not final highlight - next non-highlighted span's end idx is the start of the next highlight, otherwise - it is the end of the doc
            addition_txt = highlight_start_tkn + text[span[0]:span[1]] + highlight_end_tkn + text[span[
                                                                                                      1]:end_idx_non_highlighted]
            highlighted_text += addition_txt

        # make sure the removal of the highlights yields the original text
        assert highlighted_text.replace(highlight_start_tkn, "").replace(highlight_end_tkn, "") == text
        return highlighted_text

    def _generate_prompts(self, dataset, prompt_dict):

        used_demos = []
        head_prompt_reg, head_prompt_CoT = "", ""
        # add demonstrations
        for train_item in prompt_dict["demos"]:
            used_demos.append(train_item)

            # get regular prompt
            curr_prompt_demo_reg, _ = self._make_demo(
                instance=train_item, prompt_dict=prompt_dict, with_CoT=False,
                is_test=False
            )
            head_prompt_reg += curr_prompt_demo_reg
            head_prompt_reg += prompt_dict["demo_sep"]

            # get CoT prompt
            curr_prompt_demo_CoT, _ = self._make_demo(
                instance=train_item, prompt_dict=prompt_dict, with_CoT=True,
                is_test=False
            )
            head_prompt_CoT += curr_prompt_demo_CoT
            head_prompt_CoT += prompt_dict["demo_sep"]

        # add actual instances
        final_prompts_reg, final_prompts_CoT, all_highlights = [], [], []
        for instance in dataset:
            all_reviews = {f"review_{rev_i}": self._add_highlights(text=instance[f'review_{rev_i}_text'],
                                                                   highlights=json.loads(instance[f'review_{rev_i}_offsets']),
                                                                   highlight_start_tkn="{HS}",
                                                                   highlight_end_tkn="{HE}") for rev_i in
                           range(self.TOTAL_REVIEWS)}
            # shuffle reviews
            shuffled_reviews = {f"review_{i}": all_reviews[f"review_{self.shuffled_doc_ids[i]}"]
                                for i in range(self.TOTAL_REVIEWS)}
            # get regular prompt
            curr_inst_prompt_reg, highlights = self._make_demo(instance={"reviews": shuffled_reviews},
                                                               prompt_dict=prompt_dict, with_CoT=False,
                                                               is_test=True
                                                               )
            full_prompt_reg = head_prompt_reg + curr_inst_prompt_reg
            final_prompts_reg.append(full_prompt_reg)
            all_highlights.append(highlights)

            # get CoT prompt
            curr_inst_prompt_CoT, _ = self._make_demo(instance={"reviews": all_reviews}, prompt_dict=prompt_dict,
                                                      with_CoT=True, is_test=True
                                                      )
            full_prompt_CoT = head_prompt_CoT + curr_inst_prompt_CoT
            final_prompts_CoT.append(full_prompt_CoT)
        return final_prompts_reg, final_prompts_CoT, all_highlights

    @staticmethod
    def _get_all_review_side_alignments(row):
        review_names = sorted([elem.replace("_offsets", "") for elem in row.keys() if elem.endswith("offsets")], key=lambda x: int(x.replace("review_", "")))
        all_alignments = []
        for review_name in review_names:
            review_alignments = json.loads(row[f"{review_name}_alignments"])
            all_alignments += [{"review_name" : review_name,
                                "review_align_dict" : align["review"],
                                "review_span_text" : align["review"]['text']} for align in review_alignments]
        unique_jsons = set([json.dumps(align) for align in all_alignments]) # take unique spans
        return [json.loads(j)['review_span_text'] for j in unique_jsons]

    def get_prompt(self):
        random_generator = self.common_data['random']
        if self.cur_prompt is None:
            random_prompt = random_generator.choice(self.all_prompts['prompts'])
            selected_demonstrations = random_generator.sample(self.all_prompts['demos'], k=self.common_data['num_demos']) # no replacements sampling
            prompt_dict = self.all_prompts.copy()
            prompt_dict["instructions"] = random_prompt["instructions"]
            prompt_dict["demos"] = selected_demonstrations
            prompt_dict.pop("prompts")
            self.cur_prompt = prompt_dict
        return self.cur_prompt

    def pre_process(self):
        prompt = self.get_prompt()
        all_samples = {}
        for split_name, split_data in self.all_data.items():
            df_to_list = [*split_data.T.to_dict().values()]
            all_prompts_reg, all_prompts_CoT, highlights = self._generate_prompts(
                dataset=df_to_list, prompt_dict=prompt)
            results_data = []
            for sample_id, row in split_data.iterrows():
                if split_name != 'test':
                    all_review_side_alignments = self._get_all_review_side_alignments(row)
                else:
                    all_review_side_alignments = []
                messages = [
                    {"role": "user", "content": all_prompts_reg[sample_id]}]
                results_data.append({
                    'guid': sample_id,
                    'final_msgs': messages,
                    'highlights_concat': highlights[sample_id],
                    'review_side_alignments': sorted(all_review_side_alignments)
                })
            all_samples[split_name] = results_data
        return all_samples
