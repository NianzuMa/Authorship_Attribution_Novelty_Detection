# pip install pyyaml
import yaml
from yaml.loader import SafeLoader
import csv
import json


class GenerateAmazonTestDataSpecificationFile:
    """
    (1) generate complete specification file
    (2) generate test trials based on each line of specification file
    """
    def __init__(self):
        self.load_config()

    def load_config(self):
        with open("SAIL_ON_amazon_review_config_Jan_27_2022.yaml", mode="r") as fin:
            config = yaml.load(fin, Loader=SafeLoader)
        # endwith
        self.config = config
        pass

    # ################################ (1) generate specification file ############################
    def generate_specification_file(self):
        csv_header = "protocol,task,test_id,red_light_level,prop_novel,seed,dist_type,difficulty," \
                     "known_source,unknown_source,n_rounds,round_size,pre_novelty_batches,feedback_max_ids," \
                     "class_level_novelty,object_level_novelty,sentiment_level_novelty,interaction_level_novelty,action_level_novelty".split(",")

        with open("all_NLT_test_specification_Jan_27_2022.csv", mode="w", encoding="utf16") as fout:
            csv_writer = csv.DictWriter(fout, fieldnames=csv_header)
            csv_writer.writeheader()

            test_id = 10000
            seed = 3000000

            # (1) novelty type
            for class_level_novelty, object_level_novelty, sentiment_level_novelty, interaction_level_novelty, action_level_novelty in self.config["novelty_type_list"]:
                # (2)
                # red_light_batch_level_indices:
                #   E: [13,14,15]
                #   M: [18,19,20,21]
                #   L: [24,25,26]
                for red_light_level in sorted(self.config["red_light_batch_level_indices"].keys()):
                    for difficulty in ["easy", "hard"]:
                        # (3) low, mid, high, flat
                        for dist_type in self.config["beta_dist_params"].keys():
                            # (4) prop_novel_list: [0.3, 0.5, 0.8]
                            for prop_novel in sorted(self.config["prop_novel_list"]):

                                test_id += 1
                                seed += 10000

                                # OND,nlt,10001,E,0.4,3621391,flat,easy,amazon_reviews,amazon_reviews,40,30,5,4,1,0,1
                                entry = {
                                    "protocol": "OND",
                                    "task": "nlt",
                                    "test_id": test_id,
                                    "red_light_level": red_light_level,
                                    "prop_novel": prop_novel,
                                    "seed": seed,
                                    "dist_type": dist_type,
                                    "difficulty": difficulty,
                                    "known_source": "amazon",
                                    "unknown_source": "amazon",
                                    "n_rounds": self.config["n_rounds"],
                                    "round_size": self.config["round_size"],
                                    "pre_novelty_batches": self.config["pre_novelty_batches"],
                                    "feedback_max_ids": self.config["n_rounds"],
                                    "class_level_novelty": class_level_novelty,
                                    "object_level_novelty": object_level_novelty,
                                    "sentiment_level_novelty": sentiment_level_novelty,
                                    "interaction_level_novelty": interaction_level_novelty,
                                    "action_level_novelty": action_level_novelty
                                }
                                csv_writer.writerow(entry)
                            #endfor
                        #endfor
                    #endfor
                #endfor
            #endfor
        #endwith


def main_generate_amazon_test_data_specification_file():
    generator = GenerateAmazonTestDataSpecificationFile()
    generator.generate_specification_file()
    pass





if __name__ == '__main__':
    main_generate_amazon_test_data_specification_file()
