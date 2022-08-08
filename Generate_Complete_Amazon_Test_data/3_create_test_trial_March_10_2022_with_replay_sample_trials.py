"""
This version:
(1) drop the novelty proportion 0.3 and add proportion of novelty 0.9
(2) Try to add a little more density unknowns -> those writers we need to accommodate
    More examples of those writers. How many? 10-20?
    ------> let the introduced unknown having more examples in the pre-novelty phase

++ --------------- ++
For the last four batches, replay the previous examples, turn off feedback (post process)
-----------------
We sample examples start from the post process to the end.


"""

from collections import defaultdict
import json
import csv
import sys
import time
from datetime import datetime
import random
import os
import copy
import numpy as np
from scipy import stats
import yaml
from yaml.loader import SafeLoader
from tqdm import tqdm
from multiprocessing import Process, Pool


def argument_parser():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_root', type=str, default="", help="")
    parser.add_argument('--seed', type=int, default=42, help="")

    args = parser.parse_args()

    return args


class TestDataSampler:
    """
    Edit: Oct 31, 2021, based on the newest document

    known writers: w_1, ... w_10
    unknown writers: w_11, ... w_20

    r_1, r_2, r3 .. r_11, r_12, ... r_j, ... r_n
    The target is to create writer assignment for r_j. Suppose before j, only r_11 = "w_15", r_12 = "w_16" are unknown
    writers, others are all known writers.

    ------------------------------------------------------------------------------------
                    assigned writer    actual writer
    ------------------------------------------------------------------------------------
    scenario_1:         w_1                w_1
    Both assigned writer and actual writer are known writer. The assigned == actual
    NORMAL

    scenario_2:         w_1                w_2
    Both assigned writer and actual writer are known writer. The assigned != actual
    NORMAL

    scenario_3:         w_12               w_3
    assigned is unknown writer, actual writer is known writer. The assigned != actual
    NORMAL

    scenario_4:         w_15               w_3
    assigned is unknown writer seen before, actual is known writer.
    NORMAL post novelty

    scenario_5:         w_3                w_13
    assigned is known writer, but actual writer is unknown writer
    NOVEL

    scenario_6:         w_3                w_15
    assigned is known writer, but actual writer is unknown writer seen before
    NORMAL-Pre-novelty learning

    scenario_7**:       w_17              w_17
    The assigned is unknown, the actual is unknown. The model should learn some information of this writer.
    **And the sampler should memorize this writer for the future sample generation.
    NOVEL

    scenario_8:        w_18               w_17
    The assigned is unknown, the actual is unknown. Both are not seen before.
    NOVEL

    scenario_9:        w_15                w_15
    The assigned is unknown, the actual is unknown, assigned == actual. The unknown is the one seen before.
    NORMAL pre-novelty learning

    scenario_10:        w_15                w_16
    The assigned is unknown, the actual is unknown, assigned != actual. Both unknown is seen before.
    NORMAL pre-novelty learning

    scenario_11:       w_12                 w_15
    The assigned is unknown, the actual is unknown but seen before. assigned != actual. The assigned is not seen before.
    NORMAL pre-novelty learning

    scenario_12:       w_15                 w_19
    The assigned is unknown, seen before.  The actual is unknown, not seen before.
    NORMAL pre-novelty learning


    ==> Amazon Review Json Example:

    id: 610cb484f1c4eaad2cb93c90

    {'overall': 5.0,
    'vote': '6',
    'verified': True,
    'reviewTime': '12 15, 2008',
    'reviewerID': 'A3HPFC4WMB2LZD',
    'asin': '1932836438',
    'reviewerName': 'kre8iv1',
    'reviewText': 'I happened to find this book light by accident, and what a happy accident it was! ...',
    'summary': 'Best book light I have yet to find.',
    'unixReviewTime': 1229299200,
    'category': ['Home & Kitchen', 'Home & Kitchen>Home Dcor', "Home & Kitchen>Kids' Room Dcor",
                 'Home & Kitchen>Lamps & Lighting']}

    """

    def __init__(self,
                 config_param,
                 dataset_source,
                 output_folder,
                 tmp_output_folder,
                 output_file_name_base,

                 known_writer_non_novel_dict,
                 known_writer_novel_dict,
                 unknown_writer_non_novel_dict,
                 unknown_writer_novel_dict,
                 all_review_id_to_json_dict,
                 all_writer_id_to_num_dict,

                 known_writer_id_set,
                 unknown_writer_id_set,
                 debug_mode=False):
        """
        :param dataset_source: We build dataset from two publicly released dataset: Amazon and Yelp
        :param known_writer_non_novel_dict:
        :param known_writer_novel_dict:
        :param unknown_writer_non_novel_dict:
        :param unknown_writer_novel_dict:
        """
        self.debug_mode = debug_mode

        self.config_param = config_param

        # output folder
        # now = datetime.now()  # current date and time
        # date_time_str = now.strftime("%m_%d_%Y_%H_%M_%S")
        # self.output_dir = os.path.join(output_folder, date_time_str)
        # if not os.path.exists(self.output_dir):
        #     os.makedirs(self.output_dir)
        # #endif
        self.output_dir = output_folder
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # endif

        self.tmp_output_folder = tmp_output_folder
        if not os.path.exists(self.tmp_output_folder):
            os.makedirs(self.tmp_output_folder)
        # endif

        # self.tmp_output_dir = os.path.join("./tmp", date_time_str)
        # self.tmp_output_dir = "./tmp"
        # if not os.path.exists(self.tmp_output_dir):
        #     os.makedirs(self.tmp_output_dir)
        # # endif

        self.output_file_name_base = output_file_name_base
        assert self.output_file_name_base is not None

        self.red_light_instance = None
        self.found_red_light = False
        # False when it is in normal phrase, True if it is in novel phrase
        self.novel_flag = False

        self.all_writer_id_to_num_dict = all_writer_id_to_num_dict

        self.dataset_source = dataset_source
        assert self.dataset_source in {"Amazon", "Yelp"}

        # To make each review instance of the generated test trial unique, no duplication
        self.review_id_pool = set()
        self.test_trial_json_list = []

        # some unknown writer appear in the test trial sequence, the model should memorize this writer id
        self.unknown_writer_learned_on_the_fly_id_set = set()

        self.known_writer_non_novel_dict = known_writer_non_novel_dict
        self.known_writer_novel_dict = known_writer_novel_dict
        self.unknown_writer_non_novel_dict = unknown_writer_non_novel_dict
        self.unknown_writer_novel_dict = unknown_writer_novel_dict
        self.all_review_id_to_json_dict = all_review_id_to_json_dict

        self.known_writer_id_set = known_writer_id_set
        self.unknown_writer_id_set = unknown_writer_id_set

        self.config = TestDataSampler.load_config()
        pass

    def initialize_class_obj(self):
        """
        after

        known_writer_non_novel_dict
        known_writer_novel_dict
        unknown_writer_non_novel_dict
        unknown_writer_novel_dict

        are updated, the following two dictionary should also be updated.
        :return:
        """
        # ############ Get all known_writer_non_novel_dict review_id set ################
        self.known_writer_non_novel_review_set = set()
        if self.known_writer_non_novel_dict is not None:
            for writer_id, review_id_set in self.known_writer_non_novel_dict.items():
                self.known_writer_non_novel_review_set.update(review_id_set)
            # endfor
        # endif

        # ############ Get all known_writer_novel_dict review_id set ####################
        self.known_writer_novel_review_set = set()
        if self.known_writer_novel_dict is not None:
            for writer_id, review_id_set in self.known_writer_novel_dict.items():
                self.known_writer_novel_review_set.update(review_id_set)
            # endfor
        # endif

        pass

    @staticmethod
    def load_config():
        with open("SAIL_ON_amazon_review_config_Jan_27_2022.yaml", mode="r") as fin:
            config = yaml.load(fin, Loader=SafeLoader)
        # endwith
        return config

    @staticmethod
    def load_specification_file(config_file_path):
        json_list = []
        with open(config_file_path, mode="r", encoding="utf16") as fin:
            csv_reader = csv.DictReader(fin)
            for row in csv_reader:
                print(row["task"], row["test_id"])
                json_list.append(row)
            # endfor
        # endwith
        return json_list

    @staticmethod
    def create_list_novel_sizes(config, spec_config, red_light_batch_index):
        """
        Use beta distribution to determine where novel examples appear.

        Returns: list x corresponding to the novel batches, where x[i] is the number of novel examples
         for the i-th batch (where i is the index into the list of novel batches, length(x) = number of novel batches)

         "beta_dist_params": {
            "low": [1.2,1.8],
            "mid": [2,2],
            "high": [1.8,1.2],
            "flat": [1,1]
        }
        """
        distparams = config["beta_dist_params"][spec_config["dist_type"]]

        n_batch_novel = int(
            (config["n_rounds"] - red_light_batch_index) * float(spec_config["prop_novel"]))
        novel_size = n_batch_novel * config["round_size"]

        # Prepare a probability bin for a given novel distribution
        # red_light_batch_index are also counted as one of novel batch
        # For instance, there are totally 40 batches, with batch size 30. The red_light_batch_index is 6
        # The normal batches are 0~5, totally 6 normal batches. and 40 - 6 = 34 novel batches (including red_light_batch_index)
        # It needs 34 slices under CDF curve, to add up to the whole area
        # So, it needs 35 numbers as boundaries. So -> config["batch_number"] - red_light_batch_index + 1
        bin_prob = np.linspace(
            0, 1, config["n_rounds"] - red_light_batch_index + 1).tolist()

        # The following will calculate the slice under the CDF curve
        list_unknown_sizes = []
        for i in range(len(bin_prob) - 1):
            prob_num = stats.beta.cdf(bin_prob[i + 1], distparams[0], distparams[1], loc=0, scale=1) - \
                       stats.beta.cdf(bin_prob[i], distparams[0],
                                      distparams[1], loc=0, scale=1)
            novel_num = int(prob_num * novel_size)
            list_unknown_sizes.append(novel_num)
            # list_unknown_sizes.append(
            #     int((stats.beta.cdf(bin_prob[i + 1], distparams[0], distparams[1], loc=0, scale=1) -
            #          stats.beta.cdf(bin_prob[i], distparams[0], distparams[1], loc=0, scale=1)) * novel_size))
        # endfor

        # pro_list_unknown_sizes = []
        # for i in range(len(bin_prob) - 1):
        #     pro_list_unknown_sizes.append(stats.beta.cdf(bin_prob[i + 1], distparams[0], distparams[1], loc=0, scale=1) - stats.beta.cdf(bin_prob[i], distparams[0], distparams[1], loc=0, scale=1))
        # # endfor

        assert len(list_unknown_sizes) == (
                config["n_rounds"] - red_light_batch_index)
        # Make sure the numbers are correct
        # sometimes, the num is 0. Use max(1, num) to make sure it is at lease 1.
        # otherwise, it this batch is red_light_batch, it will report error if it does not have novel instance
        list_unknown_sizes = [max(1, min(config["round_size"], i))
                              for i in list_unknown_sizes]
        for item in list_unknown_sizes:
            assert item >= 1
        # endfor
        return list_unknown_sizes

    @staticmethod
    def sample_the_other_item_in_set(a_set, item):
        assert item in a_set
        a_list = list(a_set)
        n = len(a_list)

        random_item = None
        while random_item is None or random_item == item:
            random_int = random.randint(0, n - 1)
            random_item = a_list[random_int]
        # endwhile

        return random_item

    @staticmethod
    def sample_from_set(a_set):
        a_list = list(a_set)
        n = len(a_list)

        random_int = random.randint(0, n - 1)
        random_item = a_list[random_int]
        return random_item

    def sample_func_evenly(self, func_list, example_size):
        """
        make sure the sampled json object list is of example_size
        :param func_list:
        :param example_size:
        :return:
        """
        func_dict = {}
        for index, func in enumerate(func_list):
            func_dict[index] = func

        size = len(func_list)

        cur_json_obj_list = []
        index = 0
        while len(cur_json_obj_list) < example_size:
            index += 1
            if index > 1000:
                print("stuck here:", index)
                print(f"{func_list}")
            # endif
            random_int = random.randint(0, size - 1)
            flag, json_obj = func_dict[random_int]()
            if json_obj is not None and flag is True:
                cur_json_obj_list.append(json_obj)
            # endif
        # endwhile
        assert len(cur_json_obj_list) == example_size

        random.shuffle(cur_json_obj_list)

        return cur_json_obj_list

    def walmup_sampling_for_hard_version(self, example_size):
        """
        This walmup session is used to introduce unknown in the pre-novelty phase.
        So that the model can learn new unknown writers on the fly.
        The first round should be the walmup for hard version
        :return:
        """
        round_json_obj_list = []

        while len(round_json_obj_list) < example_size:
            random_int = random.randint(0, 9)
            if random_int in [0, 1, 2]:
                flag, json_obj = self.scenario_1()
                assert json_obj is not None
                # in the walmup phase, novel_instance should always be normal
                round_json_obj_list.append(json_obj)
            if random_int in [3, 4, 5, 6]:
                flag, json_obj = self.scenario_2()
                assert json_obj is not None
                # in the walmup phase, novel_instance should always be normal
                round_json_obj_list.append(json_obj)
            if random_int in [7, 8, 9] and len(round_json_obj_list) < example_size - 5:
                flag, json_obj_list = self.scenario_7_multiple(20, 20)  # should be OK

                new_json_obj_list = []
                for item in json_obj_list:
                    new_json_obj_list.append(item)
                # endfor

                round_json_obj_list.extend(new_json_obj_list)
            else:
                continue
            # endif
        # endfor
        round_json_obj_list = round_json_obj_list[:example_size]

        for json_obj in round_json_obj_list:
            json_obj["final_novel"] = False
        # endfor

        self.test_trial_json_list.extend(round_json_obj_list)
        pass

    def walmup_sampling_for_hard_version_interaction_level(self, example_size):
        """
        This walmup session is used to introduce unknown in the pre-novelty phase.
        So that the model can learn new unknown writers on the fly.
        The first round should be the walmup for hard version
        :return:
        """
        round_json_obj_list = []

        while len(round_json_obj_list) < example_size:
            random_int = random.randint(0, 9)
            if random_int in [0, 1, 2]:
                flag, json_obj = self.scenario_1()
                assert json_obj is not None
                # in the walmup phase, novel_instance should always be normal
                round_json_obj_list.append(json_obj)
            if random_int in [3, 4, 5, 6]:
                flag, json_obj = self.scenario_2()
                assert json_obj is not None
                # in the walmup phase, novel_instance should always be normal
                round_json_obj_list.append(json_obj)
            if random_int in [7, 8, 9] and len(round_json_obj_list) < example_size - 10:
                flag, json_obj_list = self.scenario_7_multiple(10, 20)  # because shipping review is not much

                if flag is False:
                    continue

                # new_json_obj_list = []
                # for item in json_obj_list:
                #     new_json_obj_list.append(item)
                # # endfor

                round_json_obj_list.extend(json_obj_list)
            else:
                continue
            # endif
        # endfor
        round_json_obj_list = round_json_obj_list[:example_size]

        for json_obj in round_json_obj_list:
            json_obj["final_novel"] = False
        # endfor

        self.test_trial_json_list.extend(round_json_obj_list)
        pass

    def sample_pre_novelty_phase_easy(self, example_size):
        """
        EASY VERSION, only contains normal cases
        :param example_size:
        :return:
        """
        cur_json_obj_list = self.sample_func_evenly([self.scenario_1,
                                                     self.scenario_2,
                                                     self.scenario_3], example_size)

        for json_obj in cur_json_obj_list:
            json_obj["final_novel"] = False
        # endfor

        self.test_trial_json_list.extend(cur_json_obj_list)
        pass

    def sample_pre_novelty_phase_hard(self, example_size):
        """
        HARD VERSION
        :param example_size:
        :return:
        """
        cur_json_obj_list = self.sample_func_evenly([self.scenario_1,
                                                     self.scenario_2,
                                                     self.scenario_3,
                                                     self.scenario_4,
                                                     self.scenario_6,
                                                     self.scenario_9,
                                                     self.scenario_10,
                                                     self.scenario_11,
                                                     self.scenario_12], example_size)

        for json_obj in cur_json_obj_list:
            json_obj["final_novel"] = False
        # endfor

        self.test_trial_json_list.extend(cur_json_obj_list)
        pass

    def sample_normal_and_novel_examples_for_1_0_0_0_0(self, normal_example_size, novel_example_size,
                                                       if_check_red_light_instance=None):

        assert if_check_red_light_instance in {True, False}

        pre_novelty_phase_normal_func_list = [self.scenario_1,
                                              self.scenario_2,
                                              self.scenario_3]

        post_novelty_phase_normal_func_list = pre_novelty_phase_normal_func_list + [
            self.scenario_4,
            self.scenario_6,
            self.scenario_9,
            self.scenario_10,
            self.scenario_11,
            self.scenario_12
        ]

        post_novelty_phase_novel_func_list = [self.scenario_5,
                                              self.scenario_7,
                                              self.scenario_8]

        # ###### sample normal examples #########
        normal_cur_json_obj_list = self.sample_func_evenly(
            post_novelty_phase_normal_func_list, normal_example_size)

        # ###### example novel examples #########
        novel_cur_json_obj_list = self.sample_func_evenly(
            post_novelty_phase_novel_func_list, novel_example_size)

        total_json_list = []
        total_json_list.extend(normal_cur_json_obj_list)
        total_json_list.extend(novel_cur_json_obj_list)
        random.shuffle(total_json_list)

        for exp_index, json_obj in enumerate(total_json_list):
            # text_id = json_obj["review_id"]
            assert "class_novel" in json_obj and "other_novel" in json_obj
            class_novel = json_obj["class_novel"]
            other_novel = json_obj["other_novel"]
            final_novel = class_novel or other_novel
            json_obj["final_novel"] = final_novel
        # endfor

        # ############ get red_light_instance ##############
        red_light_instance = None
        if if_check_red_light_instance:
            for json_obj in total_json_list:
                if red_light_instance is None and json_obj["final_novel"] is True:
                    red_light_instance = json_obj
                # endif
            # endfor

            assert red_light_instance is not None
            self.red_light_instance = red_light_instance

        self.test_trial_json_list.extend(total_json_list)
        pass

    def sample_normal_and_novel_examples_for_1_and_1_or_1_or_1(self,
                                                               normal_example_size,
                                                               novel_example_size,
                                                               if_check_red_light_instance=None):
        """
        For three cases:
        [1, 1, 0, 0]:  class_level_novelty and object_level_novelty
        [1, 0, 1, 0]:  class_level_novelty and sentiment_level_novelty
        [1, 0, 0, 1]:  class_level_novelty and interaction_level_novelty

        :param normal_example_size:
        :param novel_example_size:
        :return:
        """
        assert if_check_red_light_instance in {True, False}

        pre_novelty_phase_normal_func_list = [self.scenario_1,
                                              self.scenario_2,
                                              self.scenario_3]

        post_novelty_phase_normal_func_list = pre_novelty_phase_normal_func_list + [
            self.scenario_4,
            self.scenario_6,
            self.scenario_9,
            self.scenario_10,
            self.scenario_11,
            self.scenario_12
        ]

        all_with_novelty_cases = [self.scenario_1_with_novelty,
                                  self.scenario_2_with_novelty,
                                  self.scenario_3_with_novelty,
                                  self.scenario_4_with_novelty,
                                  self.scenario_5_with_novelty,
                                  self.scenario_6_with_novelty,
                                  self.scenario_7_with_novelty,
                                  self.scenario_8_with_novelty,
                                  self.scenario_9_with_novelty,
                                  self.scenario_10_with_novelty,
                                  self.scenario_11_with_novelty,
                                  self.scenario_12_with_novelty]

        post_novelty_phase_novel_func_list = [self.scenario_5,
                                              self.scenario_7,
                                              self.scenario_8] + all_with_novelty_cases

        # ###### sample normal examples #########
        normal_cur_json_obj_list = self.sample_func_evenly(
            post_novelty_phase_normal_func_list, normal_example_size)

        # ###### example novel examples #########
        novel_cur_json_obj_list = self.sample_func_evenly(
            post_novelty_phase_novel_func_list, novel_example_size)

        assert len(normal_cur_json_obj_list) == normal_example_size
        assert len(novel_cur_json_obj_list) == novel_example_size

        total_json_list = []
        total_json_list.extend(normal_cur_json_obj_list)
        total_json_list.extend(novel_cur_json_obj_list)
        random.shuffle(total_json_list)

        for json_obj in total_json_list:
            assert "class_novel" in json_obj and "other_novel" in json_obj
            class_novel = json_obj["class_novel"]
            other_novel = json_obj["other_novel"]
            final_novel = class_novel or other_novel
            json_obj["final_novel"] = final_novel
        # endfor

        # ############ get red_light_instance ##############
        red_light_instance = None
        if if_check_red_light_instance:
            for json_obj in total_json_list:
                if red_light_instance is None and json_obj["final_novel"] is True:
                    red_light_instance = json_obj
                # endif
            # endfor
            assert red_light_instance is not None
            self.red_light_instance = red_light_instance

        self.test_trial_json_list.extend(total_json_list)
        pass

    # ############ each scenario have non-novel and novel version ##############################
    # (1) in the pre-novel phrase, call the non-novel version sampling function
    # (2) in the novel phrase, call the novel version sampling function
    # ##########################################################################################
    def scenario_1(self):
        """
        ------------------------------------------------------------------------------------
                        assigned writer    actual writer
        ------------------------------------------------------------------------------------
        scenario_1:          w_1               w_1
        Both assigned writer and actual writer are known writer. The assigned == actual
        NORMAL
        :return:
        """
        # sample a known writer review
        try:
            review_id = self.known_writer_non_novel_review_set.pop()
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_1.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        # review_id does not need to be changed
        # (1) the original Amazon review does not have review id field. So the _id of MongoDB id is used
        # (2) Yelp review has review id field, but it is not _id of MongoDB, so we change it to MongoDB _id.
        original_review_json["review_id"] = review_id

        writer_id = original_review_json["reviewerID"]
        original_review_json["assigned_writer_id"] = writer_id
        assert writer_id in self.known_writer_id_set

        # novel or not?
        # class_level novelty or not?
        original_review_json["class_novel"] = False
        # other_novel: object, sentiment, interaction novelty or not?
        original_review_json["other_novel"] = False
        original_review_json["scenario"] = self.scenario_1.__name__

        return True, original_review_json

    def scenario_1_with_novelty(self):
        try:
            review_id = self.known_writer_novel_review_set.pop()
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_1_with_novelty.__name__)
        except:
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        writer_id = original_review_json["reviewerID"]
        original_review_json["assigned_writer_id"] = writer_id
        assert writer_id in self.known_writer_id_set

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = True
        original_review_json["scenario"] = self.scenario_1_with_novelty.__name__

        return True, original_review_json

    def scenario_2(self):
        """
        ------------------------------------------------------------------------------------
                        assigned writer    actual writer
        ------------------------------------------------------------------------------------
        scenario_2:         w_1                w_2
        Both assigned writer and actual writer are known writer. The assigned != actual
        NORMAL
        :return:
        """
        try:
            review_id = self.known_writer_non_novel_review_set.pop()
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_2.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        writer_id = original_review_json["reviewerID"]
        # writer_id need to be changed, sampler another known writer id
        assigned_writer_id = TestDataSampler.sample_the_other_item_in_set(
            self.known_writer_id_set, writer_id)
        original_review_json["assigned_writer_id"] = assigned_writer_id

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = False
        original_review_json["scenario"] = self.scenario_2.__name__

        return True, original_review_json

    def scenario_2_with_novelty(self):
        try:
            review_id = self.known_writer_novel_review_set.pop()
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_2_with_novelty.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        writer_id = original_review_json["reviewerID"]
        # writer_id need to be changed, sample another known writer id
        assigned_writer_id = TestDataSampler.sample_the_other_item_in_set(
            self.known_writer_id_set, writer_id)
        original_review_json["assigned_writer_id"] = assigned_writer_id

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = True
        original_review_json["scenario"] = self.scenario_2_with_novelty.__name__

        return True, original_review_json

    def scenario_3(self):
        """
        ------------------------------------------------------------------------------------
                        assigned writer    actual writer
        ------------------------------------------------------------------------------------
        scenario_3:         w_12               w_3
        assigned is unknown writer, actual writer is known writer. The assigned != actual
        NORMAL
        :return:
        """
        # actual writer is known writer
        try:
            review_id = self.known_writer_non_novel_review_set.pop()
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_3.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # assigned writer_id
        random_unknown_writer_id = TestDataSampler.sample_from_set(
            self.unknown_writer_id_set)
        original_review_json["assigned_writer_id"] = random_unknown_writer_id

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = False
        original_review_json["scenario"] = self.scenario_3.__name__

        return True, original_review_json

    def scenario_3_with_novelty(self):
        # actual writer is known writer
        try:
            review_id = self.known_writer_novel_review_set.pop()
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_3_with_novelty.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # assigned writer_id
        random_unknown_writer_id = TestDataSampler.sample_from_set(
            self.unknown_writer_id_set)
        original_review_json["assigned_writer_id"] = random_unknown_writer_id
        assert original_review_json["reviewerID"] in self.known_writer_id_set

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = True
        original_review_json["scenario"] = self.scenario_3_with_novelty.__name__

        return True, original_review_json

    def scenario_4(self):
        """
        ------------------------------------------------------------------------------------
                        assigned writer    actual writer
        ------------------------------------------------------------------------------------
        scenario_4:         w_15               w_3
        assigned is unknown writer seen before, actual is known writer.
        NORMAL
        :return:
        """
        # actual writer is known writer
        try:
            review_id = self.known_writer_non_novel_review_set.pop()
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_4.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # assigned writer_id is a unknown id that is seen before
        if len(self.unknown_writer_learned_on_the_fly_id_set) < 1:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_4.__name__)
            return False, None
        # endif
        unknown_writer_id_that_seen_before = TestDataSampler.sample_from_set(
            self.unknown_writer_learned_on_the_fly_id_set)
        original_review_json["assigned_writer_id"] = unknown_writer_id_that_seen_before

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = False
        original_review_json["scenario"] = self.scenario_4.__name__

        return True, original_review_json

    def scenario_4_with_novelty(self):
        if len(self.unknown_writer_learned_on_the_fly_id_set) < 1:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_4.__name__)
            return False, None
        # endif

        # actual writer is known writer
        try:
            review_id = self.known_writer_novel_review_set.pop()
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_4.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # assigned writer_id is a unknown id that is seen before
        unknown_writer_id_that_seen_before = TestDataSampler.sample_from_set(
            self.unknown_writer_learned_on_the_fly_id_set)
        original_review_json["assigned_writer_id"] = unknown_writer_id_that_seen_before

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = True
        original_review_json["scenario"] = self.scenario_4_with_novelty.__name__

        return True, original_review_json

    def scenario_5(self):
        """
        ------------------------------------------------------------------------------------
                        assigned writer    actual writer
        ------------------------------------------------------------------------------------
        scenario_5:         w_3                w_13
        assigned is known writer, but actual writer is unknown writer
        NOVEL

        actual writer is unknown writer that is not seen before
        :return:
        """
        if len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) == 0:
            if self.debug_mode:
                print(
                    "SAMPLE ERROR: len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) == 0",
                    self.scenario_5.__name__)
            return False, None

        unknown_writer_not_seen_before = \
            TestDataSampler.sample_from_set(
                self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set)

        try:
            review_id = self.unknown_writer_non_novel_dict[unknown_writer_not_seen_before].pop(
            )
        except:
            if self.debug_mode:
                print(
                    "SAMPLE ERROR: review_id = self.unknown_writer_non_novel_dict[unknown_writer_not_seen_before].pop()",
                    self.scenario_5.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # assigned writer id
        random_known_writer_id = TestDataSampler.sample_from_set(
            self.known_writer_id_set)
        original_review_json["assigned_writer_id"] = random_known_writer_id
        assert original_review_json["reviewerID"] in self.unknown_writer_id_set

        # novel or not?
        original_review_json["class_novel"] = True
        original_review_json["other_novel"] = False
        original_review_json["scenario"] = self.scenario_5.__name__

        return True, original_review_json

    def scenario_5_with_novelty(self):

        if len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) == 0:
            if self.debug_mode:
                print(
                    "SAMPLE ERROR: len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) == 0",
                    self.scenario_5_with_novelty.__name__)
            return False, None

        unknown_writer_not_seen_before = \
            TestDataSampler.sample_from_set(
                self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set)

        if unknown_writer_not_seen_before not in self.unknown_writer_novel_dict:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_5_with_novelty.__name__)
            return False, None

        if len(self.unknown_writer_novel_dict[unknown_writer_not_seen_before]) == 0:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_5_with_novelty.__name__)
            return False, None

        try:
            review_id = self.unknown_writer_novel_dict[unknown_writer_not_seen_before].pop(
            )
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_5_with_novelty.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # assigned writer id
        random_known_writer_id = TestDataSampler.sample_from_set(
            self.known_writer_id_set)
        original_review_json["assigned_writer_id"] = random_known_writer_id
        assert original_review_json["reviewerID"] in self.unknown_writer_id_set

        # novel or not?
        original_review_json["class_novel"] = True
        original_review_json["other_novel"] = True
        original_review_json["scenario"] = self.scenario_5_with_novelty.__name__

        return True, original_review_json

    def scenario_6(self):
        """
        ------------------------------------------------------------------------------------
                        assigned writer    actual writer
        ------------------------------------------------------------------------------------
        scenario_6:         w_3                w_15
        assigned is known writer, but actual writer is unknown writer seen before
        NORMAL-Pre-novelty learning

        actual writer is unknown writer that is seen before, assigned writer is known writer
        :return:
        """
        if len(self.unknown_writer_learned_on_the_fly_id_set) < 1:
            if self.debug_mode:
                print(f"SAMPLE ERROR:", self.scenario_6.__name__)
            return False, None

        unknown_writer_seen_before = TestDataSampler.sample_from_set(
            self.unknown_writer_learned_on_the_fly_id_set)
        try:
            review_id = self.unknown_writer_non_novel_dict[unknown_writer_seen_before].pop(
            )
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_6.__name__)
            return False, None
        # endtry

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # assigned writer id
        random_known_writer_id = TestDataSampler.sample_from_set(
            self.known_writer_id_set)
        original_review_json["assigned_writer_id"] = random_known_writer_id
        assert original_review_json["reviewerID"] in self.unknown_writer_id_set

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = False
        original_review_json["scenario"] = self.scenario_6.__name__

        return True, original_review_json

    def scenario_6_with_novelty(self):
        if len(self.unknown_writer_learned_on_the_fly_id_set) < 1:
            if self.debug_mode:
                print(f"SAMPLE ERROR:", self.scenario_6_with_novelty.__name__)
            return False, None

        unknown_writer_seen_before = TestDataSampler.sample_from_set(
            self.unknown_writer_learned_on_the_fly_id_set)

        if unknown_writer_seen_before not in self.unknown_writer_novel_dict:
            return False, None

        if len(self.unknown_writer_novel_dict[unknown_writer_seen_before]) == 0:
            if self.debug_mode:
                print(f"SAMPLE ERROR:", self.scenario_6_with_novelty.__name__)
            return False, None

        try:
            review_id = self.unknown_writer_novel_dict[unknown_writer_seen_before].pop(
            )
        except:
            if self.debug_mode:
                print(f"SAMPLE ERROR:", self.scenario_6_with_novelty.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # assigned writer id
        random_known_writer_id = TestDataSampler.sample_from_set(
            self.known_writer_id_set)
        original_review_json["assigned_writer_id"] = random_known_writer_id
        assert original_review_json["reviewerID"] in self.unknown_writer_id_set

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = True
        original_review_json["scenario"] = self.scenario_6_with_novelty.__name__

        return True, original_review_json

    def scenario_7(self):
        """
        ------------------------------------------------------------------------------------
                        assigned writer    actual writer
        ------------------------------------------------------------------------------------
        scenario_7**:       w_17              w_17
        The assigned is unknown, the actual is unknown. The model should learn some information of this writer.
        **And the sampler should memorize this writer for the future sample generation.
        NOVEL

        ==== Notes ====
        This function is used in the warmup session (pre-novelty phase) of hard version
        So that, it is important that:
            self.unknown_writer_learned_on_the_fly_id_set.add(actual_writer_id)

        However, for scenario_7(), it is used in the (post-novelty phase), so it should not
        have the operation:
        self.unknown_writer_learned_on_the_fly_id_set.add(actual_writer_id)
        :return:
        """
        if len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) == 0:
            if self.debug_mode:
                print(
                    "SAMPLE ERROR: len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) == 0",
                    self.scenario_7.__name__)
            return False, None

        # (1) actual writer is unknown writer, the assigned is the same as actual writer
        # (2) the writer id is not seen before, not in the self.unknown_writer_learned_on_the_fly_set
        # (3) the sampler should memorize this writer id
        unknown_writer_not_seen_before = None
        tmp_index = 0
        if self.unknown_writer_non_novel_dict is not None:
            while unknown_writer_not_seen_before not in self.unknown_writer_non_novel_dict:
                tmp_index += 1
                if tmp_index > 500:
                    sys.exit(">>>>>>>>>> ERROR <<<<<<<<<<< scenario_7")
                unknown_writer_not_seen_before = \
                    TestDataSampler.sample_from_set(
                        self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set)
            # endwhile
        # endif

        try:
            review_id = self.unknown_writer_non_novel_dict[unknown_writer_not_seen_before].pop(
            )
        except:
            if self.debug_mode:
                print(
                    "SAMPLE ERROR: review_id = self.unknown_writer_non_novel_dict[unknown_writer_not_seen_before].pop()",
                    self.scenario_7.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # assigned writer id is the same as the actual writer id
        actual_writer_id = original_review_json["reviewerID"]
        original_review_json["assigned_writer_id"] = actual_writer_id
        assert actual_writer_id in self.unknown_writer_id_set

        # ########### Note: this is important!! ###############
        assert actual_writer_id not in self.unknown_writer_learned_on_the_fly_id_set
        # self.unknown_writer_learned_on_the_fly_id_set.add(actual_writer_id)

        # novel or not?
        original_review_json["class_novel"] = True
        original_review_json["other_novel"] = False
        original_review_json["scenario"] = self.scenario_7.__name__

        return True, original_review_json

    def scenario_7_multiple(self, min_num, max_num):
        """
        The model memorizes the unknown writer appears in the sequence
        Generate multiple instance for a single unknown writer

        ==== Notes ====
        This function is used in the warmup session (pre-novelty phase) of hard version
        So that, it is important that:
            self.unknown_writer_learned_on_the_fly_id_set.add(actual_writer_id)

        However, for scenario_7(), it is used in the (post-novelty phase), so it should not
        have the operation:
        self.unknown_writer_learned_on_the_fly_id_set.add(actual_writer_id)

        :return:
        """
        if len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) == 0:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_7_multiple.__name__)
            sys.exit(">>>>>>>>>> ERROR <<<<<<<<<<< scenario_7_multiple")
        # (1) actual writer is unknown writer, the assigned is the same as actual writer
        # (2) the writer id is not seen before, not in the self.unknown_writer_learned_on_the_fly_set
        # (3) the sampler should memorize this writer id
        unknown_writer_not_seen_before = None
        tmp_index = 0
        if self.unknown_writer_non_novel_dict is not None:
            while unknown_writer_not_seen_before not in self.unknown_writer_non_novel_dict:
                tmp_index += 1
                if tmp_index > 500:
                    sys.exit(">>>>>>>>>> ERROR <<<<<<<<<<< scenario_7_multiple")
                unknown_writer_not_seen_before = \
                    TestDataSampler.sample_from_set(
                        self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set)
            # endwhile
        # endif

        # generate multiple instances from a single writer
        random_size = random.randint(min_num, max_num)

        sample_json_obj_list = []

        for i in range(random_size):
            try:
                review_id = self.unknown_writer_non_novel_dict[unknown_writer_not_seen_before].pop(
                )
            except:
                continue

            # assert review_id not in self.review_id_pool

            if review_id in self.review_id_pool:
                continue

            self.review_id_pool.add(review_id)
            original_review_json = self.all_review_id_to_json_dict[review_id]
            original_review_json["review_id"] = review_id

            # assigned writer id is the same as the actual writer id
            actual_writer_id = original_review_json["reviewerID"]
            original_review_json["assigned_writer_id"] = actual_writer_id
            assert actual_writer_id in self.unknown_writer_id_set
            # assert actual_writer_id not in self.unknown_writer_learned_on_the_fly_id_set
            self.unknown_writer_learned_on_the_fly_id_set.add(actual_writer_id)

            # novel or not?
            original_review_json["class_novel"] = True
            original_review_json["other_novel"] = False
            original_review_json["scenario"] = "scenario_7"

            sample_json_obj_list.append(original_review_json)
        # endfor

        return True, sample_json_obj_list

    @DeprecationWarning
    def scenario_7_with_novelty_multiple_interaction_level_novelty(self, min_num, max_num):
        """
        The model memorizes the unknown writer appears in the sequence
        Generate multiple instance for a single unknown writer

        ==== Notes ====
        This function is used in the warmup session (pre-novelty phase) of hard version
        So that, it is important that:
            self.unknown_writer_learned_on_the_fly_id_set.add(actual_writer_id)

        However, for scenario_7(), it is used in the (post-novelty phase), so it should not
        have the operation:
        self.unknown_writer_learned_on_the_fly_id_set.add(actual_writer_id)

        :return:
        """
        productive_reviewer_set = set()
        # load reviewers that write multiple reviews
        with open("./shipping_review_input_March_10_2022/productive_reviewer_and_reviews/productive_reviewer_stats.txt",
                  mode="r") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                # endif
                parts = line.split()
                productive_reviewer_set.add(parts[0])
            # endfor
        # endwith

        if len(productive_reviewer_set - self.unknown_writer_learned_on_the_fly_id_set) == 0:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_7_multiple.__name__)
            # sys.exit(">>>>>>>>>> ERROR <<<<<<<<<<< scenario_7_with_novelty_multiple")
            print(">>>>>>>>>> ERROR <<<<<<<<<<< scenario_7_with_novelty_multiple")
            return False, None
        # (1) actual writer is unknown writer, the assigned is the same as actual writer
        # (2) the writer id is not seen before, not in the self.unknown_writer_learned_on_the_fly_set
        # (3) the sampler should memorize this writer id
        unknown_writer_not_seen_before = None
        tmp_index = 0
        if self.unknown_writer_novel_dict is not None:
            while unknown_writer_not_seen_before not in self.unknown_writer_novel_dict:
                tmp_index += 1
                if tmp_index > 500:
                    sys.exit(">>>>>>>>>> ERROR <<<<<<<<<<< scenario_7_with_novelty_multiple")
                unknown_writer_not_seen_before = \
                    TestDataSampler.sample_from_set(
                        productive_reviewer_set - self.unknown_writer_learned_on_the_fly_id_set)
            # endwhile
        # endif

        # generate multiple instances from a single writer
        random_size = random.randint(min_num, max_num)

        sample_json_obj_list = []

        for i in range(random_size):
            try:
                review_id = self.unknown_writer_novel_dict[unknown_writer_not_seen_before].pop(
                )
            except:
                continue

            # assert review_id not in self.review_id_pool

            if review_id in self.review_id_pool:
                continue

            self.review_id_pool.add(review_id)
            original_review_json = self.all_review_id_to_json_dict[review_id]
            original_review_json["review_id"] = review_id

            # assigned writer id is the same as the actual writer id
            actual_writer_id = original_review_json["reviewerID"]
            original_review_json["assigned_writer_id"] = actual_writer_id
            assert actual_writer_id in self.unknown_writer_id_set
            # assert actual_writer_id not in self.unknown_writer_learned_on_the_fly_id_set
            self.unknown_writer_learned_on_the_fly_id_set.add(actual_writer_id)

            # novel or not?
            original_review_json["class_novel"] = True
            original_review_json["other_novel"] = True
            original_review_json["scenario"] = "scenario_7_with_novelty"

            sample_json_obj_list.append(original_review_json)
        # endfor

        return True, sample_json_obj_list

    @DeprecationWarning
    def scenario_7_with_novelty_multiple(self, min_num, max_num):
        """
        The model memorizes the unknown writer appears in the sequence
        Generate multiple instance for a single unknown writer

        ==== Notes ====
        This function is used in the warmup session (pre-novelty phase) of hard version
        So that, it is important that:
            self.unknown_writer_learned_on_the_fly_id_set.add(actual_writer_id)

        However, for scenario_7(), it is used in the (post-novelty phase), so it should not
        have the operation:
        self.unknown_writer_learned_on_the_fly_id_set.add(actual_writer_id)

        :return:
        """
        if len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) == 0:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_7_multiple.__name__)
            sys.exit(">>>>>>>>>> ERROR <<<<<<<<<<< scenario_7_with_novelty_multiple")
        # (1) actual writer is unknown writer, the assigned is the same as actual writer
        # (2) the writer id is not seen before, not in the self.unknown_writer_learned_on_the_fly_set
        # (3) the sampler should memorize this writer id
        unknown_writer_not_seen_before = None
        tmp_index = 0
        if self.unknown_writer_novel_dict is not None:
            while unknown_writer_not_seen_before not in self.unknown_writer_novel_dict:
                tmp_index += 1
                if tmp_index > 500:
                    sys.exit(">>>>>>>>>> ERROR <<<<<<<<<<< scenario_7_with_novelty_multiple")
                unknown_writer_not_seen_before = \
                    TestDataSampler.sample_from_set(
                        self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set)
            # endwhile
        # endif

        # generate multiple instances from a single writer
        random_size = random.randint(min_num, max_num)

        sample_json_obj_list = []

        for i in range(random_size):
            try:
                review_id = self.unknown_writer_novel_dict[unknown_writer_not_seen_before].pop(
                )
            except:
                continue

            # assert review_id not in self.review_id_pool

            if review_id in self.review_id_pool:
                continue

            self.review_id_pool.add(review_id)
            original_review_json = self.all_review_id_to_json_dict[review_id]
            original_review_json["review_id"] = review_id

            # assigned writer id is the same as the actual writer id
            actual_writer_id = original_review_json["reviewerID"]
            original_review_json["assigned_writer_id"] = actual_writer_id
            assert actual_writer_id in self.unknown_writer_id_set
            # assert actual_writer_id not in self.unknown_writer_learned_on_the_fly_id_set
            self.unknown_writer_learned_on_the_fly_id_set.add(actual_writer_id)

            # novel or not?
            original_review_json["class_novel"] = True
            original_review_json["other_novel"] = True
            original_review_json["scenario"] = "scenario_7_with_novelty"

            sample_json_obj_list.append(original_review_json)
        # endfor

        return True, sample_json_obj_list

    def scenario_7_with_novelty(self):
        """
        The model memorizes the unknown writer appears in the sequence
        :return:
        """
        if len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) == 0:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_7_with_novelty.__name__)
            return False, None

        # (1) actual writer is unknown writer, the assigned is the same as actual writer
        # (2) the writer id is not seen before, not in the self.unknown_writer_learned_on_the_fly_set
        # (3) the sampler should memorize this writer id
        unknown_writer_not_seen_before = None
        tmp_index = 0

        if self.unknown_writer_novel_dict is not None:
            while unknown_writer_not_seen_before not in self.unknown_writer_novel_dict:
                tmp_index += 1
                if tmp_index > 500:
                    sys.exit(">>>>>>>>>> ERROR <<<<<<<<<<< scenario_7_with_novelty")
                unknown_writer_not_seen_before = \
                    TestDataSampler.sample_from_set(
                        self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set)
            # endwhile

        else:
            return False, None

        # if unknown_writer_not_seen_before is None:
        #     print(">>>>> WRONG ... ")
        #     print(f"unknown_writer_learned_on_the_fly_id_set: {len(self.unknown_writer_learned_on_the_fly_id_set)}")
        #     sys.exit(1)

        if len(self.unknown_writer_novel_dict[unknown_writer_not_seen_before]) == 0:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_7_with_novelty.__name__)
            return False, None

        try:
            review_id = self.unknown_writer_novel_dict[unknown_writer_not_seen_before].pop(
            )
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_7_with_novelty.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # assigned writer id is the same as the actual writer id
        actual_writer_id = original_review_json["reviewerID"]
        original_review_json["assigned_writer_id"] = actual_writer_id
        assert actual_writer_id in self.unknown_writer_id_set

        # IMPORTANT !!
        assert actual_writer_id not in self.unknown_writer_learned_on_the_fly_id_set
        self.unknown_writer_learned_on_the_fly_id_set.add(actual_writer_id)

        # novel or not?
        original_review_json["class_novel"] = True
        original_review_json["other_novel"] = True
        original_review_json["scenario"] = self.scenario_7_with_novelty.__name__

        return True, original_review_json

    def scenario_8(self):
        """
        ------------------------------------------------------------------------------------
                        assigned writer    actual writer
        ------------------------------------------------------------------------------------
        scenario_8:        w_18                w_17
        The assigned is another unknown, the actual is unknown.
        NOVEL
        :return:
        """
        if len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) < 2:
            if self.debug_mode:
                print(
                    f"SAMPLE ERROR: len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) < 2",
                    self.scenario_8.__name__)
            return False, None

        # (1) actual writer is unknown writer, the assigned is the same as actual writer
        # (2) the writer id is not seen before, not in the self.unknown_writer_learned_on_the_fly_set
        # (3) the sampler should memorize this writer id
        unknown_writer_not_seen_before = \
            TestDataSampler.sample_from_set(
                self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set)

        try:
            review_id = self.unknown_writer_non_novel_dict[unknown_writer_not_seen_before].pop(
            )
        except:
            if self.debug_mode:
                print(
                    "SAMPLE ERROR: review_id = self.unknown_writer_non_novel_dict[unknown_writer_not_seen_before].pop()",
                    self.scenario_8.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # ##################### sample assigned writer id ########################
        # assigned writer id is NOT the same as the actual writer id, it is another unknown writer not seen before
        another_unknown_writer_not_seen_before = \
            TestDataSampler.sample_the_other_item_in_set(
                (self.unknown_writer_id_set -
                 self.unknown_writer_learned_on_the_fly_id_set),
                unknown_writer_not_seen_before)

        actual_writer_id = original_review_json["reviewerID"]
        original_review_json["assigned_writer_id"] = another_unknown_writer_not_seen_before
        assert another_unknown_writer_not_seen_before != actual_writer_id
        assert actual_writer_id in self.unknown_writer_id_set
        assert actual_writer_id not in self.unknown_writer_learned_on_the_fly_id_set

        # novel or not?
        original_review_json["class_novel"] = True
        original_review_json["other_novel"] = False
        original_review_json["scenario"] = self.scenario_8.__name__

        return True, original_review_json

    def scenario_8_with_novelty(self):
        """
        ------------------------------------------------------------------------------------
                        assigned writer    actual writer
        ------------------------------------------------------------------------------------
        scenario_8:        w_18                w_17
        The assigned is another unknown, the actual is unknown.
        NOVEL
        :return:
        """
        if len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) < 2:
            if self.debug_mode:
                print(f"SAMPLE ERROR:", self.scenario_8_with_novelty.__name__)
            return False, None

        if len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) == 0:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_8_with_novelty.__name__)
            return False, None

        # (1) actual writer is unknown writer, the assigned is the same as actual writer
        # (2) the writer id is not seen before, not in the self.unknown_writer_learned_on_the_fly_set
        # (3) the sampler should memorize this writer id
        unknown_writer_not_seen_before = \
            TestDataSampler.sample_from_set(
                self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set)

        try:
            review_id = self.unknown_writer_novel_dict[unknown_writer_not_seen_before].pop(
            )
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_8_with_novelty.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # ##################### sample assigned writer id ########################
        # assigned writer id is NOT the same as the actual writer id, it is another unknown writer not seen before
        another_unknown_writer_not_seen_before = \
            TestDataSampler.sample_the_other_item_in_set(
                (self.unknown_writer_id_set -
                 self.unknown_writer_learned_on_the_fly_id_set),
                unknown_writer_not_seen_before)

        actual_writer_id = original_review_json["reviewerID"]
        original_review_json["assigned_writer_id"] = another_unknown_writer_not_seen_before
        assert another_unknown_writer_not_seen_before != actual_writer_id
        assert actual_writer_id in self.unknown_writer_id_set
        assert actual_writer_id not in self.unknown_writer_learned_on_the_fly_id_set

        # novel or not?
        original_review_json["class_novel"] = True
        original_review_json["other_novel"] = True
        original_review_json["scenario"] = self.scenario_8_with_novelty.__name__

        return True, original_review_json

    def scenario_9(self):
        """
        ------------------------------------------------------------------------------------
                        assigned writer    actual writer
        ------------------------------------------------------------------------------------
        scenario_9:        w_15                w_15
        The assigned is unknown, the actual is unknown, assigned == actual. The unknown is the one seen before.
        NORMAL pre-novelty learning

        the unknown writer appear in the sequence before, here it appears again, and assigned == actual writer_id
        :return:
        """
        if len(self.unknown_writer_learned_on_the_fly_id_set) < 1:
            if self.debug_mode:
                print("SAMPLE ERROR: len(self.unknown_writer_learned_on_the_fly_id_set) < 1",
                      self.scenario_9.__name__)
            return False, None

        # sample from the unknown_writer_learned_on_the_fly_id_set with replacement
        seen_before_unknown_writer_id = TestDataSampler.sample_from_set(
            self.unknown_writer_learned_on_the_fly_id_set)
        # sample one review from this unknown writer
        try:
            review_id = self.unknown_writer_non_novel_dict[seen_before_unknown_writer_id].pop(
            )
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_9.__name__)
        except:
            return False, None
        # endtry

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # assigned writer id is the same as the actual writer id
        actual_writer_id = original_review_json["reviewerID"]
        original_review_json["assigned_writer_id"] = actual_writer_id
        assert actual_writer_id in self.unknown_writer_learned_on_the_fly_id_set

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = False
        original_review_json["scenario"] = self.scenario_9.__name__

        return True, original_review_json

    def scenario_9_with_novelty(self):
        if len(self.unknown_writer_learned_on_the_fly_id_set) < 1:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_9_with_novelty.__name__)
            return False, None

        # sample from the unknown_writer_learned_on_the_fly_id_set with replacement
        seen_before_unknown_writer_id = TestDataSampler.sample_from_set(
            self.unknown_writer_learned_on_the_fly_id_set)

        if seen_before_unknown_writer_id not in self.unknown_writer_novel_dict:
            return False, None

        # sample one review from this unknown writer
        if len(self.unknown_writer_novel_dict[seen_before_unknown_writer_id]) == 0:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_9_with_novelty.__name__)
            return False, None
        # endif

        try:
            review_id = self.unknown_writer_novel_dict[seen_before_unknown_writer_id].pop(
            )
        except:
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # assigned writer id is the same as the actual writer id
        actual_writer_id = original_review_json["reviewerID"]
        original_review_json["assigned_writer_id"] = actual_writer_id
        assert actual_writer_id in self.unknown_writer_learned_on_the_fly_id_set

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = True
        original_review_json["scenario"] = self.scenario_9_with_novelty.__name__

        return True, original_review_json

    def scenario_10(self):
        """
        ------------------------------------------------------------------------------------
                        assigned writer    actual writer
        ------------------------------------------------------------------------------------
        scenario_10:        w_15                w_16
        The assigned is unknown, the actual is unknown, assigned != actual. Both unknown is seen before.
        NORMAL pre-novelty learning

        the unknown writer appear in the sequence before, here it appears again, and assigned != actual writer_id
        :return:
        """
        if self.unknown_writer_non_novel_dict is None:
            return False, None

        if len(self.unknown_writer_learned_on_the_fly_id_set) < 2:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_10.__name__)
            return False, None
        # endif

        # sample from the unknown_writer_learned_on_the_fly_id_set with replacement
        unknown_writer_learned_on_the_fly_id_set_copy = copy.deepcopy(
            self.unknown_writer_learned_on_the_fly_id_set)
        try:
            seen_before_unknown_writer_id = unknown_writer_learned_on_the_fly_id_set_copy.pop()
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_10.__name__)
            return False, None

        if seen_before_unknown_writer_id not in self.unknown_writer_non_novel_dict:
            return False, None

        # sample one review from this unknown writer
        if len(self.unknown_writer_non_novel_dict[seen_before_unknown_writer_id]) == 0:
            if self.debug_mode:
                print(f"SAMPLE ERROR:", self.scenario_10.__name__)
            return False, None
        # endif

        # sample one review from this unknown writer
        try:
            review_id = self.unknown_writer_non_novel_dict[seen_before_unknown_writer_id].pop(
            )
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_10.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # assigned writer id is not the same as the actual writer id
        actual_writer_id = original_review_json["reviewerID"]
        try:
            assigned_writer_id = unknown_writer_learned_on_the_fly_id_set_copy.pop()
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_10.__name__)
            return False, None

        original_review_json["assigned_writer_id"] = assigned_writer_id

        assert actual_writer_id != assigned_writer_id
        assert actual_writer_id in self.unknown_writer_learned_on_the_fly_id_set
        assert assigned_writer_id in self.unknown_writer_learned_on_the_fly_id_set

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = False
        original_review_json["scenario"] = self.scenario_10.__name__

        return True, original_review_json

    def scenario_10_with_novelty(self):
        if len(self.unknown_writer_learned_on_the_fly_id_set) < 2:
            if self.debug_mode:
                print(f"SAMPLE ERROR:", self.scenario_10_with_novelty.__name__)
            return False, None
        # endif

        # sample from the unknown_writer_learned_on_the_fly_id_set with replacement
        unknown_writer_learned_on_the_fly_id_set_copy = copy.deepcopy(
            self.unknown_writer_learned_on_the_fly_id_set)
        try:
            seen_before_unknown_writer_id = unknown_writer_learned_on_the_fly_id_set_copy.pop()
        except:
            if self.debug_mode:
                print(f"SAMPLE ERROR:", self.scenario_10_with_novelty.__name__)
            return False, None

        if seen_before_unknown_writer_id not in self.unknown_writer_novel_dict:
            return False, None

        # sample one review from this unknown writer
        if len(self.unknown_writer_novel_dict[seen_before_unknown_writer_id]) == 0:
            if self.debug_mode:
                print(f"SAMPLE ERROR:", self.scenario_10_with_novelty.__name__)
            return False, None
        # endif

        try:
            review_id = self.unknown_writer_novel_dict[seen_before_unknown_writer_id].pop(
            )
        except:
            if self.debug_mode:
                print(f"SAMPLE ERROR:", self.scenario_10_with_novelty.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # assigned writer id is not the same as the actual writer id
        actual_writer_id = original_review_json["reviewerID"]
        try:
            assigned_writer_id = unknown_writer_learned_on_the_fly_id_set_copy.pop()
        except:
            if self.debug_mode:
                print(f"SAMPLE ERROR:", self.scenario_10_with_novelty.__name__)
            return False, None

        original_review_json["assigned_writer_id"] = assigned_writer_id

        assert actual_writer_id != assigned_writer_id
        assert actual_writer_id in self.unknown_writer_learned_on_the_fly_id_set
        assert assigned_writer_id in self.unknown_writer_learned_on_the_fly_id_set

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = True
        original_review_json["scenario"] = self.scenario_10_with_novelty.__name__

        return True, original_review_json

    def scenario_11(self):
        """
        ------------------------------------------------------------------------------------
                        assigned writer    actual writer
        ------------------------------------------------------------------------------------
        scenario_11:       w_12                 w_15
        The assigned is unknown, the actual is unknown, but seen before. assigned != actual. The assigned is not seen before.
        NORMAL pre-novelty learning

        actual writer is unknown writer that is seen before, the assigned is the unknown writer that is not seen before
        :return:
        """
        if len(self.unknown_writer_learned_on_the_fly_id_set) < 1:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_11.__name__)
            return False, None

        actual_writer = TestDataSampler.sample_from_set(
            self.unknown_writer_learned_on_the_fly_id_set)
        # sample review from this actual writer
        try:
            review_id = self.unknown_writer_non_novel_dict[actual_writer].pop()
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_11.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # assigned writer id is not seen before
        if len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) == 0:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_11.__name__)
            return False, None
        # endif
        assigned_unknown_writer_not_seen_before = \
            TestDataSampler.sample_from_set(
                self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set)
        original_review_json["assigned_writer_id"] = assigned_unknown_writer_not_seen_before

        assert actual_writer != assigned_unknown_writer_not_seen_before
        assert actual_writer in self.unknown_writer_id_set and actual_writer in self.unknown_writer_learned_on_the_fly_id_set
        assert assigned_unknown_writer_not_seen_before in self.unknown_writer_id_set
        assert assigned_unknown_writer_not_seen_before not in self.unknown_writer_learned_on_the_fly_id_set

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = False
        original_review_json["scenario"] = self.scenario_11.__name__

        return True, original_review_json

    def scenario_11_with_novelty(self):
        if len(self.unknown_writer_learned_on_the_fly_id_set) < 1:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_11_with_novelty.__name__)
            return False, None

        actual_writer = TestDataSampler.sample_from_set(
            self.unknown_writer_learned_on_the_fly_id_set)
        # sample review from this actual writer

        if actual_writer not in self.unknown_writer_novel_dict:
            return False, None

        if len(self.unknown_writer_novel_dict[actual_writer]) == 0:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_11_with_novelty.__name__)
            return False, None

        try:
            review_id = self.unknown_writer_novel_dict[actual_writer].pop()
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_11_with_novelty.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # assigned writer id is not seen before
        if len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) == 0:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_11.__name__)
            return False, None
        # endif
        assigned_unknown_writer_not_seen_before = \
            TestDataSampler.sample_from_set(
                self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set)
        original_review_json["assigned_writer_id"] = assigned_unknown_writer_not_seen_before

        assert actual_writer != assigned_unknown_writer_not_seen_before
        assert actual_writer in self.unknown_writer_id_set and actual_writer in self.unknown_writer_learned_on_the_fly_id_set
        assert assigned_unknown_writer_not_seen_before in self.unknown_writer_id_set
        assert assigned_unknown_writer_not_seen_before not in self.unknown_writer_learned_on_the_fly_id_set

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = True
        original_review_json["scenario"] = self.scenario_11_with_novelty.__name__

        return True, original_review_json

    def scenario_12(self):
        """
        ------------------------------------------------------------------------------------
                        assigned writer    actual writer
        ------------------------------------------------------------------------------------
        scenario_12:         w_15               w_17
        The assigned is unknown (seen before), the actual is another unknown not seen before.
        NORMAL pre-novelty learning

        :return:
        """
        if len(self.unknown_writer_learned_on_the_fly_id_set) < 1:
            if self.debug_mode:
                print(f"SAMPLE ERROR:", self.scenario_12.__name__)
            return False, None

        if len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) == 0:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_12.__name__)
            return False, None

        # (1) actual writer is unknown writer, the assigned is the same as actual writer
        # (2) the writer id is not seen before, not in the self.unknown_writer_learned_on_the_fly_set
        # (3) the sampler should memorize this writer id
        unknown_writer_not_seen_before = \
            TestDataSampler.sample_from_set(
                self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set)

        try:
            review_id = self.unknown_writer_non_novel_dict[unknown_writer_not_seen_before].pop(
            )
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_12.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # ##################### sample assigned writer id ########################
        # assigned writer id is NOT the same as the actual writer id, it is another unknown writer not seen before
        another_unknown_writer_seen_before = TestDataSampler.sample_from_set(
            self.unknown_writer_learned_on_the_fly_id_set)

        actual_writer_id = original_review_json["reviewerID"]
        original_review_json["assigned_writer_id"] = another_unknown_writer_seen_before
        assert another_unknown_writer_seen_before != actual_writer_id
        assert actual_writer_id in self.unknown_writer_id_set
        assert actual_writer_id not in self.unknown_writer_learned_on_the_fly_id_set

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = False
        original_review_json["scenario"] = self.scenario_12.__name__

        return True, original_review_json

    def scenario_12_with_novelty(self):
        """
        ------------------------------------------------------------------------------------
                        assigned writer    actual writer
        ------------------------------------------------------------------------------------
        scenario_12_with_novelty:         w_15               w_17
        The assigned is unknown (seen before), the actual is another unknown not seen before.
        NORMAL pre-novelty learning

        :return:
        """
        if len(self.unknown_writer_learned_on_the_fly_id_set) < 1:
            if self.debug_mode:
                print(f"SAMPLE ERROR:", self.scenario_12.__name__)
            return False, None

        if len(self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set) == 0:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_12.__name__)
            return False, None

        # (1) actual writer is unknown writer, the assigned is the same as actual writer
        # (2) the writer id is not seen before, not in the self.unknown_writer_learned_on_the_fly_set
        # (3) the sampler should memorize this writer id
        unknown_writer_not_seen_before = \
            TestDataSampler.sample_from_set(
                self.unknown_writer_id_set - self.unknown_writer_learned_on_the_fly_id_set)

        try:
            review_id = self.unknown_writer_novel_dict[unknown_writer_not_seen_before].pop(
            )
        except:
            if self.debug_mode:
                print("SAMPLE ERROR:", self.scenario_12.__name__)
            return False, None

        # assert review_id not in self.review_id_pool

        if review_id in self.review_id_pool:
            return False, None

        self.review_id_pool.add(review_id)
        original_review_json = self.all_review_id_to_json_dict[review_id]
        original_review_json["review_id"] = review_id

        # ##################### sample assigned writer id ########################
        # assigned writer id is NOT the same as the actual writer id, it is another unknown writer not seen before
        another_unknown_writer_seen_before = TestDataSampler.sample_from_set(
            self.unknown_writer_learned_on_the_fly_id_set)

        actual_writer_id = original_review_json["reviewerID"]
        original_review_json["assigned_writer_id"] = another_unknown_writer_seen_before
        assert another_unknown_writer_seen_before != actual_writer_id
        assert actual_writer_id in self.unknown_writer_id_set
        assert actual_writer_id not in self.unknown_writer_learned_on_the_fly_id_set

        # novel or not?
        original_review_json["class_novel"] = False
        original_review_json["other_novel"] = True
        original_review_json["scenario"] = self.scenario_12.__name__

        return True, original_review_json


    def text_data_preprocessing(self, text):
        """
        :return:
        """
        text = text.replace("\0", "")
        return text


    def write_to_csv_and_json_format(self):
        csv_file_path = os.path.join(
            self.output_dir, f"{self.output_file_name_base}_single_df.csv")
        # json_file_path = os.path.join(self.output_dir, f"{self.output_file_name_base}_single_df.json")
        # tmp_writer_id_path = os.path.join(self.tmp_output_dir, f"{self.output_file_name_base}_writer_id.txt")
        tmp_file_path = os.path.join(
            self.tmp_output_folder, f"{self.output_file_name_base}_single_df.csv")

        with open(csv_file_path, mode="w", encoding="utf16") as csv_fout, \
                open(tmp_file_path, mode="w", encoding="utf16") as tmp_fout:
            # open(json_file_path, mode="w", encoding="utf16") as json_fout:
            # instance_id, text, writer_id, sentiment, product/service id, product category
            fieldnames = ["instanceid", "text", "reported_writer_id", "real_writer_id", "sentiment", "product",
                          "novelty_indicator", "novel_instance", "text_id"]
            csv_writer = csv.DictWriter(
                csv_fout, fieldnames=fieldnames, quotechar="|", delimiter=",")
            csv_writer.writeheader()

            tmp_fieldnames = ["instanceid", "reported_writer_id", "real_writer_id", "sentiment", "product",
                              "novelty_indicator", "novel_instance", "text_id", "scenario"]
            tmp_csv_writer = csv.DictWriter(
                tmp_fout, fieldnames=tmp_fieldnames, quotechar="|", delimiter=",")
            tmp_csv_writer.writeheader()

            index = -1
            for json_obj in self.test_trial_json_list:
                index += 1
                assert "assigned_writer_id" in json_obj
                # Note that it is assigned writer_id
                assigned_writer_id = self.all_writer_id_to_num_dict[json_obj["assigned_writer_id"]]
                actual_writer_id = self.all_writer_id_to_num_dict[json_obj["reviewerID"]]
                review_rating = int(json_obj["overall"])
                text = json_obj["reviewText"]
                text = text.replace("\n", "</p>")
                text = self.text_data_preprocessing(text)

                words = text.split()
                if len(words) > 15:
                    words = words[5:-5]
                    # random.shuffle(words)
                    text = " ".join(words)

                product_id = json_obj["asin"]


                review_id = json_obj["review_id"]
                product_category = json_obj["category"][0]

                if review_id == self.red_light_instance["review_id"]:
                    self.novel_flag = True
                    self.red_light_index = index
                # endif

                # tmp_fout.write(f"{assigned_writer_id}\t{json_obj['assigned_writer_id']}\t{actual_writer_id}\t{json_obj['reviewerID']}\n")

                # novelty_indicator = detection column, and novel_instance = novel column

                import uuid
                review_id = str(uuid.uuid4())

                entry = {"instanceid": index,
                         "text": text,
                         "reported_writer_id": assigned_writer_id,
                         "real_writer_id": actual_writer_id,
                         "sentiment": review_rating,
                         "product": json_obj["asin"],
                         "novelty_indicator": 1 if self.novel_flag else 0,
                         "novel_instance": 1 if json_obj["final_novel"] else 0,
                         "text_id": review_id
                         }

                # output csv file
                csv_writer.writerow(entry)

                del entry["text"]
                entry["scenario"] = json_obj["scenario"]
                tmp_csv_writer.writerow(entry)
                # json_fout.write(f"{json.dumps(json_obj)}\n")
            # endfor

    def output_meta_data(self, meta_data_obj):
        meta_data_file = os.path.join(
            self.output_dir, f"{self.output_file_name_base}_metadata.json")
        with open(meta_data_file, mode="w", encoding="utf16") as fout:
            json.dump(meta_data_obj, fout, indent=4)
        pass




def load_all_writer_cate_to_pos_neg_reviews():
    """
    load writer_cate_pos_neg_dict

    writer_cate_pos_neg_dict structure
    ---
    writerID:
       category:
           pos: [review_id_list]
           neg: [review_id_list]
    ---
    :return:
    """
    all_review_to_json_obj_dict_file = "input/all_review_id_to_json_dict_uuid.json"

    with open(all_review_to_json_obj_dict_file, mode="r") as fin:
        all_review_to_json_obj_dict = json.load(fin)
    # endwith

    writer_stats = {}
    writer_cate_pos_neg_dict = {}
    for review_id, json_obj in all_review_to_json_obj_dict.items():
        rating = float(json_obj["overall"])
        category = json_obj["category"][0]
        writer_id = json_obj["reviewerID"]

        if writer_id not in writer_stats:
            writer_stats[writer_id] = {"pos": 0, "neg": 0}
        # endif
        if writer_id not in writer_cate_pos_neg_dict:
            writer_cate_pos_neg_dict[writer_id] = {}
        # endif

        if category not in writer_stats[writer_id]:
            writer_stats[writer_id][category] = {"pos": 0, "neg": 0}
        # endif
        if category not in writer_cate_pos_neg_dict[writer_id]:
            writer_cate_pos_neg_dict[writer_id][category] = {
                "pos": [], "neg": []}
        # endif

        if rating > 3.0:
            writer_stats[writer_id]["pos"] += 1
            writer_stats[writer_id][category]["pos"] += 1
            writer_cate_pos_neg_dict[writer_id][category]["pos"].append(
                review_id)
        # endif

        if rating < 3.0:
            writer_stats[writer_id]["neg"] += 1
            writer_stats[writer_id][category]["neg"] += 1
            writer_cate_pos_neg_dict[writer_id][category]["neg"].append(
                review_id)
        # endif

    # endfor
    return writer_cate_pos_neg_dict


def object_level_novelty_load_user_and_review_info_for_four_cases(train_category,
                                                                  novel_category_list,
                                                                  known_writer_num,
                                                                  known_writer_review_num):
    """
    For object_level_novelty, we control the variable - sentiment. Make sure we only have reviews in "positive"
    category.

    known_writer_non_novel_dict : positive reviews -> {'Electronics'}
    known_writer_novel_dict : positive reviews -> {'Home & Kitchen', 'Clothing, Shoes & Jewelry'}
    unknown_writer_non_novel_dict : positive reviews -> {'Electronics'}
    unknown_writer_novel_dict : positive reviews -> {'Home & Kitchen', 'Clothing, Shoes & Jewelry'}

    :param train_category:
    :param novel_category_list:
    :param known_writer_num:
    :param known_writer_review_num:
    :return:
    """
    output_base = "./output"
    output_dir = os.path.join(
        output_base, f"writer_{known_writer_num}_review_{known_writer_review_num}")
    assert os.path.exists(output_dir)

    # -------------- load known & unknown writer_id ----------------
    known_writer_id_file = os.path.join(
        output_dir, "known_writer_id_list.json")
    unknown_writer_id_file = os.path.join(
        output_dir, "unknown_writer_id_list.json")
    with open(known_writer_id_file, mode="r") as fin:
        known_writer_id_set = json.load(fin)
        known_writer_id_set = set(known_writer_id_set)
    # endwith

    with open(unknown_writer_id_file, mode="r") as fin:
        unknown_writer_id_set = json.load(fin)
        unknown_writer_id_set = set(unknown_writer_id_set)
    # endwith

    # ###################### load train_data review_id dict ######################
    train_data_known_writer_review_id_file = os.path.join(
        output_dir, "training_data_writer_to_review_id.json")
    with open(train_data_known_writer_review_id_file, mode="r") as fin:
        train_data_known_writer_review_id_dict = json.load(fin)
    # endwith

    # get all training data review_id set
    all_train_review_id_set = set()
    for writer_id, review_id_list in train_data_known_writer_review_id_dict.items():
        assert writer_id in known_writer_id_set
        all_train_review_id_set.update(review_id_list)
    # endfor

    known_writer_non_novel_dict = {}
    known_writer_novel_dict = {}
    unknown_writer_non_novel_dict = {}
    unknown_writer_novel_dict = {}

    writer_cate_pos_neg_dict = load_all_writer_cate_to_pos_neg_reviews()

    for writer_id, cate_pos_neg_review_list_dict in writer_cate_pos_neg_dict.items():

        # known writer
        if writer_id in known_writer_id_set:
            # (1) known non-novel
            non_novel_review_set = set(
                cate_pos_neg_review_list_dict[train_category]["pos"])
            intersect_set = non_novel_review_set.intersection(
                all_train_review_id_set)
            assert len(intersect_set) == known_writer_review_num
            # known non-novel
            known_writer_non_novel_dict[writer_id] = non_novel_review_set - \
                                                     all_train_review_id_set

            # (2) known novel
            novel_review_set = set()
            for cate in novel_category_list:
                novel_review_set.update(
                    cate_pos_neg_review_list_dict[cate]["pos"])
            # endfor
            known_writer_novel_dict[writer_id] = novel_review_set
        # endif

        # unknown writer
        if writer_id in unknown_writer_id_set:
            # unknown non-novel
            non_novel_review_set = set(
                cate_pos_neg_review_list_dict[train_category]["pos"])
            intersect_set = non_novel_review_set.intersection(
                all_train_review_id_set)
            assert len(intersect_set) == 0
            # unknown non-novel
            unknown_writer_non_novel_dict[writer_id] = non_novel_review_set

            # unknown novel
            novel_review_set = set()
            for cate in novel_category_list:
                novel_review_set.update(
                    cate_pos_neg_review_list_dict[cate]["pos"])
            # endfor
            unknown_writer_novel_dict[writer_id] = novel_review_set
        # endif
    # endfor

    return known_writer_non_novel_dict, \
           known_writer_novel_dict, \
           unknown_writer_non_novel_dict, \
           unknown_writer_novel_dict, \
           known_writer_id_set, \
           unknown_writer_id_set


def class_level_novelty_load_user_and_review_info_for_four_cases(train_category,
                                                                 known_writer_num,
                                                                 known_writer_review_num):
    """
    Jan 28, 2022 added

    For sentiment level novelty, we control the variable - category and sentiment.

    Make sure we only have reviews in "Electronics" category and reviews are positive.

    The novelty in the test trial only comes from the new writers.

    -------------

    known_writer_non_novel_dict : positive reviews

    known_writer_novel_dict : should be None

    unknown_writer_non_novel_dict : positive reviews

    unknown_writer_novel_dict : should be None

    :param known_writer_num:
    :param known_writer_review_num:
    :return:
    """
    output_base = "./output"
    output_dir = os.path.join(
        output_base, f"writer_{known_writer_num}_review_{known_writer_review_num}")
    assert os.path.exists(output_dir)

    # -------------- load known & unknown writer_id ----------------
    known_writer_id_file = os.path.join(
        output_dir, "known_writer_id_list.json")
    unknown_writer_id_file = os.path.join(
        output_dir, "unknown_writer_id_list.json")
    with open(known_writer_id_file, mode="r") as fin:
        known_writer_id_set = json.load(fin)
        known_writer_id_set = set(known_writer_id_set)
    # endwith

    with open(unknown_writer_id_file, mode="r") as fin:
        unknown_writer_id_set = json.load(fin)
        unknown_writer_id_set = set(unknown_writer_id_set)
    # endwith

    # --------------- load train_data review_id dict ---------------
    train_data_known_writer_review_id_file = os.path.join(
        output_dir, "training_data_writer_to_review_id.json")
    with open(train_data_known_writer_review_id_file, mode="r") as fin:
        train_data_known_writer_review_id_dict = json.load(fin)
    # endwith

    # get all training data review_id set
    all_train_review_id_set = set()
    for writer_id, review_id_list in train_data_known_writer_review_id_dict.items():
        assert writer_id in known_writer_id_set
        all_train_review_id_set.update(review_id_list)
    # endfor

    known_writer_non_novel_dict = {}
    known_writer_novel_dict = None
    unknown_writer_non_novel_dict = {}
    unknown_writer_novel_dict = None

    writer_cate_pos_neg_dict = load_all_writer_cate_to_pos_neg_reviews()

    for writer_id, cate_pos_neg_review_list_dict in writer_cate_pos_neg_dict.items():

        # known writer
        if writer_id in known_writer_id_set:
            pos_review_set = set(
                cate_pos_neg_review_list_dict[train_category]["pos"])
            intersect_set = pos_review_set.intersection(
                all_train_review_id_set)
            assert len(intersect_set) == known_writer_review_num
            # known non-novel
            known_writer_non_novel_dict[writer_id] = pos_review_set - \
                                                     all_train_review_id_set
            # known novel
            # known_writer_novel_dict[writer_id] = set(
            #     cate_pos_neg_review_list_dict[train_category]["neg"])
        # endif

        # unknown writer
        if writer_id in unknown_writer_id_set:
            pos_review_set = set(
                cate_pos_neg_review_list_dict[train_category]["pos"])
            intersect_set = pos_review_set.intersection(
                all_train_review_id_set)
            assert len(intersect_set) == 0
            # unknown non-novel
            unknown_writer_non_novel_dict[writer_id] = pos_review_set
            # unknown novel
            # unknown_writer_novel_dict[writer_id] = set(
            #     cate_pos_neg_review_list_dict[train_category]["neg"])
        # endif
    # endfor

    validation_split(known_writer_non_novel_dict,
                     known_writer_novel_dict,
                     unknown_writer_non_novel_dict,
                     unknown_writer_novel_dict)

    return known_writer_non_novel_dict, \
           known_writer_novel_dict, \
           unknown_writer_non_novel_dict, \
           unknown_writer_novel_dict, \
           known_writer_id_set, \
           unknown_writer_id_set


def sentiment_level_novelty_load_user_and_review_info_for_four_cases(train_category,
                                                                     known_writer_num,
                                                                     known_writer_review_num):
    """
    For sentiment level novelty, we control the variable - category. Make sure we only have reviews in "Electronics"
    category. And we treat the reviews in Electronics category with negative sentiment as novel reviews.

    known_writer_non_novel_dict : positive reviews
    known_writer_novel_dict : negative reviews
    unknown_writer_non_novel_dict : positive reviews
    unknown_writer_novel_dict : negative reviews

    :param known_writer_num:
    :param known_writer_review_num:
    :return:
    """
    output_base = "./output"
    output_dir = os.path.join(
        output_base, f"writer_{known_writer_num}_review_{known_writer_review_num}")
    assert os.path.exists(output_dir)

    # -------------- load known & unknown writer_id ----------------
    known_writer_id_file = os.path.join(
        output_dir, "known_writer_id_list.json")
    unknown_writer_id_file = os.path.join(
        output_dir, "unknown_writer_id_list.json")
    with open(known_writer_id_file, mode="r") as fin:
        known_writer_id_set = json.load(fin)
        known_writer_id_set = set(known_writer_id_set)
    # endwith

    with open(unknown_writer_id_file, mode="r") as fin:
        unknown_writer_id_set = json.load(fin)
        unknown_writer_id_set = set(unknown_writer_id_set)
    # endwith

    # --------------- load train_data review_id dict ---------------
    train_data_known_writer_review_id_file = os.path.join(
        output_dir, "training_data_writer_to_review_id.json")
    with open(train_data_known_writer_review_id_file, mode="r") as fin:
        train_data_known_writer_review_id_dict = json.load(fin)
    # endwith

    # get all training data review_id set
    all_train_review_id_set = set()
    for writer_id, review_id_list in train_data_known_writer_review_id_dict.items():
        assert writer_id in known_writer_id_set
        all_train_review_id_set.update(review_id_list)
    # endfor

    known_writer_non_novel_dict = {}
    known_writer_novel_dict = {}
    unknown_writer_non_novel_dict = {}
    unknown_writer_novel_dict = {}

    writer_cate_pos_neg_dict = load_all_writer_cate_to_pos_neg_reviews()

    for writer_id, cate_pos_neg_review_list_dict in writer_cate_pos_neg_dict.items():

        # known writer
        if writer_id in known_writer_id_set:
            pos_review_set = set(
                cate_pos_neg_review_list_dict[train_category]["pos"])
            intersect_set = pos_review_set.intersection(
                all_train_review_id_set)
            assert len(intersect_set) == known_writer_review_num
            # known non-novel
            known_writer_non_novel_dict[writer_id] = pos_review_set - \
                                                     all_train_review_id_set
            # known novel
            known_writer_novel_dict[writer_id] = set(
                cate_pos_neg_review_list_dict[train_category]["neg"])
        # endif

        # unknown writer
        if writer_id in unknown_writer_id_set:
            pos_review_set = set(
                cate_pos_neg_review_list_dict[train_category]["pos"])
            intersect_set = pos_review_set.intersection(
                all_train_review_id_set)
            assert len(intersect_set) == 0
            # unknown non-novel
            unknown_writer_non_novel_dict[writer_id] = pos_review_set
            # known novel
            unknown_writer_novel_dict[writer_id] = set(
                cate_pos_neg_review_list_dict[train_category]["neg"])
        # endif
    # endfor

    validation_split(known_writer_non_novel_dict,
                     known_writer_novel_dict,
                     unknown_writer_non_novel_dict,
                     unknown_writer_novel_dict)

    return known_writer_non_novel_dict, \
           known_writer_novel_dict, \
           unknown_writer_non_novel_dict, \
           unknown_writer_novel_dict, \
           known_writer_id_set, \
           unknown_writer_id_set


def load_all_shipping_review_id_to_json_dict_old():
    """
    old one used before March 2022, contains only around 800 reviews
    :return:
    """
    all_shipping_review_id_to_json_dict = {}
    with open("./shipping_review_input/shipping_review_json_list.txt", mode="r") as fin:
        for line in fin:
            line = line.strip()
            json_obj = json.loads(line)
            review_id = json_obj["review_id"]
            all_shipping_review_id_to_json_dict[review_id] = json_obj
        # endor
    # endwith
    print(
        f"There are {len(all_shipping_review_id_to_json_dict)} shipping reviews.")
    return all_shipping_review_id_to_json_dict


def load_all_shipping_review_id_to_json_dict():
    all_shipping_review_id_to_json_dict = {}

    # ####### (1) the shipping review, that I crawled from online March 10, 2022 #########
    with open("./shipping_review_input_March_10_2022/shipping_review_json_list.txt", mode="r") as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                continue
            # endif
            json_obj = json.loads(line)
            review_id = json_obj["review_id"]
            all_shipping_review_id_to_json_dict[review_id] = json_obj
        # endor
    # endwith

    # ####### (2) reviewers that write multiple reviews, provided by Eric #######
    with open(
            "./shipping_review_input_March_10_2022/productive_reviewer_and_reviews/shipping_reviewers_that_write_multiple_reviews.txt",
            mode="r") as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                continue
            # endif
            json_obj = json.loads(line)
            review_id = json_obj["review_id"]
            all_shipping_review_id_to_json_dict[review_id] = json_obj
        # endfor
    # endwith

    print(
        f"There are {len(all_shipping_review_id_to_json_dict)} shipping reviews.")
    return all_shipping_review_id_to_json_dict


def load_shipping_writer_to_review_id_list_dict():
    """
    March 10, 2022
    :return:
    """
    shipping_writer_to_review_id_list_dict = {}

    # ########### (1) ###########
    shipping_review_file_path = "./shipping_review_input_March_10_2022/shipping_review_json_list.txt"
    with open(shipping_review_file_path, mode="r") as fin:
        for line in fin:
            line = line.strip()
            json_obj = json.loads(line)
            review_id = json_obj["review_id"]
            writer_id = json_obj["reviewerID"]

            if writer_id not in shipping_writer_to_review_id_list_dict:
                shipping_writer_to_review_id_list_dict[writer_id] = set()
            # endif

            shipping_writer_to_review_id_list_dict[writer_id].add(review_id)
        # endfor
    # endwith

    # ########### (2) #############
    shipping_review_file_path = "./shipping_review_input_March_10_2022/productive_reviewer_and_reviews/shipping_reviewers_that_write_multiple_reviews.txt"
    with open(shipping_review_file_path, mode="r") as fin:
        for line in fin:
            line = line.strip()
            json_obj = json.loads(line)
            review_id = json_obj["review_id"]
            writer_id = json_obj["reviewerID"]

            if writer_id not in shipping_writer_to_review_id_list_dict:
                shipping_writer_to_review_id_list_dict[writer_id] = set()
            # endif

            shipping_writer_to_review_id_list_dict[writer_id].add(review_id)
        # endfor
    # endwith

    return shipping_writer_to_review_id_list_dict


def load_shipping_writer_id_set_old():
    # all the unknown writer are in the shipping review dataset
    reviewer_id_stats_file = "./shipping_review_input/shipping_reviewer_stats.txt"
    unknown_writer_id_set = set()
    with open(reviewer_id_stats_file, mode="r") as fin:
        for line in fin:
            line = line.strip()
            parts = line.split()
            unknown_writer_id_set.add(parts[0])
        # endfor
    # endwith
    return unknown_writer_id_set


def load_shipping_writer_id_set():
    """
    March 10, 2022
    :return:
    """
    # all the unknown writer are in the shipping review dataset
    reviewer_id_stats_file = "./shipping_review_input_March_10_2022/reviewer_to_num_dict.txt"
    unknown_writer_id_set = set()

    # ######## (1) ###########
    with open(reviewer_id_stats_file, mode="r") as fin:
        for line in fin:
            line = line.strip()
            parts = line.split()
            unknown_writer_id_set.add(parts[0])
        # endfor
    # endwith

    # ######### (2) ##########
    with open("./shipping_review_input_March_10_2022/productive_reviewer_and_reviews/productive_reviewer_stats.txt",
              mode="r") as fin:
        for line in fin:
            line = line.strip()
            parts = line.split()
            unknown_writer_id_set.add(parts[0])
        # endfor
    # endwith

    return unknown_writer_id_set


@DeprecationWarning
def interaction_level_novelty_load_user_and_review_info_for_four_cases(train_category,
                                                                       known_writer_num,
                                                                       known_writer_review_num):
    """
    Training data contains "Electronics" and positive reviews
    For interaction_level_novelty, "shipping review"

    known_writer_non_novel_dict : positive reviews -> {'Electronics'}
    known_writer_novel_dict : None

    unknown_writer_non_novel_dict : None
    unknown_writer_novel_dict : positive reviews -> {'Shipping review'}

    :param train_category:
    :param novel_category_list:
    :param known_writer_num:
    :param known_writer_review_num:
    :return:
    """
    output_base = "./output"
    output_dir = os.path.join(
        output_base, f"writer_{known_writer_num}_review_{known_writer_review_num}")
    assert os.path.exists(output_dir)

    # --------------------------- load known writer_id ------------------------
    known_writer_id_file = os.path.join(
        output_dir, "known_writer_id_list.json")
    with open(known_writer_id_file, mode="r") as fin:
        known_writer_id_set = json.load(fin)
        known_writer_id_set = set(known_writer_id_set)
    # endwith

    # --------------------------- load unknown writer_id -----------------------
    unknown_writer_id_set = load_shipping_writer_id_set()

    # ###################### load train_data review_id dict ######################
    train_data_known_writer_review_id_file = os.path.join(
        output_dir, "training_data_writer_to_review_id.json")
    with open(train_data_known_writer_review_id_file, mode="r") as fin:
        train_data_known_writer_review_id_dict = json.load(fin)
    # endwith

    # get all training data review_id set
    all_train_review_id_set = set()
    for writer_id, review_id_list in train_data_known_writer_review_id_dict.items():
        assert writer_id in known_writer_id_set
        all_train_review_id_set.update(review_id_list)
    # endfor

    known_writer_non_novel_dict = {}
    known_writer_novel_dict = None
    unknown_writer_non_novel_dict = None
    unknown_writer_novel_dict = {}

    # ################ get known_writer_non_novel_dict ###############
    writer_cate_pos_neg_dict = load_all_writer_cate_to_pos_neg_reviews()
    for writer_id, cate_pos_neg_review_list_dict in writer_cate_pos_neg_dict.items():

        # known writer
        if writer_id in known_writer_id_set:
            # (1) known non-novel
            non_novel_review_set = set(
                cate_pos_neg_review_list_dict[train_category]["pos"])
            intersect_set = non_novel_review_set.intersection(
                all_train_review_id_set)
            assert len(intersect_set) == known_writer_review_num
            # known non-novel
            known_writer_non_novel_dict[writer_id] = non_novel_review_set - \
                                                     all_train_review_id_set
        # endif
    # endfor

    # ############### get unknown_writer_novel_dict #################
    unknown_writer_novel_dict = load_shipping_writer_to_review_id_list_dict()

    return known_writer_non_novel_dict, \
           known_writer_novel_dict, \
           unknown_writer_non_novel_dict, \
           unknown_writer_novel_dict, \
           known_writer_id_set, \
           unknown_writer_id_set


def load_EASY_all_story_writer_to_review_id_list_dict():
    """
    We use novelists' text as novel text, like the shipping reviews.

    longer story text, which is easier to detect
    """
    story_writer_to_review_id_list_dict = defaultdict(set)

    story_review_file_path = "../5_Novelist_Dataset/output/story_json_list_easy.txt"
    with open(story_review_file_path, mode="r", encoding="utf16") as fin:
        for line in fin:
            line = line.strip()
            json_obj = json.loads(line)
            review_id = json_obj["review_id"]
            writer_id = json_obj["reviewerID"]

            story_writer_to_review_id_list_dict[writer_id].add(review_id)
        # endfor
    # endwith

    return story_writer_to_review_id_list_dict


def load_HARD_all_story_writer_to_review_id_list_dict():
    """
    We use novelists' text as novel text, like the shipping reviews.

    shorter story text, which is harder to detect
    """
    story_writer_to_review_id_list_dict = defaultdict(set)

    story_review_file_path = "../5_Novelist_Dataset/output/story_json_list_hard.txt"
    with open(story_review_file_path, mode="r", encoding="utf16") as fin:
        for line in fin:
            line = line.strip()
            json_obj = json.loads(line)
            review_id = json_obj["review_id"]
            writer_id = json_obj["reviewerID"]

            story_writer_to_review_id_list_dict[writer_id].add(review_id)
        # endfor
    # endwith

    return story_writer_to_review_id_list_dict


def load_story_writer_id_set():
    # all the unknown writer are in the shipping review dataset
    reviewer_id_stats_file = "../5_Novelist_Dataset/output/author_to_num_dict.txt"
    unknown_writer_id_set = set()
    with open(reviewer_id_stats_file, mode="r") as fin:
        for line in fin:
            line = line.strip()
            parts = line.split()
            unknown_writer_id_set.add(parts[0])
        # endfor
    # endwith
    return unknown_writer_id_set


def load_EASY_all_story_id_to_json_dict():
    all_story_review_id_to_json_dict = {}
    with open("../5_Novelist_Dataset/output/story_json_list_easy.txt", mode="r", encoding="utf16") as fin:
        for line in fin:
            line = line.strip()
            json_obj = json.loads(line)
            review_id = json_obj["review_id"]
            all_story_review_id_to_json_dict[review_id] = json_obj
        # endor
    # endwith
    print(
        f"There are {len(all_story_review_id_to_json_dict)} easy story reviews.")
    return all_story_review_id_to_json_dict


def load_HARD_all_story_id_to_json_dict():
    all_story_review_id_to_json_dict = {}
    with open("../5_Novelist_Dataset/output/story_json_list_hard.txt", mode="r", encoding="utf16") as fin:
        for line in fin:
            line = line.strip()
            json_obj = json.loads(line)
            review_id = json_obj["review_id"]
            all_story_review_id_to_json_dict[review_id] = json_obj
        # endor
    # endwith
    print(
        f"There are {len(all_story_review_id_to_json_dict)} hard story reviews.")
    return all_story_review_id_to_json_dict


def interaction_level_novelty_PRE_NOVELTY_load_user_and_review_info_for_four_cases(train_category,
                                                                                   known_writer_num,
                                                                                   known_writer_review_num
                                                                                   ):
    """
    March 10, 2022

    Interaction level novelty is special.
    In hard mode, in the pre-novelty phase, there could be unknown writer in Amazon review
    But in the post-novelty phase, the unknown writer and novelty should only come from novelist's story

    # ----------------- pre-novelty phase ------------------
    # known_writer_non_novel_dict = from training data
    # known_writer_novel_dict = None
    # unknown_writer_non_novel_dict = CLASS_unknown_writer_non_novel_dict
    # unknown_writer_novel_dict = author_unknown_novel_dict
    #
    # known_writer_id_set = from training data
    # unknown_writer_id_set = CLASS unknown

    # ----------------- post-novelty phase -----------------
    # known_writer_non_novel_dict = from training data
    # known_writer_novel_dict = None
    # unknown_writer_non_novel_dict = None
    # unknown_writer_novel_dict = author_unknown_novel_dict
    #
    # known_writer_id_set = from training data
    # unknown_writer_id_set = shipping authors


    :param train_category:
    :param novel_category_list:
    :param known_writer_num:
    :param known_writer_review_num:
    :return:
    """

    output_base = "./output"
    output_dir = os.path.join(
        output_base, f"writer_{known_writer_num}_review_{known_writer_review_num}")
    assert os.path.exists(output_dir)

    # --------------------------- load known writer_id ------------------------
    known_writer_id_file = os.path.join(
        output_dir, "known_writer_id_list.json")
    with open(known_writer_id_file, mode="r") as fin:
        known_writer_id_set = json.load(fin)
        known_writer_id_set = set(known_writer_id_set)
    # endwith

    # ###################### load train_data review_id dict ######################
    train_data_known_writer_review_id_file = os.path.join(
        output_dir, "training_data_writer_to_review_id.json")
    with open(train_data_known_writer_review_id_file, mode="r") as fin:
        train_data_known_writer_review_id_dict = json.load(fin)
    # endwith

    # get all training data review_id set
    all_train_review_id_set = set()
    for writer_id, review_id_list in train_data_known_writer_review_id_dict.items():
        assert writer_id in known_writer_id_set
        all_train_review_id_set.update(review_id_list)
    # endfor

    known_writer_non_novel_dict = {}
    known_writer_novel_dict = None
    unknown_writer_non_novel_dict = None
    unknown_writer_novel_dict = {}

    # ############ load class_level_novelty ###########
    CLASS_known_writer_non_novel_dict, \
    CLASS_known_writer_novel_dict, \
    CLASS_unknown_writer_non_novel_dict, \
    CLASS_unknown_writer_novel_dict, \
    CLASS_known_writer_id_set, \
    CLASS_unknown_writer_id_set = class_level_novelty_load_user_and_review_info_for_four_cases(
        train_category, known_writer_num, known_writer_review_num)

    # assign
    unknown_writer_id_set = set()
    unknown_writer_non_novel_dict = CLASS_unknown_writer_non_novel_dict
    unknown_writer_id_set.update(CLASS_unknown_writer_id_set)

    # ################ get known_writer_non_novel_dict ###############
    writer_cate_pos_neg_dict = load_all_writer_cate_to_pos_neg_reviews()
    for writer_id, cate_pos_neg_review_list_dict in writer_cate_pos_neg_dict.items():

        # known writer
        if writer_id in known_writer_id_set:
            # (1) known non-novel
            non_novel_review_set = set(
                cate_pos_neg_review_list_dict[train_category]["pos"])
            intersect_set = non_novel_review_set.intersection(
                all_train_review_id_set)
            assert len(intersect_set) == known_writer_review_num
            # known non-novel
            known_writer_non_novel_dict[writer_id] = non_novel_review_set - \
                                                     all_train_review_id_set
        # endif
    # endfor

    # ############### get unknown_writer_novel_dict ################
    unknown_writer_novel_dict = load_shipping_writer_to_review_id_list_dict()

    return known_writer_non_novel_dict, \
           known_writer_novel_dict, \
           unknown_writer_non_novel_dict, \
           unknown_writer_novel_dict, \
           known_writer_id_set, \
           unknown_writer_id_set


def interaction_level_novelty_POST_NOVELTY_load_user_and_review_info_for_four_cases(train_category,
                                                                                    known_writer_num,
                                                                                    known_writer_review_num
                                                                                    ):
    """
    :param train_category:
    :param novel_category_list:
    :param known_writer_num:
    :param known_writer_review_num:
    :return:
    """

    output_base = "./output"
    output_dir = os.path.join(
        output_base, f"writer_{known_writer_num}_review_{known_writer_review_num}")
    assert os.path.exists(output_dir)

    # --------------------------- load known writer_id ------------------------
    known_writer_id_file = os.path.join(
        output_dir, "known_writer_id_list.json")
    with open(known_writer_id_file, mode="r") as fin:
        known_writer_id_set = json.load(fin)
        known_writer_id_set = set(known_writer_id_set)
    # endwith

    # --------------------------- load unknown writer_id -----------------------
    unknown_writer_id_set = load_shipping_writer_id_set()

    # ###################### load train_data review_id dict ######################
    train_data_known_writer_review_id_file = os.path.join(
        output_dir, "training_data_writer_to_review_id.json")
    with open(train_data_known_writer_review_id_file, mode="r") as fin:
        train_data_known_writer_review_id_dict = json.load(fin)
    # endwith

    # get all training data review_id set
    all_train_review_id_set = set()
    for writer_id, review_id_list in train_data_known_writer_review_id_dict.items():
        assert writer_id in known_writer_id_set
        all_train_review_id_set.update(review_id_list)
    # endfor

    known_writer_non_novel_dict = {}
    known_writer_novel_dict = None
    unknown_writer_non_novel_dict = None

    # ################ get known_writer_non_novel_dict ###############
    writer_cate_pos_neg_dict = load_all_writer_cate_to_pos_neg_reviews()
    for writer_id, cate_pos_neg_review_list_dict in writer_cate_pos_neg_dict.items():

        # known writer
        if writer_id in known_writer_id_set:
            # (1) known non-novel
            non_novel_review_set = set(
                cate_pos_neg_review_list_dict[train_category]["pos"])
            intersect_set = non_novel_review_set.intersection(
                all_train_review_id_set)
            assert len(intersect_set) == known_writer_review_num
            # known non-novel
            known_writer_non_novel_dict[writer_id] = non_novel_review_set - \
                                                     all_train_review_id_set
        # endif
    # endfor

    # ############### get unknown_writer_novel_dict ################
    unknown_writer_novel_dict = load_shipping_writer_to_review_id_list_dict()

    assert unknown_writer_id_set == set(unknown_writer_novel_dict.keys())

    return known_writer_non_novel_dict, \
           known_writer_novel_dict, \
           unknown_writer_non_novel_dict, \
           unknown_writer_novel_dict, \
           known_writer_id_set, \
           unknown_writer_id_set


def action_level_novelty_PRE_NOVELTY_load_user_and_review_info_for_four_cases(train_category,
                                                                              known_writer_num,
                                                                              known_writer_review_num,
                                                                              mode):
    """
    Feb 1, 2022

    Action level novelty is special.
    In hard mode, in the pre-novelty phase, there could be unknown in Amazon review
    But in the post-novelty phase, the novelty should only come from novelist's story

    # ----------------- pre-novelty phase ------------------
    # known_writer_non_novel_dict = from training data
    # known_writer_novel_dict = None
    # unknown_writer_non_novel_dict = CLASS_unknown_writer_non_novel_dict
    # unknown_writer_novel_dict = author_unknown_novel_dict
    #
    # known_writer_id_set = from training data
    # unknown_writer_id_set = CLASS unknown

    # ----------------- post-novelty phase -----------------
    # known_writer_non_novel_dict = from training data
    # known_writer_novel_dict = None
    # unknown_writer_non_novel_dict = None
    # unknown_writer_novel_dict = author_unknown_novel_dict
    #
    # known_writer_id_set = from training data
    # unknown_writer_id_set = novelist authors



    Training data contains "Electronics" and positive reviews

    For interaction_level_novelty, "novelists' text"

    known_writer_non_novel_dict : positive reviews -> {'Electronics'}
    known_writer_novel_dict : None

    unknown_writer_non_novel_dict : None
    unknown_writer_novel_dict : positive reviews -> {'novelists' text'}

    :param train_category:
    :param novel_category_list:
    :param known_writer_num:
    :param known_writer_review_num:
    :return:
    """
    assert mode in {"easy", "hard"}

    output_base = "./output"
    output_dir = os.path.join(
        output_base, f"writer_{known_writer_num}_review_{known_writer_review_num}")
    assert os.path.exists(output_dir)

    # --------------------------- load known writer_id ------------------------
    known_writer_id_file = os.path.join(
        output_dir, "known_writer_id_list.json")
    with open(known_writer_id_file, mode="r") as fin:
        known_writer_id_set = json.load(fin)
        known_writer_id_set = set(known_writer_id_set)
    # endwith

    # ###################### load train_data review_id dict ######################
    train_data_known_writer_review_id_file = os.path.join(
        output_dir, "training_data_writer_to_review_id.json")
    with open(train_data_known_writer_review_id_file, mode="r") as fin:
        train_data_known_writer_review_id_dict = json.load(fin)
    # endwith

    # get all training data review_id set
    all_train_review_id_set = set()
    for writer_id, review_id_list in train_data_known_writer_review_id_dict.items():
        assert writer_id in known_writer_id_set
        all_train_review_id_set.update(review_id_list)
    # endfor

    known_writer_non_novel_dict = {}
    known_writer_novel_dict = None
    unknown_writer_non_novel_dict = None
    unknown_writer_novel_dict = {}

    # ############ load class_level_novelty ###########
    CLASS_known_writer_non_novel_dict, \
    CLASS_known_writer_novel_dict, \
    CLASS_unknown_writer_non_novel_dict, \
    CLASS_unknown_writer_novel_dict, \
    CLASS_known_writer_id_set, \
    CLASS_unknown_writer_id_set = class_level_novelty_load_user_and_review_info_for_four_cases(
        train_category, known_writer_num, known_writer_review_num)

    # assign
    unknown_writer_id_set = set()
    unknown_writer_non_novel_dict = CLASS_unknown_writer_non_novel_dict
    unknown_writer_id_set.update(CLASS_unknown_writer_id_set)

    # ################ get known_writer_non_novel_dict ###############
    writer_cate_pos_neg_dict = load_all_writer_cate_to_pos_neg_reviews()
    for writer_id, cate_pos_neg_review_list_dict in writer_cate_pos_neg_dict.items():

        # known writer
        if writer_id in known_writer_id_set:
            # (1) known non-novel
            non_novel_review_set = set(
                cate_pos_neg_review_list_dict[train_category]["pos"])
            intersect_set = non_novel_review_set.intersection(
                all_train_review_id_set)
            assert len(intersect_set) == known_writer_review_num
            # known non-novel
            known_writer_non_novel_dict[writer_id] = non_novel_review_set - \
                                                     all_train_review_id_set
        # endif
    # endfor

    # ############### get unknown_writer_novel_dict ################
    unknown_writer_novel_dict = None
    if mode == "easy":
        unknown_writer_novel_dict = load_EASY_all_story_writer_to_review_id_list_dict()
    # endif

    if mode == "hard":
        unknown_writer_novel_dict = load_HARD_all_story_writer_to_review_id_list_dict()
    # endif

    return known_writer_non_novel_dict, \
           known_writer_novel_dict, \
           unknown_writer_non_novel_dict, \
           unknown_writer_novel_dict, \
           known_writer_id_set, \
           unknown_writer_id_set


def action_level_novelty_POST_NOVELTY_load_user_and_review_info_for_four_cases(train_category,
                                                                               known_writer_num,
                                                                               known_writer_review_num,
                                                                               mode):
    """
    Training data contains "Electronics" and positive reviews

    For interaction_level_novelty, "novelists' text"

    known_writer_non_novel_dict : positive reviews -> {'Electronics'}
    known_writer_novel_dict : None

    unknown_writer_non_novel_dict : None
    unknown_writer_novel_dict : positive reviews -> {'novelists' text'}

    :param train_category:
    :param novel_category_list:
    :param known_writer_num:
    :param known_writer_review_num:
    :return:
    """
    assert mode in {"easy", "hard"}

    output_base = "./output"
    output_dir = os.path.join(
        output_base, f"writer_{known_writer_num}_review_{known_writer_review_num}")
    assert os.path.exists(output_dir)

    # --------------------------- load known writer_id ------------------------
    known_writer_id_file = os.path.join(
        output_dir, "known_writer_id_list.json")
    with open(known_writer_id_file, mode="r") as fin:
        known_writer_id_set = json.load(fin)
        known_writer_id_set = set(known_writer_id_set)
    # endwith

    # --------------------------- load unknown writer_id -----------------------
    # all the unknown writer are in the shipping review dataset
    reviewer_id_stats_file = "../5_Novelist_Dataset/output/author_to_num_dict.txt"
    unknown_writer_id_set = set()
    with open(reviewer_id_stats_file, mode="r") as fin:
        for line in fin:
            line = line.strip()
            parts = line.split()
            unknown_writer_id_set.add(parts[0])
        # endfor
    # endwith

    # ###################### load train_data review_id dict ######################
    train_data_known_writer_review_id_file = os.path.join(
        output_dir, "training_data_writer_to_review_id.json")
    with open(train_data_known_writer_review_id_file, mode="r") as fin:
        train_data_known_writer_review_id_dict = json.load(fin)
    # endwith

    # get all training data review_id set
    all_train_review_id_set = set()
    for writer_id, review_id_list in train_data_known_writer_review_id_dict.items():
        assert writer_id in known_writer_id_set
        all_train_review_id_set.update(review_id_list)
    # endfor

    known_writer_non_novel_dict = {}
    known_writer_novel_dict = None
    unknown_writer_non_novel_dict = None
    unknown_writer_novel_dict = {}

    # ################ get known_writer_non_novel_dict ###############
    writer_cate_pos_neg_dict = load_all_writer_cate_to_pos_neg_reviews()
    for writer_id, cate_pos_neg_review_list_dict in writer_cate_pos_neg_dict.items():

        # known writer
        if writer_id in known_writer_id_set:
            # (1) known non-novel
            non_novel_review_set = set(
                cate_pos_neg_review_list_dict[train_category]["pos"])
            intersect_set = non_novel_review_set.intersection(
                all_train_review_id_set)
            assert len(intersect_set) == known_writer_review_num
            # known non-novel
            known_writer_non_novel_dict[writer_id] = non_novel_review_set - \
                                                     all_train_review_id_set
        # endif
    # endfor

    # ############### get unknown_writer_novel_dict ################
    unknown_writer_novel_dict = None
    if mode == "easy":
        unknown_writer_novel_dict = load_EASY_all_story_writer_to_review_id_list_dict()
    # endif

    if mode == "hard":
        unknown_writer_novel_dict = load_HARD_all_story_writer_to_review_id_list_dict()
    # endif

    return known_writer_non_novel_dict, \
           known_writer_novel_dict, \
           unknown_writer_non_novel_dict, \
           unknown_writer_novel_dict, \
           known_writer_id_set, \
           unknown_writer_id_set


def validation_split(known_writer_non_novel_dict,
                     known_writer_novel_dict,
                     unknown_writer_non_novel_dict,
                     unknown_writer_novel_dict):
    # print(f"known_writer_non_novel_dict: {len(known_writer_non_novel_dict)}")
    # print(f"known_writer_novel_dict: {len(known_writer_novel_dict)}")
    # print(f"unknown_writer_non_novel_dict: {len(unknown_writer_non_novel_dict)}")
    # print(f"unknown_writer_novel_dict: {len(unknown_writer_novel_dict)}")

    # validation
    known_writer_non_novel_review_id_set = set()
    known_writer_novel_review_id_set = set()
    unknown_writer_non_novel_review_id_set = set()
    unknown_writer_novel_review_id_set = set()

    for writer_id, review_id_set in known_writer_non_novel_dict.items():
        known_writer_non_novel_review_id_set.update(review_id_set)
    # endfor

    if known_writer_novel_dict is not None:
        for writer_id, review_id_set in known_writer_novel_dict.items():
            known_writer_novel_review_id_set.update(review_id_set)
        # endfor

    for writer_id, review_id_set in unknown_writer_non_novel_dict.items():
        unknown_writer_non_novel_review_id_set.update(review_id_set)
    # endfor

    if unknown_writer_novel_dict is not None:
        for writer_id, review_id_set in unknown_writer_novel_dict.items():
            unknown_writer_novel_review_id_set.update(review_id_set)
        # endfor

    # print("-----")
    # print(len(known_writer_non_novel_review_id_set))
    # print(len(known_writer_novel_review_id_set))
    # print(len(unknown_writer_non_novel_review_id_set))
    # print(len(unknown_writer_novel_review_id_set))
    # print("-----")

    assert len(known_writer_non_novel_review_id_set.intersection(
        known_writer_novel_review_id_set)) == 0
    assert len(known_writer_non_novel_review_id_set.intersection(
        unknown_writer_non_novel_review_id_set)) == 0
    assert len(known_writer_non_novel_review_id_set.intersection(
        unknown_writer_novel_review_id_set)) == 0

    assert len(known_writer_novel_review_id_set.intersection(
        unknown_writer_non_novel_review_id_set)) == 0
    assert len(known_writer_novel_review_id_set.intersection(
        unknown_writer_novel_review_id_set)) == 0

    assert len(unknown_writer_non_novel_review_id_set.intersection(
        unknown_writer_novel_review_id_set)) == 0
    pass


def validate_class_level_novelty_dict():
    """
    validate:
    (1) sentiment_level_novelty_load_user_and_review_info_for_four_cases
    (2) object_level_novelty_load_user_and_review_info_for_four_cases

    Make sure that these two methods share the same
    * known_writer_non_novel_dict
    * unknown_writer_non_novel_dict
    :return:
    """
    train_category = "Electronics"
    novel_category_list = {'Home & Kitchen', 'Clothing, Shoes & Jewelry'}

    known_writer_num = 100
    known_writer_review_num = 40

    known_writer_non_novel_dict_1, \
    known_writer_novel_dict_1, \
    unknown_writer_non_novel_dict_1, \
    unknown_writer_novel_dict_1, \
    known_writer_id_set_1, \
    unknown_writer_id_set_1 = sentiment_level_novelty_load_user_and_review_info_for_four_cases(
        train_category, known_writer_num, known_writer_review_num)

    known_writer_non_novel_dict_2, \
    known_writer_novel_dict_2, \
    unknown_writer_non_novel_dict_2, \
    unknown_writer_novel_dict_2, \
    known_writer_id_set_2, \
    unknown_writer_id_set_2 = object_level_novelty_load_user_and_review_info_for_four_cases(
        train_category, novel_category_list, known_writer_num, known_writer_review_num)

    for k in known_writer_non_novel_dict_1.keys():
        assert known_writer_non_novel_dict_1[k] == known_writer_non_novel_dict_2[k]
    # endfor

    for k in unknown_writer_non_novel_dict_1.keys():
        assert unknown_writer_non_novel_dict_1[k] == unknown_writer_non_novel_dict_2[k]
    # endfor


def single_worker_generate_test_trial(output_root,
                                      seed,
                                      config,
                                      spec_config,
                                      train_data_config,
                                      group_index,
                                      all_review_id_to_json_dict,
                                      submit_folder_name,
                                      tmp_folder_name,
                                      debug_mode):
    train_category = "Electronics"
    novel_category_list = {'Home & Kitchen', 'Clothing, Shoes & Jewelry'}

    random.seed(seed)

    # get novel instance number list
    # this red_light_batch_index, count start from 0. If it is 9. There are 0~8, totally 9 batches are normal batches
    red_light_batch_index = random.choice(
        config["red_light_batch_level_indices"][spec_config["red_light_level"]])
    post_novelty_phase_novel_instance_num_list = TestDataSampler.create_list_novel_sizes(config,
                                                                                         spec_config,
                                                                                         red_light_batch_index)
    total_novel_instance_num_list = [0] * (red_light_batch_index) + post_novelty_phase_novel_instance_num_list
    assert len(total_novel_instance_num_list) == config["n_rounds"]

    # get normal instance number list
    total_normal_instance_num_list = [(config["round_size"] - item) for item in total_novel_instance_num_list]

    #
    known_writer_num = train_data_config["known_writer_num"]
    known_writer_review_num = train_data_config["known_writer_review_num"]

    # ############################### novelty_type #################################
    # novelty_type =
    # [class_level_novelty, object_level_novelty, sentiment_level_novelty, interaction_level_novelty]
    novelty_type = [int(spec_config["class_level_novelty"]),
                    int(spec_config["object_level_novelty"]),
                    int(spec_config["sentiment_level_novelty"]),
                    int(spec_config["interaction_level_novelty"]),
                    int(spec_config["action_level_novelty"])]

    # class_level_novelty must be 1
    assert int(spec_config["class_level_novelty"]) == 1
    # either one of these novelty is 1
    assert int(spec_config["object_level_novelty"]) + \
           int(spec_config["sentiment_level_novelty"]) + \
           int(spec_config["interaction_level_novelty"]) + \
           int(spec_config["action_level_novelty"]) <= 1
    # ###############################################################################

    known_writer_non_novel_dict = None
    known_writer_novel_dict = None
    unknown_writer_non_novel_dict = None
    unknown_writer_novel_dict = None
    known_writer_id_set = None
    unknown_writer_id_set = None

    # (1) class_level_novelty
    if novelty_type == [1, 0, 0, 0, 0]:
        known_writer_non_novel_dict, \
        known_writer_novel_dict, \
        unknown_writer_non_novel_dict, \
        unknown_writer_novel_dict, \
        known_writer_id_set, \
        unknown_writer_id_set = class_level_novelty_load_user_and_review_info_for_four_cases(
            train_category, known_writer_num, known_writer_review_num)
        # here, either use sentiment_level_novelty or object_level_novelty to load is OK

    # (2) sentiment_level_novelty
    if novelty_type == [1, 0, 1, 0, 0]:
        known_writer_non_novel_dict, \
        known_writer_novel_dict, \
        unknown_writer_non_novel_dict, \
        unknown_writer_novel_dict, \
        known_writer_id_set, \
        unknown_writer_id_set = sentiment_level_novelty_load_user_and_review_info_for_four_cases(
            train_category, known_writer_num, known_writer_review_num)

    # (3) object_level_novelty
    if novelty_type == [1, 1, 0, 0, 0]:
        known_writer_non_novel_dict, \
        known_writer_novel_dict, \
        unknown_writer_non_novel_dict, \
        unknown_writer_novel_dict, \
        known_writer_id_set, \
        unknown_writer_id_set = object_level_novelty_load_user_and_review_info_for_four_cases(
            train_category, novel_category_list, known_writer_num, known_writer_review_num)

    # (4) interaction_level_novelty (shipping review)
    if novelty_type == [1, 0, 0, 1, 0]:
        known_writer_non_novel_dict, \
        known_writer_novel_dict, \
        unknown_writer_non_novel_dict, \
        unknown_writer_novel_dict, \
        known_writer_id_set, \
        unknown_writer_id_set = None, None, None, None, None, None

        all_shipping_review_id_to_json_dict = load_all_shipping_review_id_to_json_dict()
        all_review_id_to_json_dict.update(all_shipping_review_id_to_json_dict)

    # (5) action_level_novelty
    if novelty_type == [1, 0, 0, 0, 1]:
        known_writer_non_novel_dict, \
        known_writer_novel_dict, \
        unknown_writer_non_novel_dict, \
        unknown_writer_novel_dict, \
        known_writer_id_set, \
        unknown_writer_id_set = None, None, None, None, None, None

        # load easy story text
        # easy_all_story_review_id_to_json_dict = load_all_story_id_to_json_dict()
        # all_review_id_to_json_dict.update(all_story_review_id_to_json_dict)

        # each review id is unique for easy and hard version, it is OK to load both
        EASY_all_story_review_id_to_json_dict = load_EASY_all_story_id_to_json_dict()
        HARD_all_story_review_id_to_json_dict = load_HARD_all_story_id_to_json_dict()
        all_review_id_to_json_dict.update(EASY_all_story_review_id_to_json_dict)
        all_review_id_to_json_dict.update(HARD_all_story_review_id_to_json_dict)

    # endif

    # ------------------------------------------------------------------------------------------
    if novelty_type in [[1, 0, 0, 0, 0]]:  # class level novelty
        assert known_writer_non_novel_dict is not None
        assert known_writer_novel_dict is None  # >>> This should be None
        assert unknown_writer_non_novel_dict is not None
        assert unknown_writer_novel_dict is None  # >>> This should be None

    if novelty_type in [[1, 1, 0, 0, 0],  # object level novelty
                        [1, 0, 1, 0, 0]]:  # sentiment level novelty

        assert known_writer_non_novel_dict is not None
        assert known_writer_novel_dict is not None
        assert unknown_writer_non_novel_dict is not None
        assert unknown_writer_novel_dict is not None
        assert known_writer_id_set is not None
        assert unknown_writer_id_set is not None
    # endif

    output_file_name_base = f"{spec_config['protocol']}.{str(group_index)}.{spec_config['test_id']}.{spec_config['seed']}"
    output_folder_base = os.path.join(
        output_root, f"writer_{known_writer_num}_review_{known_writer_review_num}/{submit_folder_name}")
    tmp_output_folder_base = os.path.join(
        output_root, f"writer_{known_writer_num}_review_{known_writer_review_num}/{tmp_folder_name}")

    if not os.path.exists(output_folder_base):
        os.makedirs(output_folder_base)
    # endif
    if not os.path.exists(tmp_output_folder_base):
        os.makedirs(tmp_output_folder_base)
    # endif

    # output_folder = os.path.join(output_folder_base, spec_config["test_id"])
    # tmp_output_folder = os.path.join(tmp_output_folder_base, spec_config["test_id"])

    output_folder = output_folder_base
    tmp_output_folder = tmp_output_folder_base

    # get all_writer_id_to_num_dict
    with open(
            f"./output/writer_{known_writer_num}_review_{known_writer_review_num}/writer_id_to_num_mapping_dict.json",
            mode="r") as fin:
        all_writer_id_to_num_dict = json.load(fin)
    # endwith

    # ####################### add more writer mapping to dict #######################
    # (1) add writer for shipping service
    # (2) add writer for story novelist

    shipping_review_writer_id_set = load_shipping_writer_id_set()
    story_writer_id_set = load_story_writer_id_set()

    for tmp_review_id in shipping_review_writer_id_set:
        all_writer_id_to_num_dict[tmp_review_id] = len(
            all_writer_id_to_num_dict) + 1
    # endfor

    for tmp_review_id in story_writer_id_set:
        all_writer_id_to_num_dict[tmp_review_id] = len(
            all_writer_id_to_num_dict) + 1
    # endfor

    # ######## output all writer mapping dict ########
    with open(
            f"./output/writer_{known_writer_num}_review_{known_writer_review_num}/four_types_of_novelty_writer_id_to_num_mapping_dict.json",
            mode="w") as fout:
        json.dump(all_writer_id_to_num_dict, fout)
    # endwith

    test_sampler = TestDataSampler(config_param=spec_config,
                                   dataset_source="Amazon",
                                   output_folder=output_folder,
                                   tmp_output_folder=tmp_output_folder,
                                   output_file_name_base=output_file_name_base,

                                   known_writer_non_novel_dict=known_writer_non_novel_dict,
                                   known_writer_novel_dict=known_writer_novel_dict,
                                   unknown_writer_non_novel_dict=unknown_writer_non_novel_dict,
                                   unknown_writer_novel_dict=unknown_writer_novel_dict,
                                   all_review_id_to_json_dict=all_review_id_to_json_dict,
                                   all_writer_id_to_num_dict=all_writer_id_to_num_dict,

                                   known_writer_id_set=known_writer_id_set,
                                   unknown_writer_id_set=unknown_writer_id_set,
                                   debug_mode=debug_mode
                                   )
    # This is necessary
    # every time assign these
    # ----------------------------------------------------
    # known_writer_non_novel_dict=known_writer_non_novel_dict,
    # known_writer_novel_dict=known_writer_novel_dict,
    # unknown_writer_non_novel_dict=unknown_writer_non_novel_dict,
    # unknown_writer_novel_dict=unknown_writer_novel_dict,
    # all_review_id_to_json_dict=all_review_id_to_json_dict,
    # all_writer_id_to_num_dict=all_writer_id_to_num_dict,
    # known_writer_id_set=known_writer_id_set,
    # unknown_writer_id_set=unknown_writer_id_set
    # ----------------------------------------------------
    test_sampler.initialize_class_obj()

    #
    #
    #
    # ########################## (I) pre-novel phrase ##########################
    #
    #
    #

    round_size = int(spec_config["round_size"])

    if novelty_type == [1, 0, 0, 0, 1]:
        # Based on the request from Eric, Jan 28, 2022
        # for action_level_novelty, the walmup sampling should not have novelty
        # it can only contains unknown writers, in the original amazon review
        # so novelty_type == [1, 0, 0, 0, 1] (the action_level_novelty) falls into this category

        # ############ SPECIAL: action level novelty ##############
        known_writer_non_novel_dict, \
        known_writer_novel_dict, \
        unknown_writer_non_novel_dict, \
        unknown_writer_novel_dict, \
        known_writer_id_set, \
        unknown_writer_id_set = action_level_novelty_PRE_NOVELTY_load_user_and_review_info_for_four_cases(
            train_category, known_writer_num, known_writer_review_num, mode=spec_config["difficulty"])
        # spec_config["difficulty"] in {"easy", "hard"} will load differently
        assert known_writer_novel_dict is None

    if novelty_type == [1, 0, 0, 1, 0]:
        # Based on the request from Eric, Jan 28, 2022
        # for action_level_novelty, the walmup sampling should not have novelty
        # it can only contains unknown writers, in the original amazon review
        # so novelty_type == [1, 0, 0, 0, 1] (the action_level_novelty) falls into this category

        # ############ SPECIAL: action level novelty ##############
        known_writer_non_novel_dict, \
        known_writer_novel_dict, \
        unknown_writer_non_novel_dict, \
        unknown_writer_novel_dict, \
        known_writer_id_set, \
        unknown_writer_id_set = interaction_level_novelty_PRE_NOVELTY_load_user_and_review_info_for_four_cases(
            train_category, known_writer_num, known_writer_review_num)
        # spec_config["difficulty"] in {"easy", "hard"} will load differently
        assert known_writer_novel_dict is None

    # --------------------------
    test_sampler.known_writer_non_novel_dict = known_writer_non_novel_dict
    test_sampler.known_writer_novel_dict = known_writer_novel_dict
    test_sampler.unknown_writer_non_novel_dict = unknown_writer_non_novel_dict
    test_sampler.unknown_writer_novel_dict = unknown_writer_novel_dict
    # writer id set
    test_sampler.known_writer_id_set = known_writer_id_set
    test_sampler.unknown_writer_id_set = unknown_writer_id_set

    # # known_writer_non_novel_dict = from training data
    # # known_writer_novel_dict = None
    # # unknown_writer_non_novel_dict = CLASS_unknown_writer_non_novel_dict
    # # unknown_writer_novel_dict = author_unknown_novel_dict

    # IMPORTANT!
    test_sampler.initialize_class_obj()
    # --------------------------

    if spec_config["difficulty"] == "easy":
        for i in range(int(spec_config["pre_novelty_batches"])):
            test_sampler.sample_pre_novelty_phase_easy(config["round_size"])
        # endfor
    # endif

    if spec_config["difficulty"] == "hard":
        walmup_round = config["hard_version_warmup_batch_num"]

        # if it is for shipping review, then walmup sampling could have novelty
        # if novelty_type in [[1, 0, 0, 1, 0]]:  # interaction_level_novelty
        #
        #     # Note: Because there are limited reviewers write multiple reviews
        #     #       interaction-level-novelty needs to be treated specially
        #     for j in range(walmup_round):
        #         test_sampler.walmup_sampling_for_hard_version_interaction_level(round_size)
        #     # endfor
        #
        # else:

        for j in range(walmup_round):
            test_sampler.walmup_sampling_for_hard_version(round_size)
        # endfor

        # endif

        for i in range(int(spec_config["pre_novelty_batches"]) - walmup_round):
            test_sampler.sample_pre_novelty_phase_hard(round_size)
        # endfor

    # endif

    #
    #
    #
    # ########################### (II) post novelty phase #############################
    #
    #
    #
    post_novelty_start_index = config["pre_novelty_batches"]

    # (1) ---------------- class level novelty -----------------
    if novelty_type == [1, 0, 0, 0, 0]:
        for i in range(config["n_rounds"]):  # iteration through all rounds
            if i < post_novelty_start_index:
                continue
            # endif
            normal_example_size = total_normal_instance_num_list[i]
            novel_example_size = total_novel_instance_num_list[i]

            test_sampler.sample_normal_and_novel_examples_for_1_0_0_0_0(normal_example_size,
                                                                        novel_example_size,
                                                                        if_check_red_light_instance=True if i == red_light_batch_index else False)
            # print(">>>>>>>", len(test_sampler.test_trial_json_list))
            # print(spec_config)
        # endfor
    # endif

    # (2) ####################
    # [1, 1, 0, 0, 0]: object level novelty
    # [1, 0, 1, 0, 0]: sentiment_level_novelty
    # [1, 0, 0, 1, 0]: interaction_level_novelty
    # [1, 0, 0, 0, 1]: action_level_novelty

    if novelty_type in [[1, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0],
                        [1, 0, 0, 1, 0],
                        [1, 0, 0, 0, 1]]:

        # --------- for action level novelty ----------------
        if novelty_type == [1, 0, 0, 0, 1]:
            # ############ SPECIAL: action level novelty ##############
            known_writer_non_novel_dict, \
            known_writer_novel_dict, \
            unknown_writer_non_novel_dict, \
            unknown_writer_novel_dict, \
            known_writer_id_set, \
            unknown_writer_id_set = action_level_novelty_POST_NOVELTY_load_user_and_review_info_for_four_cases(
                train_category, known_writer_num, known_writer_review_num, mode=spec_config["difficulty"])

            assert known_writer_novel_dict is None
            assert unknown_writer_non_novel_dict is None
        # endif

        if novelty_type == [1, 0, 0, 1, 0]:
            # ############ SPECIAL: interaction level novelty ##############
            known_writer_non_novel_dict, \
            known_writer_novel_dict, \
            unknown_writer_non_novel_dict, \
            unknown_writer_novel_dict, \
            known_writer_id_set, \
            unknown_writer_id_set = interaction_level_novelty_POST_NOVELTY_load_user_and_review_info_for_four_cases(
                train_category, known_writer_num, known_writer_review_num)
            assert known_writer_novel_dict is None
            assert unknown_writer_non_novel_dict is None
        # endif

        test_sampler.known_writer_non_novel_dict = known_writer_non_novel_dict
        test_sampler.known_writer_novel_dict = known_writer_novel_dict
        test_sampler.unknown_writer_non_novel_dict = unknown_writer_non_novel_dict
        test_sampler.unknown_writer_novel_dict = unknown_writer_novel_dict
        # writer id set
        test_sampler.known_writer_id_set = known_writer_id_set
        test_sampler.unknown_writer_id_set = unknown_writer_id_set

        # ----------------- post-novelty phase -----------------
        # known_writer_non_novel_dict = from training data
        # known_writer_novel_dict = None
        # unknown_writer_non_novel_dict = None
        # unknown_writer_novel_dict = author_unknown_novel_dict
        #
        # known_writer_id_set = from training data
        # unknown_writer_id_set = novelist authors

        # IMPORTANT!
        test_sampler.initialize_class_obj()
        # ##########################################################
        # endif

        for i in range(config["n_rounds"]):  # iteration through all rounds
            if i < post_novelty_start_index:
                continue

            normal_example_size = total_normal_instance_num_list[i]
            novel_example_size = total_novel_instance_num_list[i]
            test_sampler.sample_normal_and_novel_examples_for_1_and_1_or_1_or_1(normal_example_size,
                                                                                novel_example_size,
                                                                                if_check_red_light_instance=True if i == red_light_batch_index else False)
        # endfor
    # endif

    # #################### Here sample four batch #############

    candidate_sampling_list = copy.deepcopy(test_sampler.test_trial_json_list[
                                            red_light_batch_index * config["round_size"]: config["n_rounds"] * config[
                                                "round_size"]])
    random.shuffle(candidate_sampling_list)
    four_batch_sample_size = config["round_size"] * 4
    replay_batches = candidate_sampling_list[:four_batch_sample_size]
    test_sampler.test_trial_json_list.extend(replay_batches)
    # #########################################################

    # output json and csv file
    test_sampler.write_to_csv_and_json_format()

    # output meta data file
    meta_data_config = {}
    meta_data_config["protocol"] = spec_config["protocol"]
    meta_data_config["distribution"] = spec_config["dist_type"]
    meta_data_config["prop_novel"] = float(spec_config["prop_novel"])
    meta_data_config["difficulty"] = spec_config["difficulty"]

    meta_data_config["class_level_novelty"] = True if int(
        spec_config["class_level_novelty"]) == 1 else False

    meta_data_config["object_level_novelty"] = True if int(
        spec_config["object_level_novelty"]) == 1 else False

    meta_data_config["sentiment_level_novelty"] = True if int(
        spec_config["sentiment_level_novelty"]) == 1 else False

    meta_data_config["interaction_level_novelty"] = True if int(
        spec_config["interaction_level_novelty"]) == 1 else False

    meta_data_config["action_level_novelty"] = True if int(
        spec_config["action_level_novelty"]) == 1 else False

    meta_data_config["red_light"] = test_sampler.red_light_index
    meta_data_config["n_rounds"] = int(spec_config["n_rounds"]) + 4
    meta_data_config["round_size"] = int(spec_config["round_size"])
    meta_data_config["pre_novelty_batches"] = int(
        spec_config["pre_novelty_batches"])
    meta_data_config["seed"] = seed
    meta_data_config["feedback_max_ids"] = int(spec_config["round_size"])
    meta_data_config["red_light_batch_index"] = red_light_batch_index

    test_sampler.output_meta_data(meta_data_config)
    pass


def main_serial_process(output_root, specification_file_path, debug_mode=False):
    begin = time.time()

    # ####### output root ######
    os.makedirs(output_root, exist_ok=True)

    # (1) ######### load configuration file ###########
    config = TestDataSampler.load_config()
    # (2) ######### load specification file ###########
    specification_json_list = TestDataSampler.load_specification_file(specification_file_path)
    # ################

    with open("input/all_review_id_to_json_dict_uuid.json", mode="r") as fin:
        all_review_id_to_json_dict = json.load(fin)
    # endwith

    # train_data_config_list = [{"known_writer_num": 50, "known_writer_review_num": 40},
    #                           {"known_writer_num": 100, "known_writer_review_num": 40}]
    train_data_config_list = [
        {"known_writer_num": 100, "known_writer_review_num": 40}]

    for train_data_config in train_data_config_list:

        for spec_config in tqdm(specification_json_list, desc="all specification entries"):

            # # for each line in specification file, how many groups of dataset is generated
            # # Each group is generated with different seed
            for group_index in range(config["group_size"]):
                seed = int(spec_config["seed"]) + group_index

                # print(spec_config)
                # print(group_index)

                single_worker_generate_test_trial(output_root,
                                                  seed,
                                                  config,
                                                  spec_config,
                                                  train_data_config,
                                                  group_index,
                                                  all_review_id_to_json_dict,
                                                  "submit",
                                                  "tmp",
                                                  debug_mode)
            # endfor

        # endfor

    # endfor

    time_length = time.time() - begin
    print(f"TIME: {time_length / 60} mins.")
    pass


def main_multiprocess(specification_file_path, debug_mode):
    begin = time.time()

    args = argument_parser()

    output_root = args.output_root
    os.makedirs(output_root, exist_ok=True)
    print(f"OUTPUT: {output_root}")

    # ###### seed #######
    print(f"SEED: {args.seed}")
    time.sleep(5)

    # ######### set seed ###########
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # (1) ######### load configuration file ###########
    config = TestDataSampler.load_config()
    # (2) ######### load specification file ###########
    specification_json_list = TestDataSampler.load_specification_file(specification_file_path)
    # ###########################

    with open("input/all_review_id_to_json_dict_uuid.json", mode="r") as fin:
        all_review_id_to_json_dict = json.load(fin)
    # endwith

    # train_data_config_list = [{"known_writer_num": 50, "known_writer_review_num": 40},
    #                           {"known_writer_num": 100, "known_writer_review_num": 40}]
    train_data_config_list = [
        {"known_writer_num": 100, "known_writer_review_num": 40}]

    # ########## multiprocess generating test data ###############
    num_of_worker = os.cpu_count() - 2
    pool = Pool(processes=num_of_worker)

    job_list = []
    for train_data_config in train_data_config_list:
        for spec_config in tqdm(specification_json_list, desc="all specification entries"):
            # # for each line in specification file, how many groups of dataset is generated
            # # Each group is generated with different seed
            for group_index in range(config["group_size"]):
                seed = int(spec_config["seed"]) + group_index

                job = pool.apply_async(func=single_worker_generate_test_trial,
                                       args=(output_root,
                                             seed,
                                             config,
                                             spec_config,
                                             train_data_config,
                                             group_index,
                                             all_review_id_to_json_dict,
                                             "submit",
                                             "tmp",
                                             debug_mode))
                job_list.append(job)
            # endfor
        # endfor
    # endfor

    for job in tqdm(job_list, desc="all jobs"):
        job.get()
    # endfor

    time_length = time.time() - begin
    print(f"TIME: {time_length / 60} mins.")
    pass


if __name__ == '__main__':
    # load_story_writer_to_review_id_list_dict()
    # validate_class_level_novelty_dict()
    # main_serial_process(debug_mode=True)

    # specification_file_path = "all_NLT_test_specification_March_04_2022_full.csv"
    specification_file_path = "all_NLT_test_specification_March_04_2022_sample_trials.csv"

    # multiprocess
    main_multiprocess(specification_file_path, debug_mode=False)

    # serial process
    # main_serial_process(output_root, specification_file_path, debug_mode=False)

    print("DONE.")
