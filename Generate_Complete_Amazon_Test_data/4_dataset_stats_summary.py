"""
Jan 23, 2022
This script calculate the statistics of dataset for Jan 25 - 27, 2022 PI meeting.
"""
import json
import os
import csv
import math
from multiprocessing import Pool, Manager
from tqdm import tqdm
from collections import defaultdict
import numpy as np


class Data_Stats_Processor:
    def __init__(self) -> None:
        known_writer_id_set, unknown_writer_id_set, writer_id_to_index_mapping_dict, writer_index_to_id_mapping_dict = self.load_writer_info()
        self.known_writer_id_set = known_writer_id_set
        self.unknown_writer_id_set = unknown_writer_id_set
        self.writer_id_to_index_mapping_dict = writer_id_to_index_mapping_dict
        self.writer_index_to_id_mappping_dict = writer_index_to_id_mapping_dict
        pass

    def output_writer_num_to_id_mapping_dict(self):
        id_to_num_mapping_file_path = "./output/writer_100_review_40/writer_id_to_num_mapping_dict.json"

        with open(id_to_num_mapping_file_path, mode="r") as fin:
            id_to_num_dict = json.load(fin)
        # endwith

        num_to_id_dict = {}
        for id, index in id_to_num_dict.items():
            num_to_id_dict[int(index)] = id
        # endfor

        # ######## output #########
        output_file_path = "./output/writer_100_review_40/writer_num_to_id_mapping_dict.json"
        with open(output_file_path, mode="w") as fout:
            json.dump(num_to_id_dict, fout)
        # endwith
        pass

    def load_writer_info(self):
        # (1)
        known_writer_id_file = "./output/writer_100_review_40/known_writer_id_list.json"
        with open(known_writer_id_file, mode="r") as fin:
            known_writer_id_set = json.load(fin)
            known_writer_id_set = set(known_writer_id_set)
        # endwith

        # (2)
        unknown_writer_id_file = "./output/writer_100_review_40/unknown_writer_id_list.json"
        with open(unknown_writer_id_file, mode="r") as fin:
            unknown_writer_id_set = json.load(fin)
            unknown_writer_id_set = set(unknown_writer_id_set)
        # endwith

        # !! Include shipping writer here
        # --------------------------- load unknown writer_id -----------------------
        # all the unknown writer are in the shipping review dataset
        shipping_reviewer_id_stats_file = "./shipping_review_input/shipping_reviewer_stats.txt"
        # TODO: when generate final dataset, here it uses set, not list, so that order is lost
        # TODO: anyway, any writer id better than 310 (mapping dict start from 1), start from 311 are unknown writers.
        shipping_unknown_writer_id_list = []
        with open(shipping_reviewer_id_stats_file, mode="r") as fin:
            for line in fin:
                line = line.strip()
                parts = line.split()
                shipping_unknown_writer_id_list.append(parts[0])
            # endfor
        # endwith
        # update unknown writer
        unknown_writer_id_set.update(shipping_unknown_writer_id_list)

        # (3)
        id_to_num_mapping_file_path = "./output/writer_100_review_40/writer_id_to_num_mapping_dict.json"
        with open(id_to_num_mapping_file_path, mode="r") as fin:
            writer_id_to_index_mapping_dict = json.load(fin)
        # endwith
        for shipping_writer_id in shipping_unknown_writer_id_list:
            writer_id_to_index_mapping_dict[shipping_writer_id] = len(writer_id_to_index_mapping_dict) + 1
        # endfor

        # (4)
        writer_index_to_id_mapping_dict = {}
        for k, v in writer_id_to_index_mapping_dict.items():
            writer_index_to_id_mapping_dict[v] = k
        # endfor

        return known_writer_id_set, unknown_writer_id_set, writer_id_to_index_mapping_dict, writer_index_to_id_mapping_dict

    def single_worker_get_writer_stats(self, file_path_list):

        known_writer_sample_num_dict = defaultdict(int)
        unknown_writer_sample_num_dict = defaultdict(int)

        known_writer_doc_list_dict = defaultdict(set)
        unknown_writer_doc_list_dict = defaultdict(set)

        for file_path in tqdm(file_path_list, desc="all files"):
            with open(file_path, mode="r", encoding="utf16") as fin:
                csv_reader = csv.DictReader(fin, delimiter=",", quotechar="|")
                # instanceid,text,reported_writer_id,real_writer_id,sentiment,product,novelty_indicator,novel_instance,text_id
                for row in csv_reader:
                    real_writer_id = int(row["real_writer_id"])
                    real_writer_str = self.writer_index_to_id_mappping_dict[real_writer_id]

                    assert real_writer_str in self.known_writer_id_set or real_writer_str in self.unknown_writer_id_set

                    if real_writer_str in self.known_writer_id_set:
                        known_writer_sample_num_dict[real_writer_str] += 1
                        # add document set
                        known_writer_doc_list_dict[real_writer_str].add(file_path)
                    # endif

                    if real_writer_str in self.unknown_writer_id_set:
                        unknown_writer_sample_num_dict[real_writer_str] += 1
                        # add document set
                        unknown_writer_doc_list_dict[real_writer_str].add(file_path)
                    # endif
                # endfor
            # endwith
        # endfor

        # get document frequency dict
        known_writer_df_dict = {}
        for k, v in known_writer_doc_list_dict.items():
            known_writer_df_dict[k] = len(v)
        # endif

        unknown_writer_df_dict = {}
        for k, v in unknown_writer_doc_list_dict.items():
            unknown_writer_df_dict[k] = len(v)
        # endif

        return known_writer_sample_num_dict, unknown_writer_sample_num_dict, known_writer_df_dict, unknown_writer_df_dict


if __name__ == "__main__":
    processor = Data_Stats_Processor()
    # processor.output_writer_num_to_id_mapping_dict()
