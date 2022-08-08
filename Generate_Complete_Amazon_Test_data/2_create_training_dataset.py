"""
Create training dataset
1. The training dataset contains known writers
2. The known writers' reviews should come from the same domain
3. The known writers' sentiment should be positive (positive reviews are much more than negative review)

1000
(1) For dataset with class_level_novelty, all reviews should be from one domains and positive,
    only writer_ids are changed.

900
(2) For dataset with both class_level_novelty and object_level_novelty, all reviews should be positive.
    - class_level novel reviews are writer_ids assignment changes, positive, the domain are the same as training domain.
    - object_level_novel reviews have new domains introduced.

850
(3) For dataset with both class_level_novelty and sentiment_level_novelty
    - class_level novel reviews are writer_ids assignment changes, positive, the domain are the same as training
      data domain.
    - sentiment_level novel reviews, negative, the domain are the same as training data domain.
"""

import json
import os
import csv


def create_training_data(writer_num, review_num_each_writer):
    """
    There three categories: 'Home & Kitchen', 'Electronics', 'Clothing, Shoes & Jewelry' in the data pool.

    Based on the stats:
    1. choose category "electronics" as domain in training data
    2. reverse sorted based on neg review
    3. choose positive review with at least 40 reviews
    :return:
    """
    all_review_to_json_obj_dict_file = "input/all_review_id_to_json_dict_uuid.json"

    # ######################### (1) calculate each reviewer's review statistics ##########################
    # # The goal is to sample reviewer with sufficient negative review data to be in the training data
    # # so that, when sampling sentiment_level_novelty dataset, we can sampling enough novel data instances

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
            writer_cate_pos_neg_dict[writer_id][category] = {"pos": [], "neg": []}
        # endif

        if rating > 3.0:
            writer_stats[writer_id]["pos"] += 1
            writer_stats[writer_id][category]["pos"] += 1
            writer_cate_pos_neg_dict[writer_id][category]["pos"].append(review_id)
        # endif

        if rating < 3.0:
            writer_stats[writer_id]["neg"] += 1
            writer_stats[writer_id][category]["neg"] += 1
            writer_cate_pos_neg_dict[writer_id][category]["neg"].append(review_id)
        # endif

    # endfor

    output_folder = "./output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    #endif

    with open("output/review_id_stats.txt", mode="w") as fout:
        for review_id, stats_dict in sorted(writer_stats.items(), key=lambda x: x[1]["neg"], reverse=True):
            fout.write(f"{review_id}\t{stats_dict}\n")
        # endfor
    # endwith

    # ############################ (2) start to choose candidate training data ##################################
    # # choose "Electronics" and positive reviews

    # different groups of writers
    known_writer_list = []

    # choose "Electronics" and positive
    train_writer_size = 0
    selected_cate = "Electronics"
    training_data_writer_to_review_id_list_dict = {}
    for writer_id, stats_dict in sorted(writer_stats.items(), key=lambda x: x[1]["neg"], reverse=True):
        if train_writer_size >= writer_num:
            break

        if writer_stats[writer_id][selected_cate]["pos"] > 60:
            if writer_id not in training_data_writer_to_review_id_list_dict:
                training_data_writer_to_review_id_list_dict[writer_id] = []
            # endif

            train_writer_size += 1
            known_writer_list.append(writer_id)
            # NOTE: only choose the positive reviews
            training_data_writer_to_review_id_list_dict[writer_id] = writer_cate_pos_neg_dict[writer_id][selected_cate][
                                                                         "pos"][:review_num_each_writer]
        # endif
    # endfor
    assert len(training_data_writer_to_review_id_list_dict) == writer_num
    for writer_id, review_id_list in training_data_writer_to_review_id_list_dict.items():
        assert len(review_id_list) == review_num_each_writer
    # endfor
    print(f"Training data writer num: {len(training_data_writer_to_review_id_list_dict)}")

    # output reviews
    output_dir = os.path.join("output", f"writer_{writer_num}_review_{review_num_each_writer}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # endif

    with open(os.path.join(output_dir, "training_data_writer_to_review_id.json"), mode='w') as fout:
        json.dump(training_data_writer_to_review_id_list_dict, fout)
    # endwith

    submit_folder = os.path.join(output_dir, "submit")
    if not os.path.exists(submit_folder):
        os.makedirs(submit_folder)
    # endif

    # ############################ (3) output writers and its mapping to num ################################
    all_writer_id_list = list(writer_stats.keys())
    unknown_writer_list = list(set(all_writer_id_list) - set(known_writer_list))
    print(f"There are totally {len(all_writer_id_list)} writers.")
    print(f"There are {len(unknown_writer_list)} unknown writers.")
    print(f"There are {len(known_writer_list)} known writers.")

    writer_index = 0
    writer_id_to_num_mapping_dict = {}
    for writer_id in known_writer_list:
        writer_index += 1
        writer_id_to_num_mapping_dict[writer_id] = writer_index
    # endfor

    for writer_id in unknown_writer_list:
        writer_index += 1
        writer_id_to_num_mapping_dict[writer_id] = writer_index
    # endfor

    with open(os.path.join(output_dir, "writer_id_to_num_mapping_dict.json"), mode='w') as fout:
        json.dump(writer_id_to_num_mapping_dict, fout, indent=4)
    # endwith

    known_writer_to_num_mapping = {item: writer_id_to_num_mapping_dict[item] for item in known_writer_list}
    unknown_writer_to_num_mapping = {item: writer_id_to_num_mapping_dict[item] for item in unknown_writer_list}

    with open(os.path.join(output_dir, "known_writer_id_dict.json"), mode="w") as fout:
        json.dump(known_writer_to_num_mapping, fout, indent=4)
    #endwith

    with open(os.path.join(output_dir, "unknown_writer_id_dict.json"), mode="w") as fout:
        json.dump(unknown_writer_to_num_mapping, fout, indent=4)
    #endwith

    with open(os.path.join(output_dir, "known_writer_id_list.json"), mode="w") as fout:
        json.dump(known_writer_list, fout)
    # endwith

    with open(os.path.join(output_dir, "unknown_writer_id_list.json"), mode="w") as fout:
        json.dump(unknown_writer_list, fout)
    # endwith

    # ################################ (4) output to csv file #########################################
    train_data_submit_csv_file_path = os.path.join(output_dir, "train.csv")
    train_data_submit_json_file_path = os.path.join(output_dir, "train.json")
    train_data_tmp_csv_file_path = os.path.join(output_dir, "train_tmp.csv")

    with open(train_data_submit_csv_file_path, mode="w", encoding="utf16") as fout_csv, \
            open(train_data_submit_json_file_path, mode="w", encoding="utf16") as fout_json, \
            open(train_data_tmp_csv_file_path, mode="w", encoding="utf16") as fout_tmp:

        fieldnames = ["instanceid", "text", "writer_id", "sentiment", "product", "text_id"]
        csv_writer = csv.DictWriter(fout_csv, fieldnames=fieldnames, quotechar="|", delimiter=",")
        csv_writer.writeheader()

        tmp_fieldnames = ["instanceid", "writer_id", "sentiment", "product", "text_id"]
        tmp_csv_writer = csv.DictWriter(fout_tmp, fieldnames=tmp_fieldnames, quotechar="|", delimiter=",")
        tmp_csv_writer.writeheader()

        index = -1
        for writer_id, review_id_list in training_data_writer_to_review_id_list_dict.items():
            for review_id in review_id_list:
                index += 1
                review_json_obj = all_review_to_json_obj_dict[review_id]
                writer_id_num = writer_id_to_num_mapping_dict[review_json_obj["reviewerID"]]
                review_rating = int(review_json_obj["overall"])
                text = review_json_obj["reviewText"]
                text = text.replace("\n", "</p>")
                review_json_obj["review_id"] = review_id

                fout_json.write(f"{json.dumps(review_json_obj)}\n")

                entry = {"instanceid": index,
                         "text": text,
                         "writer_id": writer_id_num,
                         "sentiment": review_rating,
                         "product": review_json_obj["asin"],
                         "text_id": review_id
                         }

                csv_writer.writerow(entry)
                del entry["text"]
                tmp_csv_writer.writerow(entry)
            # endfor
        # endfor
    # endwith


def main_create_all_training_data_handler():
    # training_data_config_list = [{"writer_num": 50, "review_num_each_writer": 40},
    #                              {"writer_num": 100, "review_num_each_writer": 40}]
    training_data_config_list = [{"writer_num": 100, "review_num_each_writer": 40}]

    for train_config in training_data_config_list:
        writer_num = train_config["writer_num"]
        review_num_each_writer = train_config["review_num_each_writer"]
        create_training_data(writer_num, review_num_each_writer)
    # endfor


if __name__ == '__main__':
    main_create_all_training_data_handler()
