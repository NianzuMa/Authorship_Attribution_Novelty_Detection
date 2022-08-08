"""
April 1, 2022

Jiahua found that the slash 0 cause some error with the model.
I will temporarily fix the issue during dataset generation
"""
import os
import csv
import shutil

# ASU dataset
# input_folder = "./output_april_1_2022_asu_with_reply/writer_100_review_40/submit"
input_folder = "./output_april_1_2022_par_with_reply/writer_100_review_40/submit"

output_folder = f"{input_folder}_v2"


def test_read_data(input_folder, output_folder):
    """
    :param input_folder:
    :param output_folder:
    :return:
    """
    os.makedirs(output_folder, exist_ok=True)

    for root, subdir, file_list in os.walk(input_folder):
        for file_name in file_list:
            input_file_path = os.path.join(root, file_name)
            output_file_path = os.path.join(output_folder, file_name)

            if "single_df.csv" in file_name:

                if '\0' in open(input_file_path, mode="r", encoding="utf16").read():
                    print(input_file_path)
                #endif

                with open(input_file_path, mode="r", encoding="utf16") as fin:
                    csv_reader = csv.DictReader(fin, delimiter=",", quotechar="|")
                    for row in csv_reader:
                        pass
                    #endfor
                #endwith
            #endif
        #endfor
    #endfor


if __name__ == '__main__':
    test_read_data(input_folder, output_folder)







