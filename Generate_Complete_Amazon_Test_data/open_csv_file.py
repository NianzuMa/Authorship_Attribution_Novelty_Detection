"""
metadata json object also should be loaded in utf16
"""
import csv
import argparse

parser = argparse.ArgumentParser(description="Opening CSV file.")
parser.add_argument("file_name", type=str, help="file path")
args = parser.parse_args()

with open(args.file_name, mode="r", encoding="utf16") as fin:
    csv_reader = csv.DictReader(fin, delimiter=",", quotechar="|")
    # review_id,product_id,review_rating,assigned_writer_id,text
    for row in csv_reader:
        print(row["instance_id"])
        print(row["product_id"])
        print(row["sentiment"])
        print(row["writer_id"])
        print(row["text"])
        break
