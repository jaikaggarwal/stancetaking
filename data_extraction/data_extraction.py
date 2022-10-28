from hashlib import new
import os
from tqdm.auto import tqdm
import json
import time
import numpy as np 
import argparse
from utils import Serialization
import zstd
import pandas as pd
import re
from collections import defaultdict
from nltk import word_tokenize

tqdm.pandas()

DIR= '/ais/hal9000/datasets/reddit/data_dumps/'
OUT_DIR= '/ais/hal9000/datasets/reddit/stance_analysis/'



ISAAC_SUBS = pd.read_csv('~/AutismHateSpeech/Data/reddit-master-metadata.tsv', delimiter="\t")['community'].tolist()
ISAAC_SUBS = [sub.lower() for sub in ISAAC_SUBS]
FIELDS_TO_KEEP = ['author', 'body', 'controversiality', 'created_utc', 'id', 'parent_id', 'score', 'subreddit', 'author_flair_text', 'author_flair_css_class']
# quarter_to_data = {
#     "first": ["01", "02", "03"],
#     "second": ["04", "05", "06"],
#     "third": ["07", "08", "09"],
#     "fourth": ["10", "11", "12"],
# }
quarter_to_data = {
    "first": ["02"],
    "second": ["05"],
    "third": ["08"],
    "fourth": ["11"],
    "f2": ["03"],
    "f3": ["06"],
    "f4": ["09"],
    "f5":["12"]
}
# print(SUBS)


HTTP_PATTERN = re.compile(r'[\(\[]?https?:\/\/.*?(\s|\Z)[\r\n]*')
def preprocess(text):
    """
    Preprocesses text from Reddit posts and comments.
    """
    # Replace links with LINK token
    line = HTTP_PATTERN.sub(" LINK ", text)
    # Replace irregular symbol with whitespace
    line = re.sub("&amp;#x200b", " ", line)
    # Replace instances of users quoting previous posts with empty string
    line = re.sub(r"&gt;.*?(\n|\s\s|\Z)", " ", line)
    # Replace extraneous parentheses with whitespace
    line = re.sub(r'\s\(\s', " ", line)
    line = re.sub(r'\s\)\s', " ", line)
    # Replace newlines with whitespace
    line = re.sub(r"\r", " ", line)
    line = re.sub(r"\n", " ", line)
    # Replace mentions of users with USER tokens
    line = re.sub("\s/?u/[a-zA-Z0-9-_]*(\s|\Z)", " USER ", line)
    # Replace mentions of subreddits with REDDIT tokens
    line = re.sub("\s/?r/[a-zA-Z0-9-_]*(\s|\Z)", " REDDIT ", line)
    # Replace malformed quotation marks and apostrophes
    line = re.sub("’", "'", line)
    line = re.sub("”", '"', line)
    line = re.sub("“", '"', line)
    # Get rid of asterisks indicating bolded or italicized comments
    line = re.sub("\*{1,}(.+?)\*{1,}", r"\1", line)    
    # Replace emojis with EMOJI token
    # line = emoji.get_emoji_regexp().sub(" EMOJI ", line)
    # Replace all multi-whitespace characters with a single space.
    line = re.sub("\s{2,}", " ", line)
    return line

def apply_filters(new_line):
    if new_line['body'] is np.nan:
        return False, new_line
    # Remove all bots, moderators, deleted, removed authors (and spammer dollarwolf)
    if new_line['author'] in ["AutoModerator", "dollarwolf", "[deleted]", "[removed]"]:
        to_keep = False
    elif new_line['author'].endswith("Bot") or new_line['author'].endswith("bot"):
        to_keep = False
    else:
        to_keep = True
    # Now we can check to see if the body is long enough
    if to_keep:
        new_body = preprocess(new_line['body'])
        if len(word_tokenize(new_body)) < 5:
            return False, new_line
        new_line['body'] = new_body
        return True, new_line
    else:
        return False, new_line


def apply_filters_with_term(new_line, terms):
   
    curr_body = set(new_line['body'].lower().split(" "))
    any_present = 0
    for key in terms:
        present_markers = [val for val in terms[key] if val in curr_body]
        if len(present_markers) > 0:
            curr_str = "__".join(present_markers)
            new_line[key] = 1
            new_line[key + "_markers"] = curr_str
            any_present += 1
        else:
            new_line[key] = 0
            new_line[key + "_markers"] = ""
    
    if any_present > 0:
        return True, new_line
    else:
        return False, new_line

def group_to_terms():
    biber_df = pd.read_csv("biber_stance_markers.txt", header=None)
    biber_markers = set(biber_df[0])

    biber_ex_df = pd.read_csv("biber_expanded_stance_markers.txt", header=None)
    biber_ex_markers = set(biber_ex_df[0])

    dialog_df = pd.read_csv("dialog_act_stance_markers.txt", header=None)
    dialog_markers = set(dialog_df[0])

    dialog_ex_df = pd.read_csv("dialog_act_expanded_stance_markers.txt", header=None)
    dialog_ex_markers = set(dialog_ex_df[0])

    return {
        "BF": biber_markers,
        "BF_EX": biber_ex_markers,
        "DA": dialog_markers,
        "DA_EX": dialog_ex_markers
    }

def extract_data_with_term(year, quarter):
    group_to_term = group_to_terms()
    zst_files = [f'RC_{year}-{num}.zst' for num in quarter_to_data[quarter]]
    print("Number of files...", len(zst_files))
    for filename in zst_files:

        # Trackers
        print(filename) # Which files
        counter = 0 # How many bytes we've seen
        loops = 0 # How many windows we've decompressed
        a = time.time() # Overall time
        with open(f"{OUT_DIR}{filename[:-4]}_counts.json", "w") as f_out:
            with open(DIR + filename, 'rb') as fh:
                dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
                with dctx.stream_reader(fh) as reader:
                    previous_line = ""
                    while True:
                        chunk = reader.read(2**24)  # 16mb chunks
                        counter += 2**24
                        loops += 1
                        if loops % 2000 == 0:
                            print(f"{counter/10**9:.2f} GB")
                            print(f"{(time.time()-a)/60:.2f} minutes passed")
                        if not chunk:
                            break

                        string_data = chunk.decode('utf-8')
                        lines = string_data.split("\n")
                        for i, line in enumerate(lines[:-1]):
                            if i == 0:
                                line = previous_line + line
                            line = json.loads(line)
                            # if (line['subreddit'].lower() == 'askreddit'):
                            #     continue
                            # line_counter += 1
                            # if line_counter % 10 != 0:
                            #     continue
                            if line['subreddit'].lower() not in ISAAC_SUBS:
                                continue
                            
                            new_line = {field: line.get(field, np.nan) for field in FIELDS_TO_KEEP}
                            to_keep, new_line = apply_filters(new_line)
                            if not to_keep:
                                continue

                            to_keep, new_line = apply_filters_with_term(new_line, group_to_term)
                            if not to_keep:
                                continue
                            
                            f_out.write(json.dumps(new_line))
                            f_out.write("\n")
                            # do something with the object here
                        previous_line = lines[-1]


def extract_data_with_keyword(year, quarter):
    zst_files = [f'RC_{year}-{num}.zst' for num in quarter_to_data[quarter]]
    print("Number of files...", len(zst_files))
    for filename in zst_files:

        # Trackers
        print(filename) # Which files
        counter = 0 # How many bytes we've seen
        loops = 0 # How many windows we've decompressed
        line_counter = 0 # How many AskReddit lines we've seen (used for sampling)
        a = time.time() # Overall time

        with open(f"{OUT_DIR}/total_reddit_sample/{filename[:-4]}_counts.json", "w") as f_out:
            with open(DIR + filename, 'rb') as fh:
                dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
                with dctx.stream_reader(fh) as reader:
                    previous_line = ""
                    while True:
                        chunk = reader.read(2**24)  # 16mb chunks
                        counter += 2**24
                        loops += 1
                        if loops % 2000 == 0:
                            print(f"{counter/10**9:.2f} GB")
                            print(f"{(time.time()-a)/60:.2f} minutes passed")
                        if not chunk:
                            break

                        string_data = chunk.decode('utf-8')
                        lines = string_data.split("\n")
                        for i, line in enumerate(lines[:-1]):
                            if i == 0:
                                line = previous_line + line
                            line = json.loads(line)
                            # if (line['subreddit'].lower() == 'askreddit'):
                            #     continue
                            line_counter += 1
                            if line_counter % 10 != 0:
                                continue
                            if line['subreddit'].lower() not in ISAAC_SUBS:
                                continue
                            

                            # if (line['subreddit'].lower() == 'askreddit'):
                            #     askreddit_counter += 1
                            #     if (askreddit_counter % 10 != 0): # Sample 10% of data
                            #         continue
                            new_line = {field: line.get(field, np.nan) for field in FIELDS_TO_KEEP}
                            to_keep, new_line = apply_filters(new_line)
                            
                            if not to_keep:
                                continue
                            
                            # new_line = {field: new_line[field] for field in ['author', 'controversiality', 'created_utc', 'id', 'subreddit']} #reduce data size
                            f_out.write(json.dumps(new_line))
                            f_out.write("\n")
                            # do something with the object here
                        previous_line = lines[-1]


def extract_data_per_author(year, quarter, author_set_name_1, author_set_name_2):
    author_set_1 = Serialization.load_obj(author_set_name_1).tolist()
    author_set_2 = Serialization.load_obj(author_set_name_2).tolist()
    author_set = list(set(author_set_1 + author_set_2))
    print(len(author_set_1))
    print(len(author_set))
    del author_set_2
    zst_files = [f'RC_{year}-{num}.zst' for num in quarter_to_data[quarter]]
    print("Number of files...", len(zst_files))
    with open(f"{OUT_DIR}{year}_{quarter}_extra_manosphere_on_askreddit.json", "w") as f_out:
        for filename in zst_files:
            # Trackers
            print(filename) # Which files
            counter = 0 # How many bytes we've seen
            loops = 0 # How many windows we've decompressed
            askreddit_counter = 0 # How many AskReddit lines we've seen (used for sampling)
            a = time.time() # Overall time
            with open(f"{OUT_DIR}{filename[:-4]}_counts.json", "w") as f_counts:
                with open(DIR + filename, 'rb') as fh:
                    dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
                    with dctx.stream_reader(fh) as reader:
                        previous_line = ""
                        while True:
                            chunk = reader.read(2**24)  # 16mb chunks
                            counter += 2**24
                            loops += 1
                            if loops % 2000 == 0:
                                print(f"{counter/10**9:.2f} GB")
                                print(f"{(time.time()-a)/60:.2f} minutes passed")
                            if not chunk:
                                break

                            string_data = chunk.decode('utf-8')
                            lines = string_data.split("\n")
                            for i, line in enumerate(lines[:-1]):
                                if i == 0:
                                    line = previous_line + line
                                line = json.loads(line)
                                if line['author'] not in author_set:
                                    continue
                                if line['subreddit'] not in ISAAC_SUBS:
                                    continue
                                
                                if (line['subreddit'].lower() == 'askreddit') and (line['author'] not in author_set_1):
                                    continue
                                # elif (line['subreddit'].lower() == "askreddit") and (line['author'] in author_set_1):
                                #     new_line = {field: line.get(field, np.nan) for field in FIELDS_TO_KEEP}
                                #     to_keep, new_line = apply_filters(new_line)
                                #     if not to_keep:
                                #         continue
                                #     f_out.write(json.dumps(new_line))
                                #     f_out.write("\n")
                                
                                # else:
                                #     new_line = {field: line.get(field, np.nan) for field in FIELDS_TO_KEEP}
                                #     to_keep, new_line = apply_filters(new_line)
                                #     if not to_keep:
                                #         continue
                                #     new_line = {field: new_line[field] for field in ['author', 'controversiality', 'created_utc', 'id', 'subreddit']} #reduce data size
                                #     f_counts.write(json.dumps(new_line))
                                #     f_counts.write("\n")
                                # do something with the object here
                            previous_line = lines[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("year",
                        type=str)
    parser.add_argument("quarter",
                        type=str)
    args = parser.parse_args()
    print(args.year)
    print(quarter_to_data[args.quarter])
    extract_data_with_term(args.year, args.quarter)
