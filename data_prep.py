import glob
import pandas as pd

base_path = 'C:\\Users\\Chris\\Documents\\Lip Reading in the Wild Data\\lipread_mp4'

words = [
    "about", "absolutely", "abuse"
    # , "better", "building",
    # "cases", "clear", "death", "described", "emergency",
    # "energy", "following", "force", "great", "ground",
    # "heavy", "history", "hospital", "industry", "information",
    # "james", "justice", "killed", "least", "levels",
    # "little", "living", "makes", "matter", "media",
    # "night", "north", "office", "operation", "parents",
    # "people", "prices", "process", "question", "recent",
    # "result", "saying", "security", "strong", "talking",
    # "temperatures", "themselves", "understand", "warning", "watching",
    # "whole", "years", "yesterday"
]

folders = []

for word in words:
    folders.append(base_path + "\\" + word)

index_start = 0
video_count = 1000
test_val_count = 50

descriptor = "_lips"
extension = ".txt"

if test_val_count > 50:
    test_val_count = 50

train_vids = []
val_vids = []
test_vids = []

for folder in folders:
    train_vids = train_vids + glob.glob(folder + '\\train\\*.mp4')[index_start : index_start + video_count]
    val_vids = val_vids + glob.glob(folder + '\\val\\*.mp4')[index_start : index_start + test_val_count]
    test_vids = test_vids + glob.glob(folder + '\\test\\*.mp4')[index_start : index_start + test_val_count]

# Create a dataframe having video names
train = pd.DataFrame()
train['video_name'] = train_vids
# train.head()

val = pd.DataFrame()
val["video_name"] = val_vids

test = pd.DataFrame()
test["video_name"] = test_vids

def extract_tag(video_path):
    tag = video_path.split("\\")[-3]
    return tag

def separate_video_name(video_name):
    name = video_name.split("\\")[-1]
    return name

def rectify_video_name(video_name):
    name = video_name.split(" ")[0]
    return name

def apply_descriptor(video_name):
    filename_split = video_name.split(".")
    name = ""
    if len(filename_split) > 1:
        name = filename_split[0] + descriptor + extension
    return name

train["tag"] = train["video_name"].apply(extract_tag)
train["video_name"] = train["video_name"].apply(separate_video_name)

train["video_name"] = train["video_name"].apply(rectify_video_name)

val["tag"] = val["video_name"].apply(extract_tag)
val["video_name"] = val["video_name"].apply(separate_video_name)
val["video_name"] = val["video_name"].apply(rectify_video_name)

test["tag"] = test["video_name"].apply(extract_tag)
test["video_name"] = test["video_name"].apply(separate_video_name)
test["video_name"] = test["video_name"].apply(rectify_video_name)

if descriptor:
    train["video_name"] = train["video_name"].apply(apply_descriptor)
    # print(train.head())

    val["video_name"] = val["video_name"].apply(apply_descriptor)
    # print(val.head())

    test["video_name"] = test["video_name"].apply(apply_descriptor)
    # print(test.head())

train_new = train.reset_index(drop=True)
val_new = val.reset_index(drop=True)
test_new = test.reset_index(drop=True)

train_new.to_csv("train.csv", index=False)
val_new.to_csv("val.csv", index=False)
test_new.to_csv("test.csv", index=False)

print("New CSV files generated")