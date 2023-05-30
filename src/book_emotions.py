# pathing tool
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # minimize tensorflow messages
# dataframe wrangling
import pandas as pd
# randomizing dataframe
import numpy as np
# classifier used
from transformers import pipeline
###import tensorflow
# for visualisations
from matplotlib import pyplot as plt

# loading .csv data and forcing it as a string
def load_data():
    # reading input data
    data_path = os.path.join("..","data","Best_Books_Ever.csv")
    book_df = pd.read_csv(data_path)

    # making sure the desc is a string
    book_df["description"] = book_df["description"].apply(lambda x: str(x))
    return book_df

def preprocess(book_df):
    # ensuring english-only descriptions
    book_df = book_df[book_df["language"] == "English"]
    book_df = book_df.reset_index(drop=True)
    # making list from df
    book_list = list(book_df["description"])

    # truncating descriptions, focusing on first 200 chars to keep within model limits
    for idx, desc in enumerate(book_list):
        trunc_desc = desc[:200]
        book_list[idx] = trunc_desc

    # separating the descriptions categories based on their ratings
    under_four = []
    rated_four = []
    rated_five = []

    for idx, row in book_df.iterrows():
        if row["rating"] < 4:
            under_four.append(book_list[idx])
        elif row["rating"] >= 4 and row["rating"] < 5:
            rated_four.append(book_list[idx])
        elif row["rating"] == 5:
            rated_five.append(book_list[idx])
        else:
            pass

    #print(len(under_four), len(rated_four), len(rated_five)) # comment in to see sizes of partial datasets; rated_five is smaller than the two others by a factor of ~40.

    sample = 500
    np.random.shuffle(under_four)
    under_four = under_four[:sample]

    np.random.shuffle(rated_four)
    rated_four = rated_four[:sample]

    np.random.shuffle(rated_five)
    rated_five = rated_five[:sample]

    return under_four, rated_four, rated_five

# define emotion classifier
def set_cls():
    classifier = pipeline("text-classification", 
                          model="j-hartmann/emotion-english-distilroberta-base", 
                          top_k=None)
    return classifier

# run data through classifier, assuming highest prob emotion as correct
def analyse_data(classifier, under_four, rated_four, rated_five):    

    under_four_emotions = []

    print("     analysing descriptions with fewer than 4 stars")
    for description in under_four:
        emotions = classifier(description)
        # unlisting emotions by one level
        emotions = emotions[0]
        # finding dict with highest probability
        max_score = max(emotions, key=lambda x:x["score"])
        # getting label from highest prob dict
        best_label = max_score["label"]
        # adding to total list
        under_four_emotions.append(best_label)

    rated_four_emotions = []

    print("     analysing descriptions with 4-5 stars")
    for description in rated_four:
        emotions = classifier(description)
        emotions = emotions[0]
        max_score = max(emotions, key=lambda x:x["score"])
        best_label = max_score["label"]
        rated_four_emotions.append(best_label)

    rated_five_emotions = []

    print("     analysing descriptions with 5 stars")
    for description in rated_five:
        emotions = classifier(description)
        emotions = emotions[0]
        max_score = max(emotions, key=lambda x:x["score"])
        best_label = max_score["label"]
        rated_five_emotions.append(best_label)

    all_emotions = under_four_emotions + rated_four_emotions + rated_five_emotions

    return all_emotions, under_four_emotions, rated_four_emotions, rated_five_emotions

# creating tables from counts of each emotion found
def make_table(all_emotions, under_four_emotions, rated_four_emotions, rated_five_emotions):
    # counting highest probability emotions found in the descriptions
    all_anger = all_emotions.count("anger")
    all_disgust = all_emotions.count("disgust")
    all_fear = all_emotions.count("fear")
    all_joy = all_emotions.count("joy")
    all_neutral = all_emotions.count("neutral")
    all_sadness = all_emotions.count("sadness")
    all_surprise = all_emotions.count("surprise")

    under_four_anger = under_four_emotions.count("anger")
    under_four_disgust = under_four_emotions.count("disgust")
    under_four_fear = under_four_emotions.count("fear")
    under_four_joy = under_four_emotions.count("joy")
    under_four_neutral = under_four_emotions.count("neutral")
    under_four_sadness = under_four_emotions.count("sadness")
    under_four_surprise = under_four_emotions.count("surprise")

    rated_four_anger = rated_four_emotions.count("anger")
    rated_four_disgust = rated_four_emotions.count("disgust")
    rated_four_fear = rated_four_emotions.count("fear")
    rated_four_joy = rated_four_emotions.count("joy")
    rated_four_neutral = rated_four_emotions.count("neutral")
    rated_four_sadness = rated_four_emotions.count("sadness")
    rated_four_surprise = rated_four_emotions.count("surprise")

    rated_five_anger = rated_five_emotions.count("anger")
    rated_five_disgust = rated_five_emotions.count("disgust")
    rated_five_fear = rated_five_emotions.count("fear")
    rated_five_joy = rated_five_emotions.count("joy")
    rated_five_neutral = rated_five_emotions.count("neutral")
    rated_five_sadness = rated_five_emotions.count("sadness")
    rated_five_surprise = rated_five_emotions.count("surprise")


    # creating dataframe with all results
    emotion_table = pd.DataFrame({"Anger": [all_anger, under_four_anger, rated_four_anger, rated_five_anger],
                                  "Disgust": [all_disgust, under_four_disgust, rated_four_disgust, rated_five_disgust],
                                  "Fear": [all_fear, under_four_fear, rated_four_fear, rated_five_fear],
                                  "Joy": [all_joy, under_four_joy, rated_four_joy, rated_five_joy],
                                  "Neutral": [all_neutral, under_four_neutral, rated_four_neutral, rated_five_neutral],
                                  "Sadness": [all_sadness, under_four_sadness, rated_four_sadness, rated_five_sadness],
                                  "Surprise": [all_surprise, under_four_surprise, rated_four_surprise, rated_five_surprise]})

    # transposing df and naming columns for better display
    new_table = emotion_table.T
    new_table = new_table.rename(columns={0:"All descriptions",
                                          1:"Under 4 stars",
                                          2:"4-5 stars",
                                          3:"5 stars"})

    return new_table, emotion_table

def visualise_results(emotion_table):
    # separating emotions from df by rating
    all_emotions = emotion_table.iloc[0]
    under_four_emotions = emotion_table.iloc[1]
    rated_four_emotions = emotion_table.iloc[2]
    rated_five_emotions = emotion_table.iloc[3]

    # bar plot for overview of emotions contained in all descriptions
    all_df = pd.DataFrame({"All descriptions": all_emotions})
    all_vis = all_df.plot.bar()
    plt.title("Emotion distribution in all book descriptions")
    plt.xticks(rotation=45, ha="right")
    plt.xticks(fontsize=8)
    
    # creating bar plot to compare emotions in rated descriptions
    comparison_df = pd.DataFrame({"Under-4 star descriptions": under_four_emotions,
                                  "4-5 star descriptions": rated_four_emotions,
                                  "5 star descriptions": rated_five_emotions})
    rated_vis = comparison_df.plot.bar()
    plt.title("Emotion distribution in rated book descriptions")
    plt.xticks(rotation=45, ha="right")
    plt.xticks(fontsize=8)
    return all_vis, rated_vis

def save_func(new_table, all_vis, rated_vis):
    # defining path to out folder for all results
    out_dir = os.path.join("..","out")

    # saving table
    table_path = os.path.join(out_dir,"emotion_table.csv")
    new_table.to_csv(table_path)

    # saving bar plot for all desc emotions
    all_plot_path = os.path.join(out_dir,"all_plot.png")
    all_vis.get_figure().savefig(all_plot_path)
    # saving bar plot for rated desc emotions
    rated_path = os.path.join(out_dir,"rated_plot.png")
    rated_vis.get_figure().savefig(rated_path)
    return None

def main():
    # load all data and sort according to rating
    print("Loading data ...")
    book_dataframe = load_data()
    # preprocess data
    under_four, rated_four, rated_five = preprocess(book_dataframe)
    # run descriptions through defined classifier, outputting most likely emotions for each string
    emotion_cls = set_cls()
    print("   ")
    print("Processing text ...")
    all_em, under_four_em, rated_four_em, rated_five_em = analyse_data(emotion_cls, under_four, rated_four, rated_five)

    # count emotion instances and make tables from results
    print("Visualising results ...")
    final_table, vis_table = make_table(all_em, under_four_em, rated_four_em, rated_five_em)
    # make a bar plot of the results
    all_plot, rated_plot = visualise_results(vis_table)
    # save output table and plots to out folder
    save_func(final_table, all_plot, rated_plot)
    print("Visualisations saved to the out folder.")
    return None

if __name__ == "__main__":
    main()