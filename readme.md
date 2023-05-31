# Assignment 5 – Self-Assigned Project

## 1.	Contributions
This project reused large parts of the code I wrote for assignment 4, and I remain the primary contributor. Troubleshooting was primarily handled through googling and ChatGPT, and the sampling was done with the aid of a class mate, but the code is generally my own.

The Goodreads book description dataset was found here, uploaded to the open database commons by Kaggle user randomarnab who scraped it from the Goodreads website. 

## 2.	Methods
The main script, book_emotions.py, loads the dataset file, binds it to a dataframe, and ensures that the descriptions are in string format. The file is den preprocessed, keeping only the descriptions written in English before truncating them to fit within the model’s limits. Finally, the descriptions are separated based on the ratings of the corresponding books, sampled, and fed into a text classifier. The results are gathered in a single dataframe for presentation and visualizations, all saved to the out folder.

## 3.	Usage
The dataset is already present in the data folder, but can be re-downloaded from the link provided in Contributions above. The script will expect the subfolder be named dataset.

Before running the script, ensure that the current directory is lang-project and execute bash setup.sh from the command line. This will update pip and install the packages noted in requirements.txt. Then, run cd src/ and python book_emotions.py from command line to run the script. By default, the script is set to sample 500 descriptions from each ratings category to compensate for greatly unequal category sizes.

## 4.	Discussion
This project was initially envisioned quite differently, but constraints of both time, skill, and server time has somewhat reduced the scope and altered the general direction. The original attempt was to train a classifier on literary data, but data labeling and training proved too time-consuming compared to what I had set aside to get it done. As such, this project appears here in diminished form, supported in no small part by pieces of code I had written beforehand.

As for the results, the emotions are distributed unevenly, with most of the ones not deemed neutral being categorized under fear, joy, and sadness, which is likely the descriptions attempt to create a sense in the reader of suspense, wonder, and tragedy, respectively. The distribution is as seen below:

![image](https://github.com/nikolaimh/lang-project/assets/112465764/8baae77c-ef7a-4d82-bca1-087031cec77b)

More interesting (and potentially more interpretable) are the results of rated emotions distribution:

![image](https://github.com/nikolaimh/lang-project/assets/112465764/8e07adca-cf42-41f4-a6dc-b07e5721dec9)

Though no gulf between the rated descriptions is visible, a clear trend does appear: the higher a book is rated, the more likely it is for its description to be neutral in tone while the inverse is true for lower rated books and emotionally loaded descriptions. Descriptions focusing on fear, sadness, surprise, or joy trend to lower ratings for their corresponding books, which could mean several things.

Here, it may be useful to keep in mind that such back-of-the-cover blurbs are often written by the publisher, but the emotional classification may reveal how the publishers themselves view their books: whether they lean on more emotion-laden blurbs to compensate for weaker writing or permit the opposite for books they believe can succeed on neutral merits is unknown, but the tendency appears consistent in several samples. Possibly, the blurbs are simply reflective of the books whose backs they occupy, and emotion-laden books simply trend lower than their more neutral counterparts.

All this is not to say that neutrally described books all succeed wildly nor that emotional ones never do, as this is plainly untrue; the graph above show solid representation for emotionally judged five-stars and neutral under-fours. Also, it would be useful to keep in mind that the data source is not universal as it denotes only the opinions on the readership present on Goodreads, which is as unlikely to be generally representative as any self-selected community. Other inclinations might appear among other reader communities, but for Goodreads, in this sample, the trend is clear but not world-defining.

For reference, the full results table is as follows:

|Emotions|All descriptions|Under 4 stars|4-5 stars|5 stars|
|--------|----------------|-------------|---------|-------|
|Anger   |73              |18           |33       |22     |
|Disgust |101             |42           |30       |29     |
|Fear    |283             |118          |94       |71     |
|Joy     |230             |81           |77       |72     |
|Neutral |607             |159          |192      |256    |
|Sadness |151             |59           |54       |38     |
|Surprise|55              |23           |20       |12     |
