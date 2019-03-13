# Web_Info_Management
This repository contains my COEN 169 Web Information Management Project. Essentially, this project mimicked the Netflix Prize competition in which Netflix challenged teams to beat their recommendation algorithm. My project utilizes several methods to calculate vector similarity including cosine similarity, Pearson correlation, and item-based cosine similarity. Additionally it has the functionality to incorporate Inverse User Frequency and Case modification for improved results. The source code is still a little messy and work is still ongoing!

File Descriptions:

train.txt - training data that we were given in order to construct our prediction schemes. The file is organized into a 200x1000 matrix (200 users & 1000 movies).

train5.txt - A list of 100 new test users. Each new user in this file provides 5 movie ratings to help match similar users from the training data.

train10.txt - A list of 100 new test users. Each new user in this file provides 10 movie ratings to help match similar users from the training data.

train20.txt - A list of 100 new test users. Each new user in this file provides 20 movie ratings to help match similar users from the training data.

project2.py - Source code that attempts to synthesize the train data a testing file in order to predict what rating each new user will assign a corresponding movie. 
