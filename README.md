# bird-audio-classifier
*****************REPORT - DS FINAL PROJECT - RAWAN SHEHAYIB****************

I. Project Overview

1. A machine learning system that classifies three European bird species — European Robin (Erithacus rubecula), Common Chaffinch (Fringilla coelebs), and Great Tit (Parus major) — from audio recordings using a Random Forest classifier trained on mel spectrogram features, with a web interface that allows users to upload recordings and receive real-time species predictions with a timeline breakdown of when each species appears.

2. Information: 
Mel Spectograms: A mel spectrogram is a visual representation of an audio signal's spectrum over time, where frequencies are converted to the Mel scale to match human perception

Data Set:
I used the http://xeno-canto.org/ website to collect audio signals from European birds and classified them into: chaffinch, robin, and great tit. 

About the chosen birds:
http://allaboutbirds.org/guide/Great_Tit/overview
https://www.allaboutbirds.org/guide/European_Robin/overview
https://www.wildlifetrusts.org/wildlife-explorer/birds/finches-and-buntings/chaffinch

Libraries used: 
librosa — loads and processes audio files
numpy — handles arrays and math
scikit-learn — the ML toolkit (Random Forest lives here)
matplotlib — for plotting spectrograms and results

II. Data Structures

1. Arrays: 
1.1. 1D NumPy array for raw audio of 66,050 numbers per 3-second clip. 
audio, sr = librosa.load(path, sr=TARGET_SR). librosa.load() is called on the loaded wave files and it gives back a single long list of numbers. 
1.2. 2D array Mel spectrograms. After running librosa.feature.melspectrogram(), the 1D audio array gets transformed into a 2D grid, where rows represent frequency bands from low to high pitch and the cols are time steps, so each cell shows how loud the frequency is at a specific moment. We need this to train our model. 
1.3. feature_vector = mel_db.flatten(). Before feeding the spectrogram to the model, flatten the 2D into 1D array. With the flatten() function we 16,640 numbers in one row, so we can develop our labels and features parallel arrays, which store the training data, where features[i] correspond to labels[i]. 
features.append(feature_vector)  # the 16,640 numbers
labels.append(species)           # the species name
example: features[0]  =  [ -42.3, -38.1, ... ]  ←→  labels[0]  =  "robin"

2. Hash maps:
2.1. label_counts = Counter(labels)  Counter goes through all 465 labels and counts each one builds the hash map automatically:
It ends up being {"robin": 155, "chaffinch": 155, "great_tit": 155}
the function Counter internally does:
label_counts = {}
for label in labels:
    if label in label_counts:
        label_counts[label] += 1  
    else:
        label_counts[label] = 1   
This is useful, as for example, I printed the dictionary to check why I initially had a low success rate for prediction of one bird. More about this challenge below. 
2.2. vote_map is a hash map which uses np.unique and turns it into a dict to show how many times a species has appeared. Then max(vote_map, key=vote_map.get) looks through the hash map and finds whichever key has the highest value.
species, counts = np.unique(predictions, return_counts=True)
vote_map  = dict(zip(species.tolist(), counts.tolist()))
winner    = max(vote_map, key=vote_map.get)

3. Trees
The Random Forest is an algorithm where a random forest is made of 100 binary decision trees. Each internal node holds a specific feature. It selects a random subset of features to consider at each split, which ensures the trees stay uncorrelated and different from each other. each decision tree generates its own prediction independently. Each leaf stores a specific label and the process of classifying a new clip is a traversal from the root to the leaf and then takes a majority vote. 

4. Queues
4.1. As mentioned before about the three second sound extraction, explanation on the 3-second window process: grabbing what's inside the 3-second frame, processing it, then moving forward 3 seconds and repeating. Each step is one "window" — one training clip. In this way, a 3-minute robin recording becomes 60 clips, which creates more data. This is based on a FIFO as it moves window 0 -> window 1 and so on. 

III. Project Overflow
1. Inputs & Ouputs
Input from user: a .wav or a .mp3 containing at least 3 seconds of recording bird sound. This is uploaded through the REACT interface built. 
Output: Predicted species name, per species confidence percentages, vote counts, a time line showing the three 3-second windows and the frequency of how much each species appeared in the recording. Also, for data visualization features, a mel spectogram graph that has been analyzed was added for the full recording or for each 3 second window for more details. 
2. Workflow: 
Librosa opens the wav file, reads the raw binary data, and converts it into a numpy array. The sr=22050 argument tells it to resample the audio to exactly 22,050 samples per second regardless of what sample rate the original file was recorded at. 
Then, the sliding window organizes recordings into 3-second clips using window_samples = 3 * 22050 and any leftover audio shorter than 3 seconds at the end gets ignored. 
Then, each clip is turned into mel spectrogram (128 × 130 grid) runs a Fast Fourier Transform (FFT) across the window. It divides the 3 seconds into tiny overlapping time frames and measures how much energy exists at each frequency in each frame (rows represent frequency bands from low to high pitch and the cols are time steps). The result is a 2D grid which is then flattened to 16,640 numbers so that the model can analyze the data. 
Then, we use StandardScaler to normalzie all features to mean=0, std=1. This is used because 16,640 features all come from mel spectrogram values in decibels and not all features act the same way. StandardScaler gives every feature the same volume so the model listens to all of them equally. This is used to solve a challenge explained below. 
Then, Random Forest (100 trees) classifies each window independently and uses majority vote across all windows to find the final species prediction. If winner gets fewer than 40% of votes, then no match would be found. 

IV. Performance 
1. Training time complexity: O(n × d × t × log n) where n = number of clips (465), d = features considered per split (~129), t = number of trees (100). Observed training time was approximately 45–60 seconds on a standard laptop.
2. Prediction time complexity: O(t × log n) per window — each of the 100 trees traverses from root to leaf in O(log n) time. For a 30-second recording (10 windows), prediction takes under 2 seconds including feature extraction.
3. Feature extraction: O(n × w) where n = number of files and w = number of windows per file. Extracting features from 465 clips took approximately 3–5 minutes.

V. Challenges
The main problem faced was class imbalance. Despite having 10 recordings per species, robin produced 318 training clips while chaffinch only produced 200 — because robin recordings happened to be longer. On the first run, accuracy results for chaffinch was 0.58 and the accuracy for robin was 0.79. I first thought about adding more data, but I faced more problems with the model confusing chaffinch clips with robin. Then, this problem was resolved by adding a MAX_CLIPS = 200 cap in the feature extraction script, forcing equal clip counts across all species. Accuracy jumped from 85% to 94% after this fix. Regardless of how long the recordings are, they were all capped at 200.
Chaffinch/great tit confusion was the persistent challenge. These two species overlap in the 2,000–5,000 Hz frequency range, making them hard to separate with flat mel spectrogram features. Adding MFCCs (a set of features that represent the short-term power spectrum of a sound) and spectral centroid features improved chaffinch F1 from 0.71 to around 0.81, but confusion remained. Being more specific in the features was important for the decision trees as they make the voting process more precise. 

VI. Improvements 
If I had more time and more tools to develop the program, I would use a wider variety of datasets and focus more on sound detection. Instead of focusing on 3 birds, I would enlarge the selection, and work on making it predict more bird sound inputs. Most real soundscape analysis require more sound detection due to other sounds in nature and they would be longer recordings. I would improve the algorithm by implementing ways to analyze spectograms and not just rely on the three second window frequency analysis, possibly using Convolutional Neural Network instead of Random Forest, as it would be able to extract information from 2D arrays from the spectorgrams. 

VII. Learning 
Through the project, I learnt how Data Structures apply to a wider range of applications and how a concept like trees, appears in a much more complicated and fast way in such algorithms, like the Random Forest one. I could see how important that it is O(logn) and how tree decisions lead to predictions and then lead to results. I also learnt how to approach different challenges and understood how a full system works starting from a sound audio. I learnt how that can be made into arrays and how information is transmitted with numbers to build a full model. I also learnt more about how to handle data and went from thinking about just adding more audio to thinking about how features relate to the decision trees to make the model more accurate. I also learnt audio signal processing (mel spectrograms, MFCCs, sample rates), and full-stack development (connecting a Python Flask API to a React frontend). 

VIII. Real-World Relevance 
This idea is relevant in the research field of audio processing and wildlife monitoring. I learnt more about this from research projects about "hearing climate change" through digital signal processing and machine learning. Moreover, it is also important to classify bird activity in nature preserves, tracking which species are present and when without requiring human observers. This is beneficial to study how bird activity changes with seasons and over the years, which tells us a lot about the climate, through detecting the presence or absence of endangered species in large audio datasets. 

IX. Use of AI tools
I only used Claude for developing this project. Starting from brainstorming, learning about Random forest, and for building and debugging. It helped me understand how to divide the tasks of the project and to build the different pieces of it and for the user interface. However, due to limited messages and Claude restrictions, my usage of it was limited to understanding concepts clearly and writing code that I did not know how to develop beforehand (like applying Random Forest and generating mel spectograms). Therefore, I used it to set a plan at the beginning, learn new concepts, and develop certain functions of the program. I decided to only use Claude to understand my project overflow to avoid confusion, as I did not want AI to replace my understanding of the project.
