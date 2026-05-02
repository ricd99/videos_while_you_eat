# videos while you eat.

I found myself refreshing my YouTube recommendations for 10+ minutes on most nights when I ate alone. This web app helps find video essay-style YouTube channels (i.e. medium and long-form content creators that are enjoyable to watch while eating a meal) that are similar to a given channel. I am no longer refreshing my recommendations for 10+ minutes!

This repo is split into two parts. The first part develops the nearest neighbor model that identifies similar channels. This contains the ETL pipeline + model training + uploading the model to Hugging Face. The second part is the web app that handles user queries. This includes receiving a channel name as input, turning the channel name into data for the model, and retrieving and displaying the prediction. The scripts that handle model training are located in the scripts/ directory, and the ui is located at app/main.py. Both parts rely on code located in the src/ directory. Repository organization improvements are possible; it is something I'm actively thinking about. Suggestions are welcome!

## Technologies
This is a non-exhaustive list that covers the core technologies used for this project. 

- Python
  - pandas
  - scikit-learn
  - sentence-transformers
  - FastAPI
- Hugging Face
- Google Cloud Run

## My Process
I love watching hockey, but I swear I've always had a bit of a jinx. It really seemed like whenever I'd say one team would win a game, the other would always win. So, I wanted to build something that could help me make data-driven statements (!) in front of my friends to hopefully help me sound smarter!
- First, I collected season-level team data dating back to the 2008-2009 season from moneypuck.com, performed some initial data exploration, and developed a few approaches to make predictions using this dataset. In addition, playoff appearance was not in the moneypuck dataset, so I manually collected that data myself.
- For feature engineering, I tried: doing nothing (i.e. just doing the minimal amount of preprocessing for the prediction algorithms to run), manual feature engineering (by using visualizations and my (limited) domain knowledge), and feature selection using scikit-learn's RFECV.
- For the model, I tried scikit-learn's ridge regressor, random forest, K-neighbours, gradient-boosted trees, ensembles, and stacking.

The combo that gave the best results was using feature selection on a minimally preprocessed dataset (no other feture engineering) and a ridge regression model.

I also uploaded the datasets I created to Kaggle, and they got a modest amount of attention there: 
- [NHL team data by season](https://www.kaggle.com/datasets/ryanhdar/nhl-team-data-by-season-2015-christmas-2025)
- [NHL team data by season with playoff appearance column](https://www.kaggle.com/datasets/ryanhdar/nhl-team-data-and-playoff-result)

## Results and next steps
Finding a way to measure my results is not the most straightforward, as sports odds can change dramatically in just a few games. I decided a reasonable metric was to see how closely my predictions resembled moneypuck.com, a site whose predictions are relied upon by many fantasy hockey general managers and sports betters. I matched 14/16 predictions, which is not bad! The main addition I want to make to this project is to integrate a real-time data pipeline. This way, the model constantly gets fresh data from the newest NHL games, and it can update its playoff odds in real-time. 

My model's percentage odds can be viewed at the bottom of the final_model.ipynb file inside the notebooks folder. Unfortunately, but unsurprisingly and accurately, my Canucks are dead last.
