---
layout: post
title:      "Building Recommendation Systems for Boardgames"
date:       2020-12-08 18:34:21 +0000
permalink:  building_recommendation_systems_for_boardgames
---

![boardgames](https://cdn.thewirecutter.com/wp-content/uploads/2018/03/boardgamesforadults-2x1-7452.jpg)

As a consumer in the modern world it is easy to feel adrift in the sea of possibilities. There are limitless numbers of products for sale and it is often difficult to find a particular product that best suits what you're looking for. This is where recommendation systems come in. A recommendation system is designed to provide targeted product recommendations to consumers based on their preferences and history. Nearly everybody these days has come into contact with many of these systems. Everytime you see the words "products related to" you are looking at the output of a recommendation system. 

For a recent [project](https://github.com/CGrannan/building-boardgame-recommendation-systems) I delved into the world of recommendation systems, and learned how to create many different types. This post will walk through that project and explain some of the different types of systems that I created. Before we get to the details of the project, let's go over the main differences between types of recommendation systems.
 
### Content Based vs. Collaborative Filtering

There are two main types of recommendation systems, content based and collaborative filtering. Content based systems compute the similarities between items based on the features of those items. In this particular project I was looking at generating recommendations for board games. A content based system looks at details like the themes of the games, the time spent to play the games, and the number of players needed to play the games and computes how similar all of the games in the system are to each other. Then, when fed the name of a game this system will return some games that are similar so the user can narrow down their choices.  Collaborative filtering systems, on the other hand, rely on user ratings instead of descriptive features. In the board games example, a collaborative-filtering system will compute similarities between games by looking at how different users have rated that game and other games. then when fed a game, the system will return games that have been rated similarly by many different users. Generally, collaborative-filtering models are a more reliable form of recommendation systems as user enjoyment is a stronger metric than similarity between items, but there are strengths and weaknesses to both approaches. Content-based systems are much easier to get off the ground as they do not require many different users to add content. They solely rely on the nature of the items themselves. This is called the cold-start problem with collaborative-filtering systems as they often require more effort to bring into effect, especially without access to a large dataset. Ultimately both types of models are effective in connecting consumers and products when used correctly.

### Content-Based Recommendations with NLP

Now that we have covered the basics of recommendation systems, I will mention the details of my project. I used descriptive statistics and reviews of 2000 board games to create several different recomendation systems. The data was all obtained from the [boardgamegeek](https://boardgamegeek.com/) website using their api and a convenient python [wrapper](https://github.com/lcosmin/boardgamegeek). All of the scraping procedures are shown in my githup repo if you are interested and I encourage you to take a peek. Once all of the data had been collected and cleaned I decided to start with a content-based recommendation system. I decided that i would use some basic natural language processing techniques to compare the similarities between the descriptions of different games. I combined a short description of each game with tags that referenced the mechanics, categories and families of the games into a bag of words. Then I transformed each bag of words into vectors by using tf-idf vecotrization. I won't go into the details of the vectorization process, but you can read about it [here](https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a). Here is a little snippet of my code so you can follow along though:

```
# Create a vectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=5, preprocessor=' '.join)
# transform our bags of words into vectors
tfidf_matrix = tf.fit_transform(df['bag_of_words'])
# Compute similarities
cos_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend(name, names, df, cosine_sim, n):
    '''
    Returns recommendations from content-based recommendation system.
    
    Parameters:
    name - Boardgame to be compared, should be a string.
    names - Array of boardgame names.
    df - Dataframe of statistics, used to gather names for recommendations.
    cosine_sim - matrix of cosine similarities.
    n - number of recommendations to be returned.
    
    Returns:
    The names of (n) games similar to (name).
    '''
    recommended_games = []
    idx = names[names == name].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top_n_indices = list(score_series.iloc[1:n+1].index)
    for i in top_n_indices:
        recommended_games.append(list(df['name'])[i])
    return recommended_games

# Get recommendations for Gloomhaven
tfidf_gloom = recommend('Gloomhaven', names, df, cos_sim, 5)
tfidf_gloom
```
The recommendations for Gloomhaven were: 'Gloomhaven: Jaws of the Lion', 'Dragonfire', 'First Martians: Adventures on the Red Planet', 'Dungeons & Dragons: The Legend of Drizzt Board Game', and 'SeaFall.' Not a bad set of recommendations as there are some definite similarities in themes and mechanics. For those who are interested, here is a wordcloud showing the major keywords in my text data for all games:

![wordcloud](https://i.imgur.com/nTs6LJp.png)

Now we can see which words appear the most in our dataset. Several of these are unsurprising, such as player, card, game, and tile. We also see some popular bigrams like card game and dice rolling. Through this word cloud we can see that several mechanics take prominent roles (hand management, dice rolling, card drafting).

### Collaborative-Filtering With Pyspark ALS

After working through the content-based systems, I crafted a couple of simple collaborative-filtering models. These models were not particularly exciting and I won't go into the details here. For a closer look at those models check out the github repository. I do want to talk about the last model I made though. For the culmination of the project I created a collaborate-filtering model based on alternating least squares using pyspark. Pyspark is a very useful language designed for machine learning. You can read more about pyspark and its implementations [here](https://spark.apache.org/docs/latest/api/python/index.html).  Loading pyspark into a google colab notebook can be a bit tricky as there are lots of components to install. Here is some code that you can use to get started.

```
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q https://www-us.apache.org/dist/spark/spark-3.0.1/spark-3.0.1-bin-hadoop2.7.tgz
!tar xf spark-3.0.1-bin-hadoop2.7.tgz
!pip install -q findspark
!pip install pyspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.0.1-bin-hadoop2.7"

import findspark
findspark.init()


# import necessary libraries
from pyspark.sql import SparkSession

# instantiate local SparkSession object
spark = SparkSession.builder.master('local').getOrCreate()
```

From here, I loaded my data into a resilient distributed dataframe, which is very similar to loading data into a pandas dataframe. To use ALS in pyspark, all of my data had to be integer type, so I needed to change my user names into user IDs and convert all of my data from strings into integers. Once all of my datatypes were in order, it was time to begin modeling. Modeling with pyspark is very easy and intuitive, and a baseline model can be created, fitted, and evaluated in 7 lines of code. Here is an example:

```
train, test = rating_df.randomSplit([.8, .2])
als = ALS(maxIter=5, userCol='user_id', itemCol='game_id', ratingCol='rating', coldStartStrategy='drop')
model=als.fit(train)
preds = model.transform(test)
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='rating', metricName='rmse')
rmse = evaluator.evaluate(preds)
print('RMSE: ', rmse)
```

After some fine tuning, I was able to get our test RMSE error down to 0.985, meaning that we had less than 1 point in average error on a 10 point scale. This is pretty good considering the sparcity of the matrix that we are using. The final part of the project was extracting recommendations for a new user from this model. To accomplish this, we create a new RDD object of the new user's ratings, then combine it with the RDD of the whole dataset. Then we retrain the model and predict ratings for a set number of games for every user. Finally, we access the recomendations for the new user. To test this, I inserted ratings for several of my favorite games: 

```
user_ratings = [('Cthulhu Wars', 8, user_id),
                  ('Terraforming Mars', 9, user_id),
                  ('Gloomhaven', 9, user_id),
                  ('Twilight Imperium: Fourth Edition', 8, user_id),
                  ('Mage Knight Board Game', 8, user_id)]
```

And my top five recommendations were Nemesis, Go, Magic: the Gathering, Twilight Struggle, and Puerto Rico. All of these games had estimated ratings above an 8.5, and all of them have been games that I love or have been looking to try playing. These are overall very good recommendations for me.

### Conclusion

When comparing the results between our two models we can see that there is no overlap in game recommendations. This is because both models are measuring different features. The content-based model is showing games that are similar in description or theme to our orignial game while the collaborative-filtering model is showing games that it thinks I will like based on mine and other users's previous ratings. I personally like the recommendations supplied by the ALS model, but I should because they are tailored to my taste. These recommendations would be useless to someone who doesn't agree with my initial ratings. Meanwhile the content-based results could apply to anyone who likes Gloomhaven as they may be able to find a game that has a similar theme or mechanic. Ultimately, both models are effective at connecting users with games that they may enjoy, just with different methods.

Thanks for reading!

### Further Reading:

My project repository can be found [here.](https://github.com/CGrannan/building-boardgame-recommendation-systems)

For more on tf-idf vectorization in NLP, you can read this [post.](https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a)

For more on pyspark, you can check out this [article.](https://towardsdatascience.com/a-brief-introduction-to-pyspark-ff4284701873)


