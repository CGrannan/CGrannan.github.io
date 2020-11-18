---
layout: post
title:      "Real Estate Investing in Philadelphia"
date:       2020-11-18 23:46:25 +0000
permalink:  real_estate_investing_in_philadelphia
---


For this blog post, I want to go over the process of one of my recent projects. The goal of the project was to identify the five best area codes for short term (1-3 years) real estate investment in Philadelphia. To accomplish this goal I used a dataset obtained through [Zillow.com](https://www.zillow.com/research/data/). Each row of the dataset represented a zip code, and there was a row for every zip code in the country. The columns consisted of some identifying features (city, state, etc.) and a column for each month from April 1996 until April 2018. Each month contained the median home value for that zip code during that month. (104)

### Data Preprocessing

Like any project, my first step was going to be cleaning the data. To begin, I cut the data down into the zip codes that I was interested in, those in Philadelphia. Then, I removed all of the descriptive features of the zipcodes leaving me with a dataframe of 35 rows (the zip codes) and 266 columns (each month). The last step in preparing my data was converting the dataframe from wide format to long format. I needed to set the dates as the index of my dataframe and keep a column for the zip codes. It was very quick and easy to do this use pandas melt function. This was new to me, so I thought I would share it here.

```
    melted = pd.melt(df, id_vars=['RegionName'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
```

In this case, 'RegionName' was the name for my zip code column. This function kept my zip codes exactly where they were, but pivoted all of the date columns into my index and created a new column 'value' to keep track of the mean value of homes.  It was very handy for this project.  Now I was left with a dataframe with two columns, my zip codes and my values, but I needed my zip codes separated.  I split the dataframe into 35 new dataframes, one for each zip code and stored them in a list. Now I was ready to start working with my data.

### EDA

Before jumping into modeling, I wanted to inspect the data.  I plotted out the mean and median price of homes in Philadelphia over this period.  The resulting chart looked like this:


![](https://imgur.com/RiMYFGe)


The first thing that jumped out at me was the boom and collapse of the real estate market in the 2000's. My main concern at this point was that this sizable bump was going to throw off our models.  I calculated that the market returns to normal around 2012, so I dropped the data that came before this point.  
