### scrap data from website

### scrap data from Twitter
* connect to twitter and pull first 200 twitters

  log in [https://apps.twitter.com](https://apps.twitter.com) to create app and generate token
  ```
  import tweepy
  auth = tweepy.OAuthHandler(consumer_key, consumer_secret) #Fill these in
  auth.set_access_token(access_token, access_token_secret)  #Fill these in
  api = tweepy.API(auth)
  tweets = api.user_timeline(screen_name = 'chrisalbon', 
                             count = 200, 
                             include_rts = False, 
                             excludereplies = True)
  ```

* extract data from twitters
  
  * find urls data `[i.entities['urls'] for i in tweets[:5] if 'urls' in i.entities.keys()]`
  * find text data `j.text for j in tweets if 'machinelearningflashcards' in j.text`