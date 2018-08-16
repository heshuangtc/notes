## scrap data from website
* read http/https with urllib and bs4
  - no 403 error
    ```
    import re
    from urllib.request import urlopen
    from bs4 import BeautifulSoup

    url_path = 'https://www.caranddriver.com'
    html_urllib = urlopen(url_path).read()
    soup = BeautifulSoup(html_urllib,'html.parser')
    ```
  - if 403 error above, add Request first with headers
    ```
    import re
    from urllib.request import Request, urlopen
    from bs4 import BeautifulSoup

    url_path = 'https://www.google.com/search?q=hybrid+cars&sourceid=chrome&ie=UTF-8'
    html_req = Request(url_path, headers={'User-Agent': 'Mozilla/5.0'})
    html_urllib = urlopen(html_req).read()
    soup = BeautifulSoup(html_urllib,'html.parser')
    ```
* find table
  - with BeautifulSoup
  ```
  from bs4 import BeautifulSoup
  soup = soup.body.div....div.find_all('table') #find table tag, middle structure is customized
  x_head = soup[0].find_all('th')[0].contents #read one by one
  x_row = soup[0].find_all('td')[0].contents
  ```



## scrap data from Twitter
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