import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from kafka import SimpleProducer,KafkaClient

topic='twitter'

kafka=KafkaClient('localhost:9092')
producer=SimpleProducer(kafka)


consumer_key="dx554RgT5Jl5JFIiN82qfYuhh"
consumer_secret="1To1V0iY8sFwUgWQBjKJ02UNLcL0k1Kq6ApCliA975R4J4LB9M"
access_token="347852725-eqNvM2p9wqnvVJwYVzSyQOEKl2X0gBYntYFy0os4"
access_token_secret="YVqrjLDf3nI5v41vYbXE3Wq8ulWrQyG8V4QRib9x4dfF3"

baglanti=OAuthHandler(consumer_key, consumer_secret)
baglanti.set_access_token(access_token,access_token_secret)

api=tweepy.API(baglanti)

class tweetListener(tweepy.StreamListener):

    def on_status(self, status):
        print(status.text)
        producer.send_messages(topic,status.text.encode('utf-8'))

tweetstream = tweetListener()
myStream = tweepy.Stream(auth = api.auth, listener=tweetListener())
myStream.filter(track=['Syria'])


