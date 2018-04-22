import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from kafka import SimpleProducer,KafkaClient

topic='twitter'

kafka=KafkaClient('localhost:9092')
producer=SimpleProducer(kafka)


consumer_key="*************"
consumer_secret="******************"
access_token="**************"
access_token_secret="********************"

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


