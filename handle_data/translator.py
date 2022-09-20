from googletrans import Translator
from mumin import MuminDataset
from handle_data import get_text_data
import pandas as pd

translator = Translator()
# print(translator.translate(text='हद है, यहां आयुष मंत्रालय तक ग़लत दावा कर जाते है').text)


def add_translation(df):
    translations = []
    labels = []
    for i in range(len(df)):
        if df.iloc[i]['lang'] == 'und':
            continue
        print(df.iloc[i]['lang'])
        translation = translator.translate(text=df.iloc[i]['text'], dest='en')
        print(df.iloc[i]['text'])
        print(translation.text)
        translations.append(translation.text + '\n')
        labels.append(df.iloc[i]['label'])

    # df['translation'] = translations
    temp = pd.DataFrame()
    temp['text'] = translations
    temp['label'] = labels
    return temp


dataset = MuminDataset(size='medium',
                       twitter_bearer_token='AAAAAAAAAAAAAAAAAAAAAD18aQEAAAAA9cvR2Es1C7oE%2BYSdikWfevjKk6Y%3DCEpsydB03s8C6JXvZdiZEGWLMlHk8Py4lfBEFFYNieH6L5yXuq',
                       include_hashtags=False,
                       include_replies=False,
                       include_timelines=False,
                       include_mentions=False,)
dataset.compile()

tweet_df = dataset.nodes['tweet']
print(len(tweet_df.iloc[0]['lang_emb']))
# tweet_df = get_text_data.get_data()
# print(type(tweet_df))
# tweet_df = tweet_df.dropna()
# tweet_df = add_translation(tweet_df)
# for text in tweet_df[tweet_df['lang'] == 'und']['text']:
#     print(text)
# tweet_df.to_csv('dataset_trans.csv')
