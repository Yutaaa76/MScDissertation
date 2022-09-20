from mumin import MuminDataset
import pandas as pd


def get_data(size):
    # dataset = MuminDataset(size='medium',
    #                        twitter_bearer_token='AAAAAAAAAAAAAAAAAAAAAD18aQEAAAAA9cvR2Es1C7oE%2BYSdikWfevjKk6Y%3DCEpsydB03s8C6JXvZdiZEGWLMlHk8Py4lfBEFFYNieH6L5yXuq')
    dataset = MuminDataset(size=size,
                           twitter_bearer_token='AAAAAAAAAAAAAAAAAAAAAD18aQEAAAAA9cvR2Es1C7oE%2BYSdikWfevjKk6Y%3DCEpsydB03s8C6JXvZdiZEGWLMlHk8Py4lfBEFFYNieH6L5yXuq',
                           include_hashtags=False,
                           include_replies=False,
                           include_timelines=False,
                           include_mentions=False,)
    dataset.compile()

    article_df = dataset.nodes['article']
    article_df.rename(columns={'embedding': 'article_emb'}, inplace=True)

    claim_df = dataset.nodes['claim']
    claim_df.rename(columns={'embedding': 'claim_emb'}, inplace=True)

    tweet_df = dataset.nodes['tweet']
    tweet_df.dropna(inplace=True)

    has_image_df = dataset.rels[('tweet', 'has_image', 'image')]
    image_df = dataset.nodes['image']
    tweet_image_df = (tweet_df.merge(has_image_df, left_index=True, right_on='src')
                      .merge(image_df, left_on='tgt', right_index=True)
                      .reset_index(drop=True))

    # has_article_df = dataset.rels[('tweet', 'has_article', 'article')]
    # print(has_article_df)
    # tweet_image_article_df = (tweet_image_df.merge(has_article_df, left_index=True, right_on='src')
    #                           .merge(article_df, left_on='tgt', right_index=True)
    #                           .reset_index(drop=True))

    discusses_df = dataset.rels[('tweet', 'discusses', 'claim')]
    tweet_claim_df = (tweet_df.merge(discusses_df, left_index=True, right_on='src')
                      .merge(claim_df, left_on='tgt', right_index=True)
                      .reset_index(drop=True))

    tweet_claim_image_df = tweet_claim_df.merge(tweet_image_df, how='inner', on='tweet_id')
    # print(tweet_claim_image_article_df.columns)
    tweet_claim_image_df = tweet_claim_image_df[[
        'text_emb_x', 'pixels', 'label', 'claim_emb',
        'train_mask', 'val_mask', 'test_mask'
    ]]
    tweet_claim_image_df.rename(columns={'text_emb_x': 'text_emb'}, inplace=True)
    # print(tweet_claim_image_article_df.columns)

    train_set = tweet_claim_image_df.query('train_mask == True')
    pos_set = train_set[train_set['label'] != 'misinformation']
    for i in range(14):
        train_set = pd.concat([train_set, pos_set])
    train_set = train_set.sample(frac=1).reset_index(drop=True)  # shuffle

    # return tweet_claim_image_df.query('train_mask == True'), tweet_claim_image_df.query('val_mask == True'), tweet_claim_image_df.query('test_mask == True')
    return train_set, tweet_claim_image_df.query('val_mask == True'), tweet_claim_image_df.query('test_mask == True')


# train_df, val_df, test_df = get_data('medium')
#
# print(len(train_df) + len(val_df) + len(test_df))
# print(len(train_df), len(val_df), len(test_df))
# # print(len(train_df['content_emb'].iloc[0]))
# # print(len(train_df['title_emb'].iloc[0]))
# print(train_df.label.value_counts())
# print(val_df.label.value_counts())
# print(test_df.label.value_counts())
