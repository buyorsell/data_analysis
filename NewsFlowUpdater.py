import pandas as pd
from gensim.models import LdaMulticore
from gensim.corpora.dictionary import Dictionary
import numpy as np
import pickle
import json
import requests
from typing import List

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    ARRAY,
    DateTime,
    REAL,
    Float,
    ForeignKey,
    and_
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class AllNews(Base):
    __tablename__ = 'all_news'
    id = Column(Integer, primary_key=True)
    datetime = Column(DateTime)
    rubric = Column(ARRAY(String))
    link = Column(String, unique=True)
    title = Column(String)
    text = Column(String)
    locs = Column(ARRAY(String))
    pers = Column(ARRAY(String))
    orgs = Column(ARRAY(String))
    x = Column(Float)
    y = Column(Float)
    highlights = Column(String)
    tokens = Column(ARRAY(String))
    recommendations = relationship("Recommendation", backref="about")


class Recommendation(Base):
    __tablename__ = 'recommendation'
    id = Column(Integer, primary_key=True)
    quote = Column(String)
    bos = Column(Float)
    bos_positive = Column(Float)
    bos_negative = Column(Float)
    datetime = Column(DateTime)
    news_id = Column(Integer, ForeignKey("all_news.id"))



class NewsFlow():
    def __init__(self, model_path = "../LDA_model_BoS", dict_path = "../LDA_dict_BoS"):
        self.lda_model = LdaMulticore.load(model_path)
        self.dictionary = Dictionary.load(dict_path)
    def write_last_news(self, df: pd.DataFrame):
        df.index = df.id
        with open('../../tickers.json') as f:
            ticker_dict = list(json.load(f).values())
        for ticker in ticker_dict:
            result_df = self.get_ticker_df(ticker, df)
            for index, row in result_df.iterrows():
                datetime_mark = int(index.timestamp()) // (60 * 60 * 24 * 7)
                quote_mark = ticker

                dump_dict = {'news': row.top5,
                             'bos_positive': row.bos_positive,
                             'bos_negative': row.bos_negative,
                             'num_negative': row.num_negative,
                             'num_positive': row.num_positive
                            }

                dump_json = json.dumps(dump_dict)

                post_link = 'https://buyorsell.ml/redis/top/{datetime}/{quote}'.format(datetime = datetime_mark, quote = quote_mark)
                requests.post(post_link, data = dump_json)
        
    def get_ldas(self, tokens: List[str]):
        lda_scores = self.lda_model.get_document_topics(self.dictionary.doc2bow(tokens))
        asfeatures = [0. for i in range(16)]
        for theme, score in lda_scores:
            asfeatures[theme] = score
        return np.array(asfeatures)
    def get_pred(self, lda_score: np.array, coef):
        positive = 0
        negative = 0
        for i in range(16):
                if coef[i]>0:
                    positive += coef[i] * lda_score[i]
                else:
                    negative += coef[i] * lda_score[i]
        return positive, negative
    def get_ticker_df(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        df.index = df.id
        needed_ids = [item for item in df.id]
        needed_recommends = pd.read_sql(session.query(Recommendation).filter((Recommendation.quote == ticker) & (Recommendation.news_id.in_(needed_ids))).statement, session.bind)
        needed_recommends.index = needed_recommends.news_id
        needed_recommends = needed_recommends.drop(["id", "datetime"], axis = 1)
        total = pd.concat([df, needed_recommends], axis = 1)
        total = total[["id", "tokens", "bos_positive", "bos_negative", "datetime", "bos"]]
        ldas = [self.get_ldas(tokens) for tokens in total.tokens]
        total['ldas'] = ldas
        id_lists = [[i] for i in total.id]
        total['id_list'] = id_lists
        resampler = total.drop(['tokens', 'id'], axis = 1)
        resampler.index = resampler.datetime
        resampler.bos_positive = resampler.bos > 0 
        resampler.bos_negative = resampler.bos < 0
        resampler['ldas'] = resampler['ldas'].apply(lambda x: np.array(x))
        grouped = resampler.resample("W")
        aggregate = list((k, v['bos_positive'].sum(), v['bos_negative'].sum(), v["ldas"].sum(), v['id_list'].sum()) for k, v in grouped)
        new_df = pd.DataFrame(aggregate, columns=["datetime", 'num_positive', 'num_negative', "ldas", "id_list"]).set_index("datetime")
        sorted_ids = []
        for neww in new_df.id_list:
            try:
                s_ids = sorted(neww, key = lambda x: abs(total.loc[x].bos), reverse = True)
            except:
                s_ids = []
            sorted_ids.append(s_ids[:5])
        new_df['top5'] = sorted_ids

        filename = "../TickerModels/" + ticker + ".sav"
        model = pickle.load(open(filename, 'rb'))
        coef = model.coef_[0]

        pos_list = []
        neg_list = []
        for lda_score in new_df.ldas:
            try:
                pos, neg = self.get_pred(lda_score, coef)
            except:
                pos = 0
                neg = 0
            pos_list.append(pos)
            neg_list.append(neg)

        new_df['bos_positive'] = pos_list
        new_df['bos_negative'] = neg_list

        return new_df.drop(['ldas', 'id_list'], axis =1)



pushkin = NewsFlow()
pushkin.write_last_news(last_df)

