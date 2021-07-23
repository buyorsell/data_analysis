from typing import List, Tuple
import nltk
import pymorphy2
import codecs
import pickle
import numpy as np
from gensim.models import LdaMulticore
from gensim.corpora.dictionary import Dictionary
import json


class PrepareNew:
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.tokenizer = nltk.WordPunctTokenizer()
        self.stopwords = set(
            line.strip() for line in codecs.open('../../rus_stopwords.txt', "r", "utf_8_sig").readlines())  # FILEPATH

    def prepare_corp(self, news_list: List[str]) -> List[List[str]]:
        return [self.newstext2token(news_text) for news_text in news_list]

    def newstext2token(self, news_text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(news_text.lower())
        tokens_with_no_punct = [self.morph.parse(w)[0].normal_form for w in tokens if w.isalpha()]
        tokens_base_forms = [w for w in tokens_with_no_punct if w not in self.stopwords]
        tokens_last = [w for w in tokens_base_forms if len(w) > 1]
        return tokens_last


class GetBosPred():

    def __init__(self, company_name: str):
        filename = "../TickerModels/" + company_name + ".sav"  # FILEPATH
        self.model = pickle.load(open(filename, 'rb'))
        self.coefs = self.model.coef_[0]

    def get_lda_preds(self, tokens_list: List[List[str]]) -> np.array:
        self.lda_model = LdaMulticore.load("../LDA_model_BoS")  # FILEPATH
        self.dictionary = Dictionary.load("../LDA_dict_BoS")  # FILEPATH
        features = []
        for tokens in tokens_list:
            lda_scores = self.lda_model.get_document_topics(self.dictionary.doc2bow(tokens))
            asfeatures = [0. for i in range(16)]
            for theme, score in lda_scores:
                asfeatures[theme] = score
            features.append(asfeatures)
        return np.array(features)

    def get_bos_one_new(self, tokens: List[str]) -> Tuple[float]:
        tokens = self.get_lda_preds([tokens])
        positive = 0
        negative = 0
        self.coef = self.model.coef_[0]
        for i in range(16):
            if self.coef[i] > 0:
                positive += self.coef[i] * tokens[0][i]
            else:
                negative += self.coef[i] * tokens[0][i]
        return (positive, negative)


def write_tickers(new_item: AllNews) -> AllNews:
    recommendations = []
    with open('../../tickers.json') as f:  # FILEPATH
        ticker_dict = list(json.load(f).values())
    for ticker in ticker_dict:
        predictor = GetBosPred(ticker)
        pos, neg = predictor.get_bos_one_new(item.tokens)
        new_rec = Recommendation(quote=ticker,
                                 bos=pos + neg,
                                 bos_positive=pos,
                                 bos_negative=neg,
                                 datetime=item.datetime
                                 )
        recommendations.append(new_rec)
    return recommendations


def modify_item(item: AllNews) -> AllNews:
    item.tokens = PrepareNew().newstext2token(item.text)
    item.recommendations = write_tickers(item)
    return item
