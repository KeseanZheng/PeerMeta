import pandas as pd
import jieba

df = pd.read_csv('fraud_news.csv')

with open('../stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().splitlines()
jieba.load_userdict("../terms/FinWords.txt")
df['TokenizedNews'] = df['News'].apply(lambda x: list(jieba.cut(x)))

def count_fraud_terms(tokens, fraud_set):
    return sum(token in fraud_set for token in tokens)

fraud_terms = '造假 作假 涉嫌 指控 失实 爆出 虚假 弄虚作假 捏造 炒作 不实 隐瞒 利益输送 爆料 起底 查出 不属实 篡改 黑幕 舞弊 欺诈 疑点 违反 虚报 揭露 纠纷 举报 偷工减料 虚增 虚减'
fraud_set = set(fraud_terms.split())
df['TermFreq'] = df['TokenizedNews'].apply(lambda x: count_fraud_terms(x, fraud_set))

def count_stock_names(tokens, stock_names):
    return sum(token in stock_names for token in tokens)

stock_names = df['StockName'].unique().tolist()
grouped = df.groupby('ReportYear')
df['NewsFirmCount'] = grouped['TokenizedNews'].transform(lambda x: count_stock_names(x, stock_names))

df['ATF_IIF'] = (df['TermFreq'] * df['TotalFirmCount']) / df['NewsFirmCount']

grouped = df.groupby(['StockCode', 'ReportYear']).agg({
    'ATF_IIF': 'sum',
    'TotalNewsCount': 'first'
})

grouped['FSFP'] = grouped['ATF_IIF'] / grouped['TotalNewsCount']

grouped.reset_index(inplace=True)
grouped.to_csv('processed_fraud_news.csv', index=False)
