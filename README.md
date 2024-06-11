# PeerMeta
We propose a financial statement fraud detection framework that makes improvements in all three components of the detection procedure: label measurement, feature set, and detection model.
Code explaination of "Unearthing Financial statement fraud: Insights from News Coverage Analysis"
Authors: Jianqing Fan, Qingfu Liu, Bo Wang, KaixinZheng
Version: 1.0.0
Date: 2024-05-21
--------------------------------------------
The implemention of our methods in the paper is mainly driven by four scripts.

The script news_collect.py is used to collect the fraud news for each Chinese listed firm from the website http://www.bjinfobank.com/DataList.do?method=DataList. Please note that the access to the news contents in the website is not free. You need to purchase the service with your account. The file stock_historyname.xlsx saved the history names of all stock symbols as firms may change their listed names during the sample period. This data is downloaded from the database of CSMAR.

The script FSFP_calc.py is capable of calculating the FSFP measures mentioned in the paper. The text file stopwords.txt includes the general stop words. The text file FinWords.txt includes the common financial words, and it is aimed to restrain Jieba not to split these words.

The script peer_identify.py can draw the business similarity among firms and identify peer companies. The excel file business_texts.xlsx documents firms' business description which are crawled from the financial statements.

The script detection_model.py provides the code of detetion model in the paper. The input variables are included in the csv file model_input.csv.
