import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

# 抓取网页内容
url = "https://github.com/LingSiewWin/TXSA-Lab-Exercise"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# 提取正文
texts = [p.get_text() for p in soup.find_all('p') if p.get_text()]

# 转换为 TF-IDF 向量
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)

# 输出前几个关键词
feature_names = vectorizer.get_feature_names_out()
top_words = [feature_names[i] for i in X.toarray().sum(axis=0).argsort()[-5:]]
print("Top keywords:", top_words)