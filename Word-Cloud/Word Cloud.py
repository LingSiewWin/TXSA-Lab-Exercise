# importing all necessary modules
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

# Read 'Youtube04-Eminem.csv' files
# using encoding = "latin-1" to get vertical words arrangement along with horizontal once
dataFrame = pd.read_csv("D:/APU/TXSA-CT107-3-3/LAB/LAB 3/Word-Cloud-master/Youtube04-Eminem.csv", encoding = "latin-1")
dataFrame.head()
dataFrame.shape

comment_words = ''
stopwords = set(STOPWORDS) 


# iterate through the csv file
# CONTENT is a coloumn name in my dataset
for val in dataFrame.CONTENT:
    tokens = val.split()
    
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    
    for words in tokens:
        comment_words = comment_words + words + ''
        
        
wordcloud = WordCloud(width = 800, height = 800,
                     background_color ='white',
                     stopwords = stopwords, min_font_size = 10).generate(comment_words)

#plot the WordCloud image
plt.figure(figsize = (15, 8), facecolor = None) 
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()