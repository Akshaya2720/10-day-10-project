
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))
nltk.download('stopwords')
data=pd.read_csv('IMDB Dataset.csv')
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
def clean_text(text):
    text=re.sub(r'^a-zA-Z0-9\s',' ',text)
    text=text.lower()
    text=text.split()
    text=[word for word in text if not word in stopwords]
    text=' '.join(text)
    return text
data['review']=data['review'].apply(clean_text)
data.to_csv('cleaned_IMDB_Dataset.csv',index=False)
print(data.head())
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
X=data['review']
y=data['sentiment']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
vectorizer=TfidfVectorizer(max_features=5000)
X_train_vec=vectorizer.fit_transform(X_train).toarray()
X_test_vec=vectorizer.transform(X_test).toarray()
print(X_train_vec.shape)
print(X_test_vec.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
model=LogisticRegression()
model.fit(X_train_vec,y_train)
y_pred=model.predict(X_test_vec)
accuracy=accuracy_score(y_test,y_pred)
print('Accuracy:',accuracy)
print('Classification Report:\n',classification_report(y_test,y_pred))
print('Confusion Matrix:\n',confusion_matrix(y_test,y_pred))
while True:
    user_review = input("Enter a movie review (or type 'exit' to stop): ")
    if user_review.lower() == 'exit':
        print(" Exiting the program. Goodbye!")
        break
    cleaned_review = clean_text(user_review)
    review_tfidf = vectorizer.transform([cleaned_review])
    prediction = model.predict(review_tfidf)
    if prediction == 1:
        print(" Sentiment: Positive ")
    else:
        print(" Sentiment: Negative ")
