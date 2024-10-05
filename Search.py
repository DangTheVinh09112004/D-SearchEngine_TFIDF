import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def process_query(query: str):
    return query.strip().lower()
class KeywordSearchEngine:
    def __init__(self, data):
        self.data = data
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['Nội dung phim'])
    def search(self, query, top_n=5):
        query = process_query(query)
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-top_n - 1:-1]
        results = self.data.iloc[related_docs_indices].drop_duplicates(subset=['Nội dung phim'])
        return results.head(top_n)
def main():
    df = pd.read_csv("data.csv")
    st.title("🎬 Movie Search Engine")
    query = st.text_input("Nhập từ khóa tìm kiếm", "")
    if query:
        keyword_search = KeywordSearchEngine(df)
        results = keyword_search.search(query, top_n=4)
        st.write("### 🎥 Kết quả tìm kiếm:")
        for index, row in results.iterrows():
            # Hiển thị tên phim
            st.subheader(row['Tên phim'])
            # Hiển thị hình ảnh nếu có link hình ảnh
            if pd.notnull(row['Hình ảnh']):
                st.image(row['Hình ảnh'], width=300)  # Hiển thị hình ảnh với chiều rộng 300px
            # Hiển thị nội dung phim
            st.write(row['Nội dung phim'])
if __name__ == '__main__':
    main()

