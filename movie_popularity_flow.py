from metaflow import FlowSpec, step, card
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from metaflow.cards import Markdown, Image
import os


class MoviePopularityFlow(FlowSpec):

    @step
    def start(self):
        """
        Load movie data from local CSV (already downloaded into ./data/movies.csv).
        """
        print("📥 Loading local MovieLens movie data...")

        # Load the file directly from the local path
        self.data = pd.read_csv("data/movies.csv")

        print(f"✅ Loaded {len(self.data)} movies.")
        self.next(self.clean_data)

    @step
    def clean_data(self):
        """
        Preprocess and engineer a synthetic popularity label based on release year and title length.
        """
        df = self.data.dropna()

        # Extract year from title
        def extract_year(title):
            try:
                year = title.strip()[-5:-1]
                return int(year)
            except:
                return np.nan

        df['year'] = df['title'].apply(extract_year)
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)

        # Create synthetic label: "popular" if after 2010 and long title
        df['popularity'] = ((df['year'] > 2010) & (df['title'].str.len() > 20)).astype(int)

        self.df = df[['title', 'year', 'popularity']]
        print(f"🧹 Cleaned data shape: {self.df.shape}")
        self.next(self.vectorize_data)

    @step
    def vectorize_data(self):
        """
        Convert movie titles into numeric features using Bag-of-Words (CountVectorizer).
        """
        print("🔤 Vectorizing titles...")
        vectorizer = CountVectorizer(stop_words='english', max_features=1000)
        self.X = vectorizer.fit_transform(self.df['title']).toarray()
        self.y = self.df['popularity'].values
        self.vectorizer = vectorizer
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Train a logistic regression classifier on the movie title features.
        """
        print("🧠 Training logistic regression model...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.report = classification_report(y_test, y_pred, output_dict=True)
        self.y_test = y_test
        self.y_pred = y_pred

        print(f"✅ Accuracy: {self.accuracy:.2f}")
        self.next(self.report_results)

    @card
    @step
    def report_results(self):
        """
        Generate visual report using Metaflow cards.
        """
        print("📊 Generating evaluation report...")

        # Confusion matrix plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ConfusionMatrixDisplay.from_predictions(self.y_test, self.y_pred, ax=ax)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")

        # Save full classification report
        pd.DataFrame(self.report).T.to_csv("classification_report.csv")

        accuracy_percent = self.accuracy * 100

        self.card = [
            Markdown("# 🎬 Movie Popularity Classification Results"),
            Markdown(f"**Accuracy:** {accuracy_percent:.2f}%"),
            Image("confusion_matrix.png"),
            Markdown("Full classification report saved as `classification_report.csv`.")
        ]

        self.next(self.end)

    @step
    def end(self):
        print("🏁 Flow execution complete.")


if __name__ == "__main__":
    MoviePopularityFlow()