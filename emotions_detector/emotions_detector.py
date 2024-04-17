from emotions_detector.strategy import Strategy
from emotions_detector.chain_of_thought import ChainOfThoughts
import pandas as pd


class EmotionsDetector:
    strategy: Strategy
    filepath: str

    def __init__(self, filepath: str):
        self.filepath = filepath

    def __call__(self, number, shuffle=True):
        df = pd.read_csv(self.filepath)

        if shuffle:
            df = df.sample(frac=1)

        emotions = df['label']
        emotion_list = str(set(emotions))

        sentences = df.iloc[:number]['text']
        emotions = df.iloc[:number]['label']

        self.strategy.execute(sentences, emotions, emotion_list)

    def setStrategy(self, strategy: Strategy = None) -> None:
        if strategy is not None:
            self.strategy = strategy
        else:
            self.strategy = ChainOfThoughts()