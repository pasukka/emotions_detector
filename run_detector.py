from emotions_detector.emotions_detector import EmotionsDetector, ChainOfThoughts

def main():
    file_name = "data/data.csv"
    ed = EmotionsDetector(file_name)
    ed.setStrategy(ChainOfThoughts())
    ed(1)


if __name__ == '__main__':
    main()