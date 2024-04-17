from emotions_detector.emotions_detector import EmotionsDetector
from emotions_detector.chain_of_thought import ChainOfThoughts
from emotions_detector.tree_of_thought import TreeOfThoughts


def main():
    file_name = "data/data.csv"
    ed = EmotionsDetector(file_name)
    # ed.setStrategy(ChainOfThoughts())
    ed.setStrategy(TreeOfThoughts())
    ed(number=5, shuffle=False)


if __name__ == '__main__':
    main()
