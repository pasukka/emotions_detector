from abc import ABC, abstractmethod
import pandas as pd
from emotions_detector.config import Config, load_config
from rich.progress import Progress


class Strategy(ABC):
    model: str
    huggingface_hub_token: str
    log_prompts: str
    log_llm_responses: str
    show_sentence: bool
    prompt_path: str
    model_answers: list

    @abstractmethod
    def execute(self):
        pass

    def __init__(self):
        config = load_config('config.yml')
        self.model = config.llm
        self.huggingface_hub_token = config.huggingface_hub_token
        self.log_prompts = config.log_prompts
        self.log_llm_responses = config.log_llm_responses
        self.show_sentence = config.show_sentence
        self.prompt_path = config.prompt_path
        self.model_answers = []

    def write_in_file(self, number):
        df = pd.DataFrame({'sentence': self.sentences[:number],
                           'real emotion': self.emotions[:number],
                           'predicted emotion': self.model_answers})
        df.to_csv(self.result_path)

    def check_accuracy(self, all_sent):
        df = pd.DataFrame()
        df['real answer'] = self.emotions
        df.loc[:, 'model result'] = self.model_answers
        with Progress() as progress:
            accuracy_task = progress.add_task(description=f'[blue]Accuracy',
                                              total=all_sent)
            print(f'\n[ANALYSIS OF MODEL RESPONSES]:')
            res = df['model result'].compare(df['real answer'])
            progress.update(accuracy_task,
                            advance=len(self.emotions)-len(res),
                            description=f'[blue]Accuracy {len(self.emotions)-len(res)}/{all_sent}')
            res.columns = ['wrong result', 'real answer']
            print(f'\n{res}')
