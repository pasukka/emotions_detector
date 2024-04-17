import pandas as pd
from huggingface_hub import InferenceClient

from emotions_detector.emotions_detector import Strategy
from emotions_detector.config import Config, load_config

from rich.progress import Progress

class ChainOfThoughts(Strategy):
    model: str
    huggingface_hub_token: str
    log_prompts: str
    log_llm_responses: str
    show_sentence: bool
    prompt_path: str
    model_answers: list
    rounds: int

    def __init__(self):
        config = load_config('config.yml')
        self.model = config.llm
        self.huggingface_hub_token = config.huggingface_hub_token
        self.log_prompts = config.log_prompts
        self.log_llm_responses = config.log_llm_responses
        self.show_sentence = config.show_sentence
        self.prompt_path = config.prompt_path
        self.model_answers = []
        self.result_path = "answers_cot.csv"

    def execute(self, sentences, emotions, emotion_list):
        self.sentences = sentences
        self.emotions = emotions
        self.emotion_list = emotion_list
        self.llm = InferenceClient(model=self.model,
                                   timeout=8,
                                   token=self.huggingface_hub_token)
        self.rounds = 5

        self.predict()

    def write_in_file(self, number):
        df = pd.DataFrame({'sentence': self.sentences[:number],
                           'real emotion': self.emotions[:number],
                           'predicted emotion': self.model_answers})
        df.to_csv(self.result_path)

    def check_accuracy(self, number):
        true = 0
        all_sent = number
        for i in range(all_sent):
            if self.emotions.iloc[i] == self.model_answers[i]:
                true += 1

        accuracy = true/all_sent
        print(f'Accuracy: {accuracy}')

    def explain(self, explain_prompt, sentence):
        prompt = explain_prompt.replace('{text}', sentence)

        response = self.llm.text_generation(
            prompt, do_sample=False, max_new_tokens=40, stop_sequences=['.']).strip()
        response = response.replace("'", "")

        # print(f'[EXPLANATION]: {response}\n')
        return response

    def check(self, check_prompt, emotion, emotion_list):
        prompt = check_prompt.replace('{emotion}', emotion)
        prompt = prompt.replace('{emotions}', emotion_list)

        real_emotion = self.llm.text_generation(
            prompt, do_sample=False, max_new_tokens=10, stop_sequences=['.']).strip()

        real_emotion = real_emotion.replace(".", "")

        if not real_emotion:
            real_emotion = emotion
        return real_emotion

    def predict(self):
        i = 0
        total_sentences = len(self.sentences)

        try:
            explain_path = self.prompt_path + "/ChainOfThoughts/" + "explain_prompt.txt"
            with open(explain_path, 'r', encoding='utf-8') as file:
                explain_prompt = file.read()

            template_path = self.prompt_path + "/ChainOfThoughts/" + "predict_prompt.txt"
            with open(template_path, 'r', encoding='utf-8') as file:
                template_prompt = file.read()

            check_path = self.prompt_path + "/ChainOfThoughts/" + "check_prompt.txt"
            with open(check_path, 'r', encoding='utf-8') as file:
                check_prompt = file.read()

            with Progress() as progress:
                sentence_task = progress.add_task(
                    description=f'[green]Sentence {i}/{total_sentences}',
                    total=total_sentences
                )

                while i < total_sentences:
                    sentence = self.sentences.iloc[i]
                    
                    if self.show_sentence:
                         
                        print(f'[INPUT PROMPT]: {sentence}')

                    explanation = self.explain(explain_prompt, sentence)

                    prompt = template_prompt.replace(
                        '{emotions}', self.emotion_list)
                    prompt = prompt.replace('{text}', sentence)
                    prompt = prompt.replace('{explanation}', explanation)

                    emotion = self.llm.text_generation(prompt,
                                                        do_sample=False,
                                                        max_new_tokens=50).strip()
                    emotion = emotion.replace("'", "").replace(
                        ".", "").replace("Answer: ", "")

                    if self.log_prompts:
                        print(f'[INPUT PROMPT]: {prompt}\n')
                    
                    if self.log_llm_responses:
                        print(f'[RESPONSE]: {emotion}\n')
                        print(f'[REAL ANSWER]: {self.emotions.iloc[i]}\n')

                    # if emotion not in self.emotion_list:
                    #     emotion = self.check(
                    #         check_prompt, emotion, self.emotion_list)
                    #     if self.log_llm_responses:
                    #         print(f'[CHECKED EMOTION]: {emotion}\n')

                    self.model_answers.append(emotion)

                    i += 1
                    progress.update(sentence_task,
                                        advance=1,
                                        description=f'[green]Sentence {i}/{total_sentences}')

            self.write_in_file(i)
            self.check_accuracy(i)

        except Exception:
            traceback.print_exc()
            
            if i != 0:
                self.write_in_file(i)
                self.check_accuracy(i)
