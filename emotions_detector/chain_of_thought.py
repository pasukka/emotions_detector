import pandas as pd
import traceback
from rich.progress import Progress
from huggingface_hub import InferenceClient

from emotions_detector.emotions_detector import Strategy


class ChainOfThoughts(Strategy):      

    def execute(self, sentences, emotions, emotion_list):
        self.result_path = "answers_cot.csv"
        self.sentences = sentences
        self.emotions = emotions
        self.emotion_list = emotion_list
        self.llm = InferenceClient(model=self.model,
                                   timeout=8,
                                   token=self.huggingface_hub_token)
        self.predict()

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

    def predict_emotion(self, sentence):
        explanation = self.explain(self.explain_prompt, sentence)
        prompt = self.template_prompt.replace(
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
        #         self.check_prompt, emotion, self.emotion_list)
        #     if self.log_llm_responses:
        #         print(f'[CHECKED EMOTION]: {emotion}\n')

        self.model_answers.append(emotion)

    def predict(self):
        i = 0
        total_sentences = len(self.sentences)
        try:
            explain_path = self.prompt_path + "/ChainOfThoughts/" + "explain_prompt.txt"
            with open(explain_path, 'r', encoding='utf-8') as file:
                self.explain_prompt = file.read()

            template_path = self.prompt_path + "ChainOfThoughts/" + "predict_prompt.txt"
            with open(template_path, 'r', encoding='utf-8') as file:
                self.template_prompt = file.read()

            check_path = self.prompt_path + "ChainOfThoughts/" + "check_prompt.txt"
            with open(check_path, 'r', encoding='utf-8') as file:
                self.check_prompt = file.read()

            with Progress() as progress:
                sentence_task = progress.add_task(
                    description=f'[green]Sentence {i}/{total_sentences}',
                    total=total_sentences)
                while i < total_sentences:
                    sentence = self.sentences.iloc[i]
                    if self.show_sentence:
                        print(f'[INPUT PROMPT]: {sentence}')
                    self.predict_emotion(sentence)
                    i += 1
                    progress.update(sentence_task,
                                    advance=1,
                                    description=f'[green]Sentence {i}/{total_sentences}')
            self.write_in_file(i)
            print("[METHOD]: Chain of Thoughts")
            self.check_accuracy(i)
        except Exception:
            print("Something went wrong")
            if i != 0:
                self.write_in_file(i)
                self.check_accuracy(i)
            traceback.print_exc()
