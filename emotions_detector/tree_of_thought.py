import pandas as pd
import traceback
from rich.progress import Progress
from huggingface_hub import InferenceClient

from emotions_detector.emotions_detector import Strategy


class TreeOfThoughts(Strategy):
    rounds: int

    def execute(self, sentences, emotions, emotion_list):
        self.result_path = "answers_tot.csv"
        self.sentences = sentences
        self.emotions = emotions
        self.emotion_list = emotion_list
        self.llm = InferenceClient(model=self.model,
                                   timeout=8,
                                   token=self.huggingface_hub_token)
        self.rounds = 5

        self.predict()

    def get_emotion(self, reply, get_emotions_template):
        prompt = get_emotions_template.replace('{reply}', reply)
        prompt = prompt.replace('{emotions}', self.emotion_list)
        # print(f'[DISCUSSION PROMPT]: {prompt}\n')

        real_emotion = self.llm.text_generation(
            prompt, do_sample=False, max_new_tokens=10, stop_sequences=['.']).strip()
        real_emotion = real_emotion.replace(".", "").replace("'", "")
        return real_emotion, prompt

    def predict_emotion(self, sentence):
        prompt = self.template_prompt.replace('{emotions}',
                                              self.emotion_list)
        prompt = prompt.replace('{text}', sentence)
        for r in range(self.rounds):
            if self.log_prompts:
                print(f'[INPUT PROMPT]: {prompt}\n')
            reply = self.llm.text_generation(prompt,
                                             do_sample=False,
                                             max_new_tokens=50).strip()
            # print(f'[REPLY]: {reply}\n')
            prompt += f"\n{reply}"
        emotion, discussion_prompt = self.get_emotion(prompt,
                                                      self.get_emotions_prompt)
        if self.log_llm_responses:
            print(f'[RESPONSE]: {emotion}\n')
            print(f'[REAL ANSWER]: {self.emotions.iloc[i]}\n')

        # if emotion != self.emotions.iloc[i] and self.log_llm_responses:
        #     print(f'[DISCUSSION PROMPT]:\n {discussion_prompt}\n')

        self.model_answers.append(emotion)

    def predict(self):
        i = 0
        total_sentences = len(self.sentences)
        try:
            template_path = self.prompt_path + "TreeOfThoughts/" + "predict_prompt.txt"
            with open(template_path, 'r', encoding='utf-8') as file:
                self.template_prompt = file.read()

            get_emotions_path = self.prompt_path + \
                "TreeOfThoughts/" + "get_emotions_prompt.txt"
            with open(get_emotions_path, 'r', encoding='utf-8') as file:
                self.get_emotions_prompt = file.read()

            with Progress() as progress:
                self.sentence_task = progress.add_task(
                    description=f'[green]Sentence {i}/{total_sentences}',
                    total=total_sentences
                )
                while i < total_sentences:
                    sentence = self.sentences.iloc[i]
                    if self.show_sentence:
                        print(f'[INPUT PROMPT]: {sentence}')

                    self.predict_emotion(sentence)
                    i += 1
                    progress.update(self.sentence_task,
                                    advance=1,
                                    description=f'[green]Sentence {i}/{total_sentences}')
            self.write_in_file(i)
            print("[METHOD]: Tree of Thoughts")
            self.check_accuracy(i)
        except Exception:
            print("Something went wrong")
            if i != 0:
                self.write_in_file(i)
                self.check_accuracy(i)
            traceback.print_exc()
