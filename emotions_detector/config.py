import yaml


class Config:
    llm: str
    huggingface_hub_token: str
    prompt_path: str
    log_prompts: bool
    log_llm_responses: bool
    show_sentence: bool

    def __init__(
            self,
            llm: str,
            token: str,
            prompt_path: str,
            log_prompts: bool,
            log_llm_responses: bool,
            show_sentence: bool
    ):
        self.llm = llm
        self.huggingface_hub_token = token
        self.prompt_path = prompt_path
        self.log_prompts = log_prompts
        self.log_llm_responses = log_llm_responses
        self.show_sentence = show_sentence


def load_config(file_path: str) -> Config:
    with open(file_path, 'r', encoding='utf-8') as stream:
        config_dict = yaml.safe_load(stream)['app-config']

    return Config(
        config_dict['llm'],
        config_dict['huggingface-hub-token'],
        config_dict['prompt_path'],
        config_dict['log_prompts'],
        config_dict['log_llm_responses'],
        config_dict['show_sentence']
    )
