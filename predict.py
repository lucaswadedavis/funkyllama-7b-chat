from cog import BasePredictor, Input, Path
import pprint as pp
from llama_cpp import LlamaGrammar, Llama


class Predictor(BasePredictor):
    def setup(self):
        """prepare the model"""
        self.grammar = LlamaGrammar.from_file("./grammar-json.gbnf")
        model_path = '/models/llama-2-7b-chat.Q5_K_S.gguf'
        self.llm = Llama(model_path,  n_ctx=2048)

    def predict(
            self,
            prompt: str = Input(description="Prompt"),
            json_schema: str = Input(description="Json Schema for the returned data")) -> str:
        """Run a single prediction on the model"""
        prompt_prefix = """
      [INST]
      <<SYS>>
      You are a bot that always responds with correct JSON.
      Your responses always begin with an opening curly brace {
      Your responses always end with a closing curly brace }"""
        # if json_schema is not None:
        if json_schema:
            prompt_prefix += """
      Your responses always match the following JSON schema:
      """
            prompt_prefix += json_schema
        prompt_prefix += """
      <</SYS>>
      """
        prompt_suffix = """\n[/INST]"""
        prompt = prompt_prefix + prompt + prompt_suffix
        output = self.llm(prompt, grammar=self.grammar, max_tokens=1000)
        return output["choices"][0]["text"]
