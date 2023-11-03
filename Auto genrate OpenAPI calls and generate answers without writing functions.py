from langchain.chains import LLMBashChain
from langchain.llms import OpenAI
import os
from getpass import getpass
from langchain.chat_models import ChatOpenAI
from langchain.chains.api.prompt import API_RESPONSE_PROMPT
from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate

from langchain.chains.api import open_meteo_docs
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.tools.json.tool import JsonSpec
from langchain.chains import RetrievalQA
from langchain.document_loaders import JSONLoader
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="/home/duleesha/local-llm/Trelis-Llama-2-7b-chat-hf-function-calling-v2.Q4_K_S.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)

import os
os.environ['ApiKeyAuth'] = "Put API key"
headers = {"Authorization": f"Bearer {os.environ['ApiKeyAuth']}"}

chain_new = APIChain.from_llm_and_api_docs(llm, open_meteo_docs.OPEN_METEO_DOCS,headers=headers, verbose=True)
chain_new.run('What is the  weather for next 2 days in New York??')
#chain_new.run('What is the tommorrow weather in New York?')