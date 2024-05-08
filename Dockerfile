FROM mosaicml/llm-foundry:2.2.1_cu121_flash2-f044d6c

RUN mkdir /workspace
WORKDIR /workspace
RUN git clone -b feat-docker https://github.com/aisingapore/llm-foundry.git llm_foundry
WORKDIR /workspace/llm_foundry
RUN pip install -e '.[gpu]'

ENTRYPOINT ["/bin/bash", "/workspace/llm_foundry/entrypoint.sh"]
