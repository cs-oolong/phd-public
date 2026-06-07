# 1. Clonar
git clone https://github.com/NousResearch/neural-steering.git
cd neural-steering

# 2. Criar venv
python -m venv venv
source venv/bin/activate

# 3. Instalar dependências
pip install torch transformers accelerate bitsandbytes huggingface_hub sentencepiece protobuf

# 4. Instalar neural-steering
pip install -e .
