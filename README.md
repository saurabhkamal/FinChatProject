# FinChatProject

# How to run?

### Steps:
```bash
git clone: https://github.com/saurabhkamal/FinChatProject.git
```

### Step 01: Create a conda environment after opening the repository
```bash
conda create -n finbot python==3.10 -y
```

```bash
conda activate finbot
```

### Step 02: install the requirements
```bash
pip install -r requirements.txt
```

### Create a .env in the root directory and add PineCone & OpenAI credentials as follows:
```bash
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

```bash
# run the following command to store the embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now, 
```
Open up localhost:
```

### Techstack Used:
- Python
- LangChain
- Flask
- GPT
- Pinecone
- VectorDB


## AWS-CICD-Deployment-with-Github-Actions


## 3. Create ECR repo to store/save docker image
- Save the URI: 566801649228.dkr.ecr.us-east-1.amazonaws.com/financialbot

https://stockanalysis.com/stocks/goog/financials/

