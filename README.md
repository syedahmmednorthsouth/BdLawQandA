# BdLawQandA
Question Answering Bot:
Question Answering can be used in a variety of use cases. A very common one: Using it to navigate through complex knowledge bases or long documents ("search setting").

A "knowledge base" could for example be your website, an internal wiki or a collection of financial reports. 

## Project Proposal
The objective of this project is to make a model for a system where users can ask questions in legal domain(such as constitution related query and law related query) and the system try to answer   at a good accuray .  Specifically , this project focuses on the Constitution of Bangladesh , Penal Code Of Bangladesh  . The model uses NLP techniques to process the textual data from the documents and some pretrained model to understand the question and a searching query technique to find the meaningful answers

## Used Framework- Haystack:

Haystack is an open-source framework for building search systems that work intelligently over large document collections. Recent advances in NLP have enabled the application of question answering, retrieval and summarization to real world settings and Haystack is designed to be the bridge between research and industry


## Dataset Screenshot:
 
## Model Plan :
![Screenshot 2022-10-21 194214.jpg](https://github.com/syedahmmednorthsouth/BdLawQandA/blob/main/Screenshot%202022-10-21%20194214.jpg))


## Document Preprocessor :

```python
from haystack.nodes import PreProcessor

doc = converter.convert(file_path=file, meta=None)
processor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=200,
    split_respect_sentence_boundary=True,
    split_overlap=0
)
```
		







## Start Elastic Search 

   
```python
import os
from subprocess import Popen, PIPE, STDOUT

es_server = Popen(
    ["elasticsearch-7.9.2/bin/elasticsearch"], stdout=PIPE, stderr=STDOUT, preexec_fn=lambda: os.setuid(1)  # as daemon
)
# wait until ES has started
! sleep 30

 
 Connect To Elassictic Search(Document):
DocumentStores expect Documents in dictionary form, like that below. They are loaded using the DocumentStore.write_documents() method
from haystack.document_stores import ElasticsearchDocumentStore
Connect to Elasticsearch
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

```



 

## Retriever
The Retriever is a lightweight filter that can quickly go through the full document store and pass on a set of candidate documents that are relevant to the query. When used in combination with a Reader, it is a tool for sifting out irrelevant documents, saving the Reader from doing more work than it needs to and speeding up the querying process

```python
	
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    max_seq_len_query=64,
    max_seq_len_passage=256,
    batch_size=16,
    use_gpu=True,
    embed_title=True,
    use_fast_tokenizers=True,
)
```




## Reader 
	I actually use here some well known pretrained model from hugging face.
	

```python
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)


reader = TransformersReader(model_name_or_path="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)


reader = TransformersReader(model_name_or_path="deepset/xlm-roberta-large-squad2", tokenizer="deepset/xlm-roberta-large-squad2", use_gpu=-1)

```



## Final Pipeline

```python
from haystack.pipelines import ExtractiveQAPipeline

pipe = ExtractiveQAPipeline(reader, retriever)

 


q = "who has the right to transfer a judge from one high court to another high court?"

prediction = pipe.run(query=q, params={"Reader": {"top_k": 5}})
```
 

	
	
 
 
## Conclusion
The result is satisfactory in a sense that they actually give us most of the time correct result.Though we want to get the meaningful answer within a range of two - three line sentences , because it will then be relevant to whom we are actually using the model .   Training a model from a squad like question and answer dataset with large number of 
Question and answers will give us better sequential answers , but regarding Bangladesh Constitution and Penal code , there are few many QA I can find from the internet. If a professional in this domain helps us to make large amounts of QnA , then we can build that kind of model , that really helps users to know their query in legal domain
 
 
 
