{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from sentence_transformers import SentenceTransformer, util\

model = SentenceTransformer('all-MiniLM-L6-v2')\

corpus = [\
    "Python is a versatile programming language.",\
    "Machine learning improves predictions with data.",\
    "Web scraping extracts information from websites.",\
    "Cybersecurity protects systems from attacks."\
]\

corpus_embeddings = model.encode(corpus, convert_to_tensor=True)\

query = "How do I collect data from websites?"\
query_embedding = model.encode(query, convert_to_tensor=True)\

results = util.semantic_search(query_embedding, corpus_embeddings, top_k=2)\

print("Top matches:")\
for result in results[0]:\
    print(f"- \{corpus[result['corpus_id']]\} (score: \{result['score']:.4f\})")\
}
