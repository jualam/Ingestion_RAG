import os
import json
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.data.path.append(r"C:\Users\juhai\AppData\Roaming\nltk_data")
from nltk.tokenize import sent_tokenize

load_dotenv()

class RAGEvaluator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        #self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
    
    # AAccuracy check
    def answer_accuracy(self, query: str, reference: str, generated: str) -> float:
        prompt = f"""
You are evaluating the correctness of an AI-generated answer.

Question: {query}

Reference Answer: {reference}

AI Answer: {generated}

Rate from 0 to 1:
- 1.0 if the AI answer matches the meaning of the reference or provides additional relevant details (even if wording differs).
- 0.90-1.0 if mostly correct but missing minor detail.
- 0.8-0.9 if partially correct but missing some info.
- 0.4-0.7 if somewhat correct but missing significant info.
- 0.0-0.4 if mostly wrong or hallucinated.

Return only the number.
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # cheaper
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            return float(response.choices[0].message.content.strip())
        except:
            return 0.5
    
    # Factuality /hallucination check
    def factuality_score(self, answer: str, context: str) -> float:
        import json

        prompt = f"""
Is this answer supported by the context? Rate factuality from 0 (completely false) to 1 (fully supported).
Return only JSON in this format: {{"factuality": <number>}}

Context:
{context}

Answer:
{answer}
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON safely
            data = json.loads(content)
            return float(data.get("factuality", 0.5))
        except:
            # fallback if GPT returns invalid JSON
            return 0.5
    
    # Semantic similarity
    # def semantic_similarity(self, text1: str, text2: str) -> float:
    #     sents1 = sent_tokenize(text1)
    #     sents2 = sent_tokenize(text2)
    #     sims = []
    #     for s1 in sents1:
    #         for s2 in sents2:
    #             emb = self.sentence_model.encode([s1, s2])
    #             sims.append(cosine_similarity([emb[0]], [emb[1]])[0][0])
    #     return float(np.mean(sims))
    def semantic_similarity(self, text1: str, text2: str) -> float:
        try:
            # Sentence-level sim (SBERT)
            emb = self.sentence_model.encode([text1, text2])
            st_score = float(cosine_similarity([emb[0]], [emb[1]])[0][0])

            # OpenAI embedding sim (lenient)
            emb_model = OpenAIEmbeddings(
                model="text-embedding-3-large", 
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            v1, v2 = emb_model.embed_query(text1), emb_model.embed_query(text2)
            oa_score = float(cosine_similarity([v1], [v2])[0][0])

            # Weighted blend (adjust weights for leniency)
            return 0.4 * st_score + 0.6 * oa_score
        except:
            return 0.5
    
    # retrieval metrics// No required for now,so commented out
    # def retrieval_metrics(self, retrieved_texts: list, reference: str) -> dict:
    #     hits = 0
    #     first_hit_rank = len(retrieved_texts) + 1
    #     threshold = 0.7
    #     ref_emb = self.sentence_model.encode([reference])[0]
        
    #     for i, doc in enumerate(retrieved_texts):
    #         doc_emb = self.sentence_model.encode([doc.page_content])[0]
    #         sim = cosine_similarity([ref_emb], [doc_emb])[0][0]
    #         if sim >= threshold:
    #             hits += 1
    #             if first_hit_rank == len(retrieved_texts) + 1:
    #                 first_hit_rank = i + 1
    #     k = len(retrieved_texts)
    #     precision = hits / k if k > 0 else 0
    #     recall = hits / 1  # assuming 1 reference
    #     mrr = 1 / first_hit_rank
    #     return {'precision@k': precision, 'recall@k': recall, 'mrr@k': mrr}

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_rag_evaluation():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    vectorstore = PineconeVectorStore(
        index_name=os.environ["PII_INDEX_NAME"],
        embedding=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    evaluator = RAGEvaluator()
    
    # load the test cases
    with open("test_cases.txt", "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    
    all_results = []
    category_results = defaultdict(list)
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case['query']
        reference = test_case['reference_answer']
        category = test_case.get('category', 'uncategorized')
        
        retrieved_docs = retriever.invoke(query)
        context = format_docs(retrieved_docs)
        
        # Generate answer
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"""You are a helpful assistant. Your job is answer questions about patients. If you don't know the answer just say "I don't know the answer".

Context:
{context}

Question:
{query}
"""
            }]
        )
        generated_answer = response.choices[0].message.content
        
        # Compute metrics
        acc = evaluator.answer_accuracy(query,reference, generated_answer)
        factuality = evaluator.factuality_score(generated_answer, context)
        similarity = evaluator.semantic_similarity(reference, generated_answer)
        # retrieval = evaluator.retrieval_metrics(retrieved_docs, reference)
        
        result = {
            'test_number': i,
            'category': category,
            'query': query,
            'reference_answer': reference,
            'generated_answer': generated_answer,
            'retrieved_context': context,
            'metrics': {
                'answer_accuracy': acc,
                'factuality': factuality,
                'semantic_similarity': similarity,
                # **retrieval
            }
        }
        all_results.append(result)
        category_results[category].append(result)
        
        print(f"Processed Test {i} ({category}): Accuracy={acc:.3f}, Factuality={factuality:.3f}, Similarity={similarity:.3f}")
    
    #category wise averages
    category_averages = {}
    # metric_keys = all_results[0]['metrics'].keys()
    metric_keys = [k for k in all_results[0]['metrics'].keys()]
    for cat, results in category_results.items():
        category_averages[cat] = {}
        for key in metric_keys:
            category_averages[cat][key] = np.mean([r['metrics'][key] for r in results])
    
    #overall averages
    overall = {}
    for key in metric_keys:
        overall[key] = np.mean([r['metrics'][key] for r in all_results])
    
    #results
    with open("rag_evaluation_results_v5.txt", "w", encoding="utf-8") as f:
        f.write("RAG EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Test Cases: {len(all_results)}\n\n")
        
        f.write("CATEGORY-WISE AVERAGES:\n")
        for cat, metrics in category_averages.items():
            f.write(f"{cat}:\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.3f}\n")
            f.write("\n")
        
        f.write("OVERALL AVERAGES (ALL 100 TEST CASES):\n")
        for k, v in overall.items():
            f.write(f"  {k}: {v:.3f}\n")
        f.write("\nDETAILED RESULTS:\n")
        f.write("=" * 60 + "\n\n")
        
        for result in all_results:
            f.write(f"Test Case {result['test_number']} - {result['category']}\n")
            f.write(f"Query: {result['query']}\n")
            f.write(f"Reference Answer: {result['reference_answer']}\n")
            f.write(f"Generated Answer: {result['generated_answer']}\n")
            f.write(f"Retrieved Context: {result['retrieved_context'][:200]}...\n")
            f.write(f"Scores:\n")
            for k, v in result['metrics'].items():
                f.write(f"  {k}: {v:.3f}\n")
            f.write("-" * 60 + "\n\n")

if __name__ == "__main__":
    run_rag_evaluation()
