# triplets_llama_index.py

from typing import List, Tuple, Dict
import os
import nltk
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.schema import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor, SimpleLLMPathExtractor
from typing import Literal
import nest_asyncio
import json
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
import re
from collections import defaultdict
import datetime
from dotenv import load_dotenv
nest_asyncio.apply()
from llama_index.core.schema import TextNode
from graphrag.connectors.neo4j_connection import get_connection
from graphrag.utils.common import embed_text
from graphrag.utils.logger import logger

load_dotenv()

api_key=os.getenv("API_KEY")
api_version=os.getenv("API_VERSION")
azure_endpoint=os.getenv("AZURE_ENDPOINT")
engine=os.getenv("ENGINE")
model=os.getenv("MODEL")

llm = AzureOpenAI(
    model=model,
    api_key=api_key,
    api_base=azure_endpoint,
    api_version=api_version,
    engine=engine,
)

# Entities/Relations
entities = ["ENTREPRISE", "PERSONNE", "CONTRAT", "BENEFICIAIRE", "PROMETTANT", "DATE", "LOCATION"]
relations = ["WORKS_AT", "PART_OF", "ACCORDED", "IS_CALLED", "CREATED_ON", "SIGNED", "SIGNED_ON", "SIGNED_AT"]
validation_schema = [
    ["PERSONNE", "WORKS_AT", "ENTREPRISE"],
    ["PROMETTANT", "IS_CALLED", "ENTREPRISE"],
    ["BENEFICIAIRE", "IS_CALLED", "ENTREPRISE"],
    ["PERSONNE", "SIGNED", "CONTRAT"],
    ["CONTRAT", "SIGNED_ON", "DATE"],
    ["CONTRAT", "SIGNED_AT", "LOCATION"],
    ["ENTREPRISE", "CREATED_ON", "DATE"],
]

kg_extractor = SchemaLLMPathExtractor(
    llm=llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    strict=False,
)

class TripletExtractor:
    def __init__(self, neo4j_conn=None):
        """
        Initialize the triplet extractor.

        Args:
            neo4j_conn: A Neo4j connection instance.
            model_name: Hugging Face model name for triplet extraction.
        """

        self.neo4j = neo4j_conn or get_connection()

    def extract_triplets(self, chunk_text: str) -> List[Tuple[str, str, str]]:
        """
        Extract triplets from a sentence using the llama-index SchemaLLMPathExtractor model.

        Args:
            sentence: Input sentence.

        Returns:
            List of (subject, relation, object) triplets.
        """
        node = TextNode(text=chunk_text)
        res = kg_extractor([node])
        results = []

        for i in range(len(res)):
            triplets = res[i].metadata['relations']
            for triplet in triplets:
                results.append((triplet.source_id, triplet.label, triplet.target_id))
        return results

    def process_triplet(self, triplet: Tuple[str, str, str], chunk_id: str):
        """
        Process a single triplet: compute embeddings, search for similar nodes via vector queries,
        and merge the triplet into the Neo4j graph, linking subject/object to their source chunk.
        """
        subject, predicate, object_ = triplet
        logger.info(f"[Triplet] Reçu : {triplet}")
        relation_label = predicate.strip().upper().replace(" ", "_").replace("-", "_")

        subject_emb = embed_text(subject)
        predicate_emb = embed_text(predicate)
        object_emb = embed_text(object_)

        params = {
            "subject_emb": subject_emb.tolist(),
            "predicate_emb": predicate_emb.tolist(),
            "object_emb": object_emb.tolist(),
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            "chunk_id": chunk_id,
        }

        create_query = f"""
        MERGE (subjectNode:Entity {{name: toLower($subject)}})
        ON CREATE SET subjectNode.embeddings = $subject_emb, subjectNode.triplet_part = 'subject'
        ON MATCH SET subjectNode.triplet_part = 'subject'

        MERGE (objectNode:Entity {{name: toLower($object)}})
        ON CREATE SET objectNode.embeddings = $object_emb, objectNode.triplet_part = 'object'
        ON MATCH SET objectNode.triplet_part = 'object'

        MERGE (subjectNode)-[r:{relation_label}]->(objectNode)
        ON CREATE SET r.label = 'triplet', r.embeddings = $predicate_emb
        ON MATCH SET r.label = 'triplet'

        MERGE (chunk:Chunk {{id: $chunk_id}})
        MERGE (chunk)-[:MENTIONS]->(subjectNode)
        MERGE (chunk)-[:MENTIONS]->(objectNode)
        RETURN subjectNode.name AS subject, "{relation_label}" AS predicate, objectNode.name AS object
        """
        try:
            self.neo4j.run_query(create_query, params)
            logger.info(f"[Neo4j] MERGE effectué pour {triplet}")
        except Exception as e:
            logger.error(f"[Erreur] MERGE échoué pour {triplet} : {e}")

    def sanitize_relation(self, rel: str) -> str:
        """
        Sanitize the relation string to be a valid Neo4j relationship type.

        Args:
            rel: Relation string.

        Returns:
            Sanitized relation string.
        """
        rel_clean = re.sub(r"[^0-9a-zA-Z_ ]", "", rel)
        return rel_clean.replace(" ", "_").upper()
    
    def process_chunk(self, chunk_id: str, chunk_text: str) -> List[Tuple[str, str, str]]:
        logger.info(f"Processing chunk {chunk_id} for triplet extraction")
        triplets = self.extract_triplets(chunk_text)
        for triplet in triplets:
            self.process_triplet(triplet, chunk_id)
        return triplets

    def process_chunks(self, chunks: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Process a text chunk: split into sentences, extract triplets from each, and map each triplet into the graph.

        Args:
            chunk_id: Identifier for the text chunk.
            chunk_text: The text content of the chunk.

        Returns:
            List of extracted triplets.
        """
        results = {}
        for chunk_id, chunk_text in chunks:
            triplets = self.process_chunk(chunk_id, chunk_text)
            results[chunk_id] = triplets
        return results


# Convenience functions


def extract_triplets(sentence: str) -> List[Tuple[str, str, str]]:
    extractor = TripletExtractor()
    return extractor.extract_triplets(sentence)


def sanitize_relation(rel: str) -> str:
    extractor = TripletExtractor()
    return extractor.sanitize_relation(rel)


def process_chunk(chunk_id: str, chunk_text: str) -> List[Tuple[str, str, str]]:
    extractor = TripletExtractor()
    return extractor.process_chunk(chunk_id, chunk_text)


def process_chunks(
    chunks: List[Tuple[str, str]],
) -> Dict[str, List[Tuple[str, str, str]]]:
    extractor = TripletExtractor()
    return extractor.process_chunks(chunks)


if __name__ == "__main__":
    # Demo with an example sentence
    example_sentence = "Hugging Face, Inc. is an American company that develops tools for building applications using machine learning."
    print("Extracting triplets from example sentence...")
    triplets = extract_triplets(example_sentence)
    print(f"Extracted {len(triplets)} triplets:")
    for subj, rel, obj in triplets:
        print(f"  ({subj}, {rel}, {obj})")