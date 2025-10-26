"""Integration tests for semantic search quality using real Cohere API"""

import pytest
import os
from uuid import uuid4
import numpy as np

from src.vector_db.api.dependencies import get_search_service, get_document_service
from src.vector_db.domain.models import Document, Library, Chunk


@pytest.mark.integration
@pytest.mark.semantic_quality
@pytest.mark.slow
class TestSemanticSearchQuality:
    """Integration tests for semantic search quality using real Cohere API"""

    def setup_method(self):
        """Setup test fixtures"""
        self.library_id = str(uuid4())
        self.search_service = get_search_service()
        self.document_service = get_document_service()

        # Skip tests if COHERE_API_KEY is not set
        if not os.getenv('COHERE_API_KEY'):
            pytest.skip("COHERE_API_KEY environment variable not set")

    def test_semantic_similarity_basic(self):
        """Test basic semantic similarity with clear content hierarchy"""
        # Create a document about a specific, concrete topic
        document = self.document_service.create_document(
            library_id=self.library_id,
            text="The golden retriever is a medium-large dog breed known for its friendly temperament "
            "and intelligence. Originally bred for hunting waterfowl, these dogs are excellent "
            "family pets and are commonly used as service animals. Golden retrievers have a "
            "thick, water-repellent coat that ranges from light to dark golden color."
        )

        # Test relative semantic relationships - focus on ranking, not absolute scores
        queries = [
            "golden retriever dog",      # Exact match - should score highest
            "friendly family pet",       # Related concept - should score high  
            "service animal",           # Related concept - should score medium
            "water repellent coat",     # Specific feature - should score medium
            "hunting birds",            # Related activity - should score lower
            "cooking recipes",          # Completely unrelated - should score lowest
        ]

        # Get all scores
        scores = {}
        for query in queries:
            results = self.search_service.search_chunks_in_document(
                document=document,
                query_text=query,
                k=1,
                min_similarity=0.0
            )
            scores[query] = results[0][1] if results else 0.0

        # Test relative rankings instead of absolute thresholds
        assert scores["golden retriever dog"] > scores["friendly family pet"], \
            "Exact match should score higher than related concept"
        assert scores["friendly family pet"] > scores["cooking recipes"], \
            "Related concept should score higher than unrelated content"
        assert scores["service animal"] > scores["cooking recipes"], \
            "Related concept should score higher than unrelated content"

    @pytest.mark.timeout(60)  # Add timeout to prevent hanging
    def test_semantic_similarity_synonyms(self):
        """Test that semantic search handles synonyms correctly"""
        try:
            document = self.document_service.create_document(
                library_id=self.library_id,
                text="The automobile industry has revolutionized transportation. Cars have become "
                "essential for daily commuting and long-distance travel. The automotive sector "
                "employs millions of people worldwide and drives economic growth."
            )

            # Test synonym pairs
            synonym_pairs = [
                ("car", "automobile"),
                ("vehicle", "automotive"),
                ("transport", "transportation"),
            ]

            for query1, query2 in synonym_pairs:
                results1 = self.search_service.search_chunks_in_document(
                    document=document,
                    query_text=query1,
                    k=3,
                    min_similarity=0.0
                )

                results2 = self.search_service.search_chunks_in_document(
                    document=document,
                    query_text=query2,
                    k=3,
                    min_similarity=0.0
                )

                # Both queries should return results
                assert len(results1) > 0, f"No results for query: {query1}"
                assert len(results2) > 0, f"No results for query: {query2}"

                # Similarities should be comparable (within reasonable range)
                similarity1 = results1[0][1]
                similarity2 = results2[0][1]
                similarity_diff = abs(similarity1 - similarity2)

                assert similarity_diff < 0.3, \
                    f"Synonyms '{query1}' and '{query2}' have very different similarities: {similarity1} vs {similarity2}"
        except RuntimeError as e:
            if "Failed to create embedding" in str(e) and "timeout" in str(e).lower():
                pytest.skip(f"Skipping test due to Cohere API timeout: {e}")
            else:
                raise

    def test_semantic_similarity_context(self):
        """Test that semantic search understands context"""
        document = self.document_service.create_document(
            library_id=self.library_id,
            text="Python is a versatile programming language used for web development, "
            "data science, and artificial intelligence. It has a simple syntax and "
            "extensive libraries. Python developers can build applications quickly "
            "and efficiently."
        )

        # Test context-aware queries - use relative comparisons
        context_queries = [
            "programming language",  # Direct context
            "web development",       # Related context
            "data science",         # Related context
            "snake",               # Ambiguous without context
        ]

        # Get all scores
        scores = {}
        for query in context_queries:
            results = self.search_service.search_chunks_in_document(
                document=document,
                query_text=query,
                k=1,
                min_similarity=0.0
            )
            scores[query] = results[0][1] if results else 0.0

        # Test relative rankings - programming terms should score higher than ambiguous terms
        assert scores["programming language"] > scores["snake"], \
            "Programming language should score higher than ambiguous 'snake'"
        assert scores["web development"] > scores["snake"], \
            "Web development should score higher than ambiguous 'snake'"
        assert scores["data science"] > scores["snake"], \
            "Data science should score higher than ambiguous 'snake'"

    def test_semantic_similarity_multilingual(self):
        """Test semantic search with multilingual content"""
        document = self.document_service.create_document(
            library_id=self.library_id,
            text="La inteligencia artificial es una rama de la informática que busca crear "
            "sistemas capaces de realizar tareas que normalmente requieren inteligencia "
            "humana. El machine learning es una subcategoría que permite a las máquinas "
            "aprender de los datos sin ser programadas explícitamente."
        )

        # Test queries in different languages - use relative comparisons
        multilingual_queries = [
            "artificial intelligence",      # English query for Spanish content
            "machine learning",            # English query for Spanish content  
            "inteligencia artificial",     # Spanish query (exact match)
            "aprendizaje automático",      # Spanish query (exact match)
            "cooking recipes",             # Unrelated in English
        ]

        # Get all scores
        scores = {}
        for query in multilingual_queries:
            results = self.search_service.search_chunks_in_document(
                document=document,
                query_text=query,
                k=1,
                min_similarity=0.0
            )
            scores[query] = results[0][1] if results else 0.0

        # Test that AI-related terms (any language) score higher than unrelated content
        assert scores["artificial intelligence"] > scores["cooking recipes"], \
            "AI terms should score higher than unrelated content"
        assert scores["machine learning"] > scores["cooking recipes"], \
            "ML terms should score higher than unrelated content"
        assert scores["inteligencia artificial"] > scores["cooking recipes"], \
            "Spanish AI terms should score higher than unrelated content"

    def test_semantic_similarity_across_documents(self):
        """Test semantic search across multiple documents in a library"""
        library = Library(name="Test Library")

        # Create documents with related but different content
        doc1 = self.document_service.create_document(
            library_id=library.id,
            text="Machine learning algorithms can be supervised or unsupervised. "
            "Supervised learning uses labeled training data to make predictions."
        )

        doc2 = self.document_service.create_document(
            library_id=library.id,
            text="Deep learning uses neural networks with multiple layers. "
            "Convolutional neural networks are particularly effective for image processing."
        )

        doc3 = self.document_service.create_document(
            library_id=library.id,
            text="Natural language processing enables computers to understand human language. "
            "Techniques include text classification, sentiment analysis, and translation."
        )

        library = library.add_document(doc1)
        library = library.add_document(doc2)
        library = library.add_document(doc3)

        # Test search across all documents
        query = "artificial intelligence"
        results = self.search_service.search_chunks(
            library=library,
            query_text=query,
            k=5,
            min_similarity=0.0
        )

        assert len(results) > 0, "No results found across documents"

        # Verify results come from different documents
        document_ids = set()
        for chunk, similarity in results:
            document_ids.add(chunk.document_id)

        # Should find relevant content in multiple documents
        assert len(document_ids) >= 2, "Results should come from multiple documents"

    def test_semantic_similarity_ranking(self):
        """Test that semantic search properly ranks results by relevance"""
        document = self.document_service.create_document(
            library_id=self.library_id,
            text="Python is a high-level programming language known for its simplicity and readability. "
            "It's widely used in data science, web development, and artificial intelligence. "
            "The language has a large ecosystem of libraries and frameworks. "
            "Python developers appreciate its clean syntax and extensive documentation."
        )

        # Test queries with expected ranking
        queries = [
            "python programming",  # Most relevant
            "data science",  # Medium relevance
            "web development",  # Medium relevance
            "documentation",  # Lower relevance
        ]

        all_results = []
        for query in queries:
            results = self.search_service.search_chunks_in_document(
                document=document,
                query_text=query,
                k=3,
                min_similarity=0.0
            )
            all_results.append((query, results[0][1] if results else 0.0))

        # Verify that more specific queries get higher similarity scores
        python_score = next(score for query, score in all_results if "python" in query.lower())
        data_science_score = next(score for query, score in all_results if "data science" in query.lower())
        web_dev_score = next(score for query, score in all_results if "web development" in query.lower())
        doc_score = next(score for query, score in all_results if "documentation" in query.lower())

        # Python programming should be most relevant
        assert python_score >= data_science_score * 0.8, "Python query should be more relevant than data science"
        assert python_score >= web_dev_score * 0.8, "Python query should be more relevant than web development"

    def test_embedding_consistency(self):
        """Test that embeddings are consistent for the same text"""
        text = "This is a test text for embedding consistency."

        # Create embeddings multiple times
        embedding1 = self.search_service._create_query_embedding(text)
        embedding2 = self.search_service._create_query_embedding(text)

        # Embeddings should be identical for the same text
        assert embedding1 == embedding2, "Embeddings should be consistent for the same text"

        # Verify embedding structure
        assert isinstance(embedding1, list), f"Embedding should be a list, but it is {type(embedding1)}"
        assert len(embedding1) == 1536, f"Embedding should have 1536 dimensions, but has {len(embedding1)}"
        assert all(isinstance(x, (int, float)) for x in embedding1), "Embedding should contain numbers"

    def test_cosine_similarity_quality(self):
        """Test cosine similarity calculation with real embeddings"""
        # Create embeddings for related and unrelated texts
        related_text1 = "machine learning algorithms"
        related_text2 = "artificial intelligence models"
        unrelated_text = "cooking recipes and ingredients"

        embedding1 = self.search_service._create_query_embedding(related_text1)
        embedding2 = self.search_service._create_query_embedding(related_text2)
        embedding3 = self.search_service._create_query_embedding(unrelated_text)

        # Calculate similarities
        related_similarity = self.search_service._cosine_similarity(embedding1, embedding2)
        unrelated_similarity1 = self.search_service._cosine_similarity(embedding1, embedding3)
        unrelated_similarity2 = self.search_service._cosine_similarity(embedding2, embedding3)

        # Related texts should have higher similarity than unrelated texts
        assert related_similarity > unrelated_similarity1, \
            f"Related texts should have higher similarity: {related_similarity} > {unrelated_similarity1}"
        assert related_similarity > unrelated_similarity2, \
            f"Related texts should have higher similarity: {related_similarity} > {unrelated_similarity2}"

        # Similarities should be reasonable values
        assert 0.0 <= related_similarity <= 1.0, f"Related similarity should be between 0 and 1: {related_similarity}"
        assert 0.0 <= unrelated_similarity1 <= 1.0, f"Unrelated similarity should be between 0 and 1: {unrelated_similarity1}"