from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

class KnowledgeGraph:
    def __init__(self):
        # Initialize embedding model inside the class
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2") 
        try:
            self.driver = GraphDatabase.driver(
                "neo4j://127.0.0.1:7687",
                auth=("neo4j", os.getenv("NEO4J_PASSWORD", "password123"))
            )
            
            # Initialize Vector Index on startup
            self._create_vector_index()
            
            print("✅ Neo4j connection established & Vector Index checked")
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
            self.driver = None

    def _create_vector_index(self):
        """Create a vector index on Topic nodes for semantic search"""
        if not self.driver: return
        with self.driver.session() as session:
            # Check if index exists (requires Neo4j 5.x+)
            session.run("""
                CREATE VECTOR INDEX topic_embeddings IF NOT EXISTS
                FOR (t:Topic)
                ON (t.embedding)
                OPTIONS {indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
                }}
            """)

    def close(self):
        if self.driver:
            self.driver.close()

    def create_concept_graph(self, structured_data):
        """Create concepts and relationships from extracted knowledge"""
        if not self.driver:
            print("❌ No Neo4j driver available")
            return ""

        print(f"🔗 Creating knowledge graph for: {structured_data}")
        
        with self.driver.session() as session:
            try:
                # Create main topic WITH EMBEDDING
                topic_name = structured_data.get("Topic", "Unknown")
                if topic_name == "Unknown":
                    print("⚠️  No topic name found in structured data")
                    return ""
                    
                print(f"📝 Creating topic: {topic_name}")
                embedding = self.embedding_model.encode(topic_name).tolist()

                session.run("""
                    MERGE (t:Topic {name: $topic})
                    ON CREATE SET t.embedding = $embedding
                    ON MATCH SET t.embedding = $embedding
                """, topic=topic_name, embedding=embedding)
                
                print(f"✅ Created topic node: {topic_name}")
                
                # Create subtopics and link to main topic
                subtopics = structured_data.get("Subtopics", [])
                print(f"📋 Creating {len(subtopics)} subtopics: {subtopics}")
                
                for subtopic in subtopics:
                    session.run("""
                        MATCH (t:Topic {name: $topic})
                        MERGE (s:Subtopic {name: $subtopic})
                        MERGE (t)-[:HAS_SUBTOPIC]->(s)
                    """, topic=topic_name, subtopic=subtopic)
                
                # Create definitions as concepts
                definitions = structured_data.get("Definitions", [])
                print(f"📖 Creating {len(definitions)} definitions: {definitions}")
                
                for definition in definitions:
                    if ":" in definition:
                        term, desc = definition.split(":", 1)
                        session.run("""
                            MATCH (t:Topic {name: $topic})
                            MERGE (d:Definition {term: $term, description: $desc})
                            MERGE (t)-[:DEFINES]->(d)
                        """, topic=topic_name, term=term.strip(), desc=desc.strip())
                
                # Create formulas
                formulas = structured_data.get("Formulas", [])
                print(f"🧮 Creating {len(formulas)} formulas: {formulas}")
                
                for formula in formulas:
                    session.run("""
                        MATCH (t:Topic {name: $topic})
                        MERGE (f:Formula {expression: $formula})
                        MERGE (t)-[:HAS_FORMULA]->(f)
                    """, topic=topic_name, formula=formula)
                
                # Create keywords and auto-detect relationships
                keywords = structured_data.get("Terminology / Keywords", [])
                print(f"🔑 Creating {len(keywords)} keywords: {keywords}")
                
                for keyword in keywords:
                    session.run("""
                        MATCH (t:Topic {name: $topic})
                        MERGE (k:Keyword {name: $keyword})
                        MERGE (t)-[:RELATES_TO]->(k)
                    """, topic=topic_name, keyword=keyword)
                
                # Create prerequisites relationships
                prereqs = structured_data.get("Prerequisites", [])
                print(f"📚 Creating {len(prereqs)} prerequisites: {prereqs}")
                
                for prereq in prereqs:
                    session.run("""
                        MATCH (t:Topic {name: $topic})
                        MERGE (p:Topic {name: $prereq})
                        MERGE (p)-[:PREREQUISITE_FOR]->(t)
                    """, topic=topic_name, prereq=prereq)
                    
                # Create applications
                apps = structured_data.get("Applications", [])
                print(f"🎯 Creating {len(apps)} applications: {apps}")
                
                for app in apps:
                    session.run("""
                        MATCH (t:Topic {name: $topic})
                        MERGE (a:Application {name: $app})
                        MERGE (t)-[:HAS_APPLICATION]->(a)
                    """, topic=topic_name, app=app)
                
                # Auto-detect concept relationships
                related = self._detect_relationships(session, topic_name, keywords)
                print(f"🔗 Found {len(related)} related concepts: {related}")
                
                # Enhanced auto-relationship discovery
                self._create_advanced_relationships(session, topic_name)
                print(f"✅ Knowledge graph creation completed for: {topic_name}")
                
            except Exception as e:
                print(f"❌ Error creating knowledge graph: {e}")
                import traceback
                traceback.print_exc()

    def _detect_relationships(self, session, topic, keywords):
        """Auto-detect relationships between concepts"""
        # Find similar concepts based on keyword overlap
        result = session.run("""
            MATCH (k1:Keyword)<-[:RELATES_TO]-(t1:Topic {name: $topic})
            MATCH (k2:Keyword)<-[:RELATES_TO]-(t2:Topic)
            WHERE t1 <> t2 AND k1.name = k2.name
            MERGE (t1)-[:RELATED_CONCEPT]->(t2)
            RETURN t2.name as related_topic
        """, topic=topic)
        
        return [record["related_topic"] for record in result]
    
    def _create_advanced_relationships(self, session, topic_name):
        """Enhanced auto-relationship discovery using embeddings and NLP"""
        
        try:
            # 1. Find topics with similar embeddings (manual cosine similarity)
            print(f"🔍 Finding semantically similar topics to: {topic_name}")
            
            # Get current topic embedding
            current_result = session.run("""
                MATCH (current:Topic {name: $topic})
                RETURN current.embedding as embedding
            """, topic=topic_name)
            
            current_record = current_result.single()
            if not current_record or not current_record["embedding"]:
                print("⚠️  No embedding found for current topic")
                return
                
            current_embedding = np.array(current_record["embedding"])
            
            # Get all other topics with embeddings
            other_topics_result = session.run("""
                MATCH (other:Topic) 
                WHERE other.name <> $topic AND other.embedding IS NOT NULL
                RETURN other.name as name, other.embedding as embedding
            """, topic=topic_name)
            
            similar_topics = 0
            for record in other_topics_result:
                try:
                    other_embedding = np.array(record["embedding"])
                    
                    # Calculate cosine similarity manually
                    dot_product = np.dot(current_embedding, other_embedding)
                    norms = np.linalg.norm(current_embedding) * np.linalg.norm(other_embedding)
                    
                    if norms > 0:
                        similarity = dot_product / norms
                        
                        if similarity > 0.7:  # Threshold for similarity
                            session.run("""
                                MATCH (current:Topic {name: $current_topic})
                                MATCH (other:Topic {name: $other_topic})
                                MERGE (current)-[:SEMANTICALLY_SIMILAR {strength: $similarity}]->(other)
                            """, current_topic=topic_name, other_topic=record["name"], similarity=float(similarity))
                            similar_topics += 1
                            print(f"  ✅ Found similar topic: {record['name']} (similarity: {similarity:.3f})")
                            
                except Exception as e:
                    print(f"  ❌ Error calculating similarity with {record['name']}: {e}")
            
            print(f"🔗 Created {similar_topics} semantic similarity relationships")
            
            # 2. Auto-detect prerequisite relationships based on keywords
            prereq_result = session.run("""
                MATCH (t1:Topic {name: $topic})-[:RELATES_TO]->(k:Keyword)
                MATCH (t2:Topic)-[:RELATES_TO]->(k)
                WHERE t1 <> t2 AND k.name IN ['basic', 'fundamental', 'introduction', 'prerequisite']
                MERGE (t2)-[:LIKELY_PREREQUISITE]->(t1)
                RETURN count(*) as prereq_count
            """, topic=topic_name)
            
            prereq_count = prereq_result.single()["prereq_count"]
            print(f"📚 Created {prereq_count} likely prerequisite relationships")
            
            # 3. Auto-detect hierarchical relationships
            hier_result = session.run("""
                MATCH (parent:Topic {name: $topic})
                MATCH (child:Topic) 
                WHERE toLower(child.name) CONTAINS toLower(parent.name)
                AND child <> parent
                MERGE (parent)-[:PARENT_CONCEPT]->(child)
                RETURN count(*) as hier_count
            """, topic=topic_name)
            
            hier_count = hier_result.single()["hier_count"]
            print(f"🏗️  Created {hier_count} hierarchical relationships")
            
        except Exception as e:
            print(f"❌ Error in advanced relationship creation: {e}")
            import traceback
            traceback.print_exc()

    def get_concept_context(self, query_text):
        """
        UPDATED: Uses Vector Search with fallback to keyword search
        """
        if not self.driver: return ""
        
        # 1. Generate embedding for the user's query
        query_embedding = self.embedding_model.encode(query_text).tolist()

        with self.driver.session() as session:
            try:
                # 2. Try Vector Search first (Neo4j 5.x+)
                result = session.run("""
                    CALL db.index.vector.queryNodes('topic_embeddings', 3, $embedding)
                    YIELD node AS topic, score
                    
                    // 3. Traverse from these topics to get context
                    MATCH (topic)-[r]->(related)
                    
                    RETURN topic.name as MainTopic, 
                        type(r) as Relation, 
                        labels(related)[0] as Type, 
                        coalesce(related.name, related.term, related.expression) as Content,
                        related.description as Desc,
                        score
                    ORDER BY score DESC
                    LIMIT 25
                """, embedding=query_embedding)
                
                records = list(result)
                
                # If no vector results, fallback to keyword search
                if not records:
                    print("Vector search returned no results. Falling back to keyword search.")
                    search_terms = query_text.lower().split()
                    
                    result = session.run("""
                        MATCH (t:Topic)-[r]->(related)
                        WHERE ANY(term IN $terms WHERE toLower(t.name) CONTAINS term
                            OR toLower(coalesce(related.name, related.term, '')) CONTAINS term)
                        
                        RETURN t.name as MainTopic, 
                            type(r) as Relation, 
                            labels(related)[0] as Type, 
                            coalesce(related.name, related.term, related.expression) as Content,
                            related.description as Desc,
                            1.0 as score
                        LIMIT 25
                    """, terms=search_terms)
                    
                    records = list(result)
                
            except Exception as e:
                print(f"Vector search failed: {e}. Using keyword search.")
                # Fallback to keyword search
                search_terms = query_text.lower().split()
                
                result = session.run("""
                    MATCH (t:Topic)-[r]->(related)
                    WHERE ANY(term IN $terms WHERE toLower(t.name) CONTAINS term
                        OR toLower(coalesce(related.name, related.term, '')) CONTAINS term)
                    
                    RETURN t.name as MainTopic, 
                        type(r) as Relation, 
                        labels(related)[0] as Type, 
                        coalesce(related.name, related.term, related.expression) as Content,
                        related.description as Desc,
                        1.0 as score
                    LIMIT 25
                """, terms=search_terms)
                
                records = list(result)
            
            # 4. Format the output for the LLM
            if not records:
                return "No relevant context found in knowledge graph."
            
            context_lines = []
            current_topic = None
            
            for record in records:
                topic = record["MainTopic"]
                if topic != current_topic:
                    score_str = f" (Relevance: {record['score']:.2f})" if record['score'] < 1.0 else ""
                    context_lines.append(f"\n## Topic: {topic}{score_str}")
                    current_topic = topic
                
                rel_type = record["Relation"].replace("_", " ").lower()
                content = record["Content"]
                desc = f" - {record['Desc']}" if record["Desc"] else ""
                
                context_lines.append(f"- {rel_type}: {content}{desc}")

            return "\n".join(context_lines)

    def get_learning_path(self, start_concept, end_concept):
        """Find learning path between concepts"""
        if not self.driver:
            return ""
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = shortestPath(
                    (start)-[*..5]-(end)
                )
                WHERE toLower(start.name) CONTAINS toLower($start) 
                AND toLower(end.name) CONTAINS toLower($end)
                RETURN [n in nodes(path) | n.name] as path
            """, start=start_concept, end=end_concept)
            
            path_result = result.single()
            return path_result["path"] if path_result else []
        

    def get_graph_stats(self):
        """Get statistics about the knowledge graph"""
        if not self.driver:
            return {}
            
        with self.driver.session() as session:
            stats = {}
            
            # Count nodes by type
            result = session.run("""
                MATCH (n) 
                WITH labels(n)[0] as label, count(n) as count
                RETURN label, count
            """)
            
            for record in result:
                stats[record["label"]] = record["count"]
            
            # Count relationships
            rel_result = session.run("""
                MATCH ()-[r]->() 
                WITH type(r) as rel_type, count(r) as count
                RETURN rel_type, count
            """)
            
            stats["relationships"] = {}
            for record in rel_result:
                stats["relationships"][record["rel_type"]] = record["count"]
                
            return stats

    def find_related_topics(self, topic_name, max_depth=2):
        """Find topics related to a given topic"""
        if not self.driver:
            return []
            
        with self.driver.session() as session:
            result = session.run("""
                MATCH (start:Topic {name: $topic})
                MATCH (start)-[*1..$max_depth]-(related:Topic)
                WHERE related <> start
                RETURN DISTINCT related.name as topic, 
                    shortestPath((start)-[*]-(related)) as path
                LIMIT 10
            """, topic=topic_name, max_depth=max_depth)
            
            return [record["topic"] for record in result]