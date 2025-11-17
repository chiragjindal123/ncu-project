from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import json

load_dotenv()

class KnowledgeGraph:
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(
                "neo4j://127.0.0.1:7687",
                auth=("neo4j", os.getenv("NEO4J_PASSWORD", "password123"))
            )
            # Test the connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✅ Neo4j connection established")
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def create_concept_graph(self, structured_data):
        """Create concepts and relationships from extracted knowledge"""
        if not self.driver:
            return ""
    
        with self.driver.session() as session:
            # Create main topic
            topic = structured_data.get("Topic", "Unknown")
            session.run(
                "MERGE (t:Topic {name: $topic})",
                topic=topic
            )
            
            # Create subtopics and link to main topic
            for subtopic in structured_data.get("Subtopics", []):
                session.run("""
                    MATCH (t:Topic {name: $topic})
                    MERGE (s:Subtopic {name: $subtopic})
                    MERGE (t)-[:HAS_SUBTOPIC]->(s)
                """, topic=topic, subtopic=subtopic)
            
            # Create definitions as concepts
            for definition in structured_data.get("Definitions", []):
                if ":" in definition:
                    term, desc = definition.split(":", 1)
                    session.run("""
                        MATCH (t:Topic {name: $topic})
                        MERGE (d:Definition {term: $term, description: $desc})
                        MERGE (t)-[:DEFINES]->(d)
                    """, topic=topic, term=term.strip(), desc=desc.strip())
            
            # Create formulas
            for formula in structured_data.get("Formulas", []):
                session.run("""
                    MATCH (t:Topic {name: $topic})
                    MERGE (f:Formula {expression: $formula})
                    MERGE (t)-[:HAS_FORMULA]->(f)
                """, topic=topic, formula=formula)
            
            # Create keywords and auto-detect relationships
            keywords = structured_data.get("Terminology / Keywords", [])
            for keyword in keywords:
                session.run("""
                    MATCH (t:Topic {name: $topic})
                    MERGE (k:Keyword {name: $keyword})
                    MERGE (t)-[:RELATES_TO]->(k)
                """, topic=topic, keyword=keyword)
            
            # Create prerequisites relationships
            for prereq in structured_data.get("Prerequisites", []):
                session.run("""
                    MATCH (t:Topic {name: $topic})
                    MERGE (p:Topic {name: $prereq})
                    MERGE (p)-[:PREREQUISITE_FOR]->(t)
                """, topic=topic, prereq=prereq)
                
            # Create applications
            for app in structured_data.get("Applications", []):
                session.run("""
                    MATCH (t:Topic {name: $topic})
                    MERGE (a:Application {name: $app})
                    MERGE (t)-[:HAS_APPLICATION]->(a)
                """, topic=topic, app=app)
            
            # Auto-detect concept relationships
            self._detect_relationships(session, topic, keywords)

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

    def get_concept_context(self, query_terms):
        """Get relevant concepts and relationships for a query"""
        if not self.driver:
            return ""
            
        with self.driver.session() as session:
            # Enhanced search with prerequisites and applications
            result = session.run("""
                MATCH (n)
                WHERE ANY(term IN $terms WHERE 
                    toLower(n.name) CONTAINS toLower(term) OR 
                    toLower(coalesce(n.term, '')) CONTAINS toLower(term) OR 
                    toLower(coalesce(n.expression, '')) CONTAINS toLower(term) OR
                    toLower(coalesce(n.description, '')) CONTAINS toLower(term)
                )
                WITH n, 
                    size([term IN $terms WHERE toLower(n.name) CONTAINS toLower(term)]) as name_matches,
                    size([term IN $terms WHERE toLower(coalesce(n.term, '')) CONTAINS toLower(term)]) as term_matches
                ORDER BY (name_matches + term_matches) DESC
                LIMIT 15
                
                OPTIONAL MATCH (n)-[r]->(related)
                RETURN n, type(r) as relationship, related, (name_matches + term_matches) as relevance_score
            """, terms=query_terms)
            
            context = []
            processed_nodes = set()
            
            for record in result:
                node = record["n"]
                rel = record["relationship"]
                related = record["related"]
                
                # Avoid duplicates
                node_id = f"{list(node.labels)[0]}:{node.get('name', node.get('term', 'unknown'))}"
                if node_id in processed_nodes:
                    continue
                processed_nodes.add(node_id)
                
                if "Topic" in node.labels:
                    context.append(f"**Topic**: {node['name']}")
                elif "Definition" in node.labels:
                    context.append(f"**Definition**: {node['term']} - {node['description']}")
                elif "Formula" in node.labels:
                    context.append(f"**Formula**: {node['expression']}")
                elif "Keyword" in node.labels:
                    context.append(f"**Keyword**: {node['name']}")
                elif "Subtopic" in node.labels:
                    context.append(f"**Subtopic**: {node['name']}")
                elif "Application" in node.labels:
                    context.append(f"**Application**: {node['name']}")
                
                if rel and related:
                    related_name = related.get('name', related.get('term', related.get('expression', 'unknown')))
                    context.append(f"  → {rel.replace('_', ' ')}: {related_name}")
            
            return "\n".join(context)

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