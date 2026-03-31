import pandas as pd
from neo4j import GraphDatabase
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")

class GDSAnalyticsPart1:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="arnavlm10"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        import os
        os.makedirs('output', exist_ok=True)

    def run_query1_pagerank(self):
        logging.info("\n--- GDS Query 1: PageRank on User->Business Bipartite ---")
        with self.driver.session() as session:
            # 1. Drop existing projection if it exists
            session.run("CALL gds.graph.drop('pagerankGraph', false) YIELD graphName")

            # 2. Project Cypher logic mapping User -> Review -> Business into a direct weighted edge
            projection_query = """
            CALL gds.graph.project.cypher(
              'pagerankGraph',
              'MATCH (n) WHERE n:User OR n:Business RETURN id(n) AS id, labels(n) AS labels',
              'MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business) RETURN id(u) AS source, id(b) AS target, r.stars AS weight'
            )
            YIELD graphName, nodeCount, relationshipCount
            """
            logging.info("Projecting in-memory PageRank graph...")
            proj = session.run(projection_query).single()
            if proj:
                logging.info(f"Projected Graph. Nodes: {proj['nodeCount']}, Edges: {proj['relationshipCount']}")

            # 3. Stream PageRank
            pagerank_query = """
            CALL gds.pageRank.stream('pagerankGraph', {
               maxIterations: 20,
               relationshipWeightProperty: 'weight'
            })
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS b, score
            WHERE 'Business' IN labels(b)
            RETURN b.name AS BusinessName, b.stars AS AvgStarRating, b.review_count AS ReviewCount, score AS PageRank
            ORDER BY PageRank DESC
            """
            
            logging.info("Streaming PageRank scores...")
            result = session.run(pagerank_query)
            df = pd.DataFrame([r.values() for r in result], columns=result.keys())
            
            if not df.empty:
                # Top 15 output
                top15 = df.head(15)
                logging.info("Top 15 Businesses by PageRank:")
                logging.info(top15.to_string(index=False))
                
                # Plot Spearman Correlations algebraically using pandas
                spearman_review = df['PageRank'].corr(df['ReviewCount'], method='spearman')
                spearman_stars = df['PageRank'].corr(df['AvgStarRating'], method='spearman')
                
                logging.info(f"\n[Spearman] PageRank vs ReviewCount: {spearman_review:.4f}")
                logging.info(f"[Spearman] PageRank vs AvgStarRating: {spearman_stars:.4f}")
                
                df.to_csv('output/GDS_Q1_PageRank.csv', index=False)
                
                # Visualizing PageRank vs Review Count
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x='ReviewCount', y='PageRank', data=df, alpha=0.6, color='blue')
                plt.title(f'PageRank vs Review Count (Spearman: {spearman_review:.3f})')
                plt.xlabel('Total Reviews Received')
                plt.ylabel('Topological PageRank Score')
                plt.tight_layout()
                plt.savefig('output/gds_q1_pagerank_scatter.png', dpi=300)
                plt.close()
                logging.info("[!] Generated output/gds_q1_pagerank_scatter.png visualization.")
                
            session.run("CALL gds.graph.drop('pagerankGraph', false)")

    def run_query2_louvain(self):
        logging.info("\n--- GDS Query 2: Louvain Community Detection ---")
        with self.driver.session() as session:
            session.run("CALL gds.graph.drop('friendsGraph', false) YIELD graphName")
            
            proj_query = """
            CALL gds.graph.project(
                'friendsGraph',
                'User',
                {FRIENDS_WITH: {orientation: 'UNDIRECTED'}}
            ) YIELD nodeCount, relationshipCount
            """
            proj = session.run(proj_query).single()
            logging.info(f"Projected Friends Graph: {proj['nodeCount']} nodes, {proj['relationshipCount']} edges")
            
            # Writing back is critical for Predictive Modeling (Stage 4)
            louvain_write = """
            CALL gds.louvain.write('friendsGraph', {writeProperty: 'communityId'})
            YIELD communityCount, modularity
            """
            l_result = session.run(louvain_write).single()
            logging.info(f"Louvain executed. Communities: {l_result['communityCount']}, Modularity: {l_result['modularity']:.4f}")
            
            # Extract communities >= 25 members
            com_query = """
            MATCH (u:User)
            WITH u.communityId AS cid, count(u) AS size
            WHERE size >= 25
            RETURN cid, size ORDER BY size DESC LIMIT 100
            """
            communities = session.run(com_query)
            
            results = []
            for record in communities:
                cid = record['cid']
                size = record['size']
                
                # Subquery for geometry and category density
                subquery = """
                MATCH (u:User {communityId: $cid})-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
                WITH b, count(r) AS review_count
                
                // Get Categories Unwound for this subset
                MATCH (b)-[:IN_CATEGORY]->(c:Category)
                WITH b, review_count, collect(c.name) AS catList
                RETURN {
                    state: b.state,
                    categories: catList,
                    review_count: review_count
                } AS data
                """
                
                sub_res = session.run(subquery, cid=cid)
                states = {}
                cats = {}
                total_reviews = 0
                
                for sr in sub_res:
                    data = sr['data']
                    rc = data.get('review_count', 1)
                    total_reviews += rc
                    # Count states
                    st = data['state']
                    if st:
                        states[st] = states.get(st, 0) + rc
                    # Count categories
                    for cat in data['categories']:
                        cats[cat] = cats.get(cat, 0) + rc
                        
                if total_reviews == 0:
                    continue
                    
                # Sort descending
                sorted_states = sorted(states.items(), key=lambda x: x[1], reverse=True)
                sorted_cats = sorted(cats.items(), key=lambda x: x[1], reverse=True)
                
                top_3_states = sorted_states[:3]
                top_3_cats = sorted_cats[:3]
                
                top_state_count = top_3_states[0][1] if top_3_states else 0
                geo_concentration = top_state_count / total_reviews
                
                results.append({
                    "CommunityID": cid,
                    "Size": size,
                    "Top3_States": [s[0] for s in top_3_states],
                    "Top3_Cats": [c[0] for c in top_3_cats],
                    "Geo_Concentration": geo_concentration,
                    "Total_Reviews": total_reviews
                })
                
            df = pd.DataFrame(results)
            # Rank by geo concentration
            df = df.sort_values('Geo_Concentration', ascending=False)
            logging.info("\nMost Geographically Concentrated Communities:")
            logging.info(df.head(5)[['CommunityID', 'Size', 'Geo_Concentration', 'Top3_States']].to_string(index=False))
            logging.info("\nLeast Geographically Concentrated Communities:")
            logging.info(df.tail(5)[['CommunityID', 'Size', 'Geo_Concentration', 'Top3_States']].to_string(index=False))
            
            df.to_csv('output/GDS_Q2_Louvain.csv', index=False)
            
            # Visualizing Louvain Communities
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='Size', y='Geo_Concentration', data=df, size='Size', hue='Geo_Concentration', palette='viridis', sizes=(50, 400), legend=False)
            plt.title('Louvain Communities: Geographic Concentration vs Group Size')
            plt.xlabel('Community Member Count')
            plt.ylabel('Geographic Concentration Ratio (Dominant State)')
            plt.tight_layout()
            plt.savefig('output/gds_q2_louvain_scatter.png', dpi=300)
            plt.close()
            logging.info("[!] Generated output/gds_q2_louvain_scatter.png visualization.")
            session.run("CALL gds.graph.drop('friendsGraph', false)")

if __name__ == "__main__":
    gds_runner = GDSAnalyticsPart1()
    start = time.time()
    gds_runner.run_query1_pagerank()
    gds_runner.run_query2_louvain()
    logging.info(f"Execution took {time.time() - start:.2f}s")
