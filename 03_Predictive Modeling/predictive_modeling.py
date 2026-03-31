import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")

class PredictiveModelingEngine:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="arnavlm10"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        import os
        os.makedirs('output', exist_ok=True)

    def run_stage4_useful_regression(self):
        logging.info("\n--- Stage 4: Predictive Modeling (Useful Votes Regression) ---")
        logging.info("Extracting Multi-Paradigm Feature Subsets from Neo4j & MongoDB...")
        # 1. Mongo Extraction (Review Text Length, Useful Votes, Tenure)
        import pymongo
        mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = mongo_client["yelp"]
        
        logging.info("Fetching MongoDB base features...")
        mongo_pipeline = [
            {"$lookup": {"from": "user", "localField": "user_id", "foreignField": "user_id", "as": "u"}},
            {"$unwind": "$u"},
            {"$lookup": {"from": "business", "localField": "business_id", "foreignField": "business_id", "as": "b"}},
            {"$unwind": "$b"},
            {"$project": {
                "user_id": 1, "business_id": 1, 
                "review_stars": "$stars", "target_useful": {"$ifNull": ["$useful", 0]},
                "review_length": {"$strLenCP": "$text"},
                "tenure_date": "$u.yelping_since", "user_review_count": "$u.review_count",
                "business_stars": "$b.stars"
            }},
            {"$limit": 50000}
        ]
        mongo_df = pd.DataFrame(list(db.review.aggregate(mongo_pipeline)))
        
        # 2. Neo4j Extraction (Network Degree, CommunityID)
        logging.info("Fetching Neo4j topological features...")
        with self.driver.session() as session:
            q = "MATCH (u:User) RETURN u.user_id AS user_id, u.communityId AS user_community, COUNT { (u)-[:FRIENDS_WITH]-() } AS user_degree LIMIT 100000"
            neo_df = pd.DataFrame([r.values() for r in session.run(q)], columns=['user_id', 'user_community', 'user_degree'])
            
        # 3. Merge Paradigms
        df = mongo_df.merge(neo_df, on='user_id', how='inner')
        if df.empty:
            logging.error("Multi-paradigm merge yielded zero rows!")
            return

        # Feature Engineering (Tenure mapping)
        logging.info("Executing Advanced Feature Engineering Pipelines...")
        df['tenure_date'] = pd.to_datetime(df['tenure_date'])
        # Reference frame 2022 to calculate total days since account creation
        df['user_tenure_days'] = (pd.to_datetime('2022-01-01') - df['tenure_date']).dt.days 
        df['user_tenure_days'] = df['user_tenure_days'].fillna(0)

        # Ensure no NAs in graph features
        df['user_degree'] = df['user_degree'].fillna(0)
        df['user_community'] = df['user_community'].fillna(-1)

        X = df[['review_stars', 'review_length', 'user_tenure_days', 'user_review_count', 'user_degree', 'user_community', 'business_stars']]
        y = df['target_useful']
        
        # Bin stratify target
        def bin_useful(val):
            if val == 0: return 0
            if val <= 5: return 1
            if val <= 20: return 2
            return 3
            
        df['useful_strata'] = df['target_useful'].apply(bin_useful)

        logging.info("Executing Stratified Train/Test Allocation (80/20)...")
        # To avoid stratification errors if bucket 3 is extremely rare in sample
        strat_col = df['useful_strata']
        val_counts = strat_col.value_counts()
        valid_classes = val_counts[val_counts > 1].index
        mask = strat_col.isin(valid_classes)
        df_safe = df[mask]
        
        X = df_safe[['review_stars', 'review_length', 'user_tenure_days', 'user_review_count', 'user_degree', 'user_community', 'business_stars']]
        y = df_safe['target_useful']
        strat_safe = df_safe['useful_strata']

        X_train, X_test, y_train, y_test, strata_train, strata_test = train_test_split(
            X, y, strat_safe, test_size=0.2, stratify=strat_safe, random_state=42
        )

        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        logging.info("Initiating Random Forest Regressor Training...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Global Evaluation
        global_rmse = root_mean_squared_error(y_test, preds)
        global_mae = mean_absolute_error(y_test, preds)
        global_r2 = r2_score(y_test, preds)

        logging.info("\n--- Regression Evaluation Matrix (Global) ---")
        logging.info(f"RMSE: {global_rmse:.4f} | MAE: {global_mae:.4f} | R2: {global_r2:.4f}")

        # Bucket Evaluation
        logging.info("\n--- Skew-Segmented Evaluation Matrix ---")
        for bucket_val, bucket_name in zip([0, 1, 2, 3], ["0 Useful", "1-5 Useful", "6-20 Useful", "21+ Useful"]):
            mask = strata_test == bucket_val
            if mask.sum() > 0:
                y_sub = y_test[mask]
                p_sub = preds[mask]
                rmse_s = root_mean_squared_error(y_sub, p_sub)
                mae_s = mean_absolute_error(y_sub, p_sub)
                logging.info(f"Bucket [{bucket_name}] -> RMSE: {rmse_s:.4f} | MAE: {mae_s:.4f}")

        # Feature Importance Visualization
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        importances = model.feature_importances_
        feature_names = X.columns
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values('Importance', ascending=False)
        logging.info("\n--- Algorithmic Feature Importances (Top Drivers) ---")
        logging.info(fi_df.to_string(index=False))
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis", hue="Feature", legend=False)
        plt.title("Random Forest Feature Importances (Useful Votes Regression)")
        plt.tight_layout()
        plt.savefig("output/stage4_feature_importance.png", dpi=300)
        plt.close()
        logging.info("\n[!] Generated visualization: output/stage4_feature_importance.png")

    def close(self):
        self.driver.close()

if __name__ == "__main__":
    engine = PredictiveModelingEngine()
    start = time.time()
    engine.run_stage4_useful_regression()
    logging.info(f"\nPipeline Executed in {time.time() - start:.2f}s")
    engine.close()
