import pymongo
import pandas as pd
import numpy as np
import logging
import time
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")

class MongoPart2Analytics:
    def __init__(self, db_name="yelp"):
        try:
            self.client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            self.db = self.client[db_name]
            import os
            os.makedirs('output', exist_ok=True)
            logging.info(f"Connected to {db_name}")
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            raise e

    def run_query_1_cohorts(self):
        logging.info("--- Executing Q1: Cohort Analysis ---")
        pipeline = [
            {"$lookup": {"from": "user", "localField": "user_id", "foreignField": "user_id", "as": "u"}},
            {"$unwind": "$u"},
            # Parse Dates natively
            {"$addFields": {
                "user_date": {"$dateFromString": {"dateString": "$u.yelping_since"}},
                "char_length": {"$strLenCP": "$text"},
                # Ensure useful is treated as an integer (fallback to 0 if null)
                "useful_votes": {"$ifNull": ["$useful", 0]}
            }},
            {"$addFields": {
                "cohort_year": {"$year": "$user_date"}
            }},
            # Conditionally map star variables for proportion computation
            {"$addFields": {
                "is_1": {"$cond": [{"$eq": ["$stars", 1]}, 1, 0]},
                "is_2": {"$cond": [{"$eq": ["$stars", 2]}, 1, 0]},
                "is_3": {"$cond": [{"$eq": ["$stars", 3]}, 1, 0]},
                "is_4": {"$cond": [{"$eq": ["$stars", 4]}, 1, 0]},
                "is_5": {"$cond": [{"$eq": ["$stars", 5]}, 1, 0]}
            }},
            # Group by Cohort Year
            {"$group": {
                "_id": "$cohort_year",
                "total_reviews": {"$sum": 1},
                "mean_stars": {"$avg": "$stars"},
                "std_stars": {"$stdDevPop": "$stars"},
                "mean_char_length": {"$avg": "$char_length"},
                "mean_useful_votes": {"$avg": "$useful_votes"},
                "sum_1": {"$sum": "$is_1"},
                "sum_2": {"$sum": "$is_2"},
                "sum_3": {"$sum": "$is_3"},
                "sum_4": {"$sum": "$is_4"},
                "sum_5": {"$sum": "$is_5"}
            }},
            {"$sort": {"_id": 1}}
        ]

        cursor = self.db.review.aggregate(pipeline, allowDiskUse=True)
        df = pd.DataFrame(list(cursor))
        if df.empty:
            logging.error("Q1 returned empty DataFrame! Check pipeline fields.")
            return

        df.rename(columns={"_id": "Cohort_Year"}, inplace=True)
        # Calculate Proportions
        for col in ['1', '2', '3', '4', '5']:
            df[f'prop_{col}_star'] = df[f'sum_{col}'] / df['total_reviews']

        df.drop(columns=['sum_1', 'sum_2', 'sum_3', 'sum_4', 'sum_5'], inplace=True)
        
        # Identify peaks
        highest_stars = df.loc[df['mean_stars'].idxmax()]
        highest_useful = df.loc[df['mean_useful_votes'].idxmax()]
        
        logging.info(f"Q1 Final Cohort Data Shape: {df.shape}")
        logging.info(f"Highest Mean Stars Cohort: {highest_stars['Cohort_Year']} ({highest_stars['mean_stars']:.2f})")
        logging.info(f"Highest Useful Votes Cohort: {highest_useful['Cohort_Year']} ({highest_useful['mean_useful_votes']:.2f})")
        
        df.to_csv('output/Q1_Cohort_Analysis.csv', index=False)
        
        # Generation of Q1 Plot
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Cohort_Year', y='mean_useful_votes', data=df, palette='viridis', hue='Cohort_Year', legend=False)
        plt.title('Cohort Analysis: Mean Useful Votes by Yelp Join Year')
        plt.ylabel('Mean Useful Votes')
        plt.xlabel('Cohort Year')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('output/q1_cohorts_useful.png', dpi=300)
        plt.close()
        logging.info("[!] Generated output/q1_cohorts_useful.png visualization.")

    def run_query_2_mom_trends(self):
        logging.info("\n--- Executing Q2: Month-over-Month Category Trends ---")

        pipeline = [
            {"$lookup": {"from": "business", "localField": "business_id", "foreignField": "business_id", "as": "b"}},
            {"$unwind": "$b"},
            {"$unwind": "$b.categories"},
            {"$addFields": {
                "parsed_date": {"$dateFromString": {"dateString": "$date"}}
            }},
            {"$addFields": {
                # Format exactly as YYYY-MM for month-over-month
                "year_month": {"$dateToString": {"format": "%Y-%m", "date": "$parsed_date"}}
            }},
            {"$group": {
                "_id": {
                    "category": "$b.categories",
                    "month": "$year_month"
                },
                "monthly_reviews": {"$sum": 1},
                "monthly_avg_stars": {"$avg": "$stars"}
            }}
        ]
        
        cursor = self.db.review.aggregate(pipeline, allowDiskUse=True)
        raw_list = list(cursor)
        if not raw_list:
            logging.error("Q2 returned empty data.")
            return
            
        # Flatten and load into Pandas
        flattened = [{
            "category": r["_id"]["category"],
            "month": r["_id"]["month"],
            "reviews": r["monthly_reviews"],
            "avg_stars": r["monthly_avg_stars"]
        } for r in raw_list]
        
        df = pd.DataFrame(flattened)
        df['month'] = pd.to_datetime(df['month'])
        
        # 1. Filter out Categories with < 500 total reviews platform-wide
        category_counts = df.groupby('category')['reviews'].sum()
        valid_categories = category_counts[category_counts >= 500].index
        df = df[df['category'].isin(valid_categories)]
        
        # 2. Sort explicitly to ensure chronological window comparison
        df = df.sort_values(by=['category', 'month'])
        
        results = []
        for cat in valid_categories:
            subset = df[df['category'] == cat].copy()
            # If a category doesn't have many months, skip it
            if len(subset) < 3:
                continue
                
            # Month over month delta
            subset['mom_delta'] = subset['avg_stars'].diff()
            # Drop the first NaN row
            subset = subset.dropna(subset=['mom_delta'])
            
            # Count consecutive pairs. 
            # Question calls for "proportion of consecutive month-pairs showing an increase (or decrease)"
            # Let's define it generally: fraction of >0 steps and fraction of <0 steps
            total_pairs = len(subset)
            inc = (subset['mom_delta'] > 0).sum() / total_pairs
            dec = (subset['mom_delta'] < 0).sum() / total_pairs
            
            results.append({
                "category": cat,
                "total_months_active": total_pairs + 1,
                "inc_proportion": inc,
                "dec_proportion": dec
            })
            
        trend_df = pd.DataFrame(results)
        
        top3_inc = trend_df.nlargest(3, 'inc_proportion')
        top3_dec = trend_df.nlargest(3, 'dec_proportion')
        
        logging.info("Top 3 Consistent Upward Trends:")
        for _, row in top3_inc.iterrows():
            logging.info(f" - {row['category']}: {row['inc_proportion']*100:.2f}% increasing pairs")
            
        logging.info("Top 3 Consistent Downward Trends:")
        for _, row in top3_dec.iterrows():
            logging.info(f" - {row['category']}: {row['dec_proportion']*100:.2f}% decreasing pairs")
            
        trend_df.to_csv('output/Q2_MoM_Trends.csv', index=False)
        
        # Generation of Q2 Plot (Bar Chart for Trend Consistency)
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Build proper plot: upward categories by inc_proportion, downward by dec_proportion
        top3_inc_plot = top3_inc.copy()
        top3_inc_plot['trend'] = 'Upward'
        top3_inc_plot['consistency'] = top3_inc_plot['inc_proportion']

        top3_dec_plot = top3_dec.copy()
        top3_dec_plot['trend'] = 'Downward'
        top3_dec_plot['consistency'] = top3_dec_plot['dec_proportion']

        plot_df = pd.concat([
            top3_inc_plot[['category', 'trend', 'consistency']],
            top3_dec_plot[['category', 'trend', 'consistency']]
        ])
        # Truncate long category strings for readable y-axis labels
        plot_df['short_category'] = plot_df['category'].apply(
            lambda x: (x[:42] + '...') if isinstance(x, str) and len(x) > 44 else x
        )
        plot_df = plot_df.sort_values('consistency', ascending=True)

        plt.figure(figsize=(13, 7))
        palette = {'Upward': '#2ecc71', 'Downward': '#e74c3c'}
        sns.barplot(x='consistency', y='short_category', data=plot_df,
                    hue='trend', palette=palette, dodge=False)
        plt.axvline(x=0.5, color='navy', linestyle='--', linewidth=1.2, alpha=0.7, label='50% baseline')
        plt.title('Top 3 Most Consistent Upward & Downward Monthly Trends by Category')
        plt.xlabel('Proportion of Consecutive Month-Pairs in Same Direction')
        plt.ylabel('Category')
        plt.legend(title='Trend Direction', loc='lower right')
        plt.tight_layout()
        plt.savefig('output/q2_mom_trends.png', dpi=300)
        plt.close()
        logging.info("[!] Generated output/q2_mom_trends.png visualization.")

if __name__ == "__main__":
    analytics = MongoPart2Analytics()
    start = time.time()
    analytics.run_query_1_cohorts()
    analytics.run_query_2_mom_trends()
    logging.info(f"Execution took {time.time() - start:.2f}s")
