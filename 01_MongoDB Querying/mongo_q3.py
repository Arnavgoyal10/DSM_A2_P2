import pymongo
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

class MongoPart2Query3:
    def __init__(self, db_name="yelp"):
        try:
            self.client = pymongo.MongoClient("mongodb://localhost:27017/")
            self.db = self.client[db_name]
            import os
            os.makedirs('output', exist_ok=True)
            logging.info(f"Connected to {db_name}")
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            raise e

    def run_query_3_quartiles(self):
        logging.info("\n--- Executing Q3: Check-in Quartiles & Top Categories ---")
        
        # 1. Top 10 Categories by total review count
        pipeline_top_cats = [
            {"$lookup": {"from": "business", "localField": "business_id", "foreignField": "business_id", "as": "b"}},
            {"$unwind": "$b"},
            {"$unwind": "$b.categories"},
            {"$group": {"_id": "$b.categories", "review_count": {"$sum": 1}}},
            {"$sort": {"review_count": -1}},
            {"$limit": 10}
        ]
        top_cats_cursor = self.db.review.aggregate(pipeline_top_cats, allowDiskUse=True)
        top_10_cats = [doc['_id'] for doc in top_cats_cursor]
        logging.info(f"Top 10 Categories strictly identified: {top_10_cats}")

        # 2. Extract Business Base Stats (stars, review_count, categories)
        biz_cursor = self.db.business.find({}, {"business_id": 1, "stars": 1, "review_count": 1, "categories": 1})
        biz_df = pd.DataFrame(list(biz_cursor))
        if biz_df.empty: return
        biz_df = biz_df.explode('categories')
        
        # Filter businesses down exclusively to the Top 10 categories to save RAM
        biz_df = biz_df[biz_df['categories'].isin(top_10_cats)]

        # 3. Extract Tip counts per business
        pipeline_tips = [
            {"$group": {"_id": "$business_id", "tip_count": {"$sum": 1}}}
        ]
        tips_df = pd.DataFrame(list(self.db.tip.aggregate(pipeline_tips)))
        tips_df.rename(columns={"_id": "business_id"}, inplace=True)

        # 4. Extract Checkin counts per business
        # Checkin collection stores dates as comma separated string
        checkin_cursor = self.db.checkin.find({}, {"business_id": 1, "date": 1})
        checkin_list = []
        for c in checkin_cursor:
            datestr = c.get('date', "")
            count = len(datestr.split(',')) if datestr else 0
            checkin_list.append({"business_id": c['business_id'], "checkin_count": count})
        checkin_df = pd.DataFrame(checkin_list)

        # 5. Merge DataFrames algorithmically
        merged_df = biz_df.merge(checkin_df, on='business_id', how='left')
        merged_df = merged_df.merge(tips_df, on='business_id', how='left')
        
        # Fill NaNs for missing tips/checkins with 0
        merged_df['checkin_count'] = merged_df['checkin_count'].fillna(0)
        merged_df['tip_count'] = merged_df['tip_count'].fillna(0)

        # 6. Quartile Calculation purely based on Checkin Distribution
        # The prompt defines: top quartile (high), middle two quartiles (medium), bottom quartile (low)
        # We must calculate this across ALL unique businesses mathematically, not just the exploded ones.
        unique_biz = merged_df.drop_duplicates(subset=['business_id'])
        checkin_values = unique_biz['checkin_count']
        
        # Handle the case where many checkins are 0 causing overlapping bin edges (duplicate bins)
        try:
            q_labels = ['Low (Bottom 25%)', 'Medium (25-75%)', 'High (Top 25%)']
            q_bins = [0, 0.25, 0.75, 1.0]
            # Use rank(pct=True) to avoid duplicate edge errors if skewed heavily at 0
            checkin_ranks = checkin_values.rank(pct=True, method='first')
            unique_biz['Quartile'] = pd.cut(checkin_ranks, bins=q_bins, labels=q_labels, include_lowest=True)
        except Exception as e:
            logging.error(f"Quartile mathematical binning failed: {e}")
            return
            
        merged_df = merged_df.merge(unique_biz[['business_id', 'Quartile']], on='business_id', how='left')

        # 7. Computation for Cross Tabulation
        # We need Mean Star Rating, Mean Review Count, and Ratio of Tips to Reviews.
        # Group by Quartile and Category
        grouped = merged_df.groupby(['Quartile', 'categories'])
        
        results = []
        for (quartile, category), group in grouped:
            mean_stars = group['stars'].mean()
            mean_reviews = group['review_count'].mean()
            sum_tips = group['tip_count'].sum()
            sum_reviews = group['review_count'].sum()
            tip_ratio = (sum_tips / sum_reviews) if sum_reviews > 0 else 0
            
            results.append({
                "Quartile": quartile,
                "Category": category,
                "Mean_Stars": mean_stars,
                "Mean_Review_Count": mean_reviews,
                "Tip_to_Review_Ratio": tip_ratio
            })
            
        final_df = pd.DataFrame(results)
        logging.info("\n--- Final Cross Tabulation Results ---")
        logging.info(final_df.to_string(index=False))
        
        final_df.to_csv('output/Q3_Checkin_CrossTab.csv', index=False)
        logging.info("Q3 mathematical export completed successfully.")
        
        # Generation of Q3 Plot (Heatmap for Cross-tabulation)
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Truncate Category labels for clean y-axis visualization
        final_df['Short_Category'] = final_df['Category'].apply(lambda x: ', '.join(x.split(', ')[:2]) if isinstance(x, str) else x)
        pivot_df = final_df.pivot(index='Short_Category', columns='Quartile', values='Mean_Stars')
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5)
        plt.title('Check-in Quartiles vs Top Categories: Mean Star Rating')
        plt.tight_layout()
        plt.savefig('output/q3_checkin_crosstab.png', dpi=300)
        plt.close()
        logging.info("[!] Generated output/q3_checkin_crosstab.png visualization.")

if __name__ == "__main__":
    q3 = MongoPart2Query3()
    q3.run_query_3_quartiles()
