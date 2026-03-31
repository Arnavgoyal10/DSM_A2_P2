import pymongo
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

def check_mongodb_types():
    logging.info("--- MONGODB TYPE VERIFICATION ---")
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        db = client["yelp"]  # Previously the user used 'yelp' or 'yelp_subset'. We'll check 'yelp_db' etc if this fails.
        # Actually in Part 1 we generated "fake" scripts using 'yelp_viva_db' but the real db is likely 'yelp'
        # Let's list DBs:
        dbs = client.list_database_names()
        target_db = None
        for name in ['yelp', 'yelp_subset', 'dsm_assignment2']:
            if name in dbs:
                target_db = name
                break
        
        if not target_db:
            logging.error(f"Could not find likely Yelp database. Found: {dbs}")
            return False
            
        logging.info(f"Targeting MongoDB DB: {target_db}")
        db = client[target_db]
        
        # Check review collection
        review = db.review.find_one()
        if review:
            logging.info(f"Review 'useful' type: {type(review.get('useful'))} (Value: {review.get('useful')})")
            logging.info(f"Review 'date' type: {type(review.get('date'))} (Value: {review.get('date')})")
            logging.info(f"Review 'stars' type: {type(review.get('stars'))} (Value: {review.get('stars')})")
        else:
            logging.error("No reviews found.")

        # Check user collection
        user = db.user.find_one()
        if user:
            logging.info(f"User 'yelping_since' type: {type(user.get('yelping_since'))} (Value: {user.get('yelping_since')})")
            logging.info(f"User 'elite' type: {type(user.get('elite'))} (Value: {user.get('elite')})")
        else:
            logging.error("No users found.")
            
        return True
    except Exception as e:
        logging.error(f"MongoDB connection failed: {e}")
        return False

def check_neo4j_gds():
    logging.info("\n--- NEO4J GDS VERIFICATION ---")
    # Assuming standard local auth or no auth. We'll try the common ones.
    uris = ["bolt://localhost:7687", "neo4j://localhost:7687"]
    auths = [("neo4j", "arnavlm10")]
    
    driver = None
    for uri in uris:
        for auth in auths:
            try:
                driver = GraphDatabase.driver(uri, auth=auth)
                driver.verify_connectivity()
                logging.info(f"Connected to Neo4j at {uri} with user {auth[0]}")
                break
            except Exception:
                driver = None
        if driver:
            break
            
    if not driver:
        logging.error("Could not connect to Neo4j. Is it running?")
        return False
        
    try:
        with driver.session() as session:
            result = session.run("CALL gds.version()")
            version = result.single()[0]
            logging.info(f"Neo4j GDS Library is installed. Version: {version}")
            return True
    except ClientError as e:
        logging.error("Neo4j GDS Library is NOT installed or accessible!")
        logging.error(e)
        return False
    except Exception as e:
        logging.error(f"Error checking GDS: {e}")
        return False

if __name__ == "__main__":
    check_mongodb_types()
    check_neo4j_gds()
