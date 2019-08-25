import pymongo

client = pymongo.MongoClient('mongodb://mobvoinlp:geniusnlp@db_mongo_nlp-0.uc.mobvoi-idc.com/nlp')
db = client.nlp
# collection = db['comments']
# collection = db['opinion_tags']
# collection = db['opinion_build_pairs']