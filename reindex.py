"""Re-index ChromaDB with current embedding model."""
import sys, os, boto3

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'domain'))
os.chdir(os.path.join(os.path.dirname(__file__), 'src', 'domain'))

sess = boto3.Session(profile_name='Nova', region_name='us-east-1')
frozen = sess.get_credentials().get_frozen_credentials()
os.environ['AWS_ACCESS_KEY_ID']     = frozen.access_key
os.environ['AWS_SECRET_ACCESS_KEY'] = frozen.secret_key
os.environ['AWS_DEFAULT_REGION']    = 'us-east-1'

import config
print('Embedding model:', config.EMBEDDING_MODEL_ID)

from Retrieval.database import ChromaVectorStoreManager

db = ChromaVectorStoreManager(data_folder=config.DATA_FOLDER)
db.delete_collection()
print('Old index deleted.')

db2 = ChromaVectorStoreManager(data_folder=config.DATA_FOLDER)
docs = db2.load_documents(config.PROCESSED_JSON_FILE)
print(f'Indexing {len(docs)} docs...')
db2.store(docs)
print('Done. Total:', db2.count_nodes())
