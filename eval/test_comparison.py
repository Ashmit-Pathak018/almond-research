import sys
import json
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.almond import Almond
from core.memory_pipeline_v2.query_analyzer import QueryAnalyzer
from eval.utils.cache_system import SQLiteReplayCache

def main():
    db_path = r"C:\Users\ASMIT\Almond\data\longmem_eval_100_sqlite"
    cache = SQLiteReplayCache(db_path)
    cache_key = "7aad214db27ba07f"
    
    print(f"Loading cache {cache_key}...")
    success = cache.load(cache_key)
    if not success:
        print("Cache load failed.")
        return
        
    almond = Almond()
    print(f"Entities in registry: {len(almond.controller._entity_registry.all_entities())}")
    
    analyzer = QueryAnalyzer()
    query = "Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?"
    intent = analyzer.analyze(query)
    
    print(f"Intent Type: {intent.intent_type}")
    print(f"Comparison Targets: {intent.comparison_targets}")
    
    # Run the retriever
    retriever = almond.controller._comp_ret
    result = retriever.retrieve(intent)
    
    print(f"Result flat_memories: {len(result.flat_memories)}")
    for g in result.groups:
        print(f"Group: {g.entity_name}, memories: {len(g.memories)}")
        
if __name__ == '__main__':
    main()
