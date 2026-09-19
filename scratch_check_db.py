"""Quick DB inspection script."""
import sqlite3

# Timeline DB
conn1 = sqlite3.connect('almond_timeline.db')
print("=== entity_event_map ===")
for row in conn1.execute('SELECT * FROM entity_event_map').fetchall():
    print(row)

print("\n=== timeline_events ===")
for row in conn1.execute('SELECT id, memory_id, description, event_type, entities FROM timeline_events').fetchall():
    print(row)

# Main DB
conn2 = sqlite3.connect('longmem_almond.db')
print("\n=== entities (first 30) ===")
for row in conn2.execute('SELECT id, name FROM entities LIMIT 30').fetchall():
    print(row)

print("\n=== entity_memory_map (first 20) ===")
for row in conn2.execute('SELECT * FROM entity_memory_map LIMIT 20').fetchall():
    print(row)

print("\n=== memory_blocks count ===")
print(conn2.execute('SELECT count(*) FROM memory_blocks').fetchone())

print("\n=== memory_blocks sample (first 5) ===")
for row in conn2.execute('SELECT id, substr(content, 1, 80), memory_type FROM memory_blocks LIMIT 5').fetchall():
    print(row)
