# cache.py
import os, shelve, hashlib

class SimpleCache:
    """
    Simple persistent cache using Python's shelve.
    Used for storing/retrieving retrieved chunks to reduce repeated queries.
    """
    def __init__(self, filename="cache/db"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.filename = filename

    def _key(self, k):
        return hashlib.sha256(k.encode()).hexdigest()

    def get(self, key):
        """Retrieve cached value by key, if present."""
        with shelve.open(self.filename) as db:
            value = db.get(self._key(key))
            print("🟢 Cache HIT" if value else "🔵 Cache MISS")
            return value

    def set(self, key, value):
        """Store value in cache under hashed key."""
        with shelve.open(self.filename) as db:
            db[self._key(key)] = value
