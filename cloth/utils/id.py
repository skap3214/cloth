import hashlib
import uuid

def generate_unique_id() -> str:
    return str(uuid.uuid4())

def generate_id(input_string: str, always_unique: bool = False):
    if always_unique:
        return generate_unique_id()
    return hashlib.sha256(input_string.encode()).hexdigest()