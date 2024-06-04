import hashlib
import uuid

def generate_unique_id() -> str:
    return str(uuid.uuid4())

def generate_id(**kwargs):
    if kwargs.pop('always_unique', False):
        return generate_unique_id()
    input_string = '_'.join(str(v) for v in sorted(kwargs.values())) 
    return hashlib.sha256(input_string.encode()).hexdigest()