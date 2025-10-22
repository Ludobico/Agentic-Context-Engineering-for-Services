class DeltaJsonParser:
    def __init__(self):
        self.previous_values = {}
        self.final_values = {} 

    def parse_delta(self, chunk: dict):
        deltas = {}

        for key, value in chunk.items():
            self.final_values[key] = value
            
            if isinstance(value, str):
                prev = self.previous_values.get(key, "")
                if value.startswith(prev):
                    deltas[key] = value[len(prev):]
                    self.previous_values[key] = value
                else:
                    deltas[key] = value
                    self.previous_values[key] = value
            elif isinstance(value, list):
                deltas[key] = value
                
        return deltas
    
    def get_final_values(self):
        return self.final_values.copy()
    
    def reset(self):
        self.previous_values = {}
        self.final_values = {}