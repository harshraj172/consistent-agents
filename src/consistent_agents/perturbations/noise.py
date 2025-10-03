import random
import string
from consistent_agents.perturbations.base import BasePerturbation


class NoisePerturbation(BasePerturbation):
    """Add character-level noise to text while preserving readability."""
    
    def __init__(self, 
                 noise_level: float = 0.05,
                 noise_types: list = None,
                 seed: int = None,
                 **kwargs):
        """Initialize noise perturbation."""
        super().__init__(**kwargs)
        self.noise_level = noise_level
        self.noise_types = noise_types or ['swap', 'insert', 'delete', 'replace']
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def apply(self, text: str, **kwargs) -> str:
        """Apply character-level noise to the text."""
        noise_level = kwargs.get('noise_level', self.noise_level)
        
        if not text or noise_level <= 0:
            return text
        
        chars = list(text)
        i = 0
        
        while i < len(chars):
            # Skip whitespace
            if chars[i].isspace():
                i += 1
                continue
            
            if random.random() < noise_level:
                noise_type = random.choice(self.noise_types)
                
                if noise_type == 'swap' and i < len(chars) - 1 and not chars[i + 1].isspace():
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
                    i += 2  
                    continue
                
                elif noise_type == 'insert':
                    random_char = random.choice(string.ascii_lowercase)
                    chars.insert(i + 1, random_char)
                
                elif noise_type == 'delete' and len(chars) > 1:
                    chars.pop(i)
                    continue
                
                elif noise_type == 'replace':
                    if chars[i].isupper():
                        chars[i] = random.choice(string.ascii_uppercase)
                    elif chars[i].islower():
                        chars[i] = random.choice(string.ascii_lowercase)
                    elif chars[i].isdigit():
                        chars[i] = random.choice(string.digits)
            i += 1
        
        return ''.join(chars)


# Example usage:
if __name__ == "__main__":
    perturbation = NoisePerturbation(noise_level=0.1, seed=42)
    
    original = "The quick brown fox jumps over the lazy dog."
    perturbed = perturbation.apply(original)
    
    print(f"Original:  {original}")
    print(f"Perturbed: {perturbed}")
    # Example output: "The qiuck borwn fox jmups ovre the lzay dgo."