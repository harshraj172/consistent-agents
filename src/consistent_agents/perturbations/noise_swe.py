import random
import re
from datetime import datetime
from consistent_agents.perturbations.noise import NoisePerturbation

class LLMNoiseSWEPerturbation(NoisePerturbation):
    def __init__(self, seed:int = None, noise_types: list = None, **kwargs):
        super().__init__(self, 
                         noise_types or ['emotion', 'fact', 'question'], seed,
                         **kwargs)
        
    def apply(self, text: str, **kwargs) -> str:
        noise_select = random.choice(self.noise_types)

        lines = re.findall(r'[^.!?]+[.!?]?', text, flags=re.DOTALL)
        emotions = ['happy', 'angry', 'lonely', 'exhausted', 'stoic', 'ecstatic', 'determined']
        today = datetime.today()
        pos = random.randint(0, len(lines))

        if noise_select == "emotion":
            # Inserts an irrelevant statement about mood
            insert_noise = f"\nI am feeling very {random.choice(emotions)} today.\n"
        elif noise_select == "fact":
            # Inserts a random fact within task description
            insert_noise = f"\nToday is {today.strftime("%A, %B %d, %Y")}.\n"
        elif noise_select == "question":
            # Inserts a sudden question within task description
            insert_noise = "\nHow are you doing today?\n"
        else:
            insert_noise = "\nCarpe Diem!\n"

        lines.insert(pos, insert_noise)
        return "".join(lines)
    
if __name__ == "__main__":
    perturbation = LLMNoiseSWEPerturbation(seed=21)
    
    task = """I need a Python function that parses CSV files and extracts specific columns.
    Technical context:
    - Python 3.10+
    - Using standard library only (no pandas)
    - Will process files up to 1GB in size
    """
    perturbed = perturbation.apply(task)
    
    print(f"Original:  {task}")
    print(f"Perturbed: {perturbed}")
        


            
