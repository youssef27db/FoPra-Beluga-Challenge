class Env:

    def __init__(self):
        self.state = generate_Problem()
        
        pass
        
    def reset(self):
        pass

    def step(self, action) -> tuple:

        return next_state, reward, done

    def print_info(self):
        pass