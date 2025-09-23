from pettingzoo.mpe import simple_spread_v3

class Environment:
    def reset(self):
        pass
    
    def step(self, action):
        pass

    def render(self):
        pass

environment = simple_spread_v3.parallel_env(max_cycles=200, render_mode="rgb_array")