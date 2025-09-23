from pettingzoo.mpe import simple_tag_v3

class Environment:
    def reset(self):
        pass
    
    def step(self, action):
        pass

    def render(self):
        pass

environment = simple_tag_v3.parallel_env(num_good=1, num_adversaries=1, max_cycles=500, render_mode="rgb_array")