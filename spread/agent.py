class Agent:
    def __init__(self, name):
        self.name = name

    def sample(self, obs):
        velocity = obs[self.name][0:2]
        first_entity = obs[self.name][4:6]
        second_entity = obs[self.name][6:8]
        third_entity = obs[self.name][8:10]

        if(abs(first_entity[0])+abs(first_entity[1]) < abs(second_entity[0])+abs(second_entity[1])):
            target_entity = first_entity
        else:
            target_entity = second_entity
        if(abs(third_entity[0])+abs(third_entity[1]) < abs(target_entity[0])+abs(target_entity[1])):
            target_entity = third_entity

        if(abs(velocity[0]) > 0.1 or abs(velocity[1]) > 0.1):
            return ['']
        
        if(abs(target_entity[0]) < abs(target_entity[1])):
            if(target_entity[1] > 0 and abs(target_entity[1]) > 0.1):
                return ['ArrowUp']
            elif(abs(target_entity[1]) > 0.1):
                return ['ArrowDown']
        else:
            if(target_entity[0] > 0 and abs(target_entity[0]) > 0.1):
                return ['ArrowRight']
            elif(abs(target_entity[0]) > 0.1):
                return ['ArrowLeft']
        return ['']
    
    

agents = [Agent('agent_1'), Agent('agent_2')]