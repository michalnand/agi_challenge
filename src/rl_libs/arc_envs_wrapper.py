import arc_agi

from .env_wrapper import *

class ARCEnvsWrapper:

    def __init__(self, num_envs = 10, force_envs = None):

        arcade = arc_agi.Arcade()
        available_games = arcade.get_environments()

        if force_envs is None:
            games_ids = []
            for g in available_games:
                game_id = g.game_id
                games_ids.append(str(game_id))

                print(game_id)
        else:
            games_ids = list(force_envs)


        print("total envs count ", len(games_ids))
        
        
        self.envs    = []
        self.env_ids = []

        for n in range(num_envs):
            env_id      = n%len(games_ids)
            
            env_name    = games_ids[env_id]
            env         = EnvWrapper(arcade, env_name, None)
            
            self.envs.append(env)
            self.env_ids.append(env_id)


    def __len__(self):
        return len(self.envs)

    def reset(self, env_id):
        return self.envs[env_id].reset()
    
    def step_one(self, env_id, action):
        return self.envs[env_id].step(action)
    
    def step(self, actions):
        result = []

        for n in range(len(actions)):
            obs, reward, done, info = self.envs[n].step(actions[n])
            result.append((obs, reward, done, info))

        return result
    
    def get_actions(self, env_id):
        return self.envs[env_id].get_actions()
    
    def render(self, env_id):
        return self.envs[env_id].render()
