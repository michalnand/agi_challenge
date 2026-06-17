import torch


class ModelFeatures(torch.nn.Module):
    def __init__(self, input_shape, n_features):
        super(ModelFeatures, self).__init__()

        fc_size = (input_shape[1]//8) * (input_shape[2]//8)

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(), 

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(),    

            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(),    

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(),        

            torch.nn.Flatten(),       

            torch.nn.Linear(128*fc_size, n_features)
        )

        for i in range(len(self.model)):    
            if hasattr(self.model[i], "weight"):
                torch.nn.init.orthogonal_(self.model[i].weight, 0.5)
                torch.nn.init.zeros_(self.model[i].bias)

    def forward(self, x):
        z = self.model(x)
        return z

class ModelMLP(torch.nn.Module):    
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(ModelMLP, self).__init__() 
        
        self.lin_0      = torch.nn.Linear(n_inputs, n_hidden)
        self.act        = torch.nn.SiLU()
        self.lin_1      = torch.nn.Linear(n_hidden, n_outputs)


        torch.nn.init.orthogonal_(self.lin_0.weight, 0.5)
        torch.nn.init.zeros_(self.lin_0.bias)
        torch.nn.init.orthogonal_(self.lin_1.weight, 0.01)
        torch.nn.init.zeros_(self.lin_1.bias)


    def forward(self, x):
        x  = self.lin_0(x)
        x  = self.act(x)
        x  = self.lin_1(x)

        return x



    
'''
    CNN layers for visual information extraction
    FC layers for actor + critic
'''
class ModelRawPixels(torch.nn.Module):
    def __init__(self, input_shape, n_actions, n_pos_actions):
        super(ModelRawPixels, self).__init__()

        n_features      = 512   
        n_hidden        = 512

        self.features         = ModelFeatures(input_shape, n_features)       
        self.critic           = ModelMLP(n_features, n_hidden, 1) 
        
        self.actor_main       = ModelMLP(n_features, n_hidden, n_actions) 
        
        self.actor_pos_x      = ModelMLP(n_features, n_hidden, n_pos_actions) 
        self.actor_pos_y      = ModelMLP(n_features, n_hidden, n_pos_actions) 


    # main RL agent forward func
    def forward(self, state):
        # obtain features   
        z, _ = self.features(state)
     
        # obtain actor and critic outputs
        value       = self.critic(z)

        logits_main = self.actor_main(z)
        logits_x    = self.actor_pos_x(z)
        logits_y    = self.actor_pos_y(z)

        return value, logits_main, logits_x, logits_y
