import arc_agi
import numpy

import cv2

class EnvWrapper:

    def __init__(self, arcade, env_name, render_mode = None, max_steps = 4096):
        self.env = arcade.make(env_name, render_mode=render_mode)

        self.action_space = self.env.action_space

        self.num_actions_x = 16
        self.num_actions_y = 16
        
        self.levels_completed = 0
        self.episode_step     = 0
        self.episode_id       = 0

        self.color_palette = self._generate_distinct_colors(16)
        

    def reset(self):
        self.levels_completed = 0
        self.episode_step     = 0

        self.res = self.env.reset()
        return self._get_obs(self.res), self._get_info(self.res)
    

    def step(self, action):

        action_idx  = action[0]
        x           = action[1]
        y           = action[2]


        action = self.action_space[action_idx]


        self.res = self.env.step(action,  data={"x": x, "y": y})
        
        # numpy observation
        obs = self._get_obs(self.res)

        # if level completed within run, update reward
        reward = self.res.levels_completed - self.levels_completed
        self.levels_completed = int(self.res.levels_completed)

        # check for done
        done = self.res.state != "NOT_FINISHED"

        # update iinfo
        info = self._get_info(self.res)
       
        self.episode_step+= 1
        if done:
            self.episode_id+= 1
        
        return obs, reward, done, info

    def get_actions(self):
        return self.action_space
    

    def render(self):
        obs = self._get_obs(self.res)

        obs = obs[0]

        color_image = self.color_palette[obs]

        scale_factor = 8
        display_image = cv2.resize(color_image, (64 * scale_factor, 64 * scale_factor), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("16-Color Matrix Visualization", display_image)
        cv2.waitKey(1)


    def _get_obs(self, res):
        return numpy.array(res.frame, dtype=int)
    
    def _get_info(self, res):

        info = {}
        info["game_id"]           = str(res.game_id)
        info["episode_step"]      = int(self.episode_step)
        info["episode_id"]        = int(self.episode_id)
        #info["available_actions"] = list(res.available_actions)
        info["available_actions"] = list(self.action_space)
        

        return info
    
    def _generate_distinct_colors_OLD(self, num_colors=16):
        """Generates a list of N well-separated BGR colors using HSV space."""
        colors = []
        for i in range(num_colors):
            # Calculate evenly spaced hue values (OpenCV Hue range is 0-179)
            hue = int((i * 180) / num_colors)
            # Create a 1x1 pixel in HSV: Full saturation and brightness
            hsv_pixel = numpy.uint8([[[hue, 255, 255]]])
            # Convert to BGR (OpenCV's default color space)
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
            colors.append(bgr_pixel[0, 0].tolist())


        colors[5] = [0, 0, 0]

        return numpy.array(colors, dtype=numpy.uint8)

        
    def _generate_distinct_colors(self, num_colors=16):
        """
        Generates a list of N well-separated RGB colors using phase-shifted 
        sine waves with two alternating intensity levels.
        
        Args:
            num_colors (int): Total number of colors to generate (must be even).
            
        Returns:
            np.ndarray: An array of shape (num_colors, 3) containing [R, G, B] values.
        """
        if num_colors % 2 != 0:
            raise ValueError("num_colors must be an even number.")
            
        colors = []
        
        # We divide the total colors by 2 because we use 2 intensity levels per base color
        base_steps = num_colors // 2 
        
        for i in range(num_colors):
            # 1. Calculate the phase (hue angle) based on the base steps
            # This ensures we get maximum separation around the color wheel
            phase = 2.0 * numpy.pi * (i // 2) / base_steps
            
            # 2. Alternating intensity (Level) - Fixed to alternate 0.5 and 1.0
            # level = 0.5 for even indices, 1.0 for odd indices
            level = (i % 2) * 0.5 + 0.5
            
            # 3. Phase-shifted sine waves for R, G, B channels
            # Adjusted offsets slightly to hit pure primaries (Red, Green, Blue) perfectly
            r = 255 * 0.5 * (1 + numpy.cos(phase + 0.0)) * level
            g = 255 * 0.5 * (1 + numpy.cos(phase - 2.0 * numpy.pi / 3.0)) * level
            b = 255 * 0.5 * (1 + numpy.cos(phase + 2.0 * numpy.pi / 3.0)) * level
            
            # Convert to integers safely
            colors.append([int(r), int(g), int(b)])
            
        return numpy.array(colors, dtype=numpy.uint8)

        
