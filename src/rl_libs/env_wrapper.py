import arc_agi
import numpy

import cv2

class EnvWrapper:

    def __init__(self, arcade, env_name, render_mode = None, max_steps = 1000):
        self.env = arcade.make(env_name, render_mode=render_mode)

        self.action_space = self.env.action_space
        
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

        print(">>>> ", obs.shape)

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
    
    def _generate_distinct_colors(self, num_colors=16):
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

        return numpy.array(colors, dtype=numpy.uint8)

    
