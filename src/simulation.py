"""This module works as a simulation of agent in action

It is made mainly in purpose of presentation. There is an infinite loop of
continuous environment, inside which RL Agent is trying to make his best
decisions. Simulation includes basic graphic indicators made mainly in pyGame.

usage instructions:
    1) start with
        python3 simulation.py
    2) enter a model number to execute
    3) Key      | action
       ---------+-----------
       ESC      | finish
       SPACEBAR | pause
       -        | slow down
       =        | speed up
       z        | toggle chart zoom

NOTE: if you are using multiple monitors, you NEED to specify window
width and height as parameters in Simulations init method, otherwise fullscreen
will span across all monitors.

have fun watching Agent being alive,

RL-for-decision-process team, 2018
"""
import math
import json
import os
from collections import deque
import pygame
import pygame.gfxdraw
from PIL import Image, ImageDraw
from agent import AgentUtils
from agent import Agent
from environment import HouseEnergyEnvironment
from world import World
from tools import truncate

class Simulation:

    def __init__(self, width=None, height=None, model=1):
        """Configuration for simulation object

        This method is divided into two parts, the "view" and the "model",
        roughly resembling view and model responsibilities in MVC model.
        Since pygame doesn't allow complicated design pattern and introduces
        its own event-render-loop mechanism, this is only for clearness.

        Args:
            width(int) = simulation width in pixels.
            height(int) = simulation height in pixels.
            fps(int) = frames per second, which is also the rate of making
                       world steps.
            model(int) = number of model to be used.

        To apply fullscreen, simply leave width and height unmodified to None.
        Using different values is discouraged and could potentially cause
        errors.

        Note: if you are using multiple monitors, you NEED to specify window
        width and height, otherwise fullscreen will span across all monitors.

        """

        pygame.init()
        pygame.display.set_caption("Press ESC to quit, SPACE to pause")

        # --- view settings ---
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        if width and height:
            self.screen = pygame.display.set_mode((width, height),
                                                  pygame.DOUBLEBUF)
        if width and height:
            self.width, self.height = width, height
        else:
            self.width, self.height = pygame.display.get_surface().get_size()

        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.clock = pygame.time.Clock()

        # open configuration file
        add_path = ''
        if 'tests' in os.getcwd():
            add_path = '../'
        with open(add_path + '../configuration.json') as config_file:
            self.config = json.load(config_file)

        self.fps = self.config['simulation']['fps']
        self.font = pygame.font.SysFont('mono', 10, bold=True)
        self.data = dict()
        self.colors = {
            'bg':       pygame.Color('#ececec'),  # lightgrey
            'white':    pygame.Color('#ffffff'),
            'weather1': pygame.Color('#f1e2bb'),  # mellow yellow
            'weather2': pygame.Color('#e2ebd1'),  # pastel light green
            'weather3': pygame.Color('#d0dcdc'),  # pastel light blue
            'weather4': pygame.Color('#b4c4c2'),  # pastel dark blue
            'weather5': pygame.Color('#ddcfb3'),  # i dont remember really
            'font':     pygame.Color('#9b9b9b'),  # medium grey
            'devices1': pygame.Color('#e3dcbb'),  # dirty yellow light
            'devices2': pygame.Color('#ded5ae'),  # ...
            'devices3': pygame.Color('#d6cb98'),
            'devices4': pygame.Color('#ccbe81'),
            'devices5': pygame.Color('#c4b46c'),  # dirty yellow dark
            'devices0': pygame.Color('#f9f9f9'),  # light grey
            'intense1': pygame.Color('#b77d6a'),  # pastel dark red
            'intense2': pygame.Color('#c79b8c'),  # pastel light red
            'soft1':    pygame.Color('#f1e6e2'),  # reddish light grey
            'soft2':    pygame.Color('#e3dcbb'),  # yellowish light grey
        }
        self.margin = 0.025 * self.height

        # --- model settings ---
        self.env = HouseEnergyEnvironment()
        self.agent = Agent(env=self.env)

        AgentUtils.load(self.agent, model)

        self.actions = self.env.get_actions()
        self.current_state = self.env.reset(
            world=World(time_step_in_minutes=1, duration_days=None)
        )

        # memory for charts
        maxlen = 100
        self.memory = {
            'temperature': {
                'values': deque([0] * 100, maxlen=maxlen),
                'desires': deque([0] * 100, maxlen=maxlen)
            },
            'light': {
                'values': deque([0] * 100, maxlen=maxlen),
                'desires': deque([0] * 100, maxlen=maxlen)
            }
        }
        # zooming for charts
        self.zoom = 0

        # dictionary for colorized icon images
        self.icons = {}

    def update_data(self):
        """Updates simulation with data from environment"""
        self.data = self.env.get_current_state()

    def make_world_step(self):
        """Computes world events and moves forward for one timestep"""
        action_index = \
            self.agent.get_next_action_greedy(self.current_state)

        self.current_state = self.env.step(self.actions[action_index])[0]

    def draw_background(self):
        """Draws full-width rectangle"""
        self.screen.blit(self.background, (0, 0))
        pygame.draw.rect(self.screen, self.colors['bg'],
                         (0, 0, self.width, self.height))

    def draw_weather_widget(self):
        """Draws weather widget

        Widget has timer animation and five weather indicators:
        temperature and sun / wind / clouds / rain intensity.

        """

        # bg
        x = y = self.margin
        xmax = self.width * 3 // 7 - self.margin
        ymax = self.height - self.margin
        w = xmax - x
        h = ymax - y

        pygame.draw.rect(self.screen, self.colors['white'], (x, y, w, h))

        # small rects
        for i in range(1, 5):
            pygame.draw.rect(self.screen, self.colors['weather{}'.format(i)],
                             (x, y + (0.5 + i/10) * h, w, 0.11 * h))
        # circle
        radius = 0.19
        circle_center_x = x + 0.5 * w
        circle_center_y = y + 0.22 * h

        cur_time = self.data['Daytime']
        time_color_factor = cur_time / 720 if cur_time <= 720 else\
            2 - cur_time / 720

        color_daytime = pygame.Color(
            truncate(int(time_color_factor * 45 + 235), 0, 255),
            truncate(int(time_color_factor * 30 + 235), 0, 255),
            truncate(int(time_color_factor * 30 + 235), 0, 255)
        )

        for _radius, _color in ((radius + 0.01, self.colors['weather1']),
                                (radius - 0.01, color_daytime)):
            pygame.draw.circle(
                self.screen,
                _color, 
                (int(circle_center_x), int(circle_center_y)),
                int(_radius * w)
            )
        # time indicator (moving circle)
        pygame.draw.rect(
            self.screen, self.colors['weather1'],
            (int(circle_center_x - 0.01 * w),
             int(circle_center_y + radius * h * 0.7),
             0.02 * w, 0.035 * h)
        )
        daytime = self.data['Daytime']
        _phi = (daytime / 1440) * 2 * math.pi + math.pi / 2
        x_indicator = radius * math.cos(_phi)
        y_indicator = radius * math.sin(_phi)
        radius_indicator = 0.03

        for _radius, _color in ((radius_indicator, 'weather1'),
                                (radius_indicator - 0.01, 'weather5')):
            pygame.draw.circle(
                self.screen,
                self.colors[_color],
                (int(circle_center_x + x_indicator * w),
                 int(circle_center_y + y_indicator * w)),
                int(_radius * w)
            )
        # text - clock
        font_mono = pygame.font.Font(
            '../static/fonts/droid-sans-mono/DroidSansMono.ttf',
            int(0.05 * h)
        )
        color_daytime_clock = pygame.Color(
            truncate(200 - int(time_color_factor * (200 - 221)), 0, 255),
            truncate(200 - int(time_color_factor * (200 - 207)), 0, 255),
            truncate(200 - int(time_color_factor * (200 - 179)), 0, 255),
        )
        
        font_header = pygame.font.Font('../static/fonts/Lato/Lato-Regular.ttf',
                                       int(0.09 * h))

        time = [int(x) for x in divmod(daytime, 60)]
        time = "{:02}:{:02}".format(*time)
        self.draw_text(time, circle_center_x, circle_center_y,
                       color_daytime_clock, font_header, True)
        # text - blocks
        for _off, _data in enumerate(('Outside Light', 'Wind',
                                      'Clouds', 'Rain')):
            _label = _data.upper()
            if _data == 'Outside Light':
                _label = 'SUN'

            self.draw_text("{:<13}{:>5.0f}%".format(
                _label, self.data[_data] * 100),
                x + 0.57 * w, y + (0.65 + _off / 10) * h,
                self.colors['font'], font_mono, True)

        # text - temperature
        self.draw_text("{:<3.1f}°C".format(
                            self.data['Outside Temp']
                       ), x + 0.55 * w, y + 0.5 * h,
                       self.colors['weather5'], font_header, True)

        # weather icons
        for _off, _data in enumerate(('016-sun', '013-wind',
                                     '015-cloud', '010-raining')):
            self.draw_icon('../static/icons/weather/{}.png'.format(_data),
                           x + 0.1 * w, y + (0.65 + _off / 10) * h,
                           0.08 * h, 0.08 * h, self.colors['font'], True)

        self.draw_icon('../static/icons/weather/003-temperature.png',
                       x + 0.3 * w, y + 0.49 * h,
                       0.1 * h, 0.1 * h, self.colors['weather5'], True)

        # day / night icon
        if 6 * 60 < daytime < 19 * 60:
            daynight_icon = '../static/icons/weather/016-sun.png'
            daynight_color = self.colors['weather1']
        else:
            daynight_icon = '../static/icons/weather/004-moon.png'
            daynight_color = self.colors['weather3']

        self.draw_icon(daynight_icon,
                       x + 0.13 * w, y + 0.1 * h,
                       0.13 * h, 0.13 * h, daynight_color, True)

    def draw_devices_widget(self):
        """Draws devices wigdet.
        This wigdet has devices indicators informing the user about current
        device states.

        """

        # bg
        x = self.width * 3 // 7
        y = (self.height - 2 * self.margin) * 0.6 + self.margin
        xmax = self.width - self.margin
        ymax = self.height - self.margin
        w = xmax - x
        h = ymax - y
        pygame.draw.rect(self.screen, self.colors['white'], (x, y, w, h))

        font_small = pygame.font.Font('../static/fonts/Lato/Lato-Regular.ttf',
                                      int(0.05 * h))
        font_big = pygame.font.Font('../static/fonts/Lato/Lato-Regular.ttf',
                                    int(0.13 * h))

        def draw_indicator(data, x, y, w, h, name=None):
            for offset in range(9, -1, -1):
                clr = str((9 - offset) // 2 + 1)
                if data > (9 - offset) / 10:
                    # colourful rect
                    pygame.draw.rect(self.screen, self.colors['devices' + clr],
                                     (x, y + offset * h / 10, w, 0.095 * h))
                else:
                    # white rect
                    pygame.draw.rect(self.screen, self.colors['devices0'],
                                     (x, y + offset * h / 10, w, 0.095 * h))

            if not name:
                return

            self.draw_text(name, x + 0.5 * w, y + h + 0.1 * h,
                           self.colors['font'], font_small, True)

        x_begin = 0.35
        indicators = {
            x_begin: ('heating', '001-heater'),
            x_begin + 0.15: ('cooling', '002-machine'),
            x_begin + 0.30: ('light', '008-light-bulb'),
            x_begin + 0.45: ('curtains', '003-blinds'),
        }

        for x_o, ind in indicators.items():
            draw_indicator(self.data['{}_lvl'.format(ind[0])],
                           x + x_o * w, y + 0.1 * h,
                           0.1 * w, 0.55 * h, "{}".format(ind[0]).upper())
            # indicator icon
            self.draw_icon('../static/icons/house/{}.png'.format(ind[1]),
                           x + x_o * w + 0.05 * w, y + 0.85 * h,
                           0.08 * w, 0.08 * w, self.colors['font'], True)

        scale_x = 0.94
        for offset in range(0, 12, 2):
            self.draw_text('{}'.format(offset / 10),
                           scale_x * w + x,
                           0.65 * h + y - offset / 18 * h,
                           self.colors['font'], font_small, True)

        # energy indicator
        x_energy = 0.08
        self.draw_icon('../static/icons/house/006-electric-tower.png',
                       x + x_energy * w + 0.05 * w, y + 0.25 * h,
                       0.08 * w, 0.08 * w, self.colors['font'], True)
        self.draw_icon('../static/icons/house/004-battery.png',
                       x + x_energy * w + 0.05 * w, y + 0.53 * h,
                       0.08 * w, 0.08 * w, self.colors['font'], True)
        self.draw_icon('../static/icons/house/005-renewable-energy.png',
                       x + x_energy * w + 0.05 * w, y + 0.85 * h,
                       0.07 * w, 0.07 * w, self.colors['font'], True)
        self.draw_text("SOURCE",
                       x + x_energy * w + 0.05 * w, y + 0.705 * h,
                       self.colors['font'], font_small, True)

        triangle_y = 0.5
        if self.data['energy_src'] == 'grid':
            triangle_y = 0.2

        pygame.draw.polygon(
            self.screen,
            self.colors['devices5'],
            [
                [x + x_energy / 2 * w, triangle_y * h + y],
                [x + x_energy / 2 * w + 0.03 * w, (triangle_y + 0.05) * h + y],
                [x + x_energy / 2 * w, (triangle_y + 0.1) * h + y],
            ]
        )
        battery = self.data['battery_level'] / self.data['battery_max']
        self.draw_text('{:2.0f}%'.format(battery * 100),
                       x + (x_energy + 0.16) * w,
                       y + 0.54 * h,
                       self.colors['font'], font_big, True)

    def draw_chart_widget(self, chartmode='light', y=0):
        """Draws light and temperature charts

        Those charts are presenting latest 100 values in time, along with
        users requested value. Charts can be zoomed in to show
        range (-10%, +10%) of total range based on requested value

        Args:
            chartmode(string): 'light' or 'temperature'
            y(int): y position on screen

        """
        # bg
        x = self.width * 3 // 7
        xmax = self.width - self.margin
        w = xmax - x
        h = 0.15 * self.height - self.margin
        pygame.draw.rect(self.screen, self.colors['white'], (x, y, w, h))

        font_small = pygame.font.Font('../static/fonts/Lato/Lato-Regular.ttf',
                                      int(0.15 * h))

        # chartmode options
        if chartmode == 'light':
            soft_color = 'soft2'
            level = self.data['first.light']
            desired = self.data['light_desired']

        elif chartmode == 'temperature':
            soft_color = 'soft1'
            level = (self.data['first.temperature'] + 20) / 60
            desired = (self.data['temp_desired'] + 20) / 60
        else:
            raise AttributeError('wrong chartmode')
        # title
        self.draw_text(
            chartmode.upper() + " CHART",
            x + 0.02 * w, y + 0.05 * h,
            self.colors['font'], font_small
        )
        # chart
        chartx = x + 0.02 * w
        charty = y + 0.3 * h
        chartw = 0.96 * w
        charth = 0.55 * h
        pygame.draw.rect(
            self.screen,
            self.colors['white'],
            (chartx, charty, chartw, charth)
        )

        self.memory[chartmode]['values'].append(level)
        self.memory[chartmode]['desires'].append(desired)

        offset = math.ceil(chartw / 100)
        xs = list(range(int(chartx), int(chartx + chartw), offset))
        if not self.zoom:
            ys_values = [charty - 2 + charth - v * charth for v in
                         self.memory[chartmode]['values']]
            ys_desires = [charty - 2 + charth - v * charth for v in
                          self.memory[chartmode]['desires']]
        else:
            deltas = [
                max(min(v - d, 0.1), -0.1) for
                v, d in zip(self.memory[chartmode]['values'],
                            self.memory[chartmode]['desires'])
            ]
            ys_values = [charty - 2 + charth - (d * 5 + 0.5) * charth for d in
                         deltas]
            ys_desires = [charty - 2 + 0.5 * charth for v in
                          self.memory[chartmode]['desires']]

        points_des = list(zip(xs, ys_desires))

        points_val = list(zip(xs, ys_values))
        points_val.append((chartx + chartw, charty + charth))  # for nice area
        points_val.insert(0, (chartx, charty + charth))

        pygame.draw.polygon(
            self.screen,
            self.colors[soft_color],
            points_val
        )
        pygame.draw.aalines(
            self.screen,
            self.colors['weather4'],
            False,
            points_des,
            2
        )
        # legend
        self.draw_text(
            'CURRENT ',
            x + 0.3 * w, y + 0.05 * h,
            self.colors[soft_color], font_small
        )
        pygame.draw.rect(self.screen, self.colors[soft_color],
                         (x + 0.27 * w, y + 0.07 * h, 0.02 * w, 0.15 * h))
        self.draw_text(
            'DESIRED ',
            x + 0.45 * w, y + 0.05 * h,
            self.colors['weather4'], font_small
        )
        pygame.draw.rect(self.screen, self.colors['weather4'],
                         (x + 0.42 * w, y + 0.07 * h, 0.02 * w, 0.15 * h))
        if self.zoom:
            self.draw_text(
                'ZOOMED',
                x + 0.9 * w, y + 0.05 * h,
                self.colors['weather4'], font_small
            )

    def draw_speedmeter_widget(self, chartmode='light', x=0, y=0):
        """Draws gauge-like indicator for light or temperature

        Args:
            chartmode(string): 'light' or 'temperature'
            x(int): x position on screen
            y(int): y position on screnn

        """
        # chartmode options
        if chartmode == 'light':
            main_color = 'weather1'
            scnd_color = 'weather2'
            level = self.data['first.light'] * 100
            desired = self.data['light_desired'] * 100
            level_normalized = level * 180 / 100
            desired_normalized = desired * 180 / 100
            lvl_format = "{:.0f}%"

        elif chartmode == 'temperature':
            main_color = 'intense1'
            scnd_color = 'intense2'
            level = self.data['first.temperature']
            level_normalized = (level + 20) * 180 / 60
            desired = self.data['temp_desired']
            desired_normalized = (desired + 20) * 180 / 60
            lvl_format = "{:.1f}°C"

        else:
            raise AttributeError('wrong chartmode')

        color_1 = (
            self.colors[main_color].r,
            self.colors[main_color].g,
            self.colors[main_color].b
        )
        color_2 = (
            self.colors[scnd_color].r,
            self.colors[scnd_color].g,
            self.colors[scnd_color].b
        )
        white = (255, 255, 255)
        grey = (250, 250, 250)

        w = 0.2875 * self.width - self.margin
        h = 0.292 * self.height - self.margin

        # bg
        pygame.draw.rect(self.screen, self.colors['white'], (x, y, w, h))

        font_small = pygame.font.Font('../static/fonts/Lato/Lato-Regular.ttf',
                                      int(0.1 * h))
        font_big = pygame.font.Font('../static/fonts/Lato/Lato-Regular.ttf',
                                    int(0.2 * h))
        # title
        self.draw_text(
            chartmode.upper() + " INSIDE",
            x + 0.05 * w, y + 0.05 * h,
            self.colors['font'], font_small
        )

        # arc
        radius = int(h * 2 / 3)
        ccenter_x = int(x + w / 2 - radius)
        ccenter_y = int(y + h * 2 / 3 - radius * 2 / 3)

        # - generate PIL image -
        pil_size = radius * 2
        pil_image = Image.new("RGBA", (pil_size, pil_size))
        pil_draw = ImageDraw.Draw(pil_image)
        arcwidth = int(0.12 * radius)

        # haha those pieslices are nice
        # who cares, nobody even reads this shit
        pil_draw.pieslice((0, 0, pil_size - 1, pil_size - 1),
                          0, 180, fill=grey)
        pil_draw.pieslice((0, 0, pil_size - 1, pil_size - 1),
                          0, level_normalized, fill=color_1)
        pil_draw.pieslice(
            (arcwidth - 1, 0, pil_size - arcwidth, pil_size - arcwidth),
            0, 180,
            fill=grey
        )
        pil_draw.pieslice(
            (arcwidth - 1, 0, pil_size - arcwidth, pil_size - arcwidth),
            0, desired_normalized,
            fill=color_2
        )
        pil_draw.pieslice(
            ((arcwidth * 2) - 1, 0,
             pil_size - (arcwidth * 2), pil_size - (arcwidth * 2)),
            0, 180,
            fill=white
        )
        pil_draw.rectangle(
            [0, pil_size / 2, pil_size, pil_size / 2 - arcwidth],
            fill=white
        )
        # - convert to PyGame image -
        mode = pil_image.mode
        size = pil_image.size
        data = pil_image.tobytes()

        image = pygame.image.fromstring(data, size, mode)
        image = pygame.transform.rotate(image, 180)
        self.screen.blit(image, (ccenter_x, ccenter_y))

        # text
        self.draw_text(
            lvl_format.format(level),
            x + w / 2, y + h * 2 / 3,
            self.colors[main_color], font_big, True
        )
        self.draw_text(
            "should be " + lvl_format.format(desired),
            x + w / 2, y + h * 2 / 3 + 0.15 * h,
            self.colors[scnd_color], font_small, True
        )

    def run(self):
        """Main events / render loop"""
        running = True
        pause = 0
        while running:
            # EVENTS -------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_SPACE:
                        pause = abs(1 - pause)
                    if event.key == pygame.K_EQUALS:
                        if self.fps < 40:
                            self.fps += 1
                    if event.key == pygame.K_MINUS:
                        if self.fps > 1:
                            self.fps -= 1
                    if event.key == pygame.K_z:
                        self.zoom = 1 - self.zoom

            # LOOP ------
            if pause:
                continue

            self.make_world_step()
            self.update_data()

            # RENDER -----
            pygame.display.flip()  # this means update all
            self.draw_background()
            self.draw_weather_widget()
            self.draw_devices_widget()
            self.draw_chart_widget(
                'temperature',
                y=(0.95 * self.height) * 0.325 + 0.0025 * self.height
            )
            self.draw_chart_widget(
                'light',
                y=(0.95 * self.height) * 0.475 + 0.0025 * self.height
            )
            self.draw_speedmeter_widget(
                'temperature',
                x=self.width * 3 // 7,
                y=0.025 * self.height
            )
            self.draw_speedmeter_widget(
                'light',
                x=self.width * 3 // 7 + 0.27 * self.width + self.margin,
                y=0.025 * self.height
            )

            # apply upper limit on fps
            self.clock.tick(self.fps)

        pygame.quit()

    def draw_text(self, text='', x=0, y=0, color=(0, 0, 0), font=None,
                  centered=False):
        """A shortcut method to draw text on screen"""
        if not font:
            font = self.font
        surface = font.render(text, True, color)

        text_width, text_height = 0, 0
        if centered:
            text_width, text_height = font.size(text)
        self.screen.blit(surface, (x - text_width // 2, y - text_height // 2))

    def fill(self, surface, color):
        """Fill all pixels of the surface with color, preserve transparency."""
        w, h = surface.get_size()
        r, g, b = color[:3]

        for x in range(w):
            for y in range(h):
                a = surface.get_at((x, y))[3]
                surface.set_at((x, y), pygame.Color(r, g, b, a))

    def draw_icon(self, filename, x=0, y=0, width=32, height=32,
                  color=(0, 0, 0), centered=False):
        """Draws icon on screen

        Fills icon with color. At first, loads icon from memory; on next
        iterations, loads icon as python object from dictionary to optimise
        i/o time.

        """

        try:
            image = self.icons[filename]
        except KeyError:
            try:
                image = pygame.image.load(filename).convert_alpha()
            except pygame.error:
                print("Cannot find icon image:{}".format(filename))

            image = pygame.transform.scale(image, (int(width), int(height)))
            self.fill(image, color)
            self.icons[filename] = image  # save for later use

        if centered:
            x -= width // 2
            y -= height // 2
        self.screen.blit(image, (x, y))


if __name__ == '__main__':
    try:
        model = int(input("Enter model number to execute: "))
    except ValueError:
        print('model number should be an integer')
    Simulation(model=model).run()
