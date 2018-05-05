import os
import sys
import math
import torch
import pygame
import pygame.gfxdraw
from PIL import Image, ImageDraw
# import array
from datetime import datetime, timedelta
from main import load_model
from agent import Agent
from environment import HouseEnergyEnvironment
from world import World

class Simulation:

    def __init__(self, width=None, height=None, fps=10):
        pygame.init()
        pygame.display.set_caption("Press ESC to quit, SPACE to pause")

        # view settings
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        if width and height:
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF)
        self.width, self.height = pygame.display.get_surface().get_size()

        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.playtime = 0.0
        self.font = pygame.font.SysFont('mono', 10, bold=True)
        self.data = dict()
        self.colors = {
            'bg': pygame.Color('#ecececff'),
            'white': pygame.Color('#ffffffff'),
            'weather1': pygame.Color('#f1e2bbff'),
            'weather2': pygame.Color('#e2ebd1ff'),
            'weather3': pygame.Color('#d0dcdcff'),
            'weather4': pygame.Color('#b4c4c2ff'),
            'weather5': pygame.Color('#ddcfb3ff'),
            'font1': pygame.Color('#9b9b9bff'),
            'devices1': pygame.Color('#e3dcbbff'),
            'devices2': pygame.Color('#ded5aeff'),
            'devices3': pygame.Color('#d6cb98ff'),
            'devices4': pygame.Color('#ccbe81ff'),
            'devices5': pygame.Color('#c4b46cff'),
            'devices0': pygame.Color('#f9f9f9'),
            'intense1': pygame.Color('#b77d6aff'),
            'intense2': pygame.Color('#c79b8cff'),
        }

        # model settings
        self.env = HouseEnergyEnvironment()
        self.agent = Agent(env=self.env)

        # model_id = input('Enter model number to load\n')
        model_id = 1  # FIXME const model nr
        load_model(self.agent, model_id)

        self.actions = self.env.get_actions()
        self.current_state = self.env.reset(world=World(time_step_in_minutes=1, duration_days=None))

    def update_data(self):
        labels, values = self.env.render[:-1]
        labels = [label.strip().strip(':') for label in labels]
        self.data = dict(zip(labels, values))

    def update_playtime(self):
        milliseconds = self.clock.tick(self.fps)
        self.playtime += milliseconds / 1000.0

    def make_world_step(self):
        action_index = \
            self.agent.get_next_action_greedy(self.current_state)

        self.current_state = self.env.step(self.actions[action_index])[0]

    def draw_background(self):
        self.screen.blit(self.background, (0, 0))
        pygame.draw.rect(self.screen, self.colors['bg'],
                         (0, 0, self.width, self.height))

    def draw_weather_widget(self):
        # bg
        margin = 0.025 * self.height
        x = margin
        y = margin
        xmax = self.width * 3 // 7 - margin
        ymax = self.height - margin
        w = xmax - x
        h = ymax - y
        pygame.draw.rect(self.screen, self.colors['white'], (x ,y, w, h))

        # small rects
        pygame.draw.rect(self.screen, self.colors['weather1'],
                         (x, y + 0.6 * h, w, 0.11 * h))
        pygame.draw.rect(self.screen, self.colors['weather2'],
                         (x, y + 0.7 * h, w, 0.11 * h))
        pygame.draw.rect(self.screen, self.colors['weather3'],
                         (x, y + 0.8 * h, w, 0.11 * h))
        pygame.draw.rect(self.screen, self.colors['weather4'],
                         (x, y + 0.9 * h, w, 0.1 * h))

        # circle
        radius = 0.19
        circle_center_x = x + 0.5 * w
        circle_center_y = y + 0.22 * h

        pygame.draw.circle(
            self.screen,
            self.colors['weather1'],
            (int(circle_center_x), int(circle_center_y)),
            int((radius + 0.01) * w)
        )
        pygame.draw.circle(
            self.screen,
            self.colors['white'],
            (int(circle_center_x), int(circle_center_y)),
            int((radius - 0.01) * w)
        )
        # time indicator (moving circle)
        pygame.draw.rect(self.screen, self.colors['weather1'],
                         (int(circle_center_x - 0.01 * w),
                          int(circle_center_y + radius * h * 0.7),
                          0.02 * w, 0.035 * h))

        daytime = self.data['Daytime //OUTSIDE']
        fi = (daytime / 1440) * 2 * math.pi + math.pi / 2
        x_indicator = radius * math.cos(fi)
        y_indicator = radius * math.sin(fi)
        radius_indicator = 0.03

        pygame.draw.circle(
            self.screen,
            self.colors['weather1'],
            (int(circle_center_x + x_indicator * w),
             int(circle_center_y + y_indicator * w)),
            int((radius_indicator) * w)
        )
        pygame.draw.circle(
            self.screen,
            self.colors['weather5'],
            (int(circle_center_x + x_indicator * w),
             int(circle_center_y + y_indicator * w)),
            int((radius_indicator - 0.01) * w)
        )
        # text - clock
        fontMono = pygame.font.Font(
            '../fonts/droid-sans-mono/DroidSansMono.ttf',
            int(0.05 * h)
        )
        fontHeader = pygame.font.Font('../fonts/Lato/Lato-Regular.ttf',
                                      int(0.09 * h))

        time = [int(x) for x in divmod(daytime, 60)]
        time = "{:02}:{:02}".format(*time)
        self.draw_text(time, circle_center_x, circle_center_y,
                       self.colors['weather5'], fontHeader, True)

        # text - blocks
        self.draw_text("{:<13}{:0<5.3f}".format(
                            "SUN",
                            self.data['Light OUT']
                       ), x + 0.57 * w, y + 0.65 * h,
                       self.colors['font1'], fontMono, True)
        self.draw_text("{:<13}{:0<5.3f}".format(
                            "WIND",
                            self.data['Wind']
                       ), x + 0.57 * w, y + 0.75 * h,
                       self.colors['font1'], fontMono, True)
        self.draw_text("{:<13}{:0<5.3f}".format(
                            "CLOUDS",
                            self.data['Clouds']
                       ), x + 0.57 * w, y + 0.85 * h,
                       self.colors['font1'], fontMono, True)
        self.draw_text("{:<13}{:0<5.3f}".format(
                            "RAIN",
                            self.data['Rain']
                       ), x + 0.57 * w, y + 0.95 * h,
                       self.colors['font1'], fontMono, True)
        # text - temperature
        self.draw_text("{:<3.1f}°C".format(
                            self.data['Temperature_outside']
                       ), x + 0.55 * w, y + 0.5 * h,
                       self.colors['weather5'], fontHeader, True)

        # weather icons
        self.draw_icon('../icons/weather/016-sun.png',
                       x + 0.1 * w, y + 0.65 * h,
                       0.08 * h, 0.08 * h, self.colors['font1'], True)
        self.draw_icon('../icons/weather/013-wind.png',
                       x + 0.1 * w, y + 0.75 * h,
                       0.08 * h, 0.08 * h, self.colors['font1'], True)
        self.draw_icon('../icons/weather/015-cloud.png',
                       x + 0.1 * w, y + 0.85 * h,
                       0.08 * h, 0.08 * h, self.colors['font1'], True)
        self.draw_icon('../icons/weather/010-raining.png',
                       x + 0.1 * w, y + 0.95 * h,
                       0.08 * h, 0.08 * h, self.colors['font1'], True)

        self.draw_icon('../icons/weather/003-temperature.png',
                       x + 0.3 * w, y + 0.49 * h,
                       0.1 * h, 0.1 * h, self.colors['weather5'], True)

        # day / night icon
        if 6 * 60 < daytime < 19 * 60:
            daynight_icon = '../icons/weather/016-sun.png'
            daynight_color = self.colors['weather1']
        else:
            daynight_icon = '../icons/weather/004-moon.png'
            daynight_color = self.colors['weather3']

        self.draw_icon(daynight_icon,
                       x + 0.13 * w, y + 0.1 * h,
                       0.13 * h, 0.13 * h, daynight_color, True)

    def draw_devices_widget(self):
        # bg
        margin = 0.025 * self.height
        x = self.width * 3 // 7
        y = (self.height - 2 * margin) * 0.6 + margin
        xmax = self.width - margin
        ymax = self.height - margin
        w = xmax - x
        h = ymax - y
        pygame.draw.rect(self.screen, self.colors['white'], (x, y, w, h))

        fontSmall = pygame.font.Font('../fonts/Lato/Lato-Regular.ttf',
                                      int(0.05 * h))
        fontBig = pygame.font.Font('../fonts/Lato/Lato-Regular.ttf',
                                      int(0.13 * h))

        def draw_indicator(data, x, y, w, h, name=None):
            for offset in range(8, -2, -2):
                pygame.draw.rect(self.screen, self.colors['devices0'],
                                 (x, y + offset * h / 10, w, 0.19 * h))

            if data > 0:
                pygame.draw.rect(self.screen, self.colors['devices1'],
                                 (x, y + 0.8 * h, w, 0.19 * h))
            if data > 0.2:
                pygame.draw.rect(self.screen, self.colors['devices2'],
                                 (x, y + 0.6 * h, w, 0.19 * h))
            if data > 0.4:
                pygame.draw.rect(self.screen, self.colors['devices3'],
                                 (x, y + 0.4 * h, w, 0.19 * h))
            if data > 0.6:
                pygame.draw.rect(self.screen, self.colors['devices4'],
                                 (x, y + 0.2 * h, w, 0.19 * h))
            if data > 0.8:
                pygame.draw.rect(self.screen, self.colors['devices5'],
                                 (x, y + 0.0 * h, w, 0.19 * h))

            if not name:
                return

            self.draw_text(name, x + 0.5 * w, y + h + 0.1 * h,
                           self.colors['font1'], fontSmall, True)

        x_begin = 0.35
        indicators = {
            x_begin: ('Heating', '001-heater'),
            x_begin + 0.15: ('Cooling', '002-machine'),
            x_begin + 0.30: ('Light', '008-light-bulb'),
            x_begin + 0.45: ('Curtains', '003-blinds'),
        }

        for x_o, ind in indicators.items():
            draw_indicator(self.data['{}_lvl'.format(ind[0])],
                           x + x_o * w, y + 0.1 * h,
                           0.1 * w, 0.55 * h, "{}".format(ind[0]).upper())
            # indicator icon
            self.draw_icon('../icons/house/{}.png'.format(ind[1]),
                           x + x_o * w + 0.05 * w, y + 0.85 * h,
                           0.08 * w, 0.08 * w, self.colors['font1'], True)

        scale_x = 0.94
        for offset in range(0, 12, 2):
            self.draw_text('{}'.format(offset / 10),
                           scale_x * w + x,
                           0.65 * h + y - offset / 18 * h,
                           self.colors['font1'], fontSmall, True)

        # energy indicators
        x_energy = 0.08
        self.draw_icon('../icons/house/006-electric-tower.png',
                       x + x_energy * w + 0.05 * w, y + 0.25 * h,
                       0.08 * w, 0.08 * w, self.colors['font1'], True)
        self.draw_icon('../icons/house/004-battery.png',
                       x + x_energy * w + 0.05 * w, y + 0.53 * h,
                       0.08 * w, 0.08 * w, self.colors['font1'], True)
        self.draw_text("SOURCE",
                       x + x_energy * w + 0.05 * w, y + 0.705 * h,
                       self.colors['font1'], fontSmall, True)
        self.draw_icon('../icons/house/005-renewable-energy.png',
                       x + x_energy * w + 0.05 * w, y + 0.85 * h,
                       0.07 * w, 0.07 * w, self.colors['font1'], True)

        triangle_y = 0.5
        if self.data['Energy_src'] == 'grid':
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
        batt = self.data['Battery_lvl'] / self.env.house.battery['max']
        self.draw_text('{:2.0f}%'.format(batt * 100),
                       x + (x_energy + 0.16) * w,
                       y + 0.54 * h,
                       self.colors['font1'], fontBig, True)

    def draw_chart_widget(self, mode='light', y=0):
        # bg
        margin = 0.025 * self.height
        x = self.width * 3 // 7
        xmax = self.width - margin
        w = xmax - x
        h = 0.15 * self.height - margin
        ymax = y + h 
        pygame.draw.rect(self.screen, self.colors['white'], (x, y, w, h))

        fontSmall = pygame.font.Font('../fonts/Lato/Lato-Regular.ttf',
                                      int(0.05 * h))
        fontBig = pygame.font.Font('../fonts/Lato/Lato-Regular.ttf',
                                      int(0.13 * h))

    def draw_speedmeter_widget(self, mode='light', x=0, y=0):
        # bg
        margin = 0.025 * self.height
        w = 0.2875 * self.width - margin
        h = 0.292 * self.height - margin
        ymax = y + h 
        xmax = x + w
        pygame.draw.rect(self.screen, self.colors['white'], (x, y, w, h))

        fontSmall = pygame.font.Font('../fonts/Lato/Lato-Regular.ttf',
                                      int(0.1 * h))
        fontBig = pygame.font.Font('../fonts/Lato/Lato-Regular.ttf',
                                      int(0.2 * h))
        # title
        self.draw_text(mode.upper() + " INSIDE", x + 0.05 * w, y + 0.05 * h,
                       self.colors['font1'], fontSmall)
        # text
        if mode == 'light':
            data = "{:.3f}".format(self.data['Light IN'])
            request = 'Light_desired'
        elif mode == 'temperature':
            data = "{:.1f}°C".format(self.data['Temperature //INSIDE'])
            request = 'Temp_desired'

        self.draw_text(data, x + w / 2, y + h * 2 / 3 ,
                       self.colors['font1'], fontBig, True)

        ccenter_x = int(x + w / 2)
        ccenter_y = int(y + h * 2 / 3)
        radius = int(h * 2 / 3)

        # pygame.draw.circle(
            # self.screen,
            # self.colors['intense1'],
            # (ccenter_x, ccenter_y),
            # radius
        # )

        # # - generate PIL image -

        # pil_size = 300

        # pil_image = Image.new("RGBA", (pil_size, pil_size))
        # pil_draw = ImageDraw.Draw(pil_image)
        # pil_draw.arc((0, 0, pil_size-1, pil_size-1), 0, 270, fill=RED)
        # GREY  = (128, 128, 128)
        # pil_draw.pieslice((0, 0, pil_size-1, pil_size-1), 0, 60, fill=GREY)
        # pil_draw.rotate(45)

        # - convert to PyGame image -

        # mode = pil_image.mode
        # size = pil_image.size
        # data = pil_image.tobytes()

        # image = pygame.image.fromstring(data, size, mode)
        # image_rect = image.get_rect(center=self.screen.get_rect().center)
        # self.screen.blit(image, image_rect)


        # pygame.gfxdraw.arc(
            # self.screen,
            # x + 0.2 * w, y + 0.3 * h, w * 0.6, h),
            # 0, math.pi,
            # self.colors['intense1'],
        # )



    def run(self):
        """The mainloop"""
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

            # LOOP ------
            if pause:
                continue

            self.update_playtime()
            self.make_world_step()
            self.update_data()

            # RENDER -----

            pygame.display.flip()  # this means update all
            self.draw_background()
            self.draw_weather_widget()
            self.draw_devices_widget()
            self.draw_chart_widget(
                'light',
                y=(0.95 * self.height) * 0.475 + 0.0025 * self.height
            )
            self.draw_chart_widget(
                'temp',
                y=(0.95 * self.height) * 0.325 + 0.0025 * self.height
            )
            self.draw_speedmeter_widget(
                'temperature',
                x=self.width * 3 // 7,
                y=0.025 * self.height
            )
            self.draw_speedmeter_widget(
                'light',
                x=self.width * 3 // 7 + 0.27 * self.width + 0.025 * self.height,
                y=0.025 * self.height
            )

        pygame.quit()

    def draw_text(self, text, x=0, y=0, color=(0, 0, 0), font=None, centered=False):
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

    def draw_icon(self, filename, x=0, y=0, width=32, height=32, color=(0, 0, 0), centered=False):
        # FIXME: icons are colorized every frame, it should be done once
        image = pygame.image.load(filename).convert_alpha()
        image = pygame.transform.scale(image, (int(width), int(height)))
        self.fill(image, color)
        if centered:
            x -= width // 2
            y -= height // 2
        self.screen.blit(image, (x, y))


if __name__ == '__main__':
    # Simulation(800, 600, fps=10).run()
    Simulation(fps=1000).run()
