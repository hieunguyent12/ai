import pygame
import numpy as np
import math


class Canvas:
    def __init__(self, screen, nn):
        self.pixels = [0] * 784
        self.points = []
        self.screen = screen
        self.pixel_size = 15
        self.nn = nn
        self.holding = False

    def listen(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.holding = True
                self.draw()
            elif event.type == pygame.MOUSEBUTTONUP:
                self.holding = False
            elif event.type == pygame.MOUSEMOTION:
                if self.holding:
                    # print(event.pos)
                    self.draw()

    def render(self):
        top = 0
        margin = 1
        self.pixels = [0] * 784
        for x, y in self.points:
            for i, pixel in enumerate(self.pixels):
                tileX = i % 28
                tileY = i // 28
                dist = math.hypot(tileX - x, tileY - y)
                penValue = 0.8 - ((dist / 2) ** 2)
                penValue = min(max(0, penValue), 1)
                prev = pixel
                self.pixels[i] = prev + (1 - prev) * penValue

        for i, pixel in enumerate(self.pixels):
            t = (i % 28) * margin
            left = ((i % 28) + 1) * self.pixel_size + t
            top = top + self.pixel_size + margin if i % 28 == 0 else top
            rect = [left, top, self.pixel_size, self.pixel_size]
            s = pygame.Surface((15, 15))
            s.set_alpha(pixel * 255 + 10)
            s.fill((255, 255, 255))
            self.screen.blit(s, rect)

    def render_reset_button(self, events, text="reset"):
        font_size = 25
        font = pygame.font.Font(None, font_size)
        text = font.render(text, True, "black")

        width = text.get_size()[0] + 26
        height = text.get_size()[1] + 26
        surface = pygame.Surface((width, height))
        surface.fill("white")
        surface.blit(text, (13, 13))
        rect = surface.get_rect(center=(15 + width // 2, 450 + height))
        self.screen.blit(surface, rect)
        a = 1
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if rect.collidepoint(event.pos):
                    self.points = []

    def render_classify_button(self, events, text="classify"):
        font_size = 25
        font = pygame.font.Font(None, font_size)
        text = font.render(text, True, "black")

        width = text.get_size()[0] + 26
        height = text.get_size()[1] + 26
        surface = pygame.Surface((width, height))
        surface.fill("white")
        surface.blit(text, (13, 13))
        rect = surface.get_rect(center=(15 + (width // 2) + 220, 450 + height))
        self.screen.blit(surface, rect)

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if rect.collidepoint(event.pos):
                    results = self.nn.feedforward([np.array(self.pixels)])
                    print(np.argmax(results))

    def render_preprocess_button(self, events, text="preprocessing"):
        font_size = 25
        font = pygame.font.Font(None, font_size)
        text = font.render(text, True, "black")

        width = text.get_size()[0] + 26
        height = text.get_size()[1] + 26
        surface = pygame.Surface((width, height))
        surface.fill("white")
        surface.blit(text, (13, 13))
        rect = surface.get_rect(center=(15 + width // 2 + 70, 450 + height))
        self.screen.blit(surface, rect)

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if rect.collidepoint(event.pos):
                    self.preprocess()

    def preprocess(self):
        left = math.inf
        right = -math.inf
        top = math.inf
        bottom = -math.inf

        centerX = 0
        centerY = 0
        totalValue = 0

        for i, pixel in enumerate(self.pixels):
            x = i % 28
            y = i // 28

            centerX += x * pixel
            centerY += y * pixel
            totalValue += pixel

            if pixel > 0.05:
                left = min(left, x)
                right = max(right, x)
                top = min(top, y)
                bottom = max(bottom, y)

        centerX /= totalValue
        centerY /= totalValue

        for i, _ in enumerate(self.points):
            old = self.points[i]
            newX = (old[0] - centerX) + 14
            newY = (old[1] - centerY) + 14
            self.points[i] = (old[0] + (newX - old[0]), old[1] + (newY - old[1]))

    def draw(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if (
            mouse_x <= 15 + (self.pixel_size + 1) * 28
            and mouse_x >= self.pixel_size
            and mouse_y <= 15 + (self.pixel_size + 1) * 28
            and mouse_y >= self.pixel_size
        ):
            x = (mouse_x / (15 + (self.pixel_size + 1) * 28)) * 28
            y = (mouse_y / (15 + (self.pixel_size + 1) * 28)) * 28
            self.points.append((x - 1, y - 1))

            # x = int((mouse_x - 15) / (self.pixel_size + 1))
            # y = int((mouse_y - 15) / (self.pixel_size + 1))
            # self.pixels[y * 28 + x] = (0, 0, 0)


def run(nn):
    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((800, 800), flags=pygame.SRCALPHA)
    clock = pygame.time.Clock()
    running = True
    canvas = Canvas(screen, nn)

    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        screen.fill("#202020")

        canvas.render()
        canvas.listen(events)
        canvas.render_reset_button(events)
        canvas.render_preprocess_button(events)
        canvas.render_classify_button(events)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
