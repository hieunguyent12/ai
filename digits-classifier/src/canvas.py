import pygame
import numpy as np


class Canvas:
    def __init__(self, screen, nn):
        self.pixels = [(255, 255, 255)] * 784
        self.screen = screen
        self.pixel_size = 15
        self.nn = nn

    def render(self):
        top = 0
        margin = 1
        for i, pixel in enumerate(self.pixels):
            t = (i % 28) * margin
            left = ((i % 28) + 1) * self.pixel_size + t
            top = top + self.pixel_size + margin if i % 28 == 0 else top
            rect = [left, top, self.pixel_size, self.pixel_size]
            pygame.draw.rect(self.screen, pixel, rect)

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

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if rect.collidepoint(event.pos):
                    self.pixels = [(255, 255, 255)] * 784

    def render_classify_button(self, events, text="classify"):
        font_size = 25
        font = pygame.font.Font(None, font_size)
        text = font.render(text, True, "black")

        width = text.get_size()[0] + 26
        height = text.get_size()[1] + 26
        surface = pygame.Surface((width, height))
        surface.fill("white")
        surface.blit(text, (13, 13))
        rect = surface.get_rect(
            center=(15 + width // 2 + text.get_size()[0] + 10, 450 + height)
        )
        self.screen.blit(surface, rect)

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if rect.collidepoint(event.pos):
                    grayscaled_pixels = []
                    for pixel in self.pixels:
                        if pixel == (255, 255, 255):
                            grayscaled_pixels.append(0)
                        else:
                            grayscaled_pixels.append((255 / 255 - 0.5) * 2)
                            # grayscaled_pixels.append(255)
                    grayscaled_pixels = np.array(grayscaled_pixels)
                    # results = self.nn.feedforward([grayscaled_pixels])
                    # print(np.argmax(results))

    def draw(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if (
            mouse_x <= 15 + (self.pixel_size + 1) * 28
            and mouse_x >= self.pixel_size
            and mouse_y <= 15 + (self.pixel_size + 1) * 28
            and mouse_y >= self.pixel_size
        ):
            x = int((mouse_x - 15) / (self.pixel_size + 1))
            y = int((mouse_y - 15) / (self.pixel_size + 1))
            self.pixels[y * 28 + x] = (0, 0, 0)


def run(nn):
    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    clock = pygame.time.Clock()
    running = True
    canvas = Canvas(screen, nn)
    holding = False

    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                holding = True
                canvas.draw()
            elif event.type == pygame.MOUSEBUTTONUP:
                holding = False
            elif event.type == pygame.MOUSEMOTION:
                if holding:
                    canvas.draw()
        screen.fill("#202020")

        canvas.render()
        canvas.render_reset_button(events)
        canvas.render_classify_button(events)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
