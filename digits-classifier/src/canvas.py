import pygame
import numpy as np
import math


# bad practice, but who cares?
def create_button_factory(screen, font, buttons):

    def create_button(text_str, rect, subscriber=None):
        button = pygame.Rect(rect)
        text = font.render(text_str, True, "black")
        textRect = text.get_rect()
        textRect.center = button.center

        buttons.append((button, subscriber))

        def render_button():
            pygame.draw.rect(screen, "white", button)
            screen.blit(text, textRect)

        return render_button

    return create_button


def notify_buttons(buttons, events):
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            for button, subscriber in buttons:
                if button.collidepoint(event.pos):
                    if callable(subscriber):
                        subscriber()


def classify(neuralnetwork, pixels, output):
    results = neuralnetwork.feedforward([np.array(pixels)])
    output.set(np.argmax(results))


class Canvas:
    def __init__(self, screen, font, nn):
        self.points = []
        self.screen = screen
        self.pixel_size = 15
        self.holding = False
        self.font = font
        self.nn = nn

    def listen(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.holding = True
                self.collect_mouse_positions()
            elif event.type == pygame.MOUSEBUTTONUP:
                self.holding = False
            elif event.type == pygame.MOUSEMOTION:
                if self.holding:
                    self.collect_mouse_positions()

    def draw(self):
        top = 0
        margin = 1

        self.pixels = [0] * 784
        for x, y in self.points:
            for i, pixel in enumerate(self.pixels):
                tileX = i % 28
                tileY = i // 28
                # https://github.com/3b1b/3Blue1Brown.com/blob/664e40d59e888f41ae2b595ac248e772bda68954/public/content/lessons/2017/neural-networks/neural-network-interactive/index.js#L921
                # adjust each pixel's alpha value based on its distance to each mouse coordinates while drawing on the canvas
                dist = math.hypot(tileX - x, tileY - y)
                penValue = 0.8 - ((dist / 2) ** 2)
                penValue = min(max(0, penValue), 1)
                self.pixels[i] = pixel + (1 - pixel) * penValue

        for i, pixel in enumerate(self.pixels):
            t = (i % 28) * margin
            left = ((i % 28) + 1) * self.pixel_size + t
            top = top + self.pixel_size + margin if i % 28 == 0 else top
            rect = [left, top, self.pixel_size, self.pixel_size]
            s = pygame.Surface((15, 15))
            s.set_alpha(pixel * 255 + 10)
            s.fill((255, 255, 255))
            self.screen.blit(s, rect)

    def reset(self):
        self.points = []

    def get_pixels(self):
        return self.pixels

    # https://github.com/3b1b/3Blue1Brown.com/blob/664e40d59e888f41ae2b595ac248e772bda68954/public/content/lessons/2017/neural-networks/neural-network-interactive/index.js#L947
    # calculate the "center of mass" of the pixels and using it to center them
    def preprocess(self):
        centerX = 0
        centerY = 0
        totalValue = 0
        # size of canvas which is 28x28 pixels, divided by 2 to get location of center
        center_offset = 28 / 2

        for i, pixel in enumerate(self.pixels):
            x = i % 28
            y = i // 28

            centerX += x * pixel
            centerY += y * pixel
            totalValue += pixel

        centerX /= totalValue
        centerY /= totalValue

        for i, _ in enumerate(self.points):
            old = self.points[i]

            # subtracting the center of mass from each individual points will shift the points
            # back to the origin 0, 0
            newX = old[0] - centerX
            newY = old[1] - centerY

            # now that the points are relative to the origin, we add a center_offset to bring them to
            # the center of the canvas
            self.points[i] = (newX + center_offset, newY + center_offset)

    def collect_mouse_positions(self):
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
            # x = (mouse_x - 15) / (self.pixel_size + 1)
            # y = (mouse_y - 15) / (self.pixel_size + 1)
            # self.points.append((x, y))


class Output:
    def __init__(self, font, initial_value=None):
        self.output = initial_value
        self.font = font

    def get(self):
        return self.output

    def set(self, new):
        self.output = new

    def draw(self, screen):
        text_str = (
            "Output: {}".format(self.output) if self.output is not None else "Output:"
        )
        text = self.font.render(text_str, True, "white")
        textRect = text.get_rect()
        textRect.center = (50, 550)
        screen.blit(text, textRect)


def run(nn):
    WIDTH, HEIGHT = 475, 600
    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font("fonts/OpenSans-Regular.ttf", 20)

    canvas = Canvas(screen, font, nn)
    output = Output(font)

    buttons = []
    create_btn = create_button_factory(screen, font, buttons)
    draw_reset_btn = create_btn(
        "reset", (15, 478, 100, 30), lambda: [canvas.reset(), output.set(None)]
    )
    draw_preprocess_btn = create_btn(
        "preprocess", (130, 478, 150, 30), canvas.preprocess
    )
    draw_classify_btn = create_btn(
        "classify",
        (290, 478, 120, 30),
        lambda: classify(nn, canvas.get_pixels(), output),
    )

    running = True
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        # "clear" screen
        screen.fill("#202020")

        canvas.draw()
        output.draw(screen)
        draw_reset_btn()
        draw_preprocess_btn()
        draw_classify_btn()

        canvas.listen(events)
        notify_buttons(buttons, events)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
