import pygame as pg
from pygame import *


def main():
    pygame.init()
    running = True
    size = width, height = 600, 500
    screen = pg.display.set_mode(size)
    clock = pygame.time.Clock()
    position = pygame.Vector2(2, 2)
    position2 = Vector2(2, 2)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill("white")
        pygame.draw.line(screen, "Black", (0, 0), (100, 100))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
