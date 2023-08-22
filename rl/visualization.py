import pygame


def visualize_numpy_array(screen: pygame.Surface, array, cell_size, wall_color: tuple):
    rows, cols = array.shape

    for y in range(rows):
        for x in range(cols):
            value = array[x, y]
            if value == 0:
                if (x + y) % 2 == 0:
                    color = (255, 255, 255)
                else:
                    color = (200, 200, 200)
            else:
                color = wall_color
            pygame.draw.rect(screen, color, (x * cell_size,
                             y * cell_size, cell_size, cell_size))


def draw_circle_on_grid(screen, pos_x, pos_y, cell_size, color):
    pygame.draw.circle(screen, color,
                       center=(pos_x * cell_size + (cell_size / 2),
                               pos_y * cell_size + (cell_size / 2)),
                       radius=cell_size / 2)


def handle_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()


def display_fps(screen: pygame.Surface, clock: pygame.time.Clock):
    font = pygame.font.Font(pygame.font.get_default_font(), 36)
    text = font.render(f'{clock.get_fps():.0f}', True, (0, 120, 0))
    screen.blit(text, dest=(0, 0))


def render_content(clock: pygame.time.Clock, fps: int):
    pygame.display.flip()
    clock.tick(fps)
