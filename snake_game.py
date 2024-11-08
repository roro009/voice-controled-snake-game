# snake_game.py
import pygame
import random
from multiprocessing import Process, Value
import ctypes
import voice_control  # Import the custom voice_control module

# Game settings
WIDTH, HEIGHT = 600, 400
SNAKE_SIZE = 20
FPS = 0.5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Voice-Controlled Snake Game with NeMo ASR")
clock = pygame.time.Clock()

# Snake and food settings
snake_pos = [[100, 50]]
food_pos = [random.randrange(1, (WIDTH // SNAKE_SIZE)) * SNAKE_SIZE,
            random.randrange(1, (HEIGHT // SNAKE_SIZE)) * SNAKE_SIZE]
food_spawn = True
score = 0

# Directions (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
DIRECTIONS = {
    0: (0, -SNAKE_SIZE),  # UP
    1: (SNAKE_SIZE, 0),   # RIGHT
    2: (0, SNAKE_SIZE),   # DOWN
    3: (-SNAKE_SIZE, 0)   # LEFT
}

def run_game(snake_direction):
    global score, food_spawn, food_pos, snake_pos

    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Move the snake in the direction controlled by voice
        new_direction = DIRECTIONS[snake_direction.value]
        new_head = [snake_pos[0][0] + new_direction[0], snake_pos[0][1] + new_direction[1]]
        snake_pos.insert(0, new_head)

        # Check if the snake ate the food
        if snake_pos[0] == food_pos:
            score += 1
            food_spawn = False
        else:
            snake_pos.pop()

        # Spawn new food if needed
        if not food_spawn:
            food_pos = [random.randrange(1, (WIDTH // SNAKE_SIZE)) * SNAKE_SIZE,
                        random.randrange(1, (HEIGHT // SNAKE_SIZE)) * SNAKE_SIZE]
        food_spawn = True

        # Background and Snake
        screen.fill(BLACK)
        for pos in snake_pos:
            pygame.draw.rect(screen, GREEN, pygame.Rect(pos[0], pos[1], SNAKE_SIZE, SNAKE_SIZE))

        # Draw food
        pygame.draw.rect(screen, RED, pygame.Rect(food_pos[0], food_pos[1], SNAKE_SIZE, SNAKE_SIZE))

        # Check for collisions with wall or itself
        if (snake_pos[0][0] < 0 or snake_pos[0][0] >= WIDTH or
                snake_pos[0][1] < 0 or snake_pos[0][1] >= HEIGHT or
                snake_pos[0] in snake_pos[1:]):
            print("Game Over!")
            print(f"Your score: {score}")
            running = False

        # Update display
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    snake_direction = Value(ctypes.c_int, 1)  # Initial direction is "RIGHT"

    # Start the voice control process
    voice_process = Process(target=voice_control.recognize_direction, args=(snake_direction,))
    voice_process.start()

    # Start the game
    run_game(snake_direction)

    # Terminate the voice control process when the game ends
    voice_process.terminate()