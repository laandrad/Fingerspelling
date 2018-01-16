from finger_features import GetFeatures
import os, sys, inspect, pygame
import pandas as pd
src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
lib_dir = os.path.abspath(os.path.join(src_dir, '../lib'))
sys.path.insert(0, lib_dir)
import Leap


def main():

    letter = sys.argv[1]

    image = pygame.image.load('images/letter{}.png'.format(str(letter).capitalize()))

    controller = Leap.Controller()
    pygame.init()
    width = 740
    height = 480
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    listener = GetFeatures()
    font = pygame.font.SysFont("comicsansms", 72)
    features_df = []
    done = 500
    i = 0

    while i < done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                i = done
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                i = done

        frame = controller.frame()
        features = listener.get_features(frame)
        print i
        print features
        features_df.append(features)

        text = font.render("Recording", True, (0, 128, 0))

        screen.fill((255, 255, 255))
        screen.blit(text,
                    (width // 2 - text.get_width() // 2, 20))
        screen.blit(image, (width // 2 - 100, 120))

        pygame.display.flip()
        clock.tick(60)
        i += 1

    features_df = pd.DataFrame(features_df)
    features_df['letter'] = letter
    # print features_df
    if os.path.isfile('data/features.csv'):
        with open('data/features.csv', 'a') as df_file:
            features_df.to_csv(df_file, header=False, index=False)
    else:
        features_df.to_csv("data/features.csv", index=False)


if __name__ == "__main__":
    main()
