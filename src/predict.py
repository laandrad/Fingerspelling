from finger_features import GetFeatures
from sklearn.externals import joblib
import os, sys, inspect, pygame
import pandas as pd
src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
lib_dir = os.path.abspath(os.path.join(src_dir, '../lib'))
sys.path.insert(0, lib_dir)
import Leap


def main():

    controller = Leap.Controller()
    pygame.init()
    width = 740
    height = 480
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    listener = GetFeatures()
    font = pygame.font.SysFont("comicsansms", 72)
    labels = ["A", "B", "C", "D", "E", ""]

    nb_model = joblib.load("tmp/trained_model.pkl")

    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                done = True

        frame = controller.frame()
        features = pd.Series(listener.get_features(frame))
        features = features.iloc[:-1].values.reshape(1, -1)
        # print features

        if features.shape[1] > 0:
            pred = int(nb_model.predict(features))
        else:
            pred = -1

        print labels[pred]

        text = font.render("Detecting: {}".format(labels[int(pred)]), True, (0, 128, 0))

        screen.fill((255, 255, 255))
        screen.blit(text,
                    (width // 2 - text.get_width() // 2, 20))
        if pred in range(0, 6):
            image = pygame.image.load('images/letter{}.png'.format(labels[pred]))
            screen.blit(image, (width // 2 - 100, 120))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
