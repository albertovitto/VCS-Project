import os
import numpy as np
from Cristian.image_processing.retrieval_utils import get_painting_info_from_csv

# distance
CENTER_DISTANCE = 0

# weighting
NONE = 0
AREA = 1
SQRT_AREA = 2


class PeopleLocator():
    def __init__(self, distance=CENTER_DISTANCE, weighting=NONE, voting=False, verbose=False,
                 data_path='../../dataset'):
        self.distance = distance
        self.weighting = weighting
        self.voting = voting
        self.verbose = verbose
        self.data_path = data_path

    def get_weights(self, painting_bbs):
        def area(bb):
            x, y, w, h = bb
            return w * h

        W = []
        if self.weighting == AREA:
            W = np.asarray([area(bb) for bb in painting_bbs])
        elif self.weighting == SQRT_AREA:
            W = np.sqrt(np.asarray([area(bb) for bb in painting_bbs]))
        else:
            W = np.asarray(W)

        if self.verbose:
            print("Weights: {}".format(W))
        return W

    def apply_weights(self, distances, painting_bbs):
        assert len(distances) == len(painting_bbs)
        if self.weighting != NONE:
            weights = self.get_weights(painting_bbs)
            # normalizing weights between 0 and 1,
            # otherwise multiplication may have very high results
            # and big gaps
            weights = weights / np.amax(weights)
            # using negative exponential in order to:
            # - normalize between 0 and 1
            # - the closest the painting, the higher the score
            d = np.exp(-np.asarray(distances) / np.mean(distances))
            weights *= d
        else:
            weights = np.exp(-np.asarray(distances) / np.mean(distances))

        if self.verbose:
            print("D*W : {}".format(weights))
        return weights

    def person_paintings_distances(self, person_bb, painting_bbs):
        def center(bb):
            x, y, w, h = bb
            return np.asarray((x + w // 2, y + h // 2))

        distances = []
        if self.distance == CENTER_DISTANCE:
            distances = [np.linalg.norm(center(bb) - center(person_bb)) for bb in painting_bbs]
        else:
            pass

        if self.verbose:
            print("Distances: {}".format(distances))
        return distances

    def get_room_id(self, weights, painting_retrievals):
        assert len(weights) == len(painting_retrievals)
        path = os.path.join(self.data_path, 'data.csv')
        if not self.voting:
            # FROM NUMPY DOCS:
            # In case of multiple occurrences of the minimum values,
            # the indices corresponding to the first occurrence are returned
            index = np.argmax(weights)
            painting_id = painting_retrievals[index]
            _, _, room = get_painting_info_from_csv(painting_id, path=path)
            return room
        else:
            votes = np.zeros(shape=(22))
            for i, pr in enumerate(painting_retrievals):
                _, _, room = get_painting_info_from_csv(pr, path=path)
                votes[room - 1] += weights[i]

            if self.verbose:
                print("Votes:\n")
                for i, v in enumerate(votes):
                    print("Room #{} = {}".format(i + 1, v))

            room = np.argmax(votes) + 1
            return room

    def localize_person(self, person_bb, painting_bbs, retrievals):
        assert len(painting_bbs) == len(retrievals)
        if len(painting_bbs) == 0:
            print("Cannot localize person with no painting detected")
            return None
        distances = self.person_paintings_distances(person_bb, painting_bbs)
        weights = self.apply_weights(distances, painting_bbs)
        room = self.get_room_id(weights, retrievals)
        return room


def main():
    # person = (270, 100, 250, 300)
    person = (1200, 100, 100, 100)
    paintings = [(600, 100, 100, 200), (1000, 50, 50, 100), (10, 700, 1000, 10)]
    retrievals = [0, 1, 2]  # rooms 19 21 20
    pl = PeopleLocator(distance=CENTER_DISTANCE, weighting=NONE, voting=False, verbose=True)
    room = pl.localize_person(person, paintings, retrievals)
    print("Room: {}".format(room))


if __name__ == '__main__':
    main()
