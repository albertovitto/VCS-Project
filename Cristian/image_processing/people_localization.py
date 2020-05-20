import numpy as np
from Cristian.image_processing.retrieval_utils import get_painting_info_from_csv

# distance metrics
CENTER_DISTANCE = 0

# weight mode
NONE = 0
AREA = 1


def get_weights(painting_bbs, mode=AREA):
    W = []
    if mode == AREA:
        for bb in painting_bbs:
            x, y, w, h = bb
            area = w * h
            W.append(area)
    else:
        pass

    W = np.asarray(W)
    # using negative exponential to avoid w = 0
    temperature = np.mean(W)
    if temperature == 0:
        raise ValueError("All paintings have 0 area")
    W = np.exp(-W / temperature)
    print("W   = {}".format(W))
    return W


def apply_weights(distances, painting_bbs, mode=AREA):
    assert len(distances) == len(painting_bbs)
    weights = [1 / len(distances) for bb in distances]
    if mode != NONE:
        weights = get_weights(painting_bbs, mode=mode)
        distances *= weights

    print("D*W = {}".format(distances))
    return distances, weights


def person_paintings_distances(person_bb, painting_bbs, metric=CENTER_DISTANCE, weighting_mode=NONE):
    def center(bb):
        x, y, w, h = bb
        return np.asarray((x + w // 2, y + h // 2))

    distances = []
    if metric == CENTER_DISTANCE:
        distances = [np.linalg.norm(center(bb) - center(person_bb)) for bb in painting_bbs]
    else:
        pass

    print("D   = {}".format(distances))
    return distances


def get_room_id(distances, painting_retrievals, voting=False):
    assert len(distances) == len(painting_retrievals)
    path = '../../dataset/data.csv'
    if not voting:
        # FROM NUMPY DOCS:
        # In case of multiple occurrences of the minimum values,
        # the indices corresponding to the first occurrence are returned
        index = np.argmin(distances)
        painting_id = painting_retrievals[index]
        _, _, room = get_painting_info_from_csv(painting_id, path=path)
        return room
    else:
        votes = np.zeros(shape=(22))
        max = np.amax(distances)
        for i, pr in enumerate(painting_retrievals):
            _, _, room = get_painting_info_from_csv(pr, path=path)
            votes[room] += (-distances[i] + max + 1)

        room = np.argmax(votes)

        return room + 1


def localize_person(person_bb, painting_bbs, painting_retrievals,
                    distance_metric=CENTER_DISTANCE, weighting_mode=AREA, voting=True):
    assert len(painting_bbs) == len(painting_retrievals)
    distances = person_paintings_distances(person_bb, painting_bbs, metric=distance_metric)
    distances, weights = apply_weights(distances, painting_bbs, mode=weighting_mode)
    room = get_room_id(distances, painting_retrievals, voting=voting)
    return room


def main():
    # person = (270, 100, 250, 300)
    person = (1200, 100, 100, 100)
    paintings = [(600, 100, 100, 200), (1000, 50, 50, 100), (10, 700, 1000, 10)]
    retrievals = [0, 1, 2]
    room = localize_person(person, paintings, retrievals)
    print("Room: {}".format(room))


if __name__ == '__main__':
    main()
