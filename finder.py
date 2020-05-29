import argparse

from Cristian.image_processing.retrieval_utils import get_painting_info_from_csv, search_paintings_by_room


def check_range(arg):
    try:
        value = int(arg)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err))

    if value < 0 or value > 95:
        message = "Expected 0 <= value <= 95, got value = {}".format(value)
        raise argparse.ArgumentTypeError(message)

    return value


def arg_parse():
    parser = argparse.ArgumentParser(description='Given a painting, find its room and other paintings near')
    parser.add_argument('-p', dest='painting', help='ID of the painting (i.e. \'010.png\' --> 10', type=check_range,
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    path = './dataset/data.csv'
    args = arg_parse()
    _, _, room = get_painting_info_from_csv(args.painting, path)
    prev = room - 1
    next = room + 1
    if prev == 0:
        prev == 22
    if next == 23:
        next = 1
    print('Current room {}:'.format(room))
    print(search_paintings_by_room(room, path))
    print('-----------------------\n\n')
    print('Previous room {}:'.format(prev))
    print(search_paintings_by_room(prev, path))
    print('-----------------------\n\n')
    print('Next room {}:'.format(next))
    print(search_paintings_by_room(next, path))
