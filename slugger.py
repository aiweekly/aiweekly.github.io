import argparse
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name')
    args = parser.parse_args()

    slug = datetime.date.today().strftime("%Y-%m-%d") + '-' + args.name.replace(' | ', ' ').replace(' - ', ' ').replace(' ', '-').lower()
    print(slug)