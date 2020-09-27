import argparse
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name')
    args = parser.parse_args()

    short = args.name.split('|')[0] + 'and more'
    clean = short.replace(' | ', ' ').replace(' - ', ' ').replace(' ', '-').lower()
    slug = datetime.date.today().strftime("%Y-%m-%d") + '-' + clean
    permalink = '/posts/' + clean
    print(slug)
    print(permalink)