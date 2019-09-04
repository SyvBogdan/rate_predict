import asyncio
import csv
import datetime
import json
from collections import OrderedDict
from time import sleep
from config import BUILDDIR
from aiohttp import ClientSession


DATE_FORMAT = "%Y-%m-%d"
FINANCE_SECTIONS = ['business', 'politics', 'better-business']
CRITICAL_PERCENT_CHANGE = 0.5
START_DATE = '2016-08-30'
END_DATE = '2019-08-30'

def date_generator(from_date, to_date):
    start = datetime.datetime.strptime(from_date, DATE_FORMAT)
    end = datetime.datetime.strptime(to_date, DATE_FORMAT)
    return [start + datetime.timedelta(days=x) for x in range(0, (end - start).days + 1)]


async def fetch(date, session, rate_batch, url, is_range=False):
    if not is_range:
        src = url.format(date.strftime(DATE_FORMAT))
    else:
        src = url.format(date.strftime(DATE_FORMAT), date.strftime(DATE_FORMAT))
    print(src)
    async with session.get(src) as response:
        text = await response.read()
        rate_batch[date] = text


async def run(dates, rate_batch, url_pattern, is_range=False):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.1.1 Safari/605.1.15',
    }

    async with ClientSession(headers=headers) as session:
        tasks = (asyncio.ensure_future(fetch(date, session, rate_batch, url_pattern, is_range)) for date in
                 dates)
        await asyncio.gather(*tasks)


def collect_batch_async(batch, url, is_range=False):
    rate_batch = dict()

    loop = asyncio.get_event_loop()
    loop.set_exception_handler(lambda x, y: None)
    loop.run_until_complete(asyncio.ensure_future(run(batch, rate_batch, url, is_range)))

    return rate_batch


def write_rates_to_csv(percent_result):
    date_fld = 'date'
    change_fld = 'change'

    with open(BUILDDIR / 'rate_change.csv', mode='w+', newline='') as csv_file:
        fieldnames = [date_fld, change_fld]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for (k, v) in percent_result.items():
            writer.writerow({date_fld: k.strftime(DATE_FORMAT), change_fld: v})


def compute_change_rate():
    date_batch = date_generator(START_DATE, END_DATE)

    urls = collect_batch_async(date_batch, "https://api.exchangeratesapi.io/{}?base=USD&symbols=EUR")

    parsed_rates = {k: json.loads(v)["rates"]["EUR"] for (k, v) in urls.items()}

    ordered_rates = OrderedDict(sorted(parsed_rates.items(), key=lambda x: x[0]), key=lambda x: x[0])

    percent_result = dict()

    for (date, cur_rate) in ordered_rates.items():

        next_rate = ordered_rates.get(date + datetime.timedelta(+1))

        if next_rate is None:
            break
        else:
            result = 0
            percent = (next_rate - cur_rate) / cur_rate * 100

            if percent >= CRITICAL_PERCENT_CHANGE:
                result = 1
            elif percent <= -CRITICAL_PERCENT_CHANGE:
                result = -1

            percent_result[date] = result

    write_rates_to_csv(percent_result)


def parse_result(batch_result):
    date_fld = 'date'
    article_fld = 'articles'

    if len(batch_result) > 0:
        with open(BUILDDIR / 'finance_articles.csv', 'a', encoding="utf-8", newline='\n') as csv_file:
            fieldnames = [date_fld, article_fld]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            for (date, daily_articles) in batch_result.items():
                articles = daily_articles.decode('UTF-8')
                collected_articles = ""
                try:
                    for nextInfo in json.loads(articles)['response']['results']:
                        collected_articles = collected_articles + " " + nextInfo['webTitle']
                    if len(collected_articles) > 0:
                        writer.writerow({date_fld: date.strftime(DATE_FORMAT), article_fld: collected_articles})
                except Exception as str:
                    print("skip exception: " + str)


def collect_finance_info():
    batch_size = 10
    dates = date_generator(START_DATE, END_DATE)

    url = "https://content.guardianapis.com/search?api-key=e7ddbb6b-7e0e-407d-a138-f08c073141d7&section={}&from-date={}&to-date={}&page-size=30"

    def read_next_batch(batch_size):

        batch = list()

        for next_date in dates:
            batch.append(next_date)
            if len(batch) >= batch_size:
                yield batch
                batch = list()
        if batch:
            yield batch

    for section in FINANCE_SECTIONS:
        url_pattern = url.format(section, '{}', '{}')

        for date_batch in read_next_batch(batch_size):
            batch_result = collect_batch_async(date_batch, url_pattern, True)
            parse_result(batch_result)
        sleep(1)  # throttling for api


def main():
    compute_change_rate()


if __name__ == '__main__':
    main()
