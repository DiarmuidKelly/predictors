from pycoingecko import CoinGeckoAPI
import datetime


cg = CoinGeckoAPI()


if __name__ == "__main__":
    d_s = datetime.datetime(2020, 1, 1).timestamp()
    d_e = datetime.datetime.now().timestamp()
    hist = cg.get_coin_market_chart_range_by_id(id='shiba-inu', vs_currency='eur', from_timestamp=d_s, to_timestamp=d_e)
    print()
