import datetime as dt
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests


BASE_URL = "http://192.168.1.10:10213"


@dataclass
class Position:
    stock_code: str
    buy_date: str
    buy_price: float
    quantity: int
    sell_time: Optional[str] = None
    sell_date: Optional[str] = None
    buy_fee: float = 0.0


@dataclass
class Trade:
    side: str
    stock_code: str
    date: str
    time: str
    price: float
    quantity: int
    amount: float
    fee: float


class StockAPI:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.quote_cache: Dict[Tuple[str, str, str], dict] = {}
        self.info_cache: Dict[Tuple[str, str], dict] = {}

    def _get(self, path: str, params: dict) -> Optional[dict]:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("code") != 0:
                return None
            return payload.get("data", {})
        except requests.RequestException:
            return None

    def get_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        data = self._get(
            "/api/get_trade_date",
            {"start_date": start_date, "end_date": end_date},
        )
        if not data:
            return []
        return [item["date"] for item in data.get("trade_dates", []) if item.get("is_open") == 1]

    def get_stock_list(self, date: str) -> List[str]:
        data = self._get("/api/get_stock_list", {"date": date})
        if not data:
            return []
        return data.get("stocklist", [])

    def get_stock_info(self, stock_code: str, date: str) -> dict:
        key = (date, stock_code)
        if key in self.info_cache:
            return self.info_cache[key]
        data = self._get("/api/get_stock_info", {"stock_code": stock_code, "date": date})
        if not data:
            return {}
        self.info_cache[key] = data
        return data

    def get_stock_quote(self, stock_code: str, date: str, qtime: str) -> dict:
        key = (date, stock_code, qtime)
        if key in self.quote_cache:
            return self.quote_cache[key]
        data = self._get(
            "/api/get_stock_quote",
            {"stock_code": stock_code, "date": date, "qtime": qtime},
        )
        if not data:
            return {}
        self.quote_cache[key] = data
        return data


def parse_int_date(date_str: str) -> dt.date:
    return dt.datetime.strptime(date_str, "%Y%m%d").date()


def extract_details(payload: dict) -> dict:
    stock_data = payload.get("stock_data") or []
    if not stock_data:
        return {}
    return stock_data[0].get("parse_details") or {}


def pick_price(details: dict) -> Optional[float]:
    for key in ("LastPrice", "OpenPrice", "BuyPrice01", "SellPrice01", "ClosePrice"):
        price = details.get(key, 0)
        if isinstance(price, (int, float)) and price > 0:
            return float(price)
    return None


def get_open_close(details: dict) -> Tuple[Optional[float], Optional[float]]:
    open_price = details.get("OpenPrice") or 0
    close_price = details.get("ClosePrice") or 0
    open_price = float(open_price) if open_price > 0 else None
    close_price = float(close_price) if close_price > 0 else None
    if close_price is None:
        close_price = pick_price(details)
    return open_price, close_price


def next_minute(hhmm: str) -> Optional[str]:
    try:
        hours = int(hhmm[:2])
        minutes = int(hhmm[2:])
    except ValueError:
        return None
    total = hours * 60 + minutes + 1
    if total > 15 * 60:
        return None
    return f"{total // 60:02d}{total % 60:02d}"


def is_st(name: str) -> bool:
    if not name:
        return False
    upper = name.upper()
    return "ST" in upper or "退" in name


def is_new_stock(listing_date: int, current_date: str, days: int = 60) -> bool:
    if listing_date <= 0:
        return False
    try:
        list_date = parse_int_date(str(listing_date))
        cur_date = parse_int_date(current_date)
    except ValueError:
        return False
    return (cur_date - list_date).days < days


def is_suspended(details: dict) -> bool:
    status_tag = details.get("SecurityStatusTag")
    if isinstance(status_tag, str) and status_tag and set(status_tag) != {"0"}:
        return True
    return False


def in_excluded_board(stock_code: str) -> bool:
    if stock_code.endswith(".BJ"):
        return True
    symbol = stock_code.split(".")[0]
    if stock_code.endswith(".SH") and symbol.startswith("688"):
        return True
    return False


class Backtester:
    def __init__(self, api: StockAPI) -> None:
        self.api = api
        self.cash = 100000.0
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[str, float]] = []
        self.free_float_mcap_cache: Dict[Tuple[str, str], float] = {}
        self.realized_pnls: List[float] = []

    def get_free_float_mcap(self, stock_code: str, date: str, price: float) -> Optional[float]:
        key = (date, stock_code)
        if key in self.free_float_mcap_cache:
            return self.free_float_mcap_cache[key]
        info = self.api.get_stock_info(stock_code, date)
        details = extract_details(info)
        outstanding = details.get("OutstandingShare") or details.get("IssuedVolume") or 0
        if not isinstance(outstanding, (int, float)) or outstanding <= 0:
            return None
        mcap = float(outstanding) * price
        self.free_float_mcap_cache[key] = mcap
        return mcap

    def check_conditions(self, stock_code: str, date: str, qtime: str) -> Optional[dict]:
        quote = self.api.get_stock_quote(stock_code, date, qtime)
        details = extract_details(quote)
        if not details:
            return None

        price = pick_price(details)
        if price is None:
            return None

        pre_close = details.get("PreClosePrice") or 0
        if not isinstance(pre_close, (int, float)) or pre_close <= 0:
            return None

        pct_change = (price - pre_close) / pre_close * 100
        if pct_change >= 6:
            return None

        amount = details.get("TotalAmount") or 0
        if (not isinstance(amount, (int, float))) or amount <= 0:
            total_volume = details.get("TotalVolume") or 0
            if isinstance(total_volume, (int, float)) and total_volume > 0:
                amount = total_volume * price
        if amount <= 20_000_000:
            return None

        info = self.api.get_stock_info(stock_code, date)
        info_details = extract_details(info)
        name = info_details.get("SecurityName", "")
        listing_date = info_details.get("ListingDate") or 0

        if is_st(name):
            return None
        if is_new_stock(int(listing_date), date):
            return None
        if in_excluded_board(stock_code):
            return None
        if is_suspended(info_details):
            return None

        mcap = self.get_free_float_mcap(stock_code, date, price)
        if mcap is None or mcap >= 50_000_000_000:
            return None

        price_up_limit = details.get("PriceUpLimit") or 0
        if isinstance(price_up_limit, (int, float)) and price_up_limit > 0 and price >= price_up_limit:
            return None

        return {
            "price": price,
            "amount": float(amount),
            "details": details,
        }

    def buy(self, date: str, time: str, stock_code: str, price: float) -> None:
        if stock_code in self.positions:
            return
        if len(self.positions) >= 6:
            return
        target_cash = self.cash * 0.30
        quantity = math.floor(target_cash / price / 100) * 100
        if quantity < 100:
            return
        trade_amount = price * quantity
        fee = trade_amount * 0.0003
        if trade_amount + fee > self.cash:
            return
        self.cash -= trade_amount + fee
        self.positions[stock_code] = Position(
            stock_code=stock_code,
            buy_date=date,
            buy_price=price,
            quantity=quantity,
            sell_time=None,
            sell_date=None,
            buy_fee=fee,
        )
        self.trades.append(
            Trade(
                side="BUY",
                stock_code=stock_code,
                date=date,
                time=time,
                price=price,
                quantity=quantity,
                amount=trade_amount,
                fee=fee,
            )
        )

    def set_sell_time(self, date: str, stock_code: str, buy_quote_time: str, next_date: Optional[str]) -> None:
        pos = self.positions.get(stock_code)
        if not pos or pos.buy_date != date:
            return
        quote = self.api.get_stock_quote(stock_code, date, "1500")
        details = extract_details(quote)
        open_price, close_price = get_open_close(details)
        if open_price is None:
            buy_quote = self.api.get_stock_quote(stock_code, date, buy_quote_time)
            buy_details = extract_details(buy_quote)
            open_price = pick_price(buy_details)
        if open_price is None or close_price is None:
            pos.sell_time = "1400"
        else:
            pos.sell_time = "1000" if close_price < open_price else "1400"
        pos.sell_date = next_date

    def try_sell(self, date: str, stock_code: str, sell_time: str) -> bool:
        current_time = sell_time
        while current_time:
            quote = self.api.get_stock_quote(stock_code, date, current_time)
            details = extract_details(quote)
            price = pick_price(details) if details else None
            if price is None:
                current_time = next_minute(current_time)
                continue
            price_down_limit = details.get("PriceDownLimit") or 0
            if (
                isinstance(price_down_limit, (int, float))
                and price_down_limit > 0
                and price <= price_down_limit
            ):
                current_time = next_minute(current_time)
                continue
            pos = self.positions.pop(stock_code, None)
            if not pos:
                return True
            trade_amount = price * pos.quantity
            fee = trade_amount * 0.0003
            self.cash += trade_amount - fee
            pnl = (price - pos.buy_price) * pos.quantity - (fee + pos.buy_fee)
            self.realized_pnls.append(pnl)
            self.trades.append(
                Trade(
                    side="SELL",
                    stock_code=stock_code,
                    date=date,
                    time=current_time,
                    price=price,
                    quantity=pos.quantity,
                    amount=trade_amount,
                    fee=fee,
                )
            )
            return True
        return False

    def update_equity(self, date: str) -> None:
        total_value = self.cash
        for stock_code, pos in list(self.positions.items()):
            quote = self.api.get_stock_quote(stock_code, date, "1500")
            details = extract_details(quote)
            price = pick_price(details)
            if price is None:
                continue
            total_value += price * pos.quantity
        self.equity_curve.append((date, total_value))

    def run(self, start_date: str, end_date: str, verbose: bool = False) -> None:
        trade_dates = self.api.get_trade_dates(start_date, end_date)
        if not trade_dates:
            if verbose:
                print("No trade dates returned for range.", flush=True)
            return

        for idx, date in enumerate(trade_dates):
            if verbose:
                print(f"DATE {date} start", flush=True)
            stock_list = self.api.get_stock_list(date)
            candidates: List[Tuple[str, float, float]] = []
            for stock_code in stock_list:
                info = self.check_conditions(stock_code, date, "0930")
                if info:
                    candidates.append((stock_code, info["amount"], info["price"]))

            candidates.sort(key=lambda x: x[1], reverse=True)
            for stock_code, amount, price in candidates[:3]:
                self.buy(date, "0930", stock_code, price)

            next_date = trade_dates[idx + 1] if idx + 1 < len(trade_dates) else None
            for stock_code in list(self.positions.keys()):
                self.set_sell_time(date, stock_code, "0930", next_date)

            for stock_code, pos in list(self.positions.items()):
                if pos.sell_time and pos.sell_date and pos.sell_date <= date:
                    self.try_sell(date, stock_code, pos.sell_time)

            self.update_equity(date)
            if verbose:
                print(
                    f"DATE {date} end positions={len(self.positions)} cash={self.cash:.2f}",
                    flush=True,
                )

    def summarize(self) -> dict:
        if not self.equity_curve:
            return {}
        start_value = self.equity_curve[0][1]
        end_value = self.equity_curve[-1][1]
        total_return = (end_value - start_value) / start_value if start_value else 0

        start_date = parse_int_date(self.equity_curve[0][0])
        end_date = parse_int_date(self.equity_curve[-1][0])
        days = (end_date - start_date).days or 1
        annual_return = (1 + total_return) ** (252 / days) - 1

        peak = self.equity_curve[0][1]
        max_drawdown = 0.0
        for _, value in self.equity_curve:
            peak = max(peak, value)
            drawdown = (peak - value) / peak if peak else 0
            max_drawdown = max(max_drawdown, drawdown)

        wins = 0
        losses = 0
        profit_sum = 0.0
        loss_sum = 0.0
        for pnl in self.realized_pnls:
            if pnl >= 0:
                wins += 1
                profit_sum += pnl
            else:
                losses += 1
                loss_sum += abs(pnl)

        win_rate = wins / (wins + losses) if wins + losses > 0 else 0
        profit_loss_ratio = (profit_sum / loss_sum) if loss_sum > 0 else 0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
        }


def main() -> None:
    api = StockAPI(BASE_URL)
    backtester = Backtester(api)
    backtester.run("20250901", "20250915", verbose=True)

    for trade in backtester.trades:
        print(
            f"{trade.side} {trade.date} {trade.time} {trade.stock_code} "
            f"price={trade.price:.2f} qty={trade.quantity} fee={trade.fee:.2f}"
        )

    for date, equity in backtester.equity_curve:
        print(f"EQUITY {date} {equity:.2f}")

    metrics = backtester.summarize()
    if metrics:
        print(f"总收益率: {metrics['total_return']:.2%}")
        print(f"年化收益: {metrics['annual_return']:.2%}")
        print(f"最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"胜率: {metrics['win_rate']:.2%}")
        print(f"盈亏比: {metrics['profit_loss_ratio']:.2f}")


if __name__ == "__main__":
    main()
