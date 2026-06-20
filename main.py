# =====================================
# 中長期投資向け 日経225分析ツール Ver6
# GitHub Actions対応版
# スコア上限撤廃 + ランク付け + CSV/Excel保存 + LINE通知対応
# =====================================

import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from ta.momentum import RSIIndicator


# =====================================
# 0. LINE設定
# =====================================

SEND_LINE = True

LINE_SEND_MODE = os.getenv("LINE_SEND_MODE", "broadcast")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_USER_ID = os.getenv("LINE_USER_ID", "")


# =====================================
# 1. 予備の日経225銘柄リスト
# =====================================

fallback_nikkei225_codes = [
    "1332","1605","1721","1801","1802","1803","1808","1812","1925","1928",
    "1963","2002","2269","2282","2501","2502","2503","2801","2802","2871",
    "2914","3101","3103","3401","3402","3405","3407","3861","4004","4005",
    "4021","4042","4043","4061","4063","4183","4188","4208","4452","4631",
    "4901","4911","6988","4151","4502","4503","4506","4507","4519","4523",
    "4568","4578","5019","5020","5101","5108","5201","5214","5232","5233",
    "5301","5332","5333","5401","5406","5411","3436","5706","5711","5713",
    "5714","5801","5802","5803","6103","6113","6301","6302","6305","6326",
    "6361","6367","6471","6472","6473","7004","7011","7012","7013","6501",
    "6503","6504","6506","6526","6594","6645","6701","6702","6723","6724",
    "6752","6753","6758","6762","6770","6841","6857","6861","6902","6920",
    "6954","6971","6976","6981","7735","7751","7752","8035","285A","7201",
    "7202","7203","7205","7211","7261","7267","7269","7270","4543","7731",
    "7733","7741","7762","7832","7911","7912","7951","7974","8001","8002",
    "8015","8031","8053","8058","3086","3092","3099","3382","7453","7532",
    "8233","8252","8267","9843","9983","8306","8308","8309","8316","8331",
    "8354","8411","7186","8253","8591","8601","8604","8628","8697","8725",
    "8750","8766","8795","3289","8801","8802","8804","8830","9001","9005",
    "9007","9008","9009","9020","9021","9022","9023","9064","9147","9101",
    "9104","9107","9201","9202","9301","9432","9433","9434","9613","9984",
    "9501","9502","9503","9531","9532","2413","2432","3659","4324","4689",
    "4704","4751","4755","6098","6178","9602","9735","9766"
]


# =====================================
# 2. 日経225銘柄を自動取得
# =====================================

def get_nikkei225_codes_auto():
    url = "https://ja.wikipedia.org/w/api.php"

    params = {
        "action": "parse",
        "page": "日経平均株価",
        "prop": "text",
        "format": "json",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 nikkei225-analysis-tool/1.0"
    }

    response = requests.get(url, params=params, headers=headers, timeout=20)
    response.raise_for_status()

    html = response.json()["parse"]["text"]["*"]
    tables = pd.read_html(html)

    codes = []

    for table in tables:
        for col in table.columns:
            for value in table[col].astype(str):
                found = re.findall(r"\b\d{4}\b|\b\d{3}[A-Z]\b", value)
                for code in found:
                    codes.append(code)

    codes = list(dict.fromkeys(codes))

    if len(codes) > 225:
        codes = codes[:225]

    if len(codes) < 200:
        raise Exception(f"自動取得数が少なすぎます：{len(codes)}件")

    return codes


# =====================================
# 3. テーマ分類
# =====================================

semiconductor_ai_codes = [
    "8035","6857","6723","6724","6981","6861","6758","6701",
    "6702","6501","6503","6504","6594","4063","6988","9984","285A"
]

high_dividend_candidate_codes = [
    "8306","8316","8308","8309","8411","8591","8058","8001",
    "8002","8031","2914","9432","9433","9434","8766","8750"
]


# =====================================
# 4. スコア関数
# =====================================

def safe_float(value):
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def score_per(per):
    if pd.isna(per) or per <= 0:
        return 0
    if per <= 10:
        return 10
    elif per <= 15:
        return 8
    elif per <= 20:
        return 6
    elif per <= 30:
        return 3
    else:
        return 0


def score_pbr(pbr):
    if pd.isna(pbr) or pbr <= 0:
        return 0
    if pbr <= 1:
        return 5
    elif pbr <= 1.5:
        return 4
    elif pbr <= 2.5:
        return 2
    else:
        return 0


def score_roe(roe):
    if pd.isna(roe):
        return 0
    roe_percent = roe * 100
    if roe_percent >= 15:
        return 15
    elif roe_percent >= 10:
        return 12
    elif roe_percent >= 8:
        return 8
    elif roe_percent >= 5:
        return 4
    else:
        return 0


def score_dividend(dividend_yield):
    if pd.isna(dividend_yield):
        return 0
    dividend_percent = dividend_yield * 100
    if dividend_percent >= 4:
        return 8
    elif dividend_percent >= 3:
        return 6
    elif dividend_percent >= 2:
        return 4
    elif dividend_percent >= 1:
        return 2
    else:
        return 0


def score_growth(growth):
    if pd.isna(growth):
        return 0
    growth_percent = growth * 100
    if growth_percent >= 20:
        return 10
    elif growth_percent >= 10:
        return 8
    elif growth_percent >= 5:
        return 5
    elif growth_percent >= 0:
        return 2
    else:
        return 0


def judge_signal(score):
    if score >= 120:
        return "強気買い"
    elif score >= 105:
        return "買い候補"
    elif score >= 90:
        return "監視候補"
    else:
        return "対象外"


def judge_rank(score):
    if score >= 130:
        return "S"
    elif score >= 120:
        return "A"
    elif score >= 110:
        return "B"
    elif score >= 100:
        return "C"
    elif score >= 90:
        return "D"
    else:
        return "E"


# =====================================
# 5. LINE送信関数
# =====================================

def send_line_message(message):
    if not SEND_LINE:
        print("LINE送信はOFFです。")
        return

    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("LINE_CHANNEL_ACCESS_TOKENが未設定です。GitHub Secretsを確認してください。")
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
    }

    if LINE_SEND_MODE == "broadcast":
        url = "https://api.line.me/v2/bot/message/broadcast"
        payload = {
            "messages": [
                {
                    "type": "text",
                    "text": message,
                }
            ]
        }

    elif LINE_SEND_MODE == "push":
        if not LINE_USER_ID:
            print("LINE_USER_IDが未設定です。push送信の場合はGitHub Secretsに設定してください。")
            return

        url = "https://api.line.me/v2/bot/message/push"
        payload = {
            "to": LINE_USER_ID,
            "messages": [
                {
                    "type": "text",
                    "text": message,
                }
            ],
        }

    else:
        print("LINE_SEND_MODEは broadcast または push にしてください。")
        return

    response = requests.post(url, headers=headers, json=payload, timeout=20)

    print("LINE送信ステータス:", response.status_code)
    print(response.text)


# =====================================
# 6. 分析処理
# =====================================

def analyze():
    try:
        nikkei225_codes = get_nikkei225_codes_auto()
        list_source = "Wikipedia APIから自動取得"
    except Exception as e:
        print("日経225銘柄の自動取得に失敗しました。")
        print("理由：", e)
        print("固定リストで続行します。")
        nikkei225_codes = fallback_nikkei225_codes
        list_source = "固定リスト"

    nikkei225_codes = list(dict.fromkeys(nikkei225_codes))
    tickers = [code + ".T" for code in nikkei225_codes]

    print("=" * 80)
    print("日経225銘柄リスト取得結果")
    print("=" * 80)
    print(f"取得方法：{list_source}")
    print(f"分析対象銘柄数：{len(tickers)}")

    results = []
    errors = []

    for i, ticker in enumerate(tickers, start=1):
        code = ticker.replace(".T", "")
        print(f"{i}/{len(tickers)} 分析中：{ticker}")

        try:
            df = yf.download(
                ticker,
                period="1y",
                progress=False,
                auto_adjust=True,
                threads=False,
            )

            if df.empty or len(df) < 220:
                errors.append(f"{ticker}：株価データ不足")
                continue

            close = df["Close"].squeeze()
            volume = df["Volume"].squeeze()

            current = float(close.iloc[-1])

            ma25 = float(close.rolling(25).mean().iloc[-1])
            ma75 = float(close.rolling(75).mean().iloc[-1])
            ma200 = float(close.rolling(200).mean().iloc[-1])

            rsi = float(RSIIndicator(close).rsi().iloc[-1])

            ret_1m = float((current / close.iloc[-21] - 1) * 100)
            ret_3m = float((current / close.iloc[-63] - 1) * 100)
            ret_6m = float((current / close.iloc[-126] - 1) * 100)

            deviation_25 = float((current / ma25 - 1) * 100)

            volatility = float(close.pct_change().rolling(20).std().iloc[-1] * 100)

            recent_high = float(close.tail(60).max())
            drawdown = float((current / recent_high - 1) * 100)

            avg_volume_20 = float(volume.tail(20).mean())

            try:
                stock = yf.Ticker(ticker)
                info = stock.info
            except Exception:
                info = {}

            name = info.get("shortName", code)

            per = safe_float(info.get("trailingPE"))
            forward_per = safe_float(info.get("forwardPE"))
            pbr = safe_float(info.get("priceToBook"))
            roe = safe_float(info.get("returnOnEquity"))
            dividend_yield = safe_float(info.get("dividendYield"))
            revenue_growth = safe_float(info.get("revenueGrowth"))
            earnings_growth = safe_float(info.get("earningsGrowth"))

            score = 0
            reasons = []

            if current > ma200:
                score += 15
                reasons.append("200日線上")

            if current > ma75:
                score += 10
                reasons.append("75日線上")

            if 45 <= rsi <= 70:
                score += 10
                reasons.append("RSI適正")

            if ret_3m > 0:
                add = min(ret_3m / 2, 10)
                score += add
                reasons.append("3か月上昇")

            if ret_6m > 0:
                add = min(ret_6m / 3, 10)
                score += add
                reasons.append("6か月上昇")

            if -8 <= deviation_25 <= 8:
                score += 5
                reasons.append("25日線付近")

            per_score = score_per(per)
            if per_score > 0:
                score += per_score
                reasons.append("PER良好")

            pbr_score = score_pbr(pbr)
            if pbr_score > 0:
                score += pbr_score
                reasons.append("PBR良好")

            roe_score = score_roe(roe)
            if roe_score > 0:
                score += roe_score
                reasons.append("ROE良好")

            dividend_score = score_dividend(dividend_yield)
            if dividend_score > 0:
                score += dividend_score
                reasons.append("配当あり")

            revenue_growth_score = score_growth(revenue_growth)
            if revenue_growth_score > 0:
                score += revenue_growth_score
                reasons.append("売上成長")

            earnings_growth_score = score_growth(earnings_growth)
            if earnings_growth_score > 0:
                score += earnings_growth_score
                reasons.append("利益成長")

            if volatility <= 3:
                score += 5
                reasons.append("値動き安定")

            if drawdown > -15:
                score += 5
                reasons.append("下落浅い")

            if rsi >= 75:
                score -= 10
                reasons.append("RSI過熱")

            if drawdown <= -25:
                score -= 10
                reasons.append("下落大")

            score = round(max(0, score), 1)

            signal = judge_signal(score)
            rank = judge_rank(score)

            buy_zone_low = round(current * 0.97, 0)
            buy_zone_high = round(current * 1.02, 0)
            stop_loss = round(current * 0.92, 0)
            take_profit = round(current * 1.15, 0)

            expected_return = ((take_profit / current) - 1) * 100
            expected_loss = ((stop_loss / current) - 1) * 100
            rr_ratio = abs(expected_return / expected_loss) if expected_loss != 0 else np.nan

            results.append({
                "銘柄名": name,
                "コード": code,
                "株価": round(current, 0),
                "総合点": score,
                "ランク": rank,
                "判定": signal,

                "PER": round(per, 1) if not pd.isna(per) else np.nan,
                "予想PER": round(forward_per, 1) if not pd.isna(forward_per) else np.nan,
                "PBR": round(pbr, 1) if not pd.isna(pbr) else np.nan,
                "ROE%": round(roe * 100, 1) if not pd.isna(roe) else np.nan,
                "配当利回り%": round(dividend_yield * 100, 2) if not pd.isna(dividend_yield) else np.nan,
                "売上成長率%": round(revenue_growth * 100, 1) if not pd.isna(revenue_growth) else np.nan,
                "利益成長率%": round(earnings_growth * 100, 1) if not pd.isna(earnings_growth) else np.nan,

                "RSI": round(rsi, 1),
                "1か月%": round(ret_1m, 1),
                "3か月%": round(ret_3m, 1),
                "6か月%": round(ret_6m, 1),
                "25日乖離%": round(deviation_25, 1),
                "下落率%": round(drawdown, 1),
                "20日平均出来高": round(avg_volume_20, 0),

                "買いゾーン下限": buy_zone_low,
                "買いゾーン上限": buy_zone_high,
                "損切り目安": stop_loss,
                "利確目安": take_profit,
                "期待利益%": round(expected_return, 1),
                "想定損失%": round(expected_loss, 1),
                "RR比": round(rr_ratio, 2),

                "半導体AI関連": "該当" if code in semiconductor_ai_codes else "",
                "高配当候補": "該当" if code in high_dividend_candidate_codes else "",
                "理由": "、".join(reasons),
            })

            time.sleep(0.2)

        except Exception as e:
            errors.append(f"{ticker}：{e}")
            continue

    ranking = pd.DataFrame(results)

    return ranking, errors


# =====================================
# 7. 保存・通知メッセージ
# =====================================

def save_results(ranking):
    today = datetime.now().strftime("%Y%m%d")

    csv_filename = f"nikkei225_ranking_{today}.csv"
    excel_filename = f"nikkei225_ranking_{today}.xlsx"

    buy_candidates = ranking[ranking["判定"].isin(["強気買い", "買い候補"])]

    semiconductor_ranking = ranking[ranking["半導体AI関連"] == "該当"]

    high_dividend_ranking = ranking[ranking["高配当候補"] == "該当"].sort_values(
        ["配当利回り%", "総合点"],
        ascending=False,
    )

    ranking.to_csv(csv_filename, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
        ranking.to_excel(writer, sheet_name="総合ランキング", index=False)
        buy_candidates.to_excel(writer, sheet_name="買い候補", index=False)
        semiconductor_ranking.to_excel(writer, sheet_name="半導体AI", index=False)
        high_dividend_ranking.to_excel(writer, sheet_name="高配当", index=False)

    print("=" * 80)
    print("保存完了")
    print("=" * 80)
    print(csv_filename)
    print(excel_filename)


def make_line_message(ranking):
    top5 = ranking.head(5)

    message = "【中長期有望銘柄 Ver6】\n"
    message += f"{datetime.now().strftime('%Y-%m-%d')}\n\n"

    for idx, row in top5.iterrows():
        message += f"{idx + 1}位 {row['銘柄名']}（{row['コード']}）\n"
        message += f"スコア：{row['総合点']}点 / ランク：{row['ランク']}\n"
        message += f"判定：{row['判定']}\n"
        message += f"株価：{row['株価']}円\n"
        message += f"買いゾーン：{row['買いゾーン下限']}〜{row['買いゾーン上限']}円\n"
        message += f"損切り：{row['損切り目安']}円 / 利確：{row['利確目安']}円\n"
        message += f"PER：{row['PER']} / PBR：{row['PBR']} / ROE：{row['ROE%']}%\n"
        message += f"理由：{row['理由']}\n\n"

    message += "※投資判断は自己責任でお願いします。"

    return message


# =====================================
# 8. メイン処理
# =====================================

if __name__ == "__main__":
    ranking, errors = analyze()

    print("\n")
    print("=" * 80)
    print(f"中長期おすすめランキング Ver6 {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 80)

    if ranking.empty:
        print("分析結果が空です。株価データを取得できていません。")
        send_line_message("中長期有望銘柄 Ver6：分析結果が空です。")
    else:
        ranking = ranking.sort_values(
            "総合点",
            ascending=False,
        ).reset_index(drop=True)

        print(ranking.head(20).to_string(index=False))

        save_results(ranking)

        message = make_line_message(ranking)

        print("\n")
        print("=" * 80)
        print("LINE通知メッセージ")
        print("=" * 80)
        print(message)

        send_line_message(message)

    print("\n")
    print("=" * 80)
    print("取得・分析できなかった銘柄")
    print("=" * 80)

    if len(errors) == 0:
        print("エラーなし")
    else:
        for error in errors[:30]:
            print(error)

        if len(errors) > 30:
            print(f"ほか {len(errors) - 30} 件のエラーがあります。")
