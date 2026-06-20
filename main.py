import os
import re
import time
import glob
import unicodedata
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")

DATA_DIR = "j_tenbagger_data"
HISTORY_DIR = os.path.join(DATA_DIR, "history")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

BENCHMARK_TICKER = "^N225"
YF_CHUNK_SIZE = 80
FUNDAMENTAL_LIMIT = int(os.environ.get("FUNDAMENTAL_LIMIT", "700"))

THEME_CODES = {
    "AI・DX": [
        "3655", "3659", "3676", "3697", "3914", "3993", "4165", "4382",
        "4384", "4475", "4478", "4480", "4483", "4488", "5132", "5574"
    ],
    "半導体・データセンター": [
        "151A", "3132", "3445", "4062", "4063", "4186", "4369", "4970",
        "6146", "6266", "6315", "6323", "6387", "6525", "6526", "6613",
        "6723", "6857", "6861", "6871", "6920", "6963", "6967", "8035"
    ],
    "宇宙・防衛": [
        "5595", "6208", "6203", "4274", "7011", "7012", "7013", "7721"
    ],
    "電力・インフラ": [
        "1942", "1944", "5801", "5802", "5803", "5805", "6501", "6503",
        "6504", "6645", "9501", "9502", "9503", "9506", "9508"
    ],
    "エンタメ・IP": [
        "2432", "3659", "3660", "3765", "3932", "5253", "5597", "6758",
        "7832", "7974", "9468"
    ]
}


def now_jst():
    return datetime.now(JST)


def today_str():
    return now_jst().strftime("%Y%m%d")


def normalize_text(s):
    s = unicodedata.normalize("NFKC", str(s))
    return s.replace(" ", "").replace("　", "").strip()


def normalize_code(code):
    code = str(code).strip().replace(".0", "").upper()
    return code


def is_stock_code(code):
    return bool(re.fullmatch(r"[0-9A-Z]{4}", str(code)))


def safe_float(v):
    try:
        if v is None:
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def send_line_message(message):
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("LINE_CHANNEL_ACCESS_TOKEN が未設定です。")
        print(message)
        return

    url = "https://api.line.me/v2/bot/message/broadcast"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
    }

    payload = {
        "messages": [
            {
                "type": "text",
                "text": message[:4900],
            }
        ]
    }

    r = requests.post(url, headers=headers, json=payload, timeout=20)
    print("LINE送信ステータス:", r.status_code)
    print(r.text)


def find_column(df, exact_names, contains_all=None):
    normalized = {normalize_text(c): c for c in df.columns}

    for name in exact_names:
        key = normalize_text(name)
        if key in normalized:
            return normalized[key]

    if contains_all:
        for col in df.columns:
            c = normalize_text(col)
            if all(normalize_text(x) in c for x in contains_all):
                return col

    return None


def get_all_japanese_stocks_from_jpx():
    print("JPXから上場銘柄一覧を取得します。")

    headers = {"User-Agent": "Mozilla/5.0 J-TENBAGGER/1.0"}
    r = requests.get(JPX_URL, headers=headers, timeout=30)
    r.raise_for_status()

    tmp = "jpx_listed_company.xls"
    with open(tmp, "wb") as f:
        f.write(r.content)

    df = pd.read_excel(tmp, dtype=str)

    print("JPX列名:", list(df.columns))

    code_col = find_column(df, ["コード"])
    name_col = find_column(df, ["銘柄名", "会社名", "名称"])
    market_col = find_column(df, ["市場・商品区分", "市場区分"], contains_all=["市場", "区分"])
    industry_col = find_column(df, ["33業種区分", "業種"], contains_all=["33", "業種"])

    if code_col is None or name_col is None or market_col is None:
        raise Exception(
            "JPXファイルの列認識に失敗しました。\n"
            f"code_col={code_col}, name_col={name_col}, market_col={market_col}\n"
            f"列名={list(df.columns)}"
        )

    market_norm = df[market_col].fillna("").astype(str).apply(normalize_text)

    target_mask = (
        market_norm.str.contains("プライム", na=False)
        | market_norm.str.contains("スタンダード", na=False)
        | market_norm.str.contains("グロース", na=False)
    )

    stocks = df[target_mask].copy()

    stocks["コード"] = stocks[code_col].astype(str).apply(normalize_code)
    stocks["銘柄名"] = stocks[name_col].astype(str).str.strip()
    stocks["市場"] = stocks[market_col].astype(str).str.strip()

    if industry_col is not None:
        stocks["業種"] = stocks[industry_col].astype(str).str.strip()
    else:
        stocks["業種"] = ""

    stocks = stocks[stocks["コード"].apply(is_stock_code)]
    stocks = stocks[["コード", "銘柄名", "市場", "業種"]].drop_duplicates("コード")
    stocks = stocks.sort_values("コード").reset_index(drop=True)

    if len(stocks) < 2500:
        raise Exception(f"対象銘柄数が少なすぎます：{len(stocks)}")

    print(f"対象銘柄数：{len(stocks)}")
    print(stocks.head(20).to_string(index=False))

    return stocks


def yf_download_batch(tickers, period="2y"):
    all_data = {}

    print(f"株価データ取得：{len(tickers)}銘柄")

    for start in range(0, len(tickers), YF_CHUNK_SIZE):
        chunk = tickers[start:start + YF_CHUNK_SIZE]
        print(f"yfinance chunk {start + 1} - {start + len(chunk)}")

        try:
            data = yf.download(
                chunk,
                period=period,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )

            if data.empty:
                continue

            if len(chunk) == 1:
                all_data[chunk[0]] = data.copy()
            else:
                level0 = data.columns.get_level_values(0)
                for ticker in chunk:
                    if ticker in level0:
                        all_data[ticker] = data[ticker].dropna(how="all").copy()

            time.sleep(1.0)

        except Exception as e:
            print("価格取得エラー:", e)

    return all_data


def score_price_growth(ret_6m, ret_1y, rs_6m):
    score = 0

    if not pd.isna(ret_6m):
        if ret_6m >= 100:
            score += 15
        elif ret_6m >= 60:
            score += 12
        elif ret_6m >= 30:
            score += 9
        elif ret_6m >= 10:
            score += 5

    if not pd.isna(ret_1y):
        if ret_1y >= 200:
            score += 15
        elif ret_1y >= 100:
            score += 12
        elif ret_1y >= 50:
            score += 8
        elif ret_1y >= 20:
            score += 4

    if not pd.isna(rs_6m):
        if rs_6m >= 80:
            score += 10
        elif rs_6m >= 40:
            score += 8
        elif rs_6m >= 20:
            score += 5
        elif rs_6m >= 5:
            score += 2

    return min(40, score)


def score_volume(volume_ratio):
    if pd.isna(volume_ratio):
        return 0
    if volume_ratio >= 2.5:
        return 10
    if volume_ratio >= 2.0:
        return 8
    if volume_ratio >= 1.5:
        return 6
    if volume_ratio >= 1.2:
        return 3
    return 0


def score_market_cap(market_cap):
    if pd.isna(market_cap) or market_cap <= 0:
        return 0

    oku = market_cap / 100_000_000

    if 50 <= oku <= 300:
        return 15
    if 300 < oku <= 1000:
        return 13
    if 1000 < oku <= 3000:
        return 10
    if 3000 < oku <= 8000:
        return 5
    if oku < 50:
        return 3
    return 1


def score_growth(growth, max_score):
    if pd.isna(growth):
        return 0

    g = growth * 100

    if g >= 80:
        return max_score
    if g >= 50:
        return max_score * 0.85
    if g >= 30:
        return max_score * 0.7
    if g >= 20:
        return max_score * 0.55
    if g >= 10:
        return max_score * 0.35
    if g >= 0:
        return max_score * 0.1
    return -max_score * 0.3


def score_profitability(roe, operating_margin, profit_margin):
    score = 0

    if not pd.isna(roe):
        r = roe * 100
        if r >= 25:
            score += 8
        elif r >= 15:
            score += 6
        elif r >= 10:
            score += 4
        elif r < 0:
            score -= 5

    for m in [operating_margin, profit_margin]:
        if not pd.isna(m):
            p = m * 100
            if p >= 20:
                score += 4
            elif p >= 10:
                score += 3
            elif p >= 5:
                score += 1
            elif p < 0:
                score -= 4

    return max(-10, min(15, score))


def score_balance(debt_to_equity, current_ratio):
    score = 0

    if not pd.isna(debt_to_equity):
        if debt_to_equity <= 30:
            score += 5
        elif debt_to_equity <= 80:
            score += 3
        elif debt_to_equity >= 200:
            score -= 5

    if not pd.isna(current_ratio):
        if current_ratio >= 2:
            score += 5
        elif current_ratio >= 1.2:
            score += 3
        elif current_ratio < 1:
            score -= 5

    return max(-10, min(10, score))


def detect_theme(code, name, industry):
    themes = []

    for theme, codes in THEME_CODES.items():
        if code in codes:
            themes.append(theme)

    text = f"{name} {industry}"

    keyword_map = {
        "AI・DX": ["AI", "人工知能", "DX", "クラウド", "データ", "SaaS", "ソフト"],
        "半導体・データセンター": ["半導体", "電子部品", "電気機器", "データセンター"],
        "宇宙・防衛": ["宇宙", "防衛", "航空", "重工"],
        "電力・インフラ": ["電力", "インフラ", "電線", "非鉄", "建設"],
        "エンタメ・IP": ["ゲーム", "アニメ", "コンテンツ", "IP", "情報・通信業"],
        "医療・バイオ": ["医薬品", "バイオ", "医療", "創薬"]
    }

    for theme, keywords in keyword_map.items():
        if any(k in text for k in keywords):
            themes.append(theme)

    themes = list(dict.fromkeys(themes))
    return "、".join(themes)


def score_theme(theme_text):
    if not theme_text:
        return 0

    score = 0

    strong_themes = ["AI・DX", "半導体・データセンター", "宇宙・防衛", "電力・インフラ"]
    for t in strong_themes:
        if t in theme_text:
            score += 5

    if "エンタメ・IP" in theme_text:
        score += 3
    if "医療・バイオ" in theme_text:
        score += 2

    return min(15, score)


def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}

    return {
        "marketCap": safe_float(info.get("marketCap")),
        "revenueGrowth": safe_float(info.get("revenueGrowth")),
        "earningsGrowth": safe_float(info.get("earningsGrowth")),
        "returnOnEquity": safe_float(info.get("returnOnEquity")),
        "operatingMargins": safe_float(info.get("operatingMargins")),
        "profitMargins": safe_float(info.get("profitMargins")),
        "debtToEquity": safe_float(info.get("debtToEquity")),
        "currentRatio": safe_float(info.get("currentRatio")),
        "trailingPE": safe_float(info.get("trailingPE")),
        "forwardPE": safe_float(info.get("forwardPE")),
    }


def classify_candidate(score, market_cap_score, theme_score, price_score):
    if score >= 80 and market_cap_score >= 8:
        return "本命候補"
    if score >= 70 and theme_score >= 5:
        return "大化け候補"
    if score >= 60 and price_score >= 20:
        return "急浮上候補"
    if score >= 50:
        return "監視候補"
    return "対象外"


def update_old_returns(current_price_map):
    files = sorted(glob.glob(os.path.join(HISTORY_DIR, "j_tenbagger_*.csv")))

    if not files:
        return

    today_dt = now_jst().date()
    horizons = [30, 90, 180, 365]

    for file in files:
        try:
            base = os.path.basename(file)
            m = re.search(r"j_tenbagger_(\d{8})", base)
            if not m:
                continue

            file_date = datetime.strptime(m.group(1), "%Y%m%d").date()
            days_passed = (today_dt - file_date).days

            if days_passed <= 0:
                continue

            df = pd.read_csv(file, dtype={"コード": str})
            changed = False

            for h in horizons:
                col = f"{h}日後リターン%"
                if col not in df.columns:
                    df[col] = np.nan

                if days_passed >= h:
                    mask = df[col].isna()

                    if mask.any():
                        def calc_return(row):
                            code = normalize_code(row["コード"])
                            entry = safe_float(row.get("株価"))
                            now_price = current_price_map.get(code, np.nan)

                            if pd.isna(entry) or entry <= 0 or pd.isna(now_price):
                                return np.nan

                            return round((now_price / entry - 1) * 100, 2)

                        df.loc[mask, col] = df.loc[mask].apply(calc_return, axis=1)
                        changed = True

            if changed:
                df.to_csv(file, index=False, encoding="utf-8-sig")
                print("過去ファイル更新:", file)

        except Exception as e:
            print("過去リターン更新エラー:", file, e)


def analyze():
    universe = get_all_japanese_stocks_from_jpx()
    tickers = [code + ".T" for code in universe["コード"].tolist()]

    benchmark = yf.download(
        BENCHMARK_TICKER,
        period="2y",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if benchmark.empty:
        raise Exception("日経平均データ取得に失敗しました。")

    benchmark_close = benchmark["Close"].squeeze()
    bench_current = float(benchmark_close.iloc[-1])
    bench_ret_6m = float((bench_current / benchmark_close.iloc[-126] - 1) * 100)

    price_data = yf_download_batch(tickers, period="2y")

    rows = []
    current_price_map = {}

    for _, row in universe.iterrows():
        code = normalize_code(row["コード"])
        ticker = code + ".T"
        name = row["銘柄名"]
        market = row["市場"]
        industry = row["業種"]

        df = price_data.get(ticker)

        if df is None or df.empty or len(df) < 130:
            continue

        try:
            close = df["Close"].dropna()
            volume = df["Volume"].dropna()

            if len(close) < 130 or len(volume) < 60:
                continue

            current = float(close.iloc[-1])
            current_price_map[code] = current

            ret_3m = float((current / close.iloc[-63] - 1) * 100)
            ret_6m = float((current / close.iloc[-126] - 1) * 100)
            ret_1y = float((current / close.iloc[-252] - 1) * 100) if len(close) >= 252 else np.nan

            rs_6m = ret_6m - bench_ret_6m

            avg_vol_20 = float(volume.tail(20).mean())
            avg_vol_60 = float(volume.tail(60).mean())
            volume_ratio = avg_vol_20 / avg_vol_60 if avg_vol_60 > 0 else np.nan

            high_52w = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
            distance_52w_high = (current / high_52w - 1) * 100 if high_52w > 0 else np.nan

            price_score = score_price_growth(ret_6m, ret_1y, rs_6m)
            volume_score = score_volume(volume_ratio)
            theme_text = detect_theme(code, name, industry)
            theme_score = score_theme(theme_text)

            base_score = price_score + volume_score + theme_score

            rows.append({
                "コード": code,
                "銘柄名": name,
                "市場": market,
                "業種": industry,
                "株価": round(current, 1),
                "3か月%": round(ret_3m, 2),
                "6か月%": round(ret_6m, 2),
                "1年%": round(ret_1y, 2) if not pd.isna(ret_1y) else np.nan,
                "日経比6か月%": round(rs_6m, 2),
                "52週高値乖離%": round(distance_52w_high, 2),
                "出来高倍率": round(volume_ratio, 2) if not pd.isna(volume_ratio) else np.nan,
                "価格成長点": round(price_score, 1),
                "出来高点": round(volume_score, 1),
                "テーマ": theme_text,
                "テーマ点": round(theme_score, 1),
                "一次点": round(base_score, 1),
            })

        except Exception as e:
            print(f"{ticker} 分析エラー:", e)

    if not rows:
        raise Exception("価格データから候補を作れませんでした。")

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values("一次点", ascending=False).reset_index(drop=True)

    shortlist = ranking.head(FUNDAMENTAL_LIMIT)["コード"].tolist()
    fundamentals_map = {}

    print(f"ファンダ取得：上位 {len(shortlist)} 銘柄")

    for i, code in enumerate(shortlist, start=1):
        ticker = code + ".T"
        print(f"fundamental {i}/{len(shortlist)} {ticker}")
        fundamentals_map[code] = get_fundamentals(ticker)
        time.sleep(0.2)

    for col in [
        "時価総額", "売上成長率%", "利益成長率%", "ROE%", "営業利益率%",
        "純利益率%", "D/E", "流動比率", "PER", "予想PER",
        "時価総額点", "売上成長点", "利益成長点", "収益性点", "財務点"
    ]:
        ranking[col] = np.nan

    ranking["時価総額点"] = 0.0
    ranking["売上成長点"] = 0.0
    ranking["利益成長点"] = 0.0
    ranking["収益性点"] = 0.0
    ranking["財務点"] = 0.0

    for idx, row in ranking.iterrows():
        code = row["コード"]
        f = fundamentals_map.get(code, {})

        market_cap = safe_float(f.get("marketCap"))
        revenue_growth = safe_float(f.get("revenueGrowth"))
        earnings_growth = safe_float(f.get("earningsGrowth"))
        roe = safe_float(f.get("returnOnEquity"))
        operating_margin = safe_float(f.get("operatingMargins"))
        profit_margin = safe_float(f.get("profitMargins"))
        debt_to_equity = safe_float(f.get("debtToEquity"))
        current_ratio = safe_float(f.get("currentRatio"))

        mc_score = score_market_cap(market_cap)
        revenue_score = score_growth(revenue_growth, 15)
        earnings_score = score_growth(earnings_growth, 25)
        profitability_score = score_profitability(roe, operating_margin, profit_margin)
        balance_score = score_balance(debt_to_equity, current_ratio)

        ranking.at[idx, "時価総額"] = market_cap
        ranking.at[idx, "売上成長率%"] = round(revenue_growth * 100, 2) if not pd.isna(revenue_growth) else np.nan
        ranking.at[idx, "利益成長率%"] = round(earnings_growth * 100, 2) if not pd.isna(earnings_growth) else np.nan
        ranking.at[idx, "ROE%"] = round(roe * 100, 2) if not pd.isna(roe) else np.nan
        ranking.at[idx, "営業利益率%"] = round(operating_margin * 100, 2) if not pd.isna(operating_margin) else np.nan
        ranking.at[idx, "純利益率%"] = round(profit_margin * 100, 2) if not pd.isna(profit_margin) else np.nan
        ranking.at[idx, "D/E"] = round(debt_to_equity, 2) if not pd.isna(debt_to_equity) else np.nan
        ranking.at[idx, "流動比率"] = round(current_ratio, 2) if not pd.isna(current_ratio) else np.nan
        ranking.at[idx, "PER"] = round(safe_float(f.get("trailingPE")), 1) if not pd.isna(safe_float(f.get("trailingPE"))) else np.nan
        ranking.at[idx, "予想PER"] = round(safe_float(f.get("forwardPE")), 1) if not pd.isna(safe_float(f.get("forwardPE"))) else np.nan

        ranking.at[idx, "時価総額点"] = round(mc_score, 1)
        ranking.at[idx, "売上成長点"] = round(revenue_score, 1)
        ranking.at[idx, "利益成長点"] = round(earnings_score, 1)
        ranking.at[idx, "収益性点"] = round(profitability_score, 1)
        ranking.at[idx, "財務点"] = round(balance_score, 1)

    ranking["総合点"] = (
        ranking["価格成長点"]
        + ranking["出来高点"]
        + ranking["テーマ点"]
        + ranking["時価総額点"]
        + ranking["売上成長点"]
        + ranking["利益成長点"]
        + ranking["収益性点"]
        + ranking["財務点"]
    ).round(1)

    ranking = ranking.sort_values("総合点", ascending=False).reset_index(drop=True)
    ranking["順位"] = ranking.index + 1

    ranking["分類"] = ranking.apply(
        lambda r: classify_candidate(
            r["総合点"],
            r["時価総額点"],
            r["テーマ点"],
            r["価格成長点"],
        ),
        axis=1
    )

    for h in [30, 90, 180, 365]:
        col = f"{h}日後リターン%"
        if col not in ranking.columns:
            ranking[col] = np.nan

    date = today_str()

    history_file = os.path.join(HISTORY_DIR, f"j_tenbagger_{date}.csv")
    latest_file = os.path.join(OUTPUT_DIR, "j_tenbagger_latest.csv")
    excel_file = os.path.join(OUTPUT_DIR, f"j_tenbagger_{date}.xlsx")

    ranking.to_csv(history_file, index=False, encoding="utf-8-sig")
    ranking.to_csv(latest_file, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        ranking.to_excel(writer, sheet_name="総合ランキング", index=False)
        ranking[ranking["分類"] == "本命候補"].to_excel(writer, sheet_name="本命候補", index=False)
        ranking[ranking["分類"] == "大化け候補"].to_excel(writer, sheet_name="大化け候補", index=False)
        ranking[ranking["分類"] == "急浮上候補"].to_excel(writer, sheet_name="急浮上候補", index=False)
        ranking[ranking["分類"] == "監視候補"].to_excel(writer, sheet_name="監視候補", index=False)

    update_old_returns(current_price_map)

    print(ranking.head(30).to_string(index=False))
    return ranking


def make_message(ranking):
    now = now_jst().strftime("%Y-%m-%d %H:%M")

    top = ranking.head(10)
    small_growth = ranking[
        (ranking["時価総額点"] >= 10)
        & (ranking["分類"].isin(["本命候補", "大化け候補", "急浮上候補", "監視候補"]))
    ].head(5)

    msg = f"【J-TENBAGGER Ver1.0】\n{now}\n\n"
    msg += "目的：3〜5年で5倍〜10倍候補の発掘\n"
    msg += "対象：プライム・スタンダード・グロース\n\n"

    msg += "◆ テンバガー候補 TOP10\n"

    for _, row in top.iterrows():
        market_cap_oku = row["時価総額"] / 100_000_000 if not pd.isna(row["時価総額"]) else np.nan

        msg += f"{int(row['順位'])}位 {row['銘柄名']}（{row['コード']}）{row['分類']}\n"
        msg += f"総合:{row['総合点']} / 市場:{row['市場']}\n"
        msg += f"テーマ:{row['テーマ'] if row['テーマ'] else 'なし'}\n"
        msg += f"6か月:{row['6か月%']}% / 1年:{row['1年%']}% / 出来高:{row['出来高倍率']}倍\n"
        msg += f"売上成長:{row['売上成長率%']}% / 利益成長:{row['利益成長率%']}%\n"
        msg += f"時価総額:{round(market_cap_oku, 1) if not pd.isna(market_cap_oku) else 'NA'}億円\n\n"

    msg += "◆ 小型・中型成長候補\n"

    if small_growth.empty:
        msg += "該当なし\n"
    else:
        for _, row in small_growth.iterrows():
            market_cap_oku = row["時価総額"] / 100_000_000 if not pd.isna(row["時価総額"]) else np.nan
            msg += f"・{row['銘柄名']}（{row['コード']}）{row['総合点']}点 / {round(market_cap_oku, 1) if not pd.isna(market_cap_oku) else 'NA'}億円\n"

    msg += "\n※これは短期売買ではなく、将来の主役候補を週次で監視するためのランキング。"
    return msg


if __name__ == "__main__":
    try:
        ranking = analyze()

        if ranking.empty:
            send_line_message("J-TENBAGGER分析に失敗しました。ランキングが空です。")
        else:
            message = make_message(ranking)

            print("=" * 80)
            print("LINE通知メッセージ")
            print("=" * 80)
            print(message)

            send_line_message(message)

    except Exception as e:
        error_msg = f"J-TENBAGGERでエラーが発生しました。\n\n{e}"
        print(error_msg)
        send_line_message(error_msg)
