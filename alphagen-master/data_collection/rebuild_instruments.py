"""
从 baostock 重建 CSI100/300/500 的 Qlib instruments 文件
格式：CODE\tSTART_DATE\tEND_DATE
"""
import baostock as bs
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

QLIB_INST_DIR = Path("/root/.qlib/qlib_data/cn_data_2024h1/instruments")

REBALANCE_DATES = []
for year in range(2010, 2025):
    for month in ["06-30", "12-31"]:
        REBALANCE_DATES.append(f"{year}-{month}")

INDEX_QUERIES = {
    "csi300": bs.query_hs300_stocks,
    "csi500": bs.query_zz500_stocks,
    "csi100": bs.query_sz50_stocks,   # baostock 用 sz50 近似 csi100
}

def bs_code_to_qlib(code: str) -> str:
    """sh.600000 -> SH600000"""
    parts = code.split(".")
    return parts[0].upper() + parts[1]

def fetch_index_history(query_fn) -> pd.DataFrame:
    """拉取某指数在所有调仓日的成分股快照"""
    records = []
    for date in REBALANCE_DATES:
        rs = query_fn(date=date)
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            records.append({"code": row[1], "snap_date": date})
    return pd.DataFrame(records)

def build_instruments_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    把快照列表转成 (code, start, end) 区间格式
    逻辑：若某股在相邻两期快照都存在 → 连续在池内
          某期出现但下期消失 → end = 下期调仓日前一天
    """
    snap_dates = sorted(df["snap_date"].unique())
    # 末尾加一个哨兵日期
    snap_dates_ext = snap_dates + ["2030-01-01"]

    results = []
    for code in df["code"].unique():
        code_dates = set(df[df["code"] == code]["snap_date"])
        in_pool = False
        seg_start = None
        for i, d in enumerate(snap_dates):
            next_d = snap_dates_ext[i + 1]
            if d in code_dates:
                if not in_pool:
                    seg_start = d
                    in_pool = True
                # 如果下一期不在池内或已到末尾，关闭区间
                if next_d not in code_dates or next_d == "2030-01-01":
                    # end = 下一期调仓日前一天
                    end_dt = datetime.strptime(next_d, "%Y-%m-%d") - timedelta(days=1)
                    results.append({
                        "code": bs_code_to_qlib(code),
                        "start": seg_start,
                        "end": end_dt.strftime("%Y-%m-%d")
                    })
                    in_pool = False
            else:
                in_pool = False
    return pd.DataFrame(results).sort_values(["code", "start"])

def write_instruments(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for _, row in df.iterrows():
            f.write(f"{row['code']}\t{row['start']}\t{row['end']}\n")
    print(f"Written {len(df)} records to {path}")

def main():
    lg = bs.login()
    print(f"baostock login: {lg.error_msg}")

    for name, query_fn in INDEX_QUERIES.items():
        print(f"\n=== Fetching {name} ===")
        raw = fetch_index_history(query_fn)
        print(f"  Raw snapshots: {len(raw)}")
        inst_df = build_instruments_df(raw)
        print(f"  Instruments intervals: {len(inst_df)}")
        out_path = QLIB_INST_DIR / f"{name}.txt"
        # 备份旧文件
        if out_path.exists():
            out_path.rename(str(out_path) + ".bak")
        write_instruments(inst_df, out_path)

    bs.logout()
    print("\n全部完成 ✓")

if __name__ == "__main__":
    main()
