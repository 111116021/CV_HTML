"""
天氣學 HW1 — 探空圖分析
46734 馬公 2025/07/25 12Z

功能:
  1. 讀取探空資料 (原始 .edt.txt 或精簡 .csv 皆可)
  2. 由 T、RH 計算露點
  3. 依課堂程式指引, 以 Δz 逐層計算地面氣塊抬升路徑
       未飽和: 乾絕熱 Γd = g/Cp
       飽和  : 濕絕熱 Γm = g (1 + Lv rv/(Rd T)) / (Cp + Lv^2 rv ε/(Rd T^2))
  4. 找出 LCL、LFC、EL, 積分 CAPE、CIN
  5. 用 MetPy 畫斜溫圖 (溫度線、露點線、氣塊線、風標)

執行:  python hw1_skewt.py [資料檔]
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units

# ---------------- 參考公式常數 ----------------
g = 9.8          # m s^-2
Rd = 287.0       # J K^-1 kg^-1
Cp = 1004.0      # J K^-1 kg^-1
Lv = 2.5e6       # J kg^-1
eps = 0.622
GAMMA_D = g / Cp  # 乾絕熱遞減率 (K/m)
MS2KT = 1.944     # 1 m/s = 1.944 knots


# ---------------- 熱力函數 ----------------
def es_pa(t_c):
    """飽和水氣壓 (Pa), T 為攝氏"""
    return 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))


def mixing_ratio(e_pa, p_pa):
    """混合比 rv = ε e / (P - e)"""
    return eps * e_pa / (p_pa - e_pa)


def vapor_pressure_from_r(r, p_pa):
    """由混合比反推水氣壓 e = rP/(ε + r)"""
    return r * p_pa / (eps + r)


def dewpoint_from_e(e_pa):
    """es 公式的反函數, 回傳露點 (°C)"""
    x = np.log(e_pa / 611.2)
    return 243.5 * x / (17.67 - x)


def gamma_moist(t_k, p_pa):
    """濕絕熱遞減率 (K/m)"""
    rvs = mixing_ratio(es_pa(t_k - 273.15), p_pa)
    return g * (1 + Lv * rvs / (Rd * t_k)) / (Cp + Lv**2 * rvs * eps / (Rd * t_k**2))


# ---------------- 讀檔 ----------------
def read_sounding(path):
    """讀取探空資料, 回傳 z(gpm), p(hPa), T(°C), RH(%), WS(m/s), WD(deg)

    支援兩種格式:
      原始 .edt.txt : Time, Height, P, T, U, WS, WD, Ascent (逗號+空白分隔, 有表頭)
      精簡 .csv     : Height, P, T, U, WS, WD
    """
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [s.strip() for s in line.replace("\t", " ").split(",")]
            parts = [s for s in parts if s]
            try:
                vals = [float(s) for s in parts]
            except ValueError:
                continue  # 表頭、空行、註解
            if len(vals) == 8:        # 原始格式: 去掉 Time 與 Ascent
                rows.append(vals[1:7])
            elif len(vals) == 6:
                rows.append(vals)
    d = np.array(rows)
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5]


# ---------------- 氣塊抬升 ----------------
def lift_parcel(z, p_hpa, t0_c, td0_c):
    """依程式指引以 Δz 逐層抬升地面氣塊, 回傳氣塊溫度、露點 (°C) 與 LCL 資訊"""
    p = p_hpa * 100.0
    n = len(z)
    tp = np.empty(n)          # 氣塊溫度 (K)
    tdp = np.empty(n)         # 氣塊露點 (°C)
    tp[0] = t0_c + 273.15
    rv = mixing_ratio(es_pa(td0_c), p[0])   # 未飽和時混合比守恆
    tdp[0] = td0_c
    saturated = False
    lcl = None

    for k in range(n - 1):
        dz = z[k + 1] - z[k]
        if not saturated:
            tp[k + 1] = tp[k] - GAMMA_D * dz
            rvs = mixing_ratio(es_pa(tp[k + 1] - 273.15), p[k + 1])
            rh = rv / rvs * 100
            tdp[k + 1] = dewpoint_from_e(vapor_pressure_from_r(rv, p[k + 1]))
            if rh >= 100:
                # 在 k 與 k+1 之間以 (T - Td) = 0 內插出 LCL
                d0 = (tp[k] - 273.15) - tdp[k]
                d1 = (tp[k + 1] - 273.15) - tdp[k + 1]
                f = d0 / (d0 - d1)
                lcl = dict(z=z[k] + f * dz,
                           p=p_hpa[k] + f * (p_hpa[k + 1] - p_hpa[k]),
                           t=tp[k] - 273.15 + f * (tp[k + 1] - tp[k]))
                saturated = True
                tdp[k + 1] = tp[k + 1] - 273.15
        else:
            tp[k + 1] = tp[k] - gamma_moist(tp[k], p[k]) * dz
            tdp[k + 1] = tp[k + 1] - 273.15   # 飽和: e = es, rv = rvs
    return tp - 273.15, tdp, lcl


def crossing(z, p, diff, k):
    """diff 在 k 與 k+1 之間變號時, 線性內插出交點的 z 與 p"""
    f = diff[k] / (diff[k] - diff[k + 1])
    return z[k] + f * (z[k + 1] - z[k]), p[k] + f * (p[k + 1] - p[k])


def find_lfc_el(z, p, tp_c, te_c, lcl_z):
    """LFC: LCL 以上氣塊由冷轉暖的第一個交點; EL: 氣塊最後一次由暖轉冷的交點"""
    diff = tp_c - te_c
    up = [k for k in range(len(z) - 1) if diff[k] <= 0 < diff[k + 1] and z[k + 1] >= lcl_z]
    if not up:
        return None, None
    k_lfc = up[0]
    down = [k for k in range(k_lfc, len(z) - 1) if diff[k] > 0 >= diff[k + 1]]
    lfc = crossing(z, p, diff, k_lfc)
    el = crossing(z, p, diff, down[-1]) if down else (z[-1], p[-1])
    return lfc, el


def integrate_buoyancy(z, tp_c, te_c, z_bot, z_top):
    """∫ g (Tpar - Tenv)/Tenv dz, 以梯形法積分 z_bot ~ z_top (端點內插)"""
    b = g * (tp_c - te_c) / (te_c + 273.15)
    zz = np.concatenate(([z_bot], z[(z > z_bot) & (z < z_top)], [z_top]))
    bb = np.interp(zz, z, b)
    return np.trapezoid(bb, zz)


# ---------------- 主程式 ----------------
def main(path="46734-2025072512.csv"):
    z, p, t, rh, ws, wd = read_sounding(path)
    td = dewpoint_from_e(rh / 100 * es_pa(t))

    tp, tdp, lcl = lift_parcel(z, p, t[0], td[0])
    (lfc_z, lfc_p), (el_z, el_p) = find_lfc_el(z, p, tp, t, lcl["z"])
    cape = integrate_buoyancy(z, tp, t, lfc_z, el_z)
    cin = integrate_buoyancy(z, tp, t, z[0], lfc_z)

    print(f"資料筆數 {len(z)}, 地面 {p[0]} hPa / {z[0]:.0f} m, "
          f"T={t[0]} °C, Td={td[0]:.2f} °C, RH={rh[0]}%")
    print(f"LCL : {lcl['p']:7.2f} hPa  ({lcl['z']:6.0f} m,  T={lcl['t']:.2f} °C)")
    print(f"LFC : {lfc_p:7.2f} hPa  ({lfc_z:6.0f} m)")
    print(f"EL  : {el_p:7.2f} hPa  ({el_z:6.0f} m)")
    print(f"CAPE: {cape:8.2f} J/kg (m²/s²)")
    print(f"CIN : {cin:8.2f} J/kg (m²/s²)")

    # ---- 紙本用: 標準層資料 (氣塊線、風標) ----
    print("\n標準層資料 (紙本繪圖用)")
    print(" P(hPa)  高度(m)   T(°C)  Td(°C)  氣塊T(°C)  WD(°)  WS(m/s)  WS(kt)")
    for lev in [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100]:
        if lev > p[0] or lev < p[-1]:
            print(f" {lev:5d}   (低於地面 {p[0]} hPa, 無資料)" if lev > p[0]
                  else f" {lev:5d}   (超出探空頂)")
            continue
        i = lambda arr: np.interp(np.log(lev), np.log(p[::-1]), arr[::-1])
        # 風向用 u/v 內插避免 360/0 跳躍
        u_ = -ws * np.sin(np.radians(wd))
        v_ = -ws * np.cos(np.radians(wd))
        ui, vi = i(u_), i(v_)
        spd = np.hypot(ui, vi)
        dirn = np.degrees(np.arctan2(-ui, -vi)) % 360
        print(f" {lev:5d}  {i(z):7.0f}  {i(t):6.1f}  {i(td):6.1f}  {i(tp):8.1f}  "
              f"{dirn:5.0f}  {spd:6.1f}  {spd * MS2KT:6.1f}")

    # ---- 繪圖 ----
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Microsoft JhengHei",
                                       "Noto Sans CJK TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    fig = plt.figure(figsize=(10, 10))
    skew = SkewT(fig, rotation=45)
    P = p * units.hPa
    skew.plot(P, t * units.degC, color="b", lw=1.6, label="環境溫度 T")  # 依課程範例: T 藍
    skew.plot(P, td * units.degC, color="r", lw=1.6, label="環境露點 Td")  # Td 紅
    skew.plot(P, tp * units.degC, color="k", lw=1.8, label="氣塊溫度")
    skew.plot(P, tdp * units.degC, color="k", lw=1.2, ls="--", label="氣塊露點")

    # CAPE / CIN 著色
    skew.shade_cape(P, t * units.degC, tp * units.degC)
    skew.shade_cin(P, t * units.degC, tp * units.degC)

    # 風標 (knots), 含所有標準層並補中間層, 畫在圖右側
    u = -ws * np.sin(np.radians(wd)) * MS2KT
    v = -ws * np.cos(np.radians(wd)) * MS2KT
    targets = [950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450,
               400, 350, 300, 250, 200, 175, 150, 125, 100]
    idx = np.unique([np.argmin(np.abs(p - lv)) for lv in targets if p[-1] <= lv <= p[0]])
    skew.plot_barbs(P[idx], u[idx] * units.knots, v[idx] * units.knots, xloc=1.1)

    skew.plot_dry_adiabats(t0=np.arange(-20, 151, 10) * units.degC, lw=0.8)
    skew.plot_moist_adiabats(t0=np.arange(-20, 46, 5) * units.degC, lw=0.8)
    skew.plot_mixing_lines(pressure=np.arange(100, 1001, 20) * units.hPa, lw=0.8)

    # LCL / LFC / EL 標記
    for name, pv, col in [("LCL", lcl["p"], "g"), ("LFC", lfc_p, "m"), ("EL", el_p, "c")]:
        skew.ax.axhline(pv, color=col, ls=":", lw=1.2)
        skew.ax.text(0.98, pv, f"{name} {pv:.0f} hPa ", color=col, fontsize=10,
                     ha="right", va="bottom", transform=skew.ax.get_yaxis_transform())

    skew.ax.set_ylim(1000, 100)
    skew.ax.set_xlim(-30, 40)
    skew.ax.set_title("46734 馬公  2025/07/25 12Z", fontsize=14)
    skew.ax.set_xlabel("Temperature (°C)")
    skew.ax.set_ylabel("Pressure (hPa)")
    skew.ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    # 風標竿子
    box = skew.ax.get_position()
    ax2 = fig.add_axes([box.x1 * 1.07, box.y0, 0, box.height])
    ax2.set_yticks([])
    ax2.set_xticks([])

    fig.text(box.x1 * 1.12, box.y0 + box.height * 0.75,
             f"LCL: {lcl['p']:.2f} hPa\nLFC: {lfc_p:.2f} hPa\nEL: {el_p:.2f} hPa\n"
             f"CAPE: {cape:.2f} m$^2$/s$^2$\nCIN: {cin:.2f} m$^2$/s$^2$",
             fontsize=12, va="top")

    fig.savefig("skewt_46734_2025072512.png", dpi=150, bbox_inches="tight")
    print("\n已輸出 skewt_46734_2025072512.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
