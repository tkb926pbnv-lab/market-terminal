"""CLI mode: python cli.py [--watch SECONDS]"""
import sys, time, os
import core
G="\033[92m";R="\033[91m";Y="\033[93m";C="\033[96m";D="\033[2m";B="\033[1m";X="\033[0m"
def show():
    d = core.get_data(max_age=0)
    print(f"{B}{C}ZEUS MARKET TERMINAL{X}  {D}{d['generated']}{X}\n")
    print(f"{B}{'SYM':7}{'LAST':>12}{'CHG%':>8}{'RSI':>6}{'%B':>6}{'BIAS':>9}  {'BUY ZONE':>24}  {'SELL ZONE':>24}{X}")
    for s, r in d["data"].items():
        col = G if r["change_pct"] >= 0 else R
        bc = G if r["bias"]=="BULLISH" else R if r["bias"]=="BEARISH" else Y
        print(f"{s:7}{r['price']:>12,.2f}{col}{r['change_pct']:>+7.2f}%{X}{r['rsi']:>6.1f}{r['pct_b']:>6.2f}{bc}{r['bias']:>9}{X}  "
              f"{G}{r['buy_zone'][0]:>11,.2f}-{r['buy_zone'][1]:<11,.2f}{X}  {R}{r['sell_zone'][0]:>11,.2f}-{r['sell_zone'][1]:<11,.2f}{X}")
    print(f"\n{B}SIGNALS{X}")
    for s in ("^GSPC","^IXIC","^DJI"):
        r=d["data"][s]; print(f"{C}{s}{X} {r['action']}\n  {D}{r['explain']}{X}")
    print(f"\n{B}ERRORS / MISPRICINGS{X}")
    for f in d["errors"]:
        c = R if f["severity"]=="HIGH" else Y if f["severity"] in ("MED","WATCH") else D
        print(f"{c}[{f['severity']:5}] {f['type']:22} {f['symbol']:13} {f['value']}{X}\n          {D}{f['note']}{X}")
    print(f"\n{Y}{d['disclaimer']}{X}")
if __name__ == "__main__":
    if "--watch" in sys.argv:
        n = int(sys.argv[sys.argv.index("--watch")+1])
        while True: os.system("clear"); show(); time.sleep(n)
    else: show()
