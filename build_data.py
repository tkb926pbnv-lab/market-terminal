"""Generate data.json for the static GitHub Pages terminal. Usage: python build_data.py [out.json]"""
import json, sys
import core
out = sys.argv[1] if len(sys.argv) > 1 else "data.json"
d = core.get_data(max_age=0)
assert all(v["price"] for v in d["data"].values()), "missing prices"
with open(out, "w") as f: json.dump(d, f, separators=(",", ":"))
print("wrote", out, d["generated"], {k: round(v["price"], 2) for k, v in d["data"].items()})
