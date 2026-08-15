"""Optimized Neo4j hierarchical graph builder using batch UNWIND Cypher queries."""
import os, sys, time, logging
sys.path.append(os.path.join(os.path.dirname(__file__)))
import pandas as pd
from py2neo import Graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "platform_data", "csv", "local", "1")
COMM = os.path.join(DATA, "community", "single")
OVERALL = os.path.join(DATA, "community", "embedding_overall")
SKIP = {"ALARM", "CONTROLLER"}
BATCH = 5000

def load_features():
    with open(os.path.join(BASE, "local_used_feature.txt")) as f:
        return [l.strip() for l in f if l.strip()]

def discover_devs():
    return [f[6:-4] for f in sorted(os.listdir(DATA))
            if f.startswith("ipraw_") and f.endswith(".csv") and f[6:-4] not in SKIP]

def discover_perspectives():
    return [d for d in sorted(os.listdir(COMM)) if d.startswith("embedding_") and os.path.isdir(os.path.join(COMM, d))]

def rel_type(feat):
    return "Has_" + feat.replace("-", "_").replace(".", "_")

def clear_db(g):
    log.info("Clearing Neo4j...")
    g.run("MATCH (n) DETACH DELETE n")

def build_layer1(g, devs, features):
    log.info("=== Layer 1 ===")
    t0 = time.time()
    stats = {}
    for dev in devs:
        path = os.path.join(DATA, f"ipraw_{dev}.csv")
        if not os.path.exists(path):
            continue
        hdr = pd.read_csv(path, nrows=0)
        feats = [f for f in features if f in hdr.columns]
        df = pd.read_csv(path, usecols=["ip"] + feats)
        for feat in feats:
            df[feat] = df[feat].fillna("").astype(str).str.strip()
        df["ip"] = df["ip"].astype(str).str.strip()
        n = len(df)
        td = time.time()
        log.info(f"[L1] {dev}: {n} devices, {len(feats)} features")

        # MERGE Device nodes
        rows = [{"ip": r["ip"], "device_type": dev} for _, r in df.iterrows()]
        for i in range(0, len(rows), BATCH):
            g.run("UNWIND $rows AS r MERGE (d:Device {ip:r.ip, device_type:r.device_type})", rows=rows[i:i+BATCH])

        # Per-feature: MERGE Feature nodes + relationships
        total_rels = 0
        for feat in feats:
            fr = [{"ip": r["ip"], "value": r[feat]} for _, r in df.iterrows() if r[feat]]
            total_rels += len(fr)
            rt = rel_type(feat)
            for i in range(0, len(fr), BATCH):
                g.run(f"UNWIND $rows AS r MATCH (d:Device {{ip:r.ip}}) "
                      f"MERGE (f:Feature {{feature_name:$feat, value:r.value}}) "
                      f"MERGE (d)-[:{rt}]-(f)", rows=fr[i:i+BATCH], feat=feat)

        el = time.time() - td
        fc = g.run("MATCH (f:Feature) WHERE f.feature_name IN $feats RETURN count(DISTINCT f) AS c", feats=feats).data()[0]["c"]
        stats[dev] = {"devices": n, "features": fc, "rels": total_rels, "time": el}
        log.info(f"[L1] {dev} done: {n} dev, {fc} feat, {total_rels} rels, {el:.1f}s")

    tt = time.time() - t0
    _print_table("Layer 1", ["Device Type","Devices","Features","Rels","Time(s)"],
                 [(k, v["devices"], v["features"], v["rels"], f'{v["time"]:.1f}') for k,v in stats.items()],
                 [sum(v["devices"] for v in stats.values()), sum(v["features"] for v in stats.values()),
                  sum(v["rels"] for v in stats.values()), f'{tt:.1f}'])
    return stats, tt

def build_layer2(g, devs, perspectives):
    log.info("=== Layer 2 ===")
    t0 = time.time()
    stats = {}
    cc = 0
    for persp in perspectives:
        pn = persp[11:]
        pdir = os.path.join(COMM, persp)
        tp = time.time()
        vc = oc = ec = 0
        for dev in devs:
            path = os.path.join(pdir, f"ipraw_{dev}_embedding_{pn}_pca.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, usecols=["ip", "cluster"])
            df["ip"] = df["ip"].astype(str).str.strip()
            df["cluster"] = df["cluster"].astype(int)
            dv = df[df["cluster"] != -1]
            do = df[df["cluster"] == -1]
            cr, er = [], []
            for hl in sorted(dv["cluster"].unique()):
                cid = cc; cc += 1
                cr.append({"cluster_id": cid, "perspective": pn, "device_type": dev, "hdbscan_label": int(hl), "is_outlier": False})
                for ip in dv[dv["cluster"]==hl]["ip"]:
                    er.append({"ip": ip, "cluster_id": cid})
            for ip in do["ip"]:
                cid = cc; cc += 1
                cr.append({"cluster_id": cid, "perspective": pn, "device_type": dev, "hdbscan_label": -1, "is_outlier": True})
                er.append({"ip": ip, "cluster_id": cid})
            for i in range(0, len(cr), BATCH):
                g.run("UNWIND $rows AS r MERGE (c:Cluster {cluster_id:r.cluster_id, perspective:r.perspective, device_type:r.device_type}) SET c.hdbscan_label=r.hdbscan_label, c.is_outlier=r.is_outlier", rows=cr[i:i+BATCH])
            for i in range(0, len(er), BATCH):
                g.run("UNWIND $rows AS r MATCH (d:Device {ip:r.ip}) MATCH (c:Cluster {cluster_id:r.cluster_id, perspective:$pn}) MERGE (d)-[:IN_COMMUNITY {perspective:$pn}]->(c)", rows=er[i:i+BATCH], pn=pn)
            vc += dv["cluster"].nunique(); oc += len(do); ec += len(er)
        el = time.time() - tp
        stats[pn] = {"valid": vc, "outlier": oc, "total": vc+oc, "edges": ec, "time": el}
        log.info(f"[L2] {pn}: {vc}+{oc}={vc+oc} clusters, {ec} edges, {el:.1f}s")
    tt = time.time() - t0
    _print_table("Layer 2", ["Perspective","Valid","Outliers","Total","Edges","Time(s)"],
                 [(k, v["valid"], v["outlier"], v["total"], v["edges"], f'{v["time"]:.1f}') for k,v in stats.items()],
                 [sum(v["valid"] for v in stats.values()), sum(v["outlier"] for v in stats.values()),
                  sum(v["total"] for v in stats.values()), sum(v["edges"] for v in stats.values()), f'{tt:.1f}'])
    return stats, tt

def build_layer3(g, devs):
    log.info("=== Layer 3 ===")
    t0 = time.time()
    stats = {}
    cc = 0
    for dev in devs:
        path = os.path.join(OVERALL, f"ipraw_{dev}_embedding_overall_pca.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=["ip", "cluster"])
        df["ip"] = df["ip"].astype(str).str.strip()
        df["cluster"] = df["cluster"].astype(int)
        dv = df[df["cluster"] != -1]
        do = df[df["cluster"] == -1]
        cr, er = [], []
        for hl in sorted(dv["cluster"].unique()):
            cid = cc; cc += 1
            cr.append({"cluster_id": cid, "perspective": "comprehensive", "device_type": dev, "hdbscan_label": int(hl), "is_outlier": False})
            for ip in dv[dv["cluster"]==hl]["ip"]:
                er.append({"ip": ip, "cluster_id": cid})
        for ip in do["ip"]:
            cid = cc; cc += 1
            cr.append({"cluster_id": cid, "perspective": "comprehensive", "device_type": dev, "hdbscan_label": -1, "is_outlier": True})
            er.append({"ip": ip, "cluster_id": cid})
        for i in range(0, len(cr), BATCH):
            g.run("UNWIND $rows AS r MERGE (c:Cluster {cluster_id:r.cluster_id, perspective:r.perspective, device_type:r.device_type}) SET c.hdbscan_label=r.hdbscan_label, c.is_outlier=r.is_outlier", rows=cr[i:i+BATCH])
        for i in range(0, len(er), BATCH):
            g.run("UNWIND $rows AS r MATCH (d:Device {ip:r.ip}) MATCH (c:Cluster {cluster_id:r.cluster_id, perspective:'comprehensive'}) MERGE (d)-[:IN_COMMUNITY {perspective:'comprehensive'}]->(c)", rows=er[i:i+BATCH])
        nv = dv["cluster"].nunique(); no = len(do)
        stats[dev] = {"valid": nv, "outlier": no, "total": nv+no, "edges": len(er)}
        log.info(f"[L3] {dev}: {nv}+{no}={nv+no} clusters, {len(er)} edges")
    tt = time.time() - t0
    _print_table("Layer 3", ["Device Type","Valid","Outliers","Total","Edges"],
                 [(k, v["valid"], v["outlier"], v["total"], v["edges"]) for k,v in stats.items()],
                 [sum(v["valid"] for v in stats.values()), sum(v["outlier"] for v in stats.values()),
                  sum(v["total"] for v in stats.values()), sum(v["edges"] for v in stats.values())])
    return stats, tt

def _print_table(title, headers, rows, totals):
    print(f"\n{'='*80}\n{title} Summary\n{'='*80}")
    fmt = "  ".join(f"{{{i}:<{w}}}" for i,w in enumerate([22,10,10,12,12,10][:len(headers)]))
    print(fmt.format(*headers))
    print("-"*80)
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))
    print("-"*80)
    print(fmt.format("TOTAL", *[str(x) for x in totals]))
    print("="*80 + "\n")

def print_final(g):
    print(f"\n{'='*60}\nFinal Neo4j Statistics\n{'='*60}")
    for r in g.run("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c ORDER BY c DESC").data():
        print(f"  {r['l']}: {r['c']:,}")
    tc = g.run("MATCH ()-[r]->() RETURN count(r) AS c").data()[0]["c"]
    print(f"  Total relationships: {tc:,}")
    print(f"\n  Cluster distribution by perspective:")
    for r in g.run("MATCH (c:Cluster) RETURN c.perspective AS p, count(*) AS cnt ORDER BY cnt DESC").data():
        print(f"    {r['p']}: {r['cnt']:,}")
    print("="*60)

if __name__ == "__main__":
    g = Graph(NEO4J_URL := "neo4j://localhost:7687", auth=(NEO4J_USER := "neo4j", NEO4J_PASS := "avs01046"))
    feats = load_features()
    devs = discover_devs()
    persps = discover_perspectives()
    log.info(f"Devices: {devs}\nPerspectives: {persps}\nFeatures: {len(feats)}")
    clear_db(g)
    _, t1 = build_layer1(g, devs, feats)
    _, t2 = build_layer2(g, devs, persps)
    _, t3 = build_layer3(g, devs)
    print(f"\nTotal build time: Layer1={t1:.1f}s  Layer2={t2:.1f}s  Layer3={t3:.1f}s  Grand={t1+t2+t3:.1f}s")
    print_final(g)
    log.info("Done!")
