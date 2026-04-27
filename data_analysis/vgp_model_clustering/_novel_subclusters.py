import pandas as pd

asgn = pd.read_csv('data_analysis/vgp_model_clustering/dna_subclusters_assignments.csv')
summ = pd.read_csv('data_analysis/vgp_model_clustering/dna_subclusters_summary.csv')

# Drop noise entries
asgn = asgn[~asgn['is_noise']].copy()

# Extract fine-grained superfamily from header e.g. "hAT_1-aAnoBae#DNA/hAT" -> "DNA/hAT"
asgn['sf_fine'] = asgn['header'].str.extract(r'#(.+)$')

# Build cluster_id: superfamily + subcluster_local
rows = []
for (sf, sc_local), grp in asgn.groupby(['superfamily', 'subcluster_local'], sort=False):
    cluster_id = f"{sf}_{sc_local}"
    counts = grp['sf_fine'].value_counts()
    total = counts.sum()
    dom_sf   = counts.index[0]
    dom_frac = counts.iloc[0] / total
    sec_sf   = counts.index[1] if len(counts) > 1 else ''
    sec_frac = counts.iloc[1] / total if len(counts) > 1 else 0.0
    rows.append({
        'cluster_id': cluster_id,
        'size': total,
        'dominant_superfamily': dom_sf,
        'dominant_fraction': round(dom_frac, 4),
        'second_superfamily': sec_sf,
        'second_fraction': round(sec_frac, 4),
    })

df = pd.DataFrame(rows)

# Filter: dominant < 80%
novel = df[df['dominant_fraction'] < 0.80].sort_values('size', ascending=False).reset_index(drop=True)

print(f"Total non-noise subclusters: {len(df)}")
print(f"Subclusters with dominant < 80%: {len(novel)}\n")

out_cols = ['cluster_id','size','dominant_superfamily','dominant_fraction',
            'second_superfamily','second_fraction']
print(novel[out_cols].to_string(index=False))

# Save CSV
novel[out_cols].to_csv('data_analysis/vgp_model_clustering/novel_subclusters.csv', index=False)
print("\nSaved: data_analysis/vgp_model_clustering/novel_subclusters.csv")

# LaTeX tabular
print("\n--- LaTeX ---")
print(r"\begin{tabular}{llrrrr}")
print(r"\toprule")
print(r"Cluster ID & Dominant SF & Size & Dom.\ Frac. & Second SF & 2nd Frac. \\")
print(r"\midrule")
for _, r in novel[out_cols].iterrows():
    dom_frac_str = f"{r['dominant_fraction']:.2f}"
    sec_frac_str = f"{r['second_fraction']:.2f}" if r['second_fraction'] > 0 else "--"
    sec_sf       = r['second_superfamily'] if r['second_superfamily'] else "--"
    cid  = r['cluster_id'].replace('_', r'\_')
    dsf  = str(r['dominant_superfamily']).replace('_', r'\_')
    ssf  = str(sec_sf).replace('_', r'\_')
    print(f"  {cid} & {dsf} & {r['size']} & {dom_frac_str} & {ssf} & {sec_frac_str} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
