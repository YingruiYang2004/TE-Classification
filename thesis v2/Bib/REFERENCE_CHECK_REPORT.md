# Bibliography Verification Report — TE Classification Thesis

All 25 `\bibitem{}` entries verified against DOIs, arXiv abstract pages, and JOSS records.  
Fixes applied directly to `thesis/TE Classification.tex`.

---

## ❌ FABRICATED REFERENCES (DOI returns 404)

### 1. `khare2022automated` — **FABRICATED, CITED**
- **Was cited at**: Introduction L315 for "ML approaches focused on a single class (e.g. Helitrons)"
- **DOI** `10.1186/s13100-021-00163-y` → HTTP 404. Paper does not exist.
- **Fix applied**: Removed `(e.g.\ Helitrons \cite{khare2022automated})` from text; bibitem commented out.
- **Note**: If a Helitron ML example is wanted, a real candidate is Xiong et al. 2014 (PNAS) or search Mobile DNA for Helitron detection papers.

### 2. `mcnulty2020genomic` — **FABRICATED, UNCITED**
- **DOI** `10.7554/eLife.55235` → HTTP 404. Paper does not exist.
- **Fix applied**: Bibitem commented out.

---

## ❌ WRONG METADATA (real papers, wrong bibliographic data)

### 3. `selvaraju2016grad` — **WRONG arXiv ID + WRONG YEAR**
- **Thesis arXiv**: `1610.02055` → This is actually *"Places: An Image Database for Deep Scene Understanding"* (Bolei Zhou, Aditya Khosla, et al.), a completely different paper.
- **Correct arXiv**: `1610.02391` — *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* (Selvaraju, Cogswell, Das, Vedantam, Parikh, Batra)
- **Year**: ICCV 2017 (not 2016)
- **Cited in text?**: No — uncited. Text uses input-level gradients (Simonyan method), not Grad-CAM.
- **Fix applied**: URL corrected to `1610.02391`, year corrected to 2017; entry **commented out** with note that it should be cited if Grad-CAM was explicitly applied.
- **Potential home**: `\ref{ssec:saliency}` alongside `\cite{simonyan2014deep}` if Grad-CAM analysis is added.

### 4. `mcginnis2021umap` — **WRONG YEAR + WRONG AUTHOR**
- **Thesis**: year 2021, authors "McInnes L, Healy J, Melville J"
- **Reality** (JOSS DOI `10.21105/joss.00861`): year **2018**, authors **McInnes L, Healy J, Saul N, Großberger L**
  - "Melville J" is a co-author on the *longer arXiv preprint* `1802.03426`, not the JOSS paper whose DOI is cited.
- **Fix applied**: Year corrected to 2018; authors corrected to `McInnes L, Healy J, Saul N, Gro{\ss}berger L`.
- **Citation added** in text at L849: `UMAP projections \cite{mcginnis2021umap}`

### 5. `zhou2019graph` — **WRONG arXiv URL + WRONG YEAR**
- **Thesis**: year 2019, arXiv `1901.00596`, AI Open 1:57–70
- **Reality**:
  - `1901.00596` = *"A Comprehensive Survey on Graph Neural Networks"* (Wu, Pan, Chen, Long, Zhang, Yu) — published in **IEEE TNNLS**, not AI Open.
  - The paper with title "Graph neural networks: a review of methods and applications" by **Zhou J et al.** is arXiv `1812.08434`, published in **AI Open 2021**, pp. 57–75.
- **Fix applied**: URL corrected to `1812.08434`; year corrected to 2021; pages corrected to 57–75.
- **Citation added** in text at L329: `\cite{kipf2016semi,hamilton2017inductive,zhou2019graph}`

### 6. `kipf2016semi` — **YEAR (minor)**
- **Thesis**: year 2016; arXiv submitted Sep 2016 but published at **ICLR 2017**.
- **Fix applied**: Year corrected to 2017.

### 7. `kingma2014adam` — **YEAR (minor)**
- **Thesis**: year 2014; arXiv submitted Dec 2014 but published at **ICLR 2015**.
- **Fix applied**: Year corrected to 2015; entry **commented out** (uncited — AdamW is cited via `loshchilov2019decoupled`).
- **Potential home**: alongside `\cite{loshchilov2019decoupled}` at training section L403.

---

## ⚠️ CLAIM-SUPPORT ISSUE

### 8. `lecun2015deep` — cited for "dilated convolutions"
- **In-text claim** (L321): *"Dilated convolutions enlarge the receptive field exponentially with depth without an equivalent growth in parameters… \cite{lecun2015deep}"*
- **Problem**: LeCun, Bengio & Hinton 2015 (*Nature* 521) is a broad deep learning review and does **not** discuss dilated convolutions. The standard citation for dilated convolutions is Yu & Koltun 2016 (*Multi-Scale Context Aggregation by Dilated Convolutions*, arXiv 1511.07122).
- **Note**: LeCun 2015 is correctly cited at the start of the paragraph for CNNs generally.
- **Fix applied**: Removed the trailing `\cite{lecun2015deep}` from the dilated-convolution sentence. LeCun 2015 remains cited for the general CNN statement at L321.
- **Recommendation**: Add `\bibitem{yu2016multi}` (Yu & Koltun 2016, arXiv 1511.07122) and cite it for this claim if word count allows.

---

## ✅ UNCITED ENTRIES — CITATIONS ADDED TO TEXT

The following entries were in the bibliography but had no `\cite{}` in the text. Natural homes were found and citations added:

| Entry | Where added |
|---|---|
| `simonyan2014deep` | L803: `I computed input-level gradients (saliency maps) \cite{simonyan2014deep}` |
| `mcginnis2021umap` | L849: `UMAP projections \cite{mcginnis2021umap}` |
| `ward1963hierarchical` | L857: `using Ward linkage \cite{ward1963hierarchical}` |
| `ganin2015unsupervised` | L918: `domain-adversarial training \cite{ganin2015unsupervised}` |
| `finn2017model` | L918: `meta-learning over species \cite{finn2017model}` |
| `zhou2019graph` | L329: `(GNN) \cite{kipf2016semi,hamilton2017inductive,zhou2019graph}` |

---

## 📌 COMMENTED OUT — potential homes noted

These entries are genuinely uncited with no strong natural home in the current text. They are commented out with notes; restore and add citations if text is expanded.

| Entry | Reason commented | Potential home |
|---|---|---|
| `goodfellow2016deep` | Uncited; CNN intro already covered by `lecun2015deep`+`krizhevsky2012imagenet` | CNN intro L321 |
| `selvaraju2016grad` | Uncited; URL corrected (was wrong); Grad-CAM not used in text | `\ref{ssec:saliency}` if Grad-CAM added |
| `chen2020simple` | Uncited; SimCLR not mentioned (only SupCon) | v5 section alongside `khosla2020supervised` as self-supervised precursor |
| `rousseeuw1987silhouettes` | Uncited; silhouette scores not mentioned in body text | `\ref{ssec:clustering}` if ARI/silhouette table added |
| `gal2016dropout` | Uncited; MC Dropout in `\iffalse` block only | Uncomment if uncertainty quantification section is added |
| `devlin2019bert` | Uncited; BERT is NLP-specific — misleading for "DNA foundation models" | Replace with DNABERT (Ji et al. 2021, Genome Biology) if cited |
| `kingma2014adam` | Uncited (year fixed: 2014→2015) | Alongside `loshchilov2019decoupled` at L403 |

---

## ✅ CONFIRMED REAL (no fixes needed)

All metadata verified via DOI, arXiv, or JOSS:

| Key | Paper | Venue | Status |
|---|---|---|---|
| `sierra2024pantera` | Sierra & Durbin, *Mobile DNA* 15:13 | 2024 | ✓ |
| `lecun2015deep` | LeCun, Bengio, Hinton, *Nature* 521 | 2015 | ✓ |
| `hamilton2017inductive` | Hamilton, Ying, Leskovec (GraphSAGE) | NeurIPS 2017 | ✓ |
| `krizhevsky2012imagenet` | Krizhevsky, Sutskever, Hinton (AlexNet) | NeurIPS 2012 | ✓ |
| `wells2020field` | Wells & Feschotte, *Annu Rev Genet* 54 | 2020 | ✓ |
| `smit2015repeatmasker` | Smit, Hubley, Green (RepeatMasker website) | 2015 | ✓ |
| `simonyan2014deep` | Simonyan, Vedaldi, Zisserman (Saliency maps) | ICLR Workshop 2014 | ✓ |
| `khosla2020supervised` | Khosla et al. (SupCon) | NeurIPS 2020 | ✓ |
| `ward1963hierarchical` | Ward Jr JH, *J Am Stat Assoc* 58 | 1963 | ✓ |
| `finn2017model` | Finn, Abbeel, Levine (MAML) | ICML 2017 | ✓ |
| `ganin2015unsupervised` | Ganin & Lempitsky (DANN) | ICML 2015 | ✓ |
| `lin2017focal` | Lin et al. (Focal Loss) | ICCV 2017 | ✓ |
| `loshchilov2019decoupled` | Loshchilov & Hutter (AdamW) | ICLR 2019 | ✓ |
| `chen2020simple` | Chen et al. (SimCLR) | ICML 2020 | ✓ |
| `gal2016dropout` | Gal & Ghahramani (MC Dropout) | ICML 2016 | ✓ |
| `devlin2019bert` | Devlin et al. (BERT) | NAACL 2019 | ✓ |
| `goodfellow2016deep` | Goodfellow, Bengio, Courville textbook | MIT Press 2016 | ✓ |
| `rousseeuw1987silhouettes` | Rousseeuw, *J Comput Appl Math* 20 | 1987 | ✓ |
| `kingma2014adam` | Kingma & Ba (Adam) | ICLR 2015 | ✓ (year fixed) |

---

## Summary of all changes applied to `thesis/TE Classification.tex`

**In-text**:
1. Removed `(e.g.\ Helitrons \cite{khare2022automated})` — fabricated reference
2. Removed trailing `\cite{lecun2015deep}` from dilated-convolution claim (wrong paper for that claim)
3. Added `zhou2019graph` to GNN intro cite list
4. Added `\cite{simonyan2014deep}` at saliency maps
5. Added `\cite{mcginnis2021umap}` at UMAP projections
6. Added `\cite{ward1963hierarchical}` at Ward linkage
7. Added `\cite{ganin2015unsupervised}` and `\cite{finn2017model}` at domain-adaptation/meta-learning

**Bibliography**:
8. `kipf2016semi`: year 2016 → 2017
9. `zhou2019graph`: year 2019 → 2021; URL 1901.00596 → 1812.08434; pages 57–70 → 57–75
10. `selvaraju2016grad`: year 2016 → 2017; URL 1610.02055 → 1610.02391; entry commented out
11. `mcginnis2021umap`: year 2021 → 2018; authors "Melville J" → "Saul N, Großberger L"
12. `kingma2014adam`: year 2014 → 2015; entry commented out
13. Commented out with notes: `goodfellow2016deep`, `khare2022automated`, `mcnulty2020genomic`, `selvaraju2016grad`, `chen2020simple`, `rousseeuw1987silhouettes`, `gal2016dropout`, `devlin2019bert`, `kingma2014adam`
