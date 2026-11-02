# How the CYP3A4 complexes were built

The structure entry for the OpenADMET CYP inhibition blind challenge predicts the
CYP3A4 ligand complex for 20 blind compounds. The submission is a zip of 20 PDB files,
one per compound, each holding a protein chain, a heme and a posed ligand. It is scored
by lDDT-PLI, a recovery rate over protein-ligand atom contacts, where higher is better;
the bundle this document describes scores 0.4565. This document is the construction: how an input is assembled, how the run is
packaged, how one pose is chosen out of many, how the submission file is cleaned, and
how we verify the thing we are about to upload is chemically the molecule we were asked
about.

Code blocks are from the files named, abridged where a line does not carry the idea.

## 1. The input

One YAML per compound. There are three chains, and the arrangement matters more than
anything else in this document:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MALYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGKVWGFYDGQQ...
  - ligand:
      id: B
      ccd: HEM
  - ligand:
      id: C
      smiles: 'O=C(NC[C@@H]1CCC2=CC=CC=C21)C1=CC=NC(N2C=CN=C2)=C1'
constraints:
  - pocket:
      binder: C
      contacts: [[A, 98], [A, 283], [A, 284], [A, 288]]
      max_distance: 8
templates:
  - cif: /tmp/boltz/1TQN.cif
```

**The heme is its own chain, declared by CCD code, not omitted and not merged into the
protein.** CYP3A4 without its heme is the wrong pocket: the porphyrin plane is one wall
of the binding site, and the metric counts protein-ligand contact pairs, so a ligand
placed where the heme should be scores contacts that cannot exist. Declaring it as
chain B lets the co-folding model place the cofactor and the ligand in the same pass.

**The ligand goes in as SMILES, not as a CCD code**, because the blind compounds have no
CCD entry, and because SMILES is how stereochemistry is specified. Section 5 is about
taking that seriously.

**The pocket constraint** keeps the binder within 8 Angstrom of four active-site
residues. These are construct numbering; the construct is offset by 21 from canonical
CYP3A4, so 98, 283, 284 and 288 are canonical 119, 304, 305 and 309.

**The template** is 1TQN, ligand-free CYP3A4 at 2.05 Angstrom, stripped of every HETATM
before use so that no ligand or heme atom from the template can leak in as a positional
hint:

```python
def strip_hetatm(src: Path, dst: Path) -> None:
    lines = [l for l in src.read_text().splitlines() if not l.startswith("HETATM")]
    dst.write_text("\n".join(lines) + "\n")

for t in templates_needed:
    strip_hetatm(find_ref(t), HERE / "templates" / f"{t}.cif")
    assert "HETATM" not in (HERE / "templates" / f"{t}.cif").read_text(), t
```

The YAML is assembled as text rather than dumped from a dict, so the file reads exactly
as above and diffs cleanly between arms:

```python
lines = [
    "version: 1",
    "sequences:",
    "  - protein:",
    "      id: A",
    f"      sequence: {SEQUENCE}",
    "  - ligand:",
    "      id: B",
    "      ccd: HEM",
    "  - ligand:",
    "      id: C",
    f"      smiles: '{smiles}'",
    "constraints:",
    "  - pocket:",
    "      binder: C",
    "      contacts: [[A, 98], [A, 283], [A, 284], [A, 288]]",
    "      max_distance: 8",
]
if info["template_pdb"]:
    lines += ["templates:", f"  - cif: /tmp/w/templates/{info['template_pdb']}.cif"]
(HERE / "inputs" / f"{case_id}.yaml").write_text("\n".join(lines) + "\n")
```

It then parses every file it just wrote, which catches a quoting error in a SMILES before
a GPU is billed for it:

```python
for cid in case_list:
    d = _yaml.safe_load((HERE / "inputs" / f"{cid}.yaml").read_text())
    assert d["sequences"][2]["ligand"]["smiles"]
```

## 2. The run

Boltz-2 (2.2.1), three diffusion samples per compound, on one `ml.g5.xlarge`:

```bash
boltz predict /tmp/boltz/inputs \
  --out_dir "$OUT/boltz" \
  --diffusion_samples 3 \
  --recycling_steps 3 \
  --sampling_steps 200 \
  --output_format pdb \
  --use_msa_server \
  --use_potentials $KERNELS \
  --override
```

`--use_potentials` turns on the steering potentials that enforce physical validity
during diffusion. Section 6 records what happens without them.

The alignment comes from the ColabFold MMseqs2 server, which Boltz calls itself under
`--use_msa_server`: `uniref.a3m` concatenated with
`bfd.mgnify30.metaeuk30.smag30.a3m`, 14,364 sequences for this construct. One
undocumented detail worth knowing if you plan an MSA experiment on this pipeline: every
alignment Boltz obtains this way is MMseqs2-filtered. `mmseqs2.py` defaults
`use_filter=True` and `main.py` calls `run_mmseqs2(..., use_env=True, use_pairing=False)`
without overriding it, so no command-line flag reaches the unfiltered alignment. To test
that you have to fetch the alignment out of band and pass it as a file.

The job script runs a one-case smoke test before the real run, because a Boltz
invocation can exit zero having written no predictions:

```bash
boltz predict /tmp/smoke_in --out_dir /tmp/smoke_out --diffusion_samples 1 \
  --recycling_steps 1 --sampling_steps 20 --output_format pdb --use_msa_server \
  --use_potentials $KERNELS --override 2>&1 | tail -12
NS=$(find /tmp/smoke_out -name "*_model_*.pdb" | wc -l)
[ "$NS" -ge 1 ] || { echo "FATAL: smoke test produced no pose, aborting"; exit 1; }
```

The launcher asserts the shape of the input bundle before uploading it. The AppleDouble
check is there because macOS `tar` writes `._`-prefixed sidecar files into archives, and
Boltz globs `inputs/*.yaml`, so it would have happily tried to fold twenty-two files of
resource-fork garbage:

```python
junk = sorted(n for n in names if pathlib.PurePath(n).name.startswith("._")
              or pathlib.PurePath(n).name == ".DS_Store")
assert not junk, f"AppleDouble/.DS_Store entries in bundle: {junk[:5]}"
```

The fix on the writing side is `COPYFILE_DISABLE=1 tar --exclude='._*' --exclude='.DS_Store'`.

## 3. Choosing one pose

Boltz writes one PDB and one confidence JSON per diffusion sample. The submitted pose is
the highest `confidence_score`:

```python
def pick_best(case_dir: Path):
    best = (None, -1.0)
    for p in sorted(case_dir.glob("confidence_*_model_*.json")):
        idx = int(p.stem.rsplit("_model_", 1)[1])
        s = json.loads(p.read_text()).get("confidence_score", -1.0)
        if s > best[1]:
            best = (idx, s)
    return best
```

In practice this always returns `model_0`, because Boltz pre-sorts its samples by
confidence. The function is written as a search anyway so that the rule is explicit in
the code rather than implicit in an output convention.

We tested whether anything better than rank-1 exists and the answer is no. Seventeen
selection signals, twelve confidence-derived and five geometric, over two independent
sets: nothing holds above noise and top-1 is exactly chance. Consensus selection is
worse than chance here, not neutral. A coordinate-medoid pick over ten samples, same
recipe otherwise, scored 0.4034 against 0.4565 for rank-1. Pose error on this target is
a displaced mode rather than scatter about the right answer, so moving a pick toward the
center of the sampled distribution moves it toward the center of the wrong mode.

## 4. Building the submission file

Raw Boltz output needs three fixes. Both the heme and the ligand come out as residue
name `LIG`, distinguished only by chain, so the heme has to be renamed. `CONECT` and
`TER` records are stripped because the scorer does not want them:

```python
def clean_pdb(text: str) -> str:
    out = []
    for line in text.splitlines():
        rec = line[:6]
        if rec in ("CONECT", "TER   ", "TER"):
            continue
        if rec == "END   " or line.strip() == "END":
            out.append("END"); continue
        if rec in ("ATOM  ", "HETATM"):
            if rec == "HETATM" and line[21] == "B":
                line = line[:17] + "HEM" + line[20:]
            out.append(line)
    return "\n".join(out) + "\n"
```

Filtering on **chain** and not on residue name is the whole trick. Anything that
selects the ligand by `resname LIG` picks up the heme as well, silently, and produces a
43-atom ligand that still scores.

Then the preflight, which is the part worth copying. Every assertion below is one we
once needed:

```python
assert len(payload) == 20, f"expected 20 poses, built {len(payload)}"
assert sorted(payload) == live_names, "file-name set differs from the live bundle"
for n, b in payload.items():
    t = b.decode()
    chains = {l[21] for l in t.splitlines() if l[:6] in ("ATOM  ", "HETATM")}
    res    = {l[17:20] for l in t.splitlines() if l[:6] == "HETATM"}
    assert chains == {"A", "B", "C"}, f"{n}: chains {sorted(chains)}"
    assert res == {"HEM", "LIG"}, f"{n}: HETATM resnames {sorted(res)}"
    assert "CONECT" not in t and "\nTER" not in t, f"{n}: CONECT/TER survived cleaning"
    nlig = sum(1 for l in t.splitlines() if l[:6] == "HETATM" and l[21] == "C")
    assert nlig > 0, f"{n}: no ligand atoms on chain C"

with zipfile.ZipFile(OUT_ZIP) as z:
    assert not [n for n in z.namelist()
                if n.startswith("__MACOSX") or Path(n).name.startswith("._")
                or Path(n).name == ".DS_Store"], "junk entries in zip"
```

The build writes a JSON record of which sample index was chosen for each compound,
its confidence, and whether the resulting bytes match the previous submission. Identify
a bundle by per-file CRC and that record, never by its filename. We have twice been
wrong about which file was live by trusting a name.

## 5. Verifying stereochemistry

The organizers added stereocenters to the test ligands partway through the challenge,
after poses had already been generated. Four of the twenty compounds carry
stereochemistry, so the shipped coordinates have to be checked against the current
reference SMILES rather than against the SMILES we happened to fold.

The direct check assigns bond orders from the reference as a template, reads CIP codes
back out of the 3D coordinates, and compares:

```python
ref = Chem.MolFromSmiles(reference_smiles)
Chem.AssignStereochemistry(ref, cleanIt=True, force=True)
want = {a.GetIdx(): a.GetProp('_CIPCode') for a in ref.GetAtoms() if a.HasProp('_CIPCode')}

lig = Chem.MolFromPDBBlock(ligand_block, sanitize=False, proximityBonding=True, removeHs=True)
mol = AllChem.AssignBondOrdersFromTemplate(ref, lig)      # PDB has no bond orders
Chem.AssignStereochemistryFrom3D(mol)
got = {a.GetIdx(): a.GetProp('_CIPCode') for a in mol.GetAtoms() if a.HasProp('_CIPCode')}
assert want == got
```

`AssignBondOrdersFromTemplate` is required because we strip `CONECT` records in section
4, so the PDB carries no bond orders and RDKit has to infer connectivity by proximity
before stereochemistry means anything.

**This check silently skips the case most likely to be wrong.** One of our four,
OCNT-2313038, is `C[C@@H]1CC[C@@H](NC(=O)...)CC1`: a 1,4-disubstituted cyclohexane. Its
two centers are not independent, the relationship is cis or trans across the ring rather
than R or S, and RDKit assigns no CIP code to either. So `want` and `got` are both empty
dictionaries, they compare equal, and the assertion passes without testing anything.

For that topology you have to measure the geometry. Fit the ring plane, then ask which
face each exocyclic substituent sits on:

```python
def relationship(mol, ring, sub_a, ring_a, sub_b, ring_b, conf_id=0):
    p = mol.GetConformer(conf_id).GetPositions()
    R = p[list(ring)]
    c = R.mean(0)
    n = np.linalg.svd(R - c)[2][-1]          # plane normal = smallest singular vector
    sa = np.dot(p[sub_a] - p[ring_a], n)     # signed offset from the ring plane
    sb = np.dot(p[sub_b] - p[ring_b], n)
    return "cis" if sa * sb > 0 else "trans"
```

Calibrate the answer by embedding the reference SMILES, which honors its own declared
ring stereochemistry, and take the majority over conformers:

```python
r3 = Chem.AddHs(Chem.MolFromSmiles(SMI))
cids = AllChem.EmbedMultipleConfs(r3, numConfs=12, randomSeed=0xC0FFEE)
AllChem.MMFFOptimizeMoleculeConfs(r3)
votes = Counter(relationship(Chem.RemoveHs(r3), ..., conf_id=c) for c in cids)
want = votes.most_common(1)[0][0]            # 12 of 12 trans for this compound
```

Our shipped pose is trans and correct, as are the other three. The point is that a
CIP-code comparison is the obvious way to run this audit and it passes the ring case
vacuously, so the audit has to branch on topology.

## 6. What moved the result, and what did not

Briefly, since the construction above is the subject of this document.

The one generator setting that matters is `--use_potentials`. The sampler knob
`--step_scale`, which is the per-step interpolation gain toward the denoised estimate
(default 1.5, meaning the sampler overshoots the mode by 50% per step), is monotonically
harmful when lowered: fold quality collapses below 1.25, and at 0.5 all 66 ligands in
that arm stopped being chemically valid molecules. The broken-ligand count across the
arms was 0, 0, 1, 9, 66.

Nothing on the input side moved it. Narrowing the alignment by pocket-residue identity
is significantly negative. A 31-times deeper unfiltered search, 450,870 sequences with
eleven times the pocket-column diversity, is a clean null on every endpoint. Close
homolog coverage is already saturated. The pocket constraint is not what keeps poses
shallow, which we established by scoring a genuinely unconstrained run and finding its
poses no deeper.

The measurement that explains the shape of that list: we ran an arm byte-identical to
its own control, same seed, and it scored 0.0143 lDDT-PLI away. That is the pipeline's
run-to-run nondeterminism, it is larger than every apparent gain we measured anywhere,
and it is larger than the gap between our entry and first place. An internal benchmark
of 22 crystal complexes cannot resolve the effect sizes that decide rank on a 20-compound
leaderboard, and most of the null results above are better read as evidence about the
instrument than about the lever.

## 7. Files

| what | where |
|---|---|
| YAML construction, template stripping | `analysis/boltz_template_ablation_20260930/build_inputs.py` |
| run script, smoke test | `analysis/boltz_live20_*/run_*.sh` |
| launcher and bundle preflight | `analysis/boltz_live20_*/launch_*.py` |
| pose selection, PDB cleaning, zip preflight | `analysis/boltz_live20_*/build_*_submission.py` |

Each ablation directory holds a `PROTOCOL.md` written before any output existed and a
`RESULT.md` recording the outcome against the predeclared gates. Where we overrode one
of our own gates, the override and its direction are recorded there, including one case
where the override was in the permissive direction and we said so.
