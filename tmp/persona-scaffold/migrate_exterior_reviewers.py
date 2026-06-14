#!/usr/bin/env python3
"""Migrate exterior_algebra/reviewers.yaml from the old technical_domains schema
to the lisa/ada personas:/short_name/experts[].association schema (kb-bp4 follow-on,
claude-config-dev). Preserves dense association strings verbatim; drops 'Claude'
anti-pattern entries (expert-review reconstructs that inline) and the weak
gpu_hip_rocm 'Claude/MEDIUM' placeholder. am-rs-setup appends firmware-RE after."""
import yaml, collections

SRC = "/home/mcelrath/Projects/ai/exterior_algebra/reviewers.yaml"
OUT = "/home/mcelrath/Projects/ai/exterior_algebra/reviewers.yaml"

d = yaml.safe_load(open(SRC))
td = d["technical_domains"]

# domain key -> (short_name, trigger_paths)
MAP = {
    "transformer_attention_algebra": ("attention-algebra",
        ["**/*.py", "**/attention*.py", "**/*svd*.py", "**/analysis/**"]),
    "linear_algebra_svd": ("linear-algebra-svd",
        ["**/*svd*.py", "**/compress*.py", "**/*rank*.py", "**/analysis/**"]),
    "quantization_model_compression": ("quantization",
        ["**/*quant*.py", "**/*compress*.py", "scripts/*quant*"]),
    "amd_rocm_internals": ("amd-rocm-internals",
        ["scripts/*.hip", "scripts/*megakernel*", "scripts/*persistent*", "**/*.hip"]),
    "pcie_coherence_atomics": ("pcie-coherence",
        ["scripts/p2p_*.hip", "scripts/p2p_*.cpp", "**/P2P.md", "scripts/*atomic*"]),
    "gpu_collectives_dispatch": ("gpu-collectives",
        ["scripts/*megakernel*", "scripts/*dispatch*", "scripts/*moe*", "scripts/*collective*"]),
    "gpu_kernel_driver_os": ("gpu-kernel-driver",
        ["scripts/*.hip", "scripts/p2p_*.cpp", "scripts/*persistent*"]),
    # gpu_hip_rocm: DROPPED (weak 'Claude/MEDIUM' placeholder; amd-rocm-internals covers it)
}

def norm(s):
    return " ".join(str(s).split()) if s else s

personas = []
for dom, (short, paths) in MAP.items():
    if dom not in td:
        continue
    experts = []
    for tier in ("primary", "secondary"):
        for e in td[dom].get(tier, []):
            name = e["name"]
            if name.strip().lower() == "claude":
                continue
            assoc = norm(e.get("association", ""))
            use_for = norm(e.get("use_for", ""))
            if use_for:
                assoc = f"{assoc} — USE FOR: {use_for}"
            experts.append({"name": name, "association": assoc})
    if experts:
        personas.append({"name": dom.replace("_", " ").title(),
                         "short_name": short, "trigger_paths": paths,
                         "experts": experts})

# composite panels (new shape: personas: [short_names]); drop Claude
PANELS = {
    "default_review": ("Attention-algebra + approximation work",
                       ["attention-algebra", "linear-algebra-svd"]),
    "quantization_review": ("Quantization experiments and compression research",
                            ["quantization", "attention-algebra"]),
    "math_identity_review": ("Mathematical proofs and algebraic identities",
                             ["linear-algebra-svd", "attention-algebra"]),
    "multi_gpu_systems": ("Low-level GPU systems: PCIe coherence/atomics, ROCm/HIP "
                          "runtime, amdkfd/amdgpu driver, persistent megakernels, "
                          "GPU-initiated dispatch, multi-GPU collectives",
                          ["amd-rocm-internals", "pcie-coherence", "gpu-collectives",
                           "gpu-kernel-driver"]),
}
panels = {k: {"description": desc, "personas": ps} for k, (desc, ps) in PANELS.items()}

out = collections.OrderedDict()
out["personas"] = personas
out["composite_panels"] = panels
out["model_calibration"] = {"calibrated": "pending",
                            "note": "Run parent calibration protocol to populate."}

class Dumper(yaml.SafeDumper):
    pass
def str_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)
Dumper.add_representer(str, str_presenter)

header = ("# Reviewer personas for exterior_algebra (transformer-analysis: SVD compression,\n"
          "# v1 structure, layer-adaptive attention; + multi-GPU systems work).\n"
          "# Migrated from technical_domains -> personas:/short_name schema for uniform\n"
          "# persona-load (lisa/ada convention). firmware-reverse-engineering persona +\n"
          "# mes.md authored by am-rs-setup. Association strings activate expert vocabulary.\n\n")
with open(OUT, "w") as f:
    f.write(header)
    yaml.dump(dict(out), f, Dumper=Dumper, default_flow_style=False,
              sort_keys=False, width=10**9, allow_unicode=True)

d2 = yaml.safe_load(open(OUT))
print("migrated personas:", [p["short_name"] for p in d2["personas"]])
print("panels:", list(d2["composite_panels"].keys()))
print("expert counts:", {p["short_name"]: len(p["experts"]) for p in d2["personas"]})
