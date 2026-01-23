#!/usr/bin/env python3
"""Test structured KB format generation using local LLM."""

import json
import os
from urllib.request import Request, urlopen

LLM_URL = os.environ.get("KB_LLM_URL", "http://tardis:9510/v1/chat/completions")

SCHEMA = {
    "type": "object",
    "required": ["summary", "framework", "formalism", "regime", "status", "key_results"],
    "properties": {
        "summary": {"type": "string", "description": "50 word max summary"},
        "framework": {
            "type": "object",
            "properties": {
                "algebra": {"type": "string", "enum": ["Cl(4,4)", "so(4,4)", "G2'", "sl(2,R)", "su(2,1)", "so(3,1)", "none"]},
                "dimension": {"type": "integer", "enum": [2, 4, 8, 16, None]},
                "field": {"type": "string", "enum": ["real", "complex", "split"]}
            }
        },
        "formalism": {
            "type": "array",
            "items": {"type": "string", "enum": [
                "clifford", "lie-algebra", "representation-theory", "group-theory",
                "thermodynamics", "hartree-fock", "bcs", "spectral-analysis",
                "svd", "heat-kernel", "polylog", "numerical"
            ]}
        },
        "regime": {
            "type": "object",
            "properties": {
                "temperature": {"type": "string", "enum": ["T=0", "finite-T", "T→∞", None]},
                "level": {"type": "string", "enum": ["tree", "one-loop", "exact", None]},
                "basis": {"type": "string", "enum": ["fock", "g2-weight", "z3-irrep", "complement", None]}
            }
        },
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "status": {
            "type": "object",
            "properties": {
                "verified": {"type": "string", "enum": ["proven", "numerical", "heuristic", "conjectured"]},
                "precision": {"type": "string"}
            }
        },
        "key_results": {
            "type": "object",
            "properties": {
                "dimensions": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "value": {"type": "integer"}}
                }},
                "eigenvalues": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"operator": {"type": "string"}, "values": {"type": "array"}}
                }},
                "states": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}, "property": {"type": "string"}}
                }}
            }
        },
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "interpretation": {
            "type": "object",
            "properties": {
                "algebraic": {"type": "string"},
                "physical": {"type": "string"}
            }
        }
    }
}

SYSTEM_PROMPT = """You extract structured data from physics findings. Output ONLY valid JSON.
CRITICAL RULES:
- Use null or [] for ANY field where the information is not EXPLICITLY stated
- Do NOT guess or infer - only extract what is directly written
- Copy numbers exactly as written (e.g., 2.83, not "approximately 3")
- algebra must be one of: Cl(4,4), so(4,4), G2', sl(2,R), su(2,1), so(3,1), or null
- verified must be one of: numerical, analytical, heuristic, or null"""

EXAMPLE_OUTPUT = """{
  "summary": "M² = 3I exactly; τ(M)² has eigenvalues {0,3,12}, not proportional to identity",
  "framework": {
    "algebra": "so(4,4)",
    "dimension": 16,
    "field": "real"
  },
  "formalism": ["lie-algebra", "spectral-analysis"],
  "regime": {
    "temperature": null,
    "level": null,
    "basis": null
  },
  "assumptions": [],
  "status": {
    "verified": "numerical",
    "precision": "exact"
  },
  "key_results": {
    "dimensions": [],
    "eigenvalues": [
      {"operator": "M²", "values": [3]},
      {"operator": "τ(M)²", "values": [0, 3, 12]}
    ],
    "norms": []
  },
  "open_questions": [],
  "interpretation": null
}"""

def convert_entry(content: str, finding_type: str) -> dict | None:
    prompt = f"""Extract structured JSON from this physics finding.

RULES:
1. Use null or empty [] if information is NOT explicitly stated - do NOT guess
2. Copy numbers exactly as written in the text
3. algebra field must be exactly one of: Cl(4,4), so(4,4), G2', sl(2,R), su(2,1), so(3,1), or null
4. verified field must be exactly one of: numerical, analytical, heuristic, or null
5. Only include eigenvalues/dimensions/norms that have explicit numerical values in the text

EXAMPLE OUTPUT:
{EXAMPLE_OUTPUT}

FINDING:
{content}

Extract to JSON. Use null for anything not explicitly stated."""

    req = Request(
        LLM_URL,
        data=json.dumps({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()
            return json.loads(content)
    except Exception as e:
        print(f"Error: {e}")
        return None


TEST_ENTRIES = [
    ("discovery", """M² = 3I BUT τ(M)² HAS EIGENVALUES {0, 3, 12}: THE KEY DISTINCTION

VERIFIED NUMERICALLY:
- M = Σ^{05}+Σ^{06}+Σ^{07} satisfies M² = 3I exactly
- τ(M)² has eigenvalues {0, 3, 12} — NOT proportional to identity
- This explains why M has simple spectrum (±√3, 8-fold) while τ(M) has complex spectrum (0, ±√3, ±2√3)

τ(M) COMPONENTS:
While M involves only (0,5), (0,6), (0,7) bivectors, τ(M) involves TWELVE bivectors:
  (0,5), (0,6), (0,7) with coefficient -0.5
  (1,4), (2,4), (3,4) with coefficient ±0.5
  (1,6), (1,7), (2,5), (2,7), (3,5), (3,6) with mixed coefficients

THE M² ∝ I CONDITION:
- This is a special algebraic property of M
- Physically: M couples all states with equal strength (isotropic)
- τ(M) couples states with DIFFERENT strengths (anisotropic)

WHAT THIS IMPLIES:
1. M is algebraically special (not just "one of three equivalent choices")
2. The M² ∝ I condition selects M from the triality orbit
3. This is the "equal coefficients" condition from composition algebra
4. BUT: the selection is within the triality-rotating eigenspace (M is NOT triality-fixed)"""),

    ("success", """VERIFIED: NO U(1) survives from oscillator sl(2,R)_L × sl(2,R)_R after M_HF condensation. SVD analysis of 6 oscillator generators (T³_L, T±_L, T³_R, T±_R) shows all 6 singular values > 0 for the commutator matrix [M, g_i]. No linear combination commutes with M. Even diagonal charges (T³_L, T³_R, N, B-L, Y) have nonzero commutators: ||[M, T³_L]|| = 2.83, ||[M, T³_R]|| = 4.00, ||[M, N]|| = 9.80, ||[M, Y]|| = 7.48. CONCLUSION: The oscillator sl(2,R)×sl(2,R) is COMPLETELY broken by M_HF = Σ^{05}+Σ^{06}+Σ^{07}. No U(1) hypercharge can come from this sector."""),

    ("failure", """CRITICAL REVIEW CORRECTIONS for triality-covariant basis work: (1) CENTRALIZER DIMENSION: Script m_squared_physical_implications.py incorrectly reported 6 preserved, 22 broken generators. CORRECT: 16-dimensional centralizer (as linear combinations), 12 truly broken generators. Individual bivectors that commute with M: only 6 (Lorentz SO(3,1) on indices {1,2,3,4}). But SVD of ad_M on 28-dim bivector space shows 16 null directions. (2) GROUND STATE ENERGY: Comparing H = λN + σM for M vs τ(M) with same σ is INVALID. The self-consistent σ = √3 only applies to M (from M² = 3I). For τ(M) with τ(M)² eigenvalues {0,3,12}, the gap equation would have different/no solution. (3) MULTIPLICITY BUG: np.isclose precision issue showed "multiplicity 0" instead of 8 for ±√3 eigenvalues."""),
]


if __name__ == "__main__":
    for finding_type, content in TEST_ENTRIES:
        print(f"\n{'='*60}")
        print(f"TYPE: {finding_type}")
        print(f"{'='*60}")
        print(f"ORIGINAL ({len(content)} chars):")
        print(content[:200] + "..." if len(content) > 200 else content)
        print(f"\n{'─'*60}")
        print("STRUCTURED:")
        result = convert_entry(content, finding_type)
        if result:
            print(json.dumps(result, indent=2))
        else:
            print("FAILED")
